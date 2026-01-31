"""
Generation utilities for ReForm rollout data collection.

This module handles the generation of autoformalization samples with
multi-step reward computation and dynamic filtering.
"""

from typing import List
import asyncio
import re
import copy
import json
import os
import os.path as osp
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample

from slime.rollout.rm_hub import async_rm, batched_async_rm

from lean_plugins.format_check import check_output_format

__all__ = ["lean_generate_rollout"]

CONFIGS = {
    "max_retries": 3,
    "search_concurrency": 256,
}

# Prompt template for autoformalization
PROMPT = """Think step by step to translate the mathematical problem in natural language to Lean 4, and verify the consistency.\n{informal_statement}\n"""


def convert_samples_to_data(samples: list[Sample]) -> list[dict]:
    """Convert samples to serializable dictionaries."""
    return [sample.to_dict() for sample in samples]


def save_info_and_data(args, rollout_id: int, info: dict, data: List[List[dict]]):
    """Save rollout information and data to disk."""
    save_path = osp.join(args.save, "rollout_data")
    os.makedirs(save_path, exist_ok=True)

    with open(osp.join(save_path, "rollout_info.jsonl"), "a") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")

    with open(osp.join(save_path, f"rollout_{rollout_id}.json"), "a") as f:
        json.dump(data, f, ensure_ascii=False)


class GenerateState(metaclass=SingletonMeta):
    """Global state for the generation process."""

    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        print(f"Rollout concurrency: {args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine}", flush=True)
        
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
            presence_penalty=1.1,
        )
        self.reset()

    def reset(self):
        """Reset state for a new rollout."""
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]):
        """Submit generation tasks for a batch of samples."""
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    """
    Generate a response for a single sample.
    
    Args:
        args: Training arguments.
        sample: Sample to generate for.
        sampling_params: Sampling parameters for generation.
        evaluation: Whether this is for evaluation.
        
    Returns:
        Updated sample with generated response.
    """
    assert not args.partial_rollout, "Partial rollout is not supported for this function."
    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status in [Sample.Status.PENDING, Sample.Status.ABORTED], \
        f"Sample status is {sample.status}"

    if len(sample.response) > 0:
        response_token_ids = state.tokenizer(sample.response, add_special_tokens=False)["input_ids"]
        sampling_params["max_new_tokens"] -= len(response_token_ids)

    assert sampling_params["max_new_tokens"] >= 0, \
        f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare input with chat template
    messages = [{"role": "user", "content": PROMPT.format(informal_statement=sample.prompt)}]
    prompt = state.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    sampling_params['max_new_tokens'] = sampling_params['max_new_tokens'] - len(input_token_ids) - 64
    
    payload = {
        "input_ids": input_token_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    output = await post(url, payload, use_http2=args.use_http2)

    # Extract response tokens and log probabilities
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens = []
        new_response_log_probs = []

    # Update sample
    sample.tokens = input_token_ids + new_response_tokens
    sample.response_length = len(new_response_tokens)
    sample.response = output["text"]
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def generate_and_rm(
    args,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False
) -> Sample:
    """Generate response and compute reward for a single sample."""
    if sample.status in [Sample.Status.COMPLETED, Sample.Status.TRUNCATED]:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params.copy(), evaluation=evaluation)

    if sample.status == Sample.Status.ABORTED:
        return sample

    if args.group_rm:
        return sample

    sample.reward = await async_rm(args, sample, evaluation=evaluation)
    return sample


async def generate_and_rm_group(
    args,
    group: list[Sample],
    sampling_params: dict,
    evaluation: bool = False
) -> list[Sample]:
    """Generate responses and compute rewards for a group of samples."""
    state = GenerateState(args)

    if state.aborted:
        return group

    group = await asyncio.gather(
        *[generate_and_rm(args, sample, sampling_params.copy(), evaluation=evaluation) for sample in group]
    )

    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group


async def abort(args, rollout_id: int) -> list[Sample]:
    """Abort ongoing generation and collect partial samples."""
    aborted_samples = []
    state = GenerateState(args)
    
    assert not state.aborted
    state.aborted = True
    
    response = await get(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers",
        use_http2=args.use_http2
    )

    # Abort all ongoing requests
    for url in response["urls"]:
        print(f"Abort request for {url}", flush=True)
        await post(f"{url}/abort_request", {"abort_all": True}, use_http2=False)

    # Wait for pending tasks to finish
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples += group
            count += len(group)

    if args.partial_rollout:
        print(f"Collected {count} partial samples into the data buffer", flush=True)

    return aborted_samples


async def generate_rollout_async(
    args,
    rollout_id: int,
    data_source
) -> tuple[list[list[Sample]], list[Sample]]:
    """
    Generate rollout data asynchronously.

    Args:
        args: Training arguments.
        rollout_id: ID for this rollout.
        data_source: Data source callable to fetch samples.

    Returns:
        Tuple of (completed_samples, aborted_samples).
    """
    assert args.rollout_global_dataset
    state = GenerateState(args)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )

    target_data_size = args.rollout_batch_size

    dapo_info = {
        "rollout_id": rollout_id,
        "reward": [],
        "truncated": [],
    }
    sampled_data = []
    data = []
    do_print = True
    
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
            group: list[Sample] = task.result()

            sampled_data.append(convert_samples_to_data(group))
            for sample_group in group:
                dapo_info['reward'].append(sample_group.reward)
                dapo_info['truncated'].append(1 if sample_group.status == Sample.Status.TRUNCATED else 0)

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            
            if dynamic_filter is not None and not dynamic_filter(args, group):
                print(f"Dynamically filtered out group, reward={group[0].reward} | count={len(group)} | current={len(data)}")
                state.remaining_batch_size -= 1
                continue

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    print(
        f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
        f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
        flush=True,
    )

    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0].index)

    save_info_and_data(args, rollout_id, dapo_info, sampled_data)
    state.reset()
    
    return data, aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args, rollout_id: int) -> tuple[dict, list]:
    """Run evaluation rollout on all configured datasets."""
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        results.update(await eval_rollout_single_dataset(args, rollout_id, name, path))
    return results, []


async def eval_rollout_single_dataset(
    args,
    rollout_id: int,
    name: str,
    path: str
) -> dict:
    """
    Run evaluation on a single dataset.

    Args:
        args: Training arguments.
        rollout_id: ID for this rollout.
        name: Dataset name.
        path: Path to the dataset.

    Returns:
        Dictionary of evaluation results.
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    if name not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[name] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
        )
    dataset = EVAL_PROMPT_DATASET[name]

    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params.copy(),
                    evaluation=True,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Eval generation", disable=not do_print)
    
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            print([sample.prompt + sample.response], sample.reward, flush=True)
            do_print = False
        data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)
    
    # Save evaluation data
    save_eval = convert_samples_to_data(data)
    save_eval_path = osp.join(args.save, "eval_data")
    os.makedirs(save_eval_path, exist_ok=True)

    with open(osp.join(save_eval_path, f"eval_rollout_{rollout_id}.jsonl"), "a") as f:
        for item in save_eval:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    reward_key = args.reward_key or args.eval_reward_key
    eval_result = {}
    
    for item in data:
        data_source = item.metadata['data_source']
        if data_source not in eval_result:
            eval_result[data_source] = {"rewards": [], "truncated": []}
            eval_result[data_source + "_verify"] = {"rewards": []}

        eval_result[data_source]["rewards"].append(
            item.reward[0] if not reward_key else item.reward[reward_key][0]
        )
        eval_result[data_source + "_verify"]["rewards"].append(
            item.reward[-1] if not reward_key else item.reward[reward_key][-1]
        )
        eval_result[data_source]["truncated"].append(item.status == Sample.Status.TRUNCATED)
    
    # Compute averages across datasets
    avg_eval = {"avg": [], "avg_verify": []}
    for dataname, eval_metric in eval_result.items():
        if '_verify' in dataname:
            avg_eval['avg_verify'].extend(eval_metric['rewards'])
        else:
            avg_eval['avg'].extend(eval_metric['rewards'])
    
    eval_result["avg"] = {"rewards": avg_eval['avg']}
    eval_result["avg_verify"] = {"rewards": avg_eval['avg_verify']}

    # Save evaluation results
    save_eval_result = {}
    for k, v in eval_result.items():
        save_eval_result[k] = {}
        for k1, v1 in v.items():
            if len(v1) > 0:
                save_eval_result[k][k1] = float(np.sum(v1))
    
    with open(osp.join(save_eval_path, "eval_result.jsonl"), "a") as f:
        json.dump(save_eval_result, f, ensure_ascii=False)
        f.write("\n")
    
    return eval_result


def lean_generate_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    """
    Main entry point for ReForm rollout generation.

    Args:
        args: Training arguments.
        rollout_id: ID for this rollout.
        data_buffer: Data buffer for managing samples.
        evaluation: Whether this is for evaluation.

    Returns:
        List of completed sample groups.
    """
    completed_samples, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return completed_samples


def generate_abortable_samples(
    args,
    rollout_id: int,
    data_source,
    evaluation: bool = False
) -> tuple[list, list]:
    """Generate samples with abort support."""
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source))
