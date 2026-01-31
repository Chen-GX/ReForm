"""
Reward computation for ReForm PBSO training.

This module implements multi-step reward functions for autoformalization tasks,
including step-level and task-level reward computation with discounted propagation.
"""

import os
import re
import numpy as np
import asyncio
import time
import json5
import random
from openai import AsyncOpenAI

from slime.utils.types import Sample

from autoformation_remote_critic_round_consistency import consistency_refine

try:
    from .verify_utils import verify_formal_statement
    from .format_check import check_output_format, check_answer_part
    from .critic_lean_prompt import critic_lean_prompt, user_prompt_template
except ImportError:
    from verify_utils import verify_formal_statement
    from format_check import check_output_format, check_answer_part
    from critic_lean_prompt import critic_lean_prompt, user_prompt_template

from sglang.srt.reasoning_parser import ReasoningParser

reasoning_parser = ReasoningParser('qwen3')
CRITICLEAN_URL = os.environ.get('CRITICLEAN_URL')

POS_REWARD = 1
NEG_REWARD = -1

critic_lean = AsyncOpenAI(
    api_key="EMPTY",
    base_url=CRITICLEAN_URL,
    max_retries=3,
    timeout=120.0,
)


def extract_round_content(text: str) -> list[str]:
    """
    Extract all content enclosed by <round> and </round> tags.

    Args:
        text: Original string containing <round> tags.

    Returns:
        List of strings, each being the content within a <round> block.
    """
    pattern = r"<round>(.*?)</round>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_lean_in_answer(text: str) -> str | None:
    """Extract Lean code from within <answer> tags."""
    pattern = r"<answer>.*?```lean4\n?(.*?)\n?```.*?</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_lean(text: str) -> str | None:
    """Extract Lean code from markdown code blocks."""
    pattern = r".*?```lean4\n?(.*?)\n?```.*?"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def propagate_reward(
    step_rewards: list[float],
    use_clip: bool = True,
    gamma: float = 0.25
) -> list[float]:
    """
    Propagate rewards backward using discounted returns with optional clipping.
    
    This implements the PBSO reward shaping strategy where future rewards
    influence earlier steps through discounted propagation.

    Args:
        step_rewards: Raw immediate rewards for each step.
        use_clip: If True, clip propagated rewards to [NEG_REWARD, POS_REWARD].
        gamma: Discount factor for reward propagation.

    Returns:
        List of processed rewards after propagation.
    """
    if not step_rewards:
        return []

    propagated_rewards = np.zeros_like(step_rewards, dtype=np.float32)
    propagated_rewards[-1] = step_rewards[-1]

    for i in range(len(step_rewards) - 2, -1, -1):
        cur_reward = step_rewards[i] + gamma * propagated_rewards[i + 1]
        propagated_rewards[i] = np.clip(cur_reward, NEG_REWARD, POS_REWARD) if use_clip else cur_reward
        
    return propagated_rewards.tolist()


async def _get_task_level_criticlean_reward(
    round_content: str,
    informal_statement: str,
    header: dict,
    model: str,
    return_verify: bool = False
) -> tuple[float, float] | float:
    """
    Calculate reward for a task step containing <answer>.
    
    Uses a critic model to verify semantic consistency between the
    formal statement and informal mathematical problem.

    Args:
        round_content: Content of the round/answer to evaluate.
        informal_statement: Original natural language problem.
        header: Lean header for verification.
        model: Model identifier for the critic.
        return_verify: If True, return both task and verification rewards.

    Returns:
        Task reward, or tuple of (task_reward, verify_reward) if return_verify is True.
    """
    formal_statement = extract_lean(round_content)
    if not formal_statement:
        return (NEG_REWARD, NEG_REWARD) if return_verify else NEG_REWARD

    # Step 1: Verify the formal statement compiles
    verify_result = await verify_formal_statement(formal_statement, header)
    verify_reward = POS_REWARD if verify_result[0] else NEG_REWARD
    if not verify_result[0]:
        return (NEG_REWARD, verify_reward) if return_verify else NEG_REWARD

    # Step 2: Check semantic consistency with the informal statement
    user_prompt = critic_lean_prompt + user_prompt_template.format(
        informal_prefix=informal_statement,
        formal_statement=formal_statement
    )

    response_json = None
    for _ in range(3):
        try:
            response = await critic_lean.chat.completions.create(
                model="EMPTY",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.6,
                max_tokens=20480,
                top_p=0.95,
                presence_penalty=1.2,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            text = response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {str(e)}")
            continue

        try:
            response_str = text.split("</think>")[-1]
            response_json = json5.loads(response_str)
            break
        except Exception as err:
            print(f"Error: {err} criticlean parser failed, {response}")
            response_json = None
    
    if response_json is None:
        print('CriticLean scoring failed')
    
    task_reward = NEG_REWARD if response_json is None or 'incorrect' in response_json['is_assistant_correct'].lower() else POS_REWARD

    return (task_reward, verify_reward) if return_verify else task_reward


async def _get_step_level_reward(
    round_content: str,
    informal_statement: str,
    model: str = None,
    max_retry: int = 2
) -> float:
    """
    Calculate reward for an intermediate reasoning step.
    
    Uses an external consistency checker to evaluate the semantic
    correctness of intermediate formalization steps.

    Args:
        round_content: Content of the intermediate round.
        informal_statement: Original natural language problem.
        model: Model to use for evaluation (optional).
        max_retry: Maximum number of retry attempts.

    Returns:
        Step reward (POS_REWARD or NEG_REWARD).
    """
    begin_time = time.time()
    model_candidate = ['deepseek-r1', 'deepseek-r1', 'deepseek-r1', 'deepseek-r1', 'deepseek-r1-0528']
    
    cur_history = f"\n<round>\n{round_content}\n</round>\n"

    for _ in range(max_retry):
        cur_model = random.choice(model_candidate)
        reward, reason = await asyncio.to_thread(
            consistency_refine,
            informal_statement=informal_statement,
            history=cur_history,
            model=cur_model,
        )
        if len(reward) != 0:
            print(f"Step reward time: {time.time() - begin_time}s")
            return NEG_REWARD if 'incorrect' in reward[0].lower() else POS_REWARD

    # Fallback to OpenRouter if primary models fail
    print("Primary model scoring failed, falling back to OpenRouter")
    reward, reason = await asyncio.to_thread(
        consistency_refine,
        informal_statement=informal_statement,
        history=cur_history,
        model='openrouter_deepseek_r1',
    )
    if len(reward) != 0:
        print(f"Step reward time: {time.time() - begin_time}s")
        return NEG_REWARD if 'incorrect' in reward[0].lower() else POS_REWARD
 
    print(f"Consistency refine failed after {max_retry} retries, time: {time.time() - begin_time}s")
    return NEG_REWARD


async def reward_func(args, sample: Sample, **kwargs) -> list[float]:
    """
    Main reward function for PBSO training.
    
    Computes multi-step rewards for autoformalization samples:
    - For evaluation: Returns task and verification rewards
    - For training: Returns step-wise rewards with discounted propagation

    Args:
        args: Training arguments containing reward_shaping and reward_shaping_gamma.
        sample: Sample to evaluate.
        **kwargs: Additional arguments (e.g., evaluation flag).

    Returns:
        List of rewards for each step in the response.
    """
    is_evaluation = kwargs.get('evaluation', False)
    start_time = time.time()
    
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    
    reasoning_text, answer_text = reasoning_parser.parse_non_stream(sample.response)

    if is_evaluation:
        task_reward, verify_reward = await _get_task_level_criticlean_reward(
            round_content=answer_text,
            informal_statement=sample.prompt,
            header=None,
            model="criticlean",
            return_verify=True,
        )
        # Convert to 0/1 for evaluation metrics
        if task_reward == NEG_REWARD:
            task_reward = 0
        if verify_reward == NEG_REWARD:
            verify_reward = 0

        eval_time = time.time() - start_time
        if eval_time > 120:
            print(f"Eval reward time: {eval_time}", flush=True)
        return [task_reward, verify_reward]

    # Training mode
    if not check_output_format(reasoning_text) or not check_answer_part(answer_text):
        num_rounds = len(re.findall(r"</round>", sample.response))
        print('Reward: format error, assigning negative reward')
        return [NEG_REWARD] * max(num_rounds, 1)
    
    rounds = extract_round_content(reasoning_text)

    # Create concurrent tasks for step-level rewards
    tasks_to_run = []
    
    # Compute step-level rewards for intermediate rounds
    for round_content in rounds[:-1]:
        task = _get_step_level_reward(
            round_content=round_content,
            informal_statement=sample.prompt,
        )
        tasks_to_run.append(task)
        
    # Add task-level reward for the final round
    task = _get_task_level_criticlean_reward(
        round_content=answer_text,
        informal_statement=sample.prompt,
        header=None,
        model="criticlean",
        return_verify=False,
    )
    tasks_to_run.append(task)

    # Execute all tasks concurrently
    step_rewards = await asyncio.gather(*tasks_to_run)

    assert len(step_rewards) == len(rounds)
    
    print(f"Concurrent reward time for {len(rounds)} steps: {time.time() - start_time}", flush=True)

    # Apply PBSO reward shaping with discounted propagation
    if args.reward_shaping == "discounted_with_clip":
        step_rewards = propagate_reward(
            step_rewards,
            use_clip=True,
            gamma=args.reward_shaping_gamma
        )
    
    print(f"[PBSO] {args.reward_shaping} gamma={args.reward_shaping_gamma}: {step_rewards=}")

    return step_rewards
