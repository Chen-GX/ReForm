"""
Advantage and returns computation utilities for PBSO (Prospective Bounded Sequence Optimization).

This module implements multi-step reward handling and advantage estimation for the ReForm framework.
"""

from typing import List, Union
import os
import torch
from collections import defaultdict
import numpy as np

from slime.utils.types import Sample

# Token ID used to identify step boundaries in the response
step_reward_token_id = int(os.environ.get("STEP_REWARD_TOKEN_ID", 151670))


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    responses: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Compute GRPO-style returns with multi-step reward assignment.
    
    Assigns step-wise rewards to token positions based on step boundary markers.
    
    Args:
        rewards: List of step-wise rewards for each sample.
        kl: KL divergence tensors for each sample.
        responses: Response token tensors for each sample.
        
    Returns:
        List of return tensors with rewards assigned to corresponding positions.
    """
    returns = []
    for i in range(len(rewards)):
        reward_list = rewards[i]
        response = responses[i]

        step_rewards_tensor = torch.zeros_like(kl[i], dtype=torch.float32)

        # Find positions of step boundary tokens
        reward_token_indices = torch.where(response == step_reward_token_id)[0]
        if reward_token_indices.size(0) == 0:
            print("Warning: <step_reward_token_id> not found")
            assert len(reward_list) == 1, f"{len(reward_list)=} should be 1 when <step_reward_token_id> not found"
            step_rewards_tensor[:] = reward_list[0]
            returns.append(step_rewards_tensor)
            continue

        assert reward_token_indices.size(0) == len(reward_list), \
            f"{reward_token_indices.size(0)=} != {len(reward_list)=} | {reward_list=} | {response.tolist()=}"

        # Assign rewards to token ranges
        start_idx = 0
        for j, end_idx_tensor in enumerate(reward_token_indices):
            end_idx = end_idx_tensor.item()
            step_rewards_tensor[start_idx : end_idx + 1] = reward_list[j]
            start_idx = end_idx + 1

        # Handle remaining tokens after the last boundary
        if start_idx < step_rewards_tensor.size(0):
            step_rewards_tensor[start_idx:] = reward_list[-1]

        returns.append(step_rewards_tensor)
    return returns


def compute_step_advantages_and_returns(
    args,
    rewards: List[List[float]],
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    responses: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute advantages and returns for multi-step PBSO training.
    
    Args:
        args: Training arguments containing advantage_estimator config.
        rewards: Multi-step rewards for each sample.
        kl: KL divergence tensors.
        loss_masks: Loss mask tensors.
        response_lengths: Response lengths.
        total_lengths: Total sequence lengths.
        responses: Response token tensors.
        
    Returns:
        Tuple of (advantages, returns) tensors.
    """
    if args.advantage_estimator in ["grpo", "gspo"]:
        returns = get_grpo_returns(rewards, kl, responses)
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus":
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported.")
    
    return advantages, returns


def group_step_reward_normalization(
    args,
    rewards: List[List[float]],
    group_ids: List,
    epsilon: float = 1e-6
) -> List[List[float]]:
    """
    Perform group-wise normalization on multi-step rewards.
    
    Normalizes rewards within each group (samples from the same prompt) to reduce variance
    and improve training stability.

    Args:
        args: Configuration object containing:
              - advantage_estimator (str)
              - rewards_normalization (bool)
              - n_samples_per_prompt (int)
              - grpo_std_normalization (bool)
        rewards: List where each element is a list of step-wise rewards for a sample.
        group_ids: Flat list of group identifiers aligned with rewards.
        epsilon: Small value to avoid division by zero.

    Returns:
        List of normalized rewards preserving the original nested structure.
    """
    if len(rewards) != len(group_ids):
        raise ValueError("Length of rewards and group_ids must be the same.")
    
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        id_to_scores = defaultdict(list)
        for group_id, reward_list in zip(group_ids, rewards):
            id_to_scores[group_id].append(reward_list)

        id_to_mean, id_to_std = {}, {}
        for group_id, list_of_reward_lists in id_to_scores.items():
            assert len(list_of_reward_lists) >= args.n_samples_per_prompt, \
                f"Group {group_id} has less than {args.n_samples_per_prompt} samples."

            scores_tensor = torch.tensor(
                [score for sublist in list_of_reward_lists for score in sublist],
                dtype=torch.float32
            )
            id_to_mean[group_id] = torch.mean(scores_tensor)
            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                id_to_std[group_id] = torch.std(scores_tensor)
        
        normalized_rewards = []
        for group_id, reward_list in zip(group_ids, rewards):
            mean = id_to_mean[group_id]

            reward_tensor = torch.tensor(reward_list, dtype=torch.float32)
            advantage_tensor = reward_tensor - mean

            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                std = id_to_std[group_id]
                advantage_tensor = advantage_tensor / (std + epsilon)
            
            normalized_rewards.append(advantage_tensor.tolist())
    
    else:
        normalized_rewards = [list(r) for r in rewards]
     
    return normalized_rewards


def custom_reward_post_process_func(
    args,
    samples: Union[list[Sample], list[list[Sample]]]
) -> tuple[List[List[float]], List[List[float]]]:
    """
    Post-process rewards with group-wise normalization for PBSO.
    
    Args:
        args: Training arguments.
        samples: List of samples to process.
        
    Returns:
        Tuple of (raw_rewards, normalized_rewards).
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    group_ids = [sample.metadata['instance_id'] for sample in samples]

    rewards = group_step_reward_normalization(args, raw_rewards, group_ids)

    return raw_rewards, rewards
