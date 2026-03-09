# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns

# 这里修改，增加了loss_mask的处理，避免部分结果为空导致的失败
@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor = None,
    group_size: int = 1,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor, optional): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    # Handle case when loss_mask is None (e.g., in AV tasks with sequence-level rewards)
    if loss_mask is not None:
        # For token-level tasks: expand advantages to match loss_mask shape [n_steps, bsz]
        advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask
    else:
        # For sequence-level tasks (AV): expand advantages to all timesteps
        # Input: [num_groups, group_size] where num_groups * group_size = bsz
        # Output: [n_steps, bsz] flattened to [n_steps * bsz]
        n_steps = kwargs.get("n_steps", 1)
        bsz = advantages.numel()  # num_groups * group_size
        advantages = advantages.reshape(-1)  # [bsz]
        # Repeat advantages for all timesteps: [bsz] -> [n_steps, bsz] -> [n_steps * bsz]
        advantages = advantages.unsqueeze(0).expand(n_steps, bsz).reshape(-1)

    return advantages, None


@register_advantage("grpo_per_step_weighted")
def compute_grpo_per_step_weighted_advantages(
    rewards: torch.Tensor,
    group_size: int = 1,
    time_weight_type: str = "linear",
    time_weight_gamma: float = 0.5,
    **kwargs,
):
    """
    Per-step GRPO advantages with temporal weighting.

    First normalizes each timestep's reward independently within the group
    (identical to grpo_per_step), then multiplies by a monotonically
    increasing weight so that later timesteps receive stronger gradient signal.

    Args:
        rewards (torch.Tensor): Per-step rewards. Shape: [n_steps, bsz].
        group_size (int): Number of rollouts per scene.
        time_weight_type (str): Weighting scheme: "linear", "exponential", or "square".
        time_weight_gamma (float): Base for exponential weighting (only used when
            time_weight_type == "exponential").

    Returns:
        Tuple[torch.Tensor, None]: (advantages [n_steps, bsz], None)
    """
    n_steps, bsz = rewards.shape
    num_groups = bsz // group_size

    # Per-step GRPO normalization
    grouped = rewards.reshape(n_steps, num_groups, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, keepdim=True)
    advantages = (grouped - mean) / (std + 1e-6)
    advantages = advantages.reshape(n_steps, bsz)

    # Temporal weighting: w_t increases with t
    if time_weight_type == "exponential":
        w = time_weight_gamma ** torch.arange(
            n_steps - 1, -1, -1, device=rewards.device, dtype=rewards.dtype
        )
    elif time_weight_type == "square":
        w = (
            torch.arange(1, n_steps + 1, device=rewards.device, dtype=rewards.dtype)
            / n_steps
        ) ** 2
    else:  # linear (default)
        w = torch.linspace(1 / n_steps, 1.0, n_steps, device=rewards.device)

    advantages = advantages * w.unsqueeze(1)

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None
