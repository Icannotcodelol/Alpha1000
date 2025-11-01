"""Rollout utilities tailored for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.distributions import Categorical
import torch.nn as nn

from rl.action import ActionCatalog
from rl.env import Observation, TysiacEnv
from rl.ppo_lstm.net import PolicyOutputs


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    stats: Dict[str, object]


def _mask_logits(logits: torch.Tensor, legal_mask) -> torch.Tensor:
    mask_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(~mask_tensor, -1e9)
    return masked


def collect_rollouts(
    envs: Sequence[TysiacEnv],
    model: nn.Module,
    *,
    catalog: ActionCatalog,
    steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    current_obs: List[Observation],
) -> Tuple[RolloutBatch, List[Observation]]:
    model.train()
    num_envs = len(envs)
    assert num_envs >= 1, "At least one environment required"
    # Use provided observations to continue episodes without unnecessary resets.
    current_obs = list(current_obs)
    observations: List[torch.Tensor] = []
    legal_masks: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []
    dones: List[torch.Tensor] = []
    episode_rewards = [0.0 for _ in envs]
    stats: Dict[str, object] = {
        "episodes": 0,
        "episode_rewards": [],
        "tricks_won": 0,
        "tricks_lost": 0,
        "agent_melds": 0,
        "opponent_melds": 0,
        "contract_attempts": 0,
        "contract_success": 0,
        "opponents": [],
    }

    hidden_states = [model.initial_state(device=device) for _ in envs]
    last_env_index = 0

    step_count = 0
    while step_count < steps:
        for env_index, env in enumerate(envs):
            obs = current_obs[env_index]
            obs_tensor = torch.tensor(obs.vector, dtype=torch.float32, device=device).unsqueeze(0)
            outputs: PolicyOutputs = model(obs_tensor, hidden_states[env_index])
            hidden_states[env_index] = tuple(t.detach() for t in outputs.hidden)

            masked_logits = _mask_logits(outputs.logits, obs.legal_mask)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()

            next_obs, reward, done, info = env.step(action.item())

            observations.append(obs_tensor.squeeze(0))
            legal_masks.append(torch.tensor(obs.legal_mask, dtype=torch.float32, device=device))
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(outputs.value.squeeze())
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
            dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            episode_rewards[env_index] += reward

            current_obs[env_index] = next_obs
            last_env_index = env_index

            if done:
                stats["episodes"] += 1
                stats["episode_rewards"].append(episode_rewards[env_index])
                episode_rewards[env_index] = 0.0
                stats["opponents"].append(env.opponent_name)
                if "episode_stats" in info:
                    ep_stats = info["episode_stats"]
                    stats["tricks_won"] += ep_stats.get("tricks_won", 0)
                    stats["tricks_lost"] += ep_stats.get("tricks_lost", 0)
                    stats["agent_melds"] += ep_stats.get("agent_melds", 0)
                    stats["opponent_melds"] += ep_stats.get("opponent_melds", 0)
                    stats["contract_attempts"] += ep_stats.get("contract_attempts", 0)
                    stats["contract_success"] += ep_stats.get("contract_success", 0)
                current_obs[env_index] = env.reset()
                hidden_states[env_index] = model.initial_state(device=device)

            step_count += 1
            if step_count >= steps:
                break
        if step_count >= steps:
            break

    # Bootstrap value for final state
    with torch.no_grad():
        bootstrap_obs = current_obs[last_env_index]
        obs_tensor = torch.tensor(bootstrap_obs.vector, dtype=torch.float32, device=device).unsqueeze(0)
        outputs: PolicyOutputs = model(obs_tensor, hidden_states[last_env_index])
        next_value = outputs.value.squeeze()

    values_tensor = torch.stack(values)
    rewards_tensor = torch.stack(rewards)
    dones_tensor = torch.stack(dones)

    advantages = torch.zeros_like(values_tensor, device=device)
    gae = torch.zeros(1, device=device)
    for step in reversed(range(steps)):
        if step == steps - 1:
            next_val = next_value
            next_non_terminal = 1.0 - dones_tensor[step]
        else:
            next_val = values_tensor[step + 1]
            next_non_terminal = 1.0 - dones_tensor[step + 1]
        delta = rewards_tensor[step] + gamma * next_val * next_non_terminal - values_tensor[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae

    returns = advantages + values_tensor

    batch = RolloutBatch(
        observations=torch.stack(observations),
        legal_masks=torch.stack(legal_masks),
        actions=torch.stack(actions),
        log_probs=torch.stack(log_probs),
        values=values_tensor,
        rewards=rewards_tensor,
        dones=dones_tensor,
        advantages=advantages,
        returns=returns,
        stats=stats,
    )

    return batch, current_obs
