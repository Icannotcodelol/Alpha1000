"""Self-play utilities for collecting rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from bots.base import BotStrategy
from bots.baseline_greedy import GreedyBot
from engine.game import GameSession

from .env import Observation, TysiacEnv


@dataclass
class Transition:
    observation: Observation
    action: int
    reward: float
    done: bool
    info: dict


def generate_rollout(
    env: TysiacEnv,
    policy,
    *,
    max_steps: int = 512,
) -> List[Transition]:
    """Run a single episode using the provided policy callable."""
    transitions: List[Transition] = []
    obs = env.reset()
    for _ in range(max_steps):
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        transitions.append(Transition(observation=obs, action=action, reward=reward, done=done, info=info))
        obs = next_obs
        if done:
            break
    return transitions


def random_policy(catalog_size: int):
    import random

    def _policy(obs: Observation) -> int:
        legal_indices = [idx for idx, flag in enumerate(obs.legal_mask) if flag]
        if not legal_indices:
            return 0
        return random.choice(legal_indices)

    return _policy
