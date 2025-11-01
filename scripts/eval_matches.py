#!/usr/bin/env python3
"""Simulate full-to-1000 matches for a PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bots.self_play import SelfPlayBot
from rl.action import ActionCatalog
from rl.env import TysiacEnv
from rl.ppo_lstm.net import build_policy_value_net
from rl.ppo_lstm.train import TrainConfig, make_env, infer_model_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full matches to 1000 points.")
    parser.add_argument("--checkpoint", required=True, help="Agent checkpoint path.")
    parser.add_argument("--matches", type=int, default=20, help="Number of matches to simulate.")
    parser.add_argument("--target-score", type=int, default=1000, help="Score threshold to win a match.")
    parser.add_argument("--max-hands", type=int, default=200, help="Safety cap on hands per match.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--opponent",
        type=str,
        default="greedy",
        choices=["greedy", "trump", "counter", "random", "self"],
    )
    parser.add_argument(
        "--selfplay-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path when using --opponent self.",
    )
    return parser.parse_args()


def load_config(checkpoint_path: Path, device: torch.device) -> Tuple[TrainConfig, dict, bool]:
    from torch.serialization import add_safe_globals
    from pathlib import PosixPath

    add_safe_globals([PosixPath])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint.get("config")
    if isinstance(raw_config, dict):
        config = TrainConfig(**raw_config)
    else:
        config = raw_config
    inferred = infer_model_sizes(checkpoint.get("model_state", {}))
    if inferred is not None:
        hidden, lstm = inferred
        config.model_hidden = hidden
        config.model_lstm = lstm
        if inferred == (256, 128):
            config.include_seen_cards = False
    legacy = checkpoint.get("use_legacy_model", False) or inferred == (256, 128)
    return config, checkpoint["model_state"], legacy


def build_env(config: TrainConfig, catalog: ActionCatalog, args: argparse.Namespace) -> TysiacEnv:
    if args.opponent == "self":
        if args.selfplay_checkpoint is None:
            raise SystemExit("--selfplay-checkpoint required when opponent=self")
        opponent = SelfPlayBot(Path(args.selfplay_checkpoint), device=args.device)
        env = TysiacEnv(
            opponent=opponent,
            agent_player=config.agent_player,
            starting_player=config.starting_player,
            seed=config.seed,
            include_seen_cards=config.include_seen_cards,
        )
        return env
    config.opponent = args.opponent
    return make_env(config, catalog)


def greedy_action(logits: torch.Tensor, legal_mask: List[int]) -> int:
    mask = torch.tensor(legal_mask, dtype=torch.bool, device=logits.device)
    masked_logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device))
    return int(masked_logits.argmax().item())


def simulate_match(
    model: torch.nn.Module,
    env: TysiacEnv,
    catalog: ActionCatalog,
    device: torch.device,
    target_score: int,
    max_hands: int,
) -> Tuple[int, List[int], int]:
    env.session.scores = [0, 0]
    total_hands = 0
    while max(env.session.scores) < target_score and total_hands < max_hands:
        obs = env.reset()
        hidden = model.initial_state(device=device)
        done = False
        while not done:
            obs_tensor = torch.tensor(obs.vector, dtype=torch.float32, device=device).unsqueeze(0)
            outputs = model(obs_tensor, hidden)
            hidden = outputs.hidden
            action_id = greedy_action(outputs.logits.squeeze(0), obs.legal_mask)
            obs, _, done, info = env.step(action_id)
        total_hands += 1
        scores = info.get("scores", env.session.scores)
        env.session.scores = list(scores)
    winner = 0 if env.session.scores[0] >= target_score else 1
    return winner, env.session.scores[:], total_hands


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    config, state_dict, legacy = load_config(checkpoint_path, device)
    catalog = ActionCatalog()
    env = build_env(config, catalog, args)
    sample_obs = env.reset()
    obs_size = len(sample_obs.vector)
    model = build_policy_value_net(
        obs_size=obs_size,
        action_size=catalog.size,
        hidden_size=config.model_hidden,
        lstm_size=config.model_lstm,
        legacy=legacy,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    results = []
    for _ in range(args.matches):
        winner, scores, hands = simulate_match(
            model=model,
            env=env,
            catalog=catalog,
            device=device,
            target_score=args.target_score,
            max_hands=args.max_hands,
        )
        results.append((winner, scores, hands))

    agent_wins = sum(1 for winner, _, _ in results if winner == config.agent_player)
    win_rate = agent_wins / args.matches * 100 if results else 0.0
    avg_hands = mean(hands for _, _, hands in results) if results else 0.0
    print(f"Matches played: {len(results)}")
    print(f"Agent wins: {agent_wins} ({win_rate:.2f}%)")
    print(f"Average hands per match: {avg_hands:.2f}")
    print("Scores per match:")
    for idx, (_, scores, hands) in enumerate(results, start=1):
        print(f"  Match {idx}: hands={hands}, scores={scores}")


if __name__ == "__main__":
    main()
