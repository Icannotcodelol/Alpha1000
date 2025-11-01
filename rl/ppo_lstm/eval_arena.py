"""Evaluate a trained PPO agent against baseline or self-play opponents."""

from __future__ import annotations

import argparse
from statistics import mean
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.action import ActionCatalog
from rl.env import TysiacEnv
from bots.self_play import SelfPlayBot
from rl.ppo_lstm.net import build_policy_value_net
from rl.ppo_lstm.selfplay import _mask_logits  # reuse helper
from rl.ppo_lstm.train import TrainConfig, make_env, infer_model_sizes


def adapt_state_dict_for_obs_size(state_dict: dict, model: nn.Module) -> dict:
    model_state = model.state_dict()
    weight_key = "obs_encoder.0.weight"
    bias_key = "obs_encoder.0.bias"
    if weight_key in state_dict and weight_key in model_state:
        old_weight = state_dict[weight_key]
        new_weight = model_state[weight_key]
        if old_weight.shape != new_weight.shape:
            if old_weight.shape[0] == new_weight.shape[0] and old_weight.shape[1] <= new_weight.shape[1]:
                adapted = new_weight.clone()
                adapted[:, : old_weight.shape[1]] = old_weight
                state_dict[weight_key] = adapted
            else:
                raise RuntimeError("Checkpoint observation size incompatible with current environment.")
    if bias_key in state_dict and bias_key in model_state:
        old_bias = state_dict[bias_key]
        new_bias = model_state[bias_key]
        if old_bias.shape != new_bias.shape:
            if old_bias.shape[0] == new_bias.shape[0]:
                state_dict[bias_key] = new_bias.clone()
            else:
                raise RuntimeError("Checkpoint bias mismatch.")
    return state_dict


def greedy_action(logits, legal_mask):
    masked = _mask_logits(logits, legal_mask)
    return torch.argmax(masked, dim=-1)


def evaluate(model: nn.Module, env: TysiacEnv, catalog: ActionCatalog, episodes: int, device: torch.device) -> dict:
    wins = 0
    returns = []
    model.eval()

    for _ in range(episodes):
        obs = env.reset()
        hidden = model.initial_state(device=device)
        total_reward = 0.0
        while True:
            obs_tensor = torch.tensor(obs.vector, dtype=torch.float32, device=device).unsqueeze(0)
            outputs = model(obs_tensor, hidden)
            hidden = outputs.hidden
            action = greedy_action(outputs.logits, obs.legal_mask)
            obs, reward, done, info = env.step(action.item())
            total_reward += reward
            if done:
                scores = info.get("scores")
                if scores is not None:
                    agent_score = scores[env.agent_player]
                    opponent_score = scores[env.opponent_player]
                    if agent_score > opponent_score:
                        wins += 1
                elif reward > 0:
                    wins += 1
                returns.append(total_reward)
                break

    return {
        "win_rate": wins / episodes if episodes else 0.0,
        "avg_return": mean(returns) if returns else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint against baseline bots.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        choices=[None, "greedy", "trump", "counter", "self"],
    )
    parser.add_argument(
        "--selfplay-checkpoint",
        type=str,
        default=None,
        help="Checkpoint for self-play opponent when --opponent self is specified.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    from torch.serialization import add_safe_globals
    from pathlib import PosixPath

    add_safe_globals([PosixPath])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    if isinstance(raw_config, dict):
        saved_config = TrainConfig(**raw_config)
    else:
        # Fallback for legacy checkpoints
        saved_config = raw_config
    if args.opponent and args.opponent != "self":
        saved_config.opponent = args.opponent
    hidden = checkpoint.get("model_hidden")
    lstm = checkpoint.get("model_lstm")
    inferred = infer_model_sizes(checkpoint.get("model_state", {}))
    if inferred is not None:
        hidden, lstm = inferred
        if inferred == (256, 128):
            saved_config.include_seen_cards = False
    legacy = False
    if inferred == (256, 128) or checkpoint.get("use_legacy_model"):
        legacy = True
        saved_config.include_seen_cards = False
    catalog = ActionCatalog()
    if args.opponent == "self":
        if args.selfplay_checkpoint is None:
            raise SystemExit("--selfplay-checkpoint must be provided when --opponent self is used.")
        opponent = SelfPlayBot(Path(args.selfplay_checkpoint), device=args.device)
        env = TysiacEnv(
            opponent=opponent,
            agent_player=saved_config.agent_player,
            starting_player=saved_config.starting_player,
            seed=saved_config.seed,
            include_seen_cards=saved_config.include_seen_cards,
        )
    else:
        env = make_env(saved_config, catalog)

    sample_obs = env.reset()
    obs_size = len(sample_obs.vector)
    model = build_policy_value_net(
        obs_size=obs_size,
        action_size=catalog.size,
        hidden_size=hidden or 512,
        lstm_size=lstm or 256,
        legacy=legacy,
    ).to(device)
    state_dict = adapt_state_dict_for_obs_size(checkpoint["model_state"], model)
    model.load_state_dict(state_dict)

    metrics = evaluate(model, env, catalog, episodes=args.episodes, device=device)
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Average return: {metrics['avg_return']:.3f}")


if __name__ == "__main__":
    main()
