"""Basic PPO training loop for the Tysiąc environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence
import csv
import random

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from bots.base import BotStrategy
from bots.self_play import SelfPlayBot
from rl.action import ActionCatalog
from rl.env import TysiacEnv
from rl.ppo_lstm.net import PolicyValueNet, build_policy_value_net
from rl.ppo_lstm.selfplay import RolloutBatch, collect_rollouts


@dataclass
class TrainConfig:
    steps_per_update: int = 1024
    updates: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    device: str = "cpu"
    seed: Optional[int] = None
    agent_player: int = 0
    starting_player: int = 0
    opponent: str = "greedy"
    opponent_mix: Optional[str] = None
    checkpoint_dir: Path = Path("data/checkpoints")
    log_csv: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None
    num_envs: int = 1
    model_hidden: int = 512
    model_lstm: int = 256
    include_seen_cards: bool = True
    use_legacy_model: bool = False
    opponent_mix_schedule: Optional[str] = None
    schedule_repeat: bool = True
    selfplay_checkpoints: List[Path] = field(default_factory=list)


@dataclass
class CurriculumPhase:
    mix: str
    updates: int


def parse_opponent_schedule(schedule: str) -> List[CurriculumPhase]:
    phases: List[CurriculumPhase] = []
    for raw in schedule.split(";"):
        segment = raw.strip()
        if not segment:
            continue
        if "@" not in segment:
            raise ValueError(f"Opponent mix schedule segment '{segment}' must contain '@' separating mix and updates.")
        mix_part, updates_part = segment.rsplit("@", 1)
        mix = mix_part.strip()
        if not mix:
            raise ValueError(f"Opponent mix schedule segment '{segment}' is missing a mix definition.")
        try:
            updates = int(updates_part)
        except ValueError as exc:  # noqa: BLE001
            raise ValueError(f"Invalid update count '{updates_part}' in schedule segment '{segment}'.") from exc
        if updates <= 0:
            raise ValueError(f"Update count in schedule segment '{segment}' must be positive.")
        phases.append(CurriculumPhase(mix=mix, updates=updates))
    if not phases:
        raise ValueError("Opponent mix schedule must include at least one phase.")
    return phases


def build_envs_for_mix(base_config: TrainConfig, catalog: ActionCatalog, mix: Optional[str]) -> List[TysiacEnv]:
    phase_config = replace(base_config, opponent_mix=mix)
    return [make_env(phase_config, catalog) for _ in range(base_config.num_envs)]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train PPO agent for Tysiąc.")
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent-player", type=int, default=0, choices=[0, 1])
    parser.add_argument("--starting-player", type=int, default=0, choices=[0, 1])
    parser.add_argument("--opponent", type=str, default="greedy")
    parser.add_argument("--opponent-mix", type=str, default=None, help="Comma-separated opponents with optional weights, e.g., 'greedy:0.5,counter:0.5'")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints")
    parser.add_argument("--log-csv", type=str, default=None, help="Optional path to append per-update training metrics")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--model-hidden", type=int, default=512)
    parser.add_argument("--model-lstm", type=int, default=256)
    parser.add_argument("--disable-seen-cards", action="store_true")
    parser.add_argument("--legacy-model", action="store_true")
    parser.add_argument(
        "--selfplay",
        action="append",
        default=[],
        help="Path to a checkpoint used for self-play opponents. Can be provided multiple times (self0, self1, ...).",
    )
    parser.add_argument(
        "--opponent-mix-schedule",
        type=str,
        default=None,
        help=(
            "Curriculum schedule as 'mix@updates;mix2@updates'. "
            "Mix strings follow --opponent-mix format and schedule repeats until --updates are exhausted."
        ),
    )
    parser.add_argument(
        "--no-schedule-repeat",
        action="store_true",
        help="Do not loop the opponent mix schedule; run it once even if --updates exceeds the total.",
    )
    args = parser.parse_args()
    selfplay_paths = [Path(p) for p in args.selfplay] if args.selfplay else []
    return TrainConfig(
        steps_per_update=args.steps_per_update,
        updates=args.updates,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        agent_player=args.agent_player,
        starting_player=args.starting_player,
        opponent=args.opponent,
        opponent_mix=args.opponent_mix,
        checkpoint_dir=Path(args.checkpoint_dir),
        log_csv=Path(args.log_csv) if args.log_csv else None,
        resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        num_envs=max(1, args.num_envs),
        model_hidden=args.model_hidden,
        model_lstm=args.model_lstm,
        include_seen_cards=not args.disable_seen_cards,
        use_legacy_model=args.legacy_model,
        opponent_mix_schedule=args.opponent_mix_schedule,
        schedule_repeat=not args.no_schedule_repeat,
        selfplay_checkpoints=selfplay_paths,
    )


def _build_opponent_factory(
    mix: str,
    seed: Optional[int],
    selfplay_paths: Sequence[Path],
    device: str,
) -> Callable[[], BotStrategy]:
    from bots.baseline_counter import CounterBot
    from bots.baseline_greedy import GreedyBot
    from bots.baseline_trump_manager import TrumpManagerBot
    from bots.random_bot import RandomBot

    registry = {
        "greedy": GreedyBot,
        "trump": TrumpManagerBot,
        "counter": CounterBot,
        "random": RandomBot,
    }

    for idx, path in enumerate(selfplay_paths):
        key = f"self{idx}"

        def make_bot(p: Path = path) -> BotStrategy:
            return SelfPlayBot(p, device=device)

        registry[key] = make_bot

    entries = []
    total = 0.0
    for part in mix.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, weight_str = part.split(":", 1)
            weight = float(weight_str)
        else:
            name, weight = part, 1.0
        name = name.strip()
        if name not in registry:
            raise ValueError(f"Unknown opponent '{name}' in mix '{mix}'")
        if weight <= 0:
            continue
        total += weight
        entries.append((name, weight))
    if not entries:
        raise ValueError("Opponent mix must specify at least one valid opponent")

    rng = random.Random(seed)

    def factory() -> BotStrategy:
        x = rng.random() * total
        cumulative = 0.0
        for name, weight in entries:
            cumulative += weight
            if x <= cumulative:
                return registry[name]()
        return registry[entries[-1][0]]()

    return factory


def make_env(config: TrainConfig, catalog: ActionCatalog) -> TysiacEnv:
    from bots.baseline_counter import CounterBot
    from bots.baseline_greedy import GreedyBot
    from bots.baseline_trump_manager import TrumpManagerBot
    from bots.random_bot import RandomBot

    opponent_map = {
        "greedy": GreedyBot(),
        "trump": TrumpManagerBot(),
        "counter": CounterBot(),
        "random": RandomBot(),
    }
    for idx, path in enumerate(config.selfplay_checkpoints):
        opponent_map[f"self{idx}"] = SelfPlayBot(path, device=config.device)
    opponent_factory = None
    if config.opponent_mix:
        opponent_factory = _build_opponent_factory(
            config.opponent_mix,
            config.seed,
            config.selfplay_checkpoints,
            config.device,
        )
        opponent = opponent_factory()
    else:
        if config.opponent not in opponent_map:
            raise ValueError(f"Unknown opponent '{config.opponent}'")
        opponent = opponent_map[config.opponent]
    env = TysiacEnv(
        opponent=opponent,
        opponent_factory=opponent_factory,
        agent_player=config.agent_player,
        starting_player=config.starting_player,
        seed=config.seed,
        include_seen_cards=config.include_seen_cards,
    )
    return env


def ppo_update(
    model: PolicyValueNet,
    optimizer: Adam,
    batch: RolloutBatch,
    config: TrainConfig,
) -> dict:
    observations = batch.observations
    actions = batch.actions
    old_log_probs = batch.log_probs.detach()
    advantages = batch.advantages.detach()
    returns = batch.returns.detach()
    legal_masks = batch.legal_masks

    for epoch in range(4):
        outputs = model(observations)
        masked_logits = torch.where(
            legal_masks.bool(),
            outputs.logits,
            torch.tensor(-1e9, device=observations.device),
        )
        dist = Categorical(logits=masked_logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        surrogate1 = ratio * adv
        surrogate2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * adv
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = F.mse_loss(outputs.value, returns)

        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


def _log_metrics_csv(path: Path, row: dict) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def infer_model_sizes(state_dict: dict) -> Optional[tuple[int, int]]:
    if not isinstance(state_dict, dict):
        return None
    weight = state_dict.get("obs_encoder.0.weight")
    lstm_weight = state_dict.get("lstm.weight_ih_l0")
    if weight is None or lstm_weight is None:
        return None
    hidden = weight.shape[0]
    lstm = lstm_weight.shape[0] // 4
    return hidden, lstm


def adapt_state_dict_for_obs_size(state_dict: dict, model: PolicyValueNet) -> tuple[dict, bool]:
    model_state = model.state_dict()
    weight_key = "obs_encoder.0.weight"
    bias_key = "obs_encoder.0.bias"
    adapted = False
    if weight_key in state_dict and weight_key in model_state:
        old_weight = state_dict[weight_key]
        new_weight = model_state[weight_key]
        if old_weight.shape != new_weight.shape:
            if old_weight.shape[0] == new_weight.shape[0] and old_weight.shape[1] <= new_weight.shape[1]:
                adapted = new_weight.clone()
                adapted[:, : old_weight.shape[1]] = old_weight
                state_dict[weight_key] = adapted
                adapted = True
            else:
                raise RuntimeError(
                    "Checkpoint input size does not match new observation size and cannot be adapted automatically."
                )
    if bias_key in state_dict and bias_key in model_state:
        old_bias = state_dict[bias_key]
        new_bias = model_state[bias_key]
        if old_bias.shape != new_bias.shape:
            if old_bias.shape[0] == new_bias.shape[0]:
                state_dict[bias_key] = new_bias.clone()
                adapted = True
            else:
                raise RuntimeError("Checkpoint bias size incompatible with new observation size.")
    return state_dict, adapted


def main() -> None:
    config = parse_args()
    device = torch.device(config.device)
    catalog = ActionCatalog()

    schedule_phases: Optional[List[CurriculumPhase]] = None
    if config.opponent_mix_schedule:
        schedule_phases = parse_opponent_schedule(config.opponent_mix_schedule)

    checkpoint_data = None
    if config.resume_checkpoint is not None:
        from torch.serialization import add_safe_globals
        from pathlib import PosixPath

        add_safe_globals([PosixPath])
        checkpoint_data = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        inferred = infer_model_sizes(checkpoint_data.get("model_state", {})) if isinstance(checkpoint_data, dict) else None
        if inferred is not None:
            config.model_hidden, config.model_lstm = inferred
            if inferred == (256, 128):
                config.include_seen_cards = False
                config.use_legacy_model = True
            # ensure schedule still parsed with updated config

    if schedule_phases:
        phase_index = 0
        current_phase = schedule_phases[phase_index]
        active_mix = current_phase.mix
        phase_updates_remaining = current_phase.updates
        print(
            f"Opponent mix schedule enabled. Starting phase 1/{len(schedule_phases)}: "
            f"{active_mix} for {phase_updates_remaining} updates."
        )
    else:
        phase_index = None
        current_phase = None
        active_mix = config.opponent_mix
        phase_updates_remaining = None

    envs = build_envs_for_mix(config, catalog, active_mix)
    current_obs = [env.reset() for env in envs]
    sample_obs = current_obs[0]
    obs_size = len(sample_obs.vector)
    model = build_policy_value_net(
        obs_size=obs_size,
        action_size=catalog.size,
        hidden_size=config.model_hidden,
        lstm_size=config.model_lstm,
        legacy=config.use_legacy_model,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    start_update = 1
    if checkpoint_data is not None:
        checkpoint = checkpoint_data
        adapted_obs = False
        if "model_state" in checkpoint:
            state_dict, adapted_obs = adapt_state_dict_for_obs_size(checkpoint["model_state"], model)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {config.resume_checkpoint}")
        if "optimizer_state" in checkpoint and not adapted_obs:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                print("Loaded optimizer state from checkpoint")
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: could not load optimizer state ({exc})")
        if "update" in checkpoint:
            start_update = int(checkpoint["update"]) + 1
            print(f"Resuming training after update {checkpoint['update']}")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_updates = config.updates
    updates_completed = 0
    update_idx = start_update

    while updates_completed < total_updates:
        batch, current_obs = collect_rollouts(
            envs,
            model,
            catalog=catalog,
            steps=config.steps_per_update,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=device,
            current_obs=current_obs,
        )

        metrics = ppo_update(model, optimizer, batch, config)

        stats = batch.stats
        episode_count = stats.get("episodes", 0)
        rewards_list = stats.get("episode_rewards", [])
        avg_episode_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
        trick_win_rate = (
            stats.get("tricks_won", 0) / episode_count if episode_count else 0.0
        )
        contract_attempts = stats.get("contract_attempts", 0)
        contract_success_rate = (
            stats.get("contract_success", 0) / contract_attempts if contract_attempts else 0.0
        )
        opponent_list = batch.stats.get("opponents", [])
        if opponent_list:
            current_opponent = ",".join(sorted(set(opponent_list)))
        else:
            opponent_set = sorted({env.opponent_name for env in envs})
            current_opponent = ",".join(opponent_set)

        is_last_update = updates_completed == total_updates - 1
        if update_idx % 5 == 0 or is_last_update:
            ckpt_path = config.checkpoint_dir / f"ppo_update_{update_idx}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(config),
                    "update": update_idx,
                    "model_hidden": config.model_hidden,
                    "model_lstm": config.model_lstm,
                    "use_legacy_model": config.use_legacy_model,
                },
                ckpt_path,
            )
            print(f"[Update {update_idx}] Saved checkpoint to {ckpt_path}")

        print(
            f"[Update {update_idx}] policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f} "
            f"episodes={episode_count} avg_reward={avg_episode_reward:.3f} "
            f"trick_win_rate={trick_win_rate:.3f} contract_sr={contract_success_rate:.3f}"
        )

        if config.log_csv is not None:
            row = {
                "update": update_idx,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "episodes": episode_count,
                "avg_reward": avg_episode_reward,
                "trick_win_rate": trick_win_rate,
                "contract_success_rate": contract_success_rate,
                "tricks_won": stats.get("tricks_won", 0),
                "tricks_lost": stats.get("tricks_lost", 0),
                "agent_melds": stats.get("agent_melds", 0),
                "opponent_melds": stats.get("opponent_melds", 0),
                "contract_attempts": contract_attempts,
                "opponent": current_opponent,
                "num_envs": config.num_envs,
            }
            _log_metrics_csv(config.log_csv, row)

        updates_completed += 1
        update_idx += 1

        if schedule_phases and phase_updates_remaining is not None:
            phase_updates_remaining -= 1
            if phase_updates_remaining <= 0 and updates_completed < total_updates:
                next_index = (phase_index + 1) if phase_index is not None else 0
                if next_index >= len(schedule_phases):
                    if config.schedule_repeat:
                        next_index = 0
                    else:
                        phase_updates_remaining = schedule_phases[-1].updates
                        continue
                phase_index = next_index
                current_phase = schedule_phases[phase_index]
                active_mix = current_phase.mix
                phase_updates_remaining = current_phase.updates
                envs = build_envs_for_mix(config, catalog, active_mix)
                current_obs = [env.reset() for env in envs]
                if current_obs:
                    obs_length = len(current_obs[0].vector)
                    if obs_length != obs_size:
                        raise ValueError(
                            f"Observation size changed between curriculum phases ({obs_length} vs {obs_size})."
                        )
                print(
                    f"Switched opponent mix to phase {phase_index + 1}/{len(schedule_phases)}: "
                    f"{active_mix} for {phase_updates_remaining} updates."
                )


if __name__ == "__main__":
    main()
