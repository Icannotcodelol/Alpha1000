#!/usr/bin/env python3
"""
Automated league training loop.

Runs alternating phases of PPO training and evaluation, promoting checkpoints
that beat existing self-play opponents while maintaining scripted win rates.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def checkpoint_id(path: Path) -> int:
    name = path.stem
    try:
        return int(name.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return -1


def sorted_checkpoints(pattern: str) -> List[Path]:
    paths = [Path(p) for p in glob.glob(pattern)]
    return sorted(paths, key=checkpoint_id)


def run_train(
    *,
    resume: Path,
    selfplay_paths: Sequence[Path],
    schedule: str,
    steps_per_update: int,
    updates: int,
    num_envs: int,
    learning_rate: float,
    entropy_coef: float,
    model_hidden: int,
    model_lstm: int,
    device: str,
    log_csv: Path,
    log_path: Path,
) -> None:
    cmd: List[str] = [
        "./.venv/bin/python",
        "-m",
        "rl.ppo_lstm.train",
        "--steps-per-update",
        str(steps_per_update),
        "--updates",
        str(updates),
        "--num-envs",
        str(num_envs),
        "--device",
        device,
        "--learning-rate",
        str(learning_rate),
        "--entropy-coef",
        str(entropy_coef),
        "--resume-checkpoint",
        str(resume),
        "--opponent-mix-schedule",
        schedule,
        "--log-csv",
        str(log_csv),
        "--model-hidden",
        str(model_hidden),
        "--model-lstm",
        str(model_lstm),
    ]
    for path in selfplay_paths:
        cmd.extend(["--selfplay", str(path)])
    print(f"[league] starting training run, resume={resume.name}")
    with log_path.open("w") as log_handle:
        proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Training command failed with code {proc.returncode}")


def run_evaluation(
    *,
    checkpoints_pattern: str,
    limit: int,
    episodes: int,
    selfplay_paths: Sequence[Path],
    device: str,
    json_path: Path,
) -> dict:
    cmd: List[str] = [
        "./scripts/evaluate_checkpoints.py",
        "--checkpoints",
        checkpoints_pattern,
        "--limit",
        str(limit),
        "--episodes",
        str(episodes),
        "--device",
        device,
        "--json",
        str(json_path),
    ]
    for path in selfplay_paths:
        cmd.extend(["--selfplay", str(path)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError("Evaluation script failed.")
    data = json.loads(json_path.read_text())
    return data


def pick_candidate(data: dict, selfplay_paths: Sequence[Path], coverage: float, self_threshold: float) -> Path | None:
    def meets_requirements(metrics: dict) -> bool:
        scripted_ok = all(metrics.get(opp, {}).get("win", 0.0) >= coverage for opp in ("greedy", "counter", "trump"))
        self_ok = all(metrics.get(f"self_{sp.stem}", {}).get("win", 0.0) >= self_threshold for sp in selfplay_paths)
        return scripted_ok and self_ok

    best_path = None
    best_score = None
    for ckpt, metrics in data.items():
        if not meets_requirements(metrics):
            continue
        score = min(metrics.get(opp, {}).get("win", 0.0) for opp in ("greedy", "counter", "trump"))
        if best_score is None or score > best_score:
            best_score = score
            best_path = Path(ckpt)
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated self-play league training loop.")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps-per-update", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.0075)
    parser.add_argument("--model-hidden", type=int, default=512)
    parser.add_argument("--model-lstm", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, required=True, help="Initial checkpoint to resume from.")
    parser.add_argument(
        "--selfplay",
        type=str,
        action="append",
        required=True,
        help="Initial self-play opponent checkpoints. Provide multiple times for multiple opponents.",
    )
    parser.add_argument("--schedule", type=str, required=True, help="Opponent mix schedule.")
    parser.add_argument("--log-csv", type=str, default="data/training_log_trump_v3.csv")
    parser.add_argument("--log-dir", type=str, default="data/run_logs")
    parser.add_argument("--eval-pattern", type=str, default="data/checkpoints/ppo_update_*.pt")
    parser.add_argument("--eval-limit", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--scripted-threshold", type=float, default=95.0, help="Minimum win rate vs each scripted opponent.")
    parser.add_argument("--self-threshold", type=float, default=10.0, help="Minimum win rate vs each self opponent for promotion.")
    parser.add_argument("--max-selfplay", type=int, default=3, help="Maximum number of self-play opponents to retain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    resume = Path(args.resume)
    selfplay_paths: List[Path] = [Path(p) for p in args.selfplay]
    for cycle in range(1, args.cycles + 1):
        log_path = log_dir / f"league_cycle_{cycle}.log"
        run_train(
            resume=resume,
            selfplay_paths=selfplay_paths,
            schedule=args.schedule,
            steps_per_update=args.steps_per_update,
            updates=args.updates,
            num_envs=args.num_envs,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            model_hidden=args.model_hidden,
            model_lstm=args.model_lstm,
            device=args.device,
            log_csv=Path(args.log_csv),
            log_path=log_path,
        )
        checkpoints = sorted_checkpoints(args.eval_pattern)
        if not checkpoints:
            raise RuntimeError("No checkpoints found after training.")
        latest_checkpoint = checkpoints[-1]
        resume = latest_checkpoint
        json_path = log_dir / f"league_eval_cycle_{cycle}.json"
        data = run_evaluation(
            checkpoints_pattern=args.eval_pattern,
            limit=args.eval_limit,
            episodes=args.eval_episodes,
            selfplay_paths=selfplay_paths,
            device=args.device,
            json_path=json_path,
        )
        candidate = pick_candidate(
            data,
            selfplay_paths=selfplay_paths,
            coverage=args.scripted_threshold,
            self_threshold=args.self_threshold,
        )
        if candidate and candidate not in selfplay_paths:
            selfplay_paths.append(candidate)
            if len(selfplay_paths) > args.max_selfplay:
                removed = selfplay_paths.pop(0)
                print(f"[league] Removed oldest self-play bot {removed.name}")
            print(f"[league] Promoted new champion: {candidate.name}")
        else:
            print("[league] No promotion this cycle.")
    print("[league] Training loop complete.")


if __name__ == "__main__":
    main()
