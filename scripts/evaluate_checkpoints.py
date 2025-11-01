#!/usr/bin/env python3
"""
Evaluate recent PPO checkpoints against scripted and self-play opponents,
and suggest the best candidate to promote.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


SCRIPTED_OPPONENTS = ("greedy", "counter", "trump")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints.")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="data/checkpoints/ppo_update_*.pt",
        help="Glob pattern for checkpoints to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Evaluation episodes per opponent.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of newest checkpoints to evaluate.",
    )
    parser.add_argument(
        "--selfplay",
        type=str,
        nargs="*",
        default=[],
        help="Paths to self-play opponent checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string passed to eval_arena.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file to append results.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional JSON file to write full results.",
    )
    return parser.parse_args()


def checkpoint_id(path: str | Path) -> int:
    name = Path(path).stem
    try:
        return int(name.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return -1


def run_eval(
    checkpoint: Path,
    opponent: str,
    episodes: int,
    device: str,
    selfplay_checkpoint: Optional[Path] = None,
) -> Tuple[float, float]:
    cmd: List[str] = [
        "./.venv/bin/python",
        "-m",
        "rl.ppo_lstm.eval_arena",
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(episodes),
        "--device",
        device,
    ]
    if opponent == "self":
        if selfplay_checkpoint is None:
            raise ValueError("selfplay_checkpoint is required for self opponent.")
        cmd.extend(
            [
                "--opponent",
                "self",
                "--selfplay-checkpoint",
                str(selfplay_checkpoint),
            ]
        )
    elif opponent != "default":
        cmd.extend(["--opponent", opponent])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    win_line = next(line for line in lines if line.lower().startswith("win rate"))
    avg_line = next(line for line in lines if line.lower().startswith("average return"))
    win = float(win_line.split(":")[1].strip().rstrip("%"))
    avg_return = float(avg_line.split(":")[1].strip())
    return win, avg_return


def evaluate(
    checkpoints: Sequence[Path],
    opponents: Iterable[str],
    selfplay_paths: Sequence[Path],
    episodes: int,
    device: str,
) -> dict:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for checkpoint in checkpoints:
        ck_results: dict[str, dict[str, float]] = {}
        for opponent in opponents:
            if opponent == "self":
                for idx, sp_ckpt in enumerate(selfplay_paths):
                    label = f"self_{sp_ckpt.stem}"
                    win, avg = run_eval(checkpoint, "self", episodes, device, sp_ckpt)
                    ck_results[label] = {"win": win, "avg_return": avg}
            else:
                win, avg = run_eval(checkpoint, opponent, episodes, device)
                ck_results[opponent] = {"win": win, "avg_return": avg}
        results[str(checkpoint)] = ck_results
    return results


def append_csv(path: Path, results: dict) -> None:
    rows: List[dict] = []
    for checkpoint, metrics in results.items():
        row = {"checkpoint": checkpoint}
        for opponent, stats in metrics.items():
            row[f"{opponent}_win"] = stats["win"]
            row[f"{opponent}_avg_return"] = stats["avg_return"]
        rows.append(row)
    header = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def score_checkpoint(metrics: dict, self_targets: Sequence[str]) -> float:
    # Score by minimum win rate across scripted + average win rate vs self opponents.
    scripted_min = min(
        metrics.get(opp, {}).get("win", 0.0) for opp in SCRIPTED_OPPONENTS
    )
    if self_targets:
        self_avg = sum(
            metrics.get(label, {}).get("win", 0.0) for label in self_targets
        ) / len(self_targets)
    else:
        self_avg = 0.0
    return scripted_min + self_avg * 0.5


def pick_best(results: dict, selfplay_paths: Sequence[Path]) -> Optional[str]:
    if not results:
        return None
    self_labels = [f"self_{path.stem}" for path in selfplay_paths]
    best_checkpoint = max(
        results.items(),
        key=lambda item: score_checkpoint(item[1], self_labels),
    )[0]
    return best_checkpoint


def main() -> None:
    args = parse_args()
    checkpoints = sorted(
        Path(p) for p in glob.glob(args.checkpoints)
    )
    checkpoints = sorted(checkpoints, key=checkpoint_id)
    checkpoints = checkpoints[-args.limit :]
    opponents = list(SCRIPTED_OPPONENTS)
    selfplay_paths = [Path(p) for p in args.selfplay]
    if selfplay_paths:
        opponents.append("self")
    results = evaluate(
        checkpoints,
        opponents=opponents,
        selfplay_paths=selfplay_paths,
        episodes=args.episodes,
        device=args.device,
    )
    if args.csv:
        append_csv(Path(args.csv), results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    best = pick_best(results, selfplay_paths)
    print(json.dumps(results, indent=2))
    if best:
        print(f"\nSuggested champion: {best}")


if __name__ == "__main__":
    main()
