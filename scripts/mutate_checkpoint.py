#!/usr/bin/env python3
"""Create a mutated variant of a PPO checkpoint by adding Gaussian noise."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mutate PPO checkpoint weights.")
    parser.add_argument("--source", required=True, help="Source checkpoint path.")
    parser.add_argument("--output", required=True, help="Destination checkpoint path.")
    parser.add_argument("--sigma", type=float, default=0.02, help="Standard deviation of Gaussian noise.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    checkpoint = torch.load(args.source, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise RuntimeError("Checkpoint missing model_state")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    mutated = {}
    for name, tensor in state_dict.items():
        if tensor.dtype.is_floating_point:
            noise = torch.randn_like(tensor) * args.sigma
            mutated[name] = tensor + noise
        else:
            mutated[name] = tensor
    checkpoint["model_state"] = mutated
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Mutated checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
