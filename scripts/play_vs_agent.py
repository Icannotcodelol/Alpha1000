#!/usr/bin/env python3
"""Interactive CLI to play a full match of Tysiąc against the current champion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

from bots.self_play import SelfPlayBot
from engine.cards import Card
from engine.game import HandPhase
from rl.action import ActionCatalog, ActionSpec, ActionType
from rl.env import TysiacEnv
from rl.ppo_lstm.net import build_policy_value_net
from rl.ppo_lstm.train import TrainConfig, infer_model_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Tysiąc against an RL champion.")
    parser.add_argument("--opponent-checkpoint", required=True, help="Checkpoint for the champion opponent.")
    parser.add_argument("--matches", type=int, default=1, help="Number of matches to play (first to 1000).")
    parser.add_argument("--target-score", type=int, default=1000, help="Score required to win a match.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_selfplay_bot(path: Path, device: torch.device) -> SelfPlayBot:
    return SelfPlayBot(path, device=device)


def describe_card(card: Card) -> str:
    return f"{card.rank.name.title()} of {card.suit.name.title()}"


def agent_hand(env: TysiacEnv) -> List[Card]:
    if env.hand is None:
        return []
    if env.hand.state is not None:
        return list(env.hand.state.hands[env.agent_player])
    return list(env.hand.hands[env.agent_player])


def card_from_id(cards: List[Card], catalog: ActionCatalog, card_id: int) -> Card | None:
    for card in cards:
        if catalog.encode_card(card) == card_id:
            return card
    return None


def describe_action(env: TysiacEnv, spec: ActionSpec, catalog: ActionCatalog) -> str:
    hand_cards = agent_hand(env)
    if spec.action_type == ActionType.BID:
        return f"Bid {spec.payload[0]}"
    if spec.action_type == ActionType.PASS:
        return "Pass"
    if spec.action_type == ActionType.CHOOSE_MUSIK:
        return f"Choose musik pile {spec.payload[0]}"
    if spec.action_type == ActionType.RETURN_CARDS:
        cards = [card_from_id(hand_cards, catalog, cid) for cid in spec.payload]
        names = ", ".join(describe_card(card) if card else f"Card#{cid}" for cid, card in zip(spec.payload, cards))
        return f"Return cards: {names}"
    if spec.action_type == ActionType.SET_CONTRACT:
        return f"Set contract {spec.payload[0]}"
    if spec.action_type == ActionType.PLAY_CARD:
        card = card_from_id(hand_cards, catalog, spec.payload[0])
        return f"Play {describe_card(card) if card else f'Card#{spec.payload[0]}'}"
    return spec.action_type.name


def list_legal_actions(env: TysiacEnv, obs, catalog: ActionCatalog) -> List[Tuple[int, int, str]]:
    actions = []
    for idx, allowed in enumerate(obs.legal_mask):
        if allowed:
            spec = catalog.spec(idx)
            label = describe_action(env, spec, catalog)
            actions.append((len(actions), idx, label))
    return actions


def print_hand(env: TysiacEnv) -> None:
    cards = agent_hand(env)
    if not cards:
        return
    print("Your hand:")
    for i, card in enumerate(sorted(cards, key=lambda c: (c.suit.name, c.rank.name))):
        print(f"  - {describe_card(card)}")


def print_state(env: TysiacEnv, obs) -> None:
    print("\n============================")
    print(f"Phase: {obs.info.get('phase')}")
    scores = env.session.scores
    print(f"Scores -> You: {scores[env.agent_player]}, Champion: {scores[env.opponent_player]}")
    if env.hand is not None and env.hand.phase == HandPhase.AUCTION:
        highest = env.hand.auction.highest_bid or 0
        bidder = env.hand.auction.highest_bidder
        print(f"Highest bid: {highest} (player {bidder})")
    if env.hand is not None and env.hand.phase == HandPhase.PLAY:
        if env.hand.state and env.hand.state.current_trick.plays:
            print("Current trick:")
            for player, card in env.hand.state.current_trick.plays:
                print(f"  Player {player} -> {describe_card(card)}")
    print_hand(env)


def choose_action(env: TysiacEnv, obs, catalog: ActionCatalog) -> int:
    options = list_legal_actions(env, obs, catalog)
    if not options:
        print("No legal actions; passing turn.")
        return catalog.action_id(ActionSpec(ActionType.NO_OP, ()))
    for option_idx, action_id, label in options:
        print(f"[{option_idx}] {label}")
    while True:
        choice = input("Select action (q to quit): ").strip()
        if choice.lower() == "q":
            raise KeyboardInterrupt
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        option_idx = int(choice)
        if 0 <= option_idx < len(options):
            return options[option_idx][1]
        print("Invalid choice. Try again.")


def play_match(env: TysiacEnv, catalog: ActionCatalog, target_score: int) -> None:
    while max(env.session.scores) < target_score:
        obs = env.reset()
        done = False
        while not done:
            print_state(env, obs)
            action_id = choose_action(env, obs, catalog)
            obs, reward, done, info = env.step(action_id)
            if done:
                scores = info.get("scores", env.session.scores)
                env.session.scores = list(scores)
                contract = info.get("contract_success")
                print("\nHand complete.")
                if contract is not None:
                    print(f"Contract success: {contract}")
                print(f"Updated scores -> You: {scores[env.agent_player]}, Champion: {scores[env.opponent_player]}\n")
                break
    print("Match finished!")
    scores = env.session.scores
    if scores[env.agent_player] >= target_score:
        print("Congratulations, you won the match!")
    else:
        print("The champion wins this match.")


def main() -> None:
    args = parse_args()
    opponent_path = Path(args.opponent_checkpoint)
    device = torch.device(args.device)
    opponent_bot = load_selfplay_bot(opponent_path, device)
    env = TysiacEnv(opponent=opponent_bot, agent_player=0, include_seen_cards=True)
    catalog = env.catalog
    for match_idx in range(1, args.matches + 1):
        env.session.scores = [0, 0]
        print(f"\n===== Match {match_idx} =====")
        try:
            play_match(env, catalog, args.target_score)
        except KeyboardInterrupt:
            print("\nExiting early.")
            break


if __name__ == "__main__":
    main()
