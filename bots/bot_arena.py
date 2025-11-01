"""Simple bot arena for Alpha-1000."""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, Sequence
from engine.game import GameSession, HandEngine, HandPhase

from .base import BotStrategy
from .baseline_counter import CounterBot
from .baseline_greedy import GreedyBot
from .baseline_trump_manager import TrumpManagerBot

BOT_REGISTRY: Dict[str, type[BotStrategy]] = {
    "greedy": GreedyBot,
    "trump": TrumpManagerBot,
    "counter": CounterBot,
}


def _resolve_auction(hand: HandEngine, bots: Sequence[BotStrategy]) -> None:
    while hand.phase == HandPhase.AUCTION:
        player = hand.auction.current_player
        decision = bots[player].offer_bid(hand, player)
        if decision is None:
            hand.pass_bid(player)
        else:
            bid = decision.get("bid")
            if bid is None:
                raise ValueError("Bid decision missing 'bid'.")
            hand.bid(player, bid, decision.get("proof"))


def _resolve_musik(hand: HandEngine, bots: Sequence[BotStrategy]) -> None:
    playing = hand.playing_player
    if playing is None:
        raise RuntimeError("Auction not resolved before musik phase.")
    index = bots[playing].choose_musik(hand, playing)
    hand.choose_musik(playing, index=index)
    cards = list(bots[playing].return_cards(hand, playing))
    if len(cards) != 2:
        raise ValueError("Bots must return exactly two cards to the unused musik.")
    hand.return_to_unused(playing, cards=cards)
    contract = bots[playing].choose_contract(hand, playing)
    minimum = hand.winning_bid or 100
    if contract < minimum:
        contract = minimum
    if contract % 10 != 0:
        contract += 10 - (contract % 10)
    hand.set_contract(contract)


def _play_out(hand: HandEngine, bots: Sequence[BotStrategy]) -> None:
    while hand.phase == HandPhase.PLAY:
        assert hand.state is not None
        player = hand.state.current_player
        card, declare, partner = bots[player].play_card(hand, player)
        hand.play_card(player, card, declare_meld=declare, meld_partner=partner)


def play_hand(hand: HandEngine, bots: Sequence[BotStrategy]) -> None:
    _resolve_auction(hand, bots)
    _resolve_musik(hand, bots)
    _play_out(hand, bots)


def run_match(
    bot_a: BotStrategy,
    bot_b: BotStrategy,
    *,
    n_hands: int = 10,
    seed: int | None = None,
) -> dict:
    session = GameSession(seed=seed)
    bots = [bot_a, bot_b]
    history = []
    for idx in range(n_hands):
        hand = session.start_hand(starting_player=idx % 2)
        play_hand(hand, bots)
        result = session.finish_hand()
        history.append(
            {
                "scores": result.new_scores,
                "contract_success": result.contract_success,
                "defender_points_added": result.defender_points_added,
            }
        )
    return {"scores": session.scores, "history": history}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a bot match.")
    parser.add_argument("--bot-a", default="greedy", choices=BOT_REGISTRY.keys())
    parser.add_argument("--bot-b", default="trump", choices=BOT_REGISTRY.keys())
    parser.add_argument("--n", type=int, default=10, help="Number of hands to play.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    bot_a = BOT_REGISTRY[args.bot_a]()
    bot_b = BOT_REGISTRY[args.bot_b]()
    results = run_match(bot_a, bot_b, n_hands=args.n, seed=args.seed)

    print(f"Scores after {args.n} hands: {results['scores']}")
    successes = sum(1 for entry in results["history"] if entry["contract_success"])
    print(f"Contract success rate: {successes}/{len(results['history'])}")


if __name__ == "__main__":
    main()
