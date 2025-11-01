"""Baseline counter bot aiming to shed points."""

from __future__ import annotations

from typing import Optional, Tuple

from engine.cards import CARD_POINTS, Card, Rank
from engine.game import HandEngine

from .baseline_greedy import GreedyBot, _should_meld


class CounterBot(GreedyBot):
    name = "Counter"

    def play_card(
        self, hand: HandEngine, player: int
    ) -> Tuple[Card, bool, Optional[Card]]:
        assert hand.state is not None
        state = hand.state
        legal = list(state.available_moves(player))

        chosen = min(legal, key=lambda c: (CARD_POINTS[c.rank], c.rank.value))

        declare = False
        partner = None
        if _should_meld(state, player, chosen):
            partner_rank = Rank.KING if chosen.rank is Rank.QUEEN else Rank.QUEEN
            partner = next(
                (card for card in state.hands[player] if card.suit is chosen.suit and card.rank is partner_rank),
                None,
            )
            declare = partner is not None

        return chosen, declare, partner
