"""Baseline bot focused on trump management."""

from __future__ import annotations

from typing import Optional, Tuple

from engine.cards import CARD_POINTS, Card, Rank
from engine.game import HandEngine
from engine.state import GameState

from .baseline_greedy import GreedyBot, _should_meld


class TrumpManagerBot(GreedyBot):
    name = "TrumpManager"

    def play_card(
        self, hand: HandEngine, player: int
    ) -> Tuple[Card, bool, Optional[Card]]:
        assert hand.state is not None
        state = hand.state
        legal = list(state.available_moves(player))
        trump = state.trump

        chosen = None
        if trump is not None:
            non_trump = [card for card in legal if card.suit is not trump]
            if non_trump:
                chosen = min(non_trump, key=lambda c: (CARD_POINTS[c.rank], c.rank.value))
            else:
                chosen = min(legal, key=lambda c: (CARD_POINTS[c.rank], c.rank.value))

        if chosen is None:
            legal.sort(key=lambda c: (CARD_POINTS[c.rank], c.rank.value))
            chosen = legal[-1]

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
