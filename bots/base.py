"""Common bot strategy interfaces."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from engine.cards import Card
from engine.game import HandEngine


class BotStrategy:
    """Base class for bot policies."""

    name: str = "BaseBot"

    def set_env_context(
        self,
        *,
        catalog=None,
        session=None,
        agent_player: int,
        opponent_player: int,
        include_seen_cards: bool,
    ) -> None:
        """Optional hook to provide environment context before a hand starts."""
        return None

    def on_hand_start(self, hand: HandEngine) -> None:
        """Optional hook invoked at the start of each hand."""
        return None

    def offer_bid(self, hand: HandEngine, player: int):
        """Return a dict with 'bid' and optional 'proof', or None to pass."""
        return None

    def choose_musik(self, hand: HandEngine, player: int) -> int:
        """Return index of musik pile to take."""
        return 0

    def return_cards(self, hand: HandEngine, player: int) -> Sequence[Card]:
        """Return exactly two cards to place back onto the unused musik."""
        return list(hand.hands[player][-2:])

    def choose_contract(self, hand: HandEngine, player: int) -> int:
        """Return the contract declaration."""
        return hand.winning_bid or 100

    def play_card(
        self, hand: HandEngine, player: int
    ) -> Tuple[Card, bool, Optional[Card]]:
        """Return (card, declare_meld, meld_partner)."""
        assert hand.state is not None
        legal = hand.state.available_moves(player)
        if not legal:
            raise RuntimeError("No legal plays available for bot.")
        return legal[0], False, None
