"""Trick representation and resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .cards import Card, Suit, beats


class TrickError(RuntimeError):
    """Raised when trick play breaks ordering constraints."""


@dataclass
class Trick:
    leader: int
    plays: List[Tuple[int, Card]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.plays

    def add_play(self, player: int, card: Card) -> None:
        if len(self.plays) >= 2:
            raise TrickError("Trick already complete.")
        if not self.plays and player != self.leader:
            raise TrickError("Only the leader can start the trick.")
        if self.plays and player == self.plays[0][0]:
            raise TrickError("Leader cannot play twice in the same trick.")
        self.plays.append((player, card))

    def led_suit(self) -> Optional[Suit]:
        return self.plays[0][1].suit if self.plays else None

    def is_full(self) -> bool:
        return len(self.plays) == 2

    def winning_play(self, trump: Optional[Suit]) -> Tuple[int, Card]:
        if not self.plays:
            raise TrickError("Cannot determine winner on empty trick.")
        led = self.led_suit()
        assert led is not None
        winning_player, winning_card = self.plays[0]
        for player, card in self.plays[1:]:
            if beats(card, winning_card, led, trump):
                winning_player, winning_card = player, card
        return winning_player, winning_card
