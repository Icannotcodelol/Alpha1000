"""Musik handling for the bidding winner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .cards import Card


class InvalidMusikOperation(RuntimeError):
    """Raised when musik handling violates the rules."""


@dataclass
class MusikPiles:
    """Manage the two face-down musiki piles during the bidding resolution."""

    piles: List[List[Card]]
    _chosen_index: int | None = None
    _returned: bool = False

    def __post_init__(self) -> None:
        if len(self.piles) != 2:
            raise InvalidMusikOperation("Exactly two musiki must be provided.")
        self.piles = [list(pile) for pile in self.piles]

    def take(self, hand: Sequence[Card], *, index: int) -> List[Card]:
        """Return a new hand with the selected musik cards added."""
        if self._chosen_index is not None:
            raise InvalidMusikOperation("Musik already taken.")
        if index not in (0, 1):
            raise InvalidMusikOperation("Musik index must be 0 or 1.")
        taken_cards = list(self.piles[index])
        if len(taken_cards) != 2:
            raise InvalidMusikOperation("Selected musik must contain exactly two cards.")

        self.piles[index] = []
        self._chosen_index = index
        return list(hand) + taken_cards

    def return_to_unused(self, hand: Sequence[Card], *, cards_to_return: Sequence[Card]) -> List[Card]:
        """Return two cards to the unused musik after the winner inspects their cards."""
        if self._chosen_index is None:
            raise InvalidMusikOperation("Must take a musik before returning cards.")
        if self._returned:
            raise InvalidMusikOperation("Cards have already been returned to the unused musik.")
        if len(cards_to_return) != 2:
            raise InvalidMusikOperation("Exactly two cards must be returned to the unused musik.")

        new_hand = list(hand)
        for card in cards_to_return:
            try:
                new_hand.remove(card)
            except ValueError as exc:
                raise InvalidMusikOperation("Returned cards must come from the current hand.") from exc

        unused_index = 1 - self._chosen_index
        self.piles[unused_index].extend(cards_to_return)
        self._returned = True
        return new_hand

    def pile(self, index: int) -> tuple[Card, ...]:
        if index not in (0, 1):
            raise InvalidMusikOperation("Musik index must be 0 or 1.")
        return tuple(self.piles[index])

    @property
    def chosen_index(self) -> int | None:
        return self._chosen_index
