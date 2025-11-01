"""Deck creation utilities for Tysiąc."""

from __future__ import annotations

from random import Random
from typing import Iterable, List, Optional, Sequence, Tuple

from .cards import Card, Rank, Suit, RANK_ORDER


def build_deck() -> List[Card]:
    """Return the ordered 24-card deck."""
    return [Card(rank, suit) for suit in Suit for rank in RANK_ORDER]


def deal_two_player(
    *,
    rng: Optional[Random] = None,
    deck: Optional[Sequence[Card]] = None,
) -> Tuple[List[List[Card]], List[List[Card]]]:
    """Deal two 10-card hands and two musiki of two cards each."""
    if deck is not None:
        cards = list(deck)
    else:
        cards = build_deck()
        if rng is None:
            rng = Random()
        rng.shuffle(cards)
    if len(cards) != 24:
        raise ValueError("Deck must contain exactly 24 cards.")

    hand_size = 10
    musik_size = 2

    hand0 = cards[0:hand_size]
    hand1 = cards[hand_size : 2 * hand_size]
    musik_a = cards[2 * hand_size : 2 * hand_size + musik_size]
    musik_b = cards[2 * hand_size + musik_size : 2 * hand_size + 2 * musik_size]

    return [hand0, hand1], [musik_a, musik_b]
