"""Card-related data structures and helpers for Tysiąc."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, Mapping, Optional, Set


class Suit(Enum):
    SPADES = auto()
    CLUBS = auto()
    DIAMONDS = auto()
    HEARTS = auto()

    def __str__(self) -> str:
        return self.name.lower()


class Rank(Enum):
    NINE = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()
    TEN = auto()
    ACE = auto()

    def __str__(self) -> str:
        return self.name.lower()


# Card point values per the official rules.
CARD_POINTS: dict[Rank, int] = {
    Rank.NINE: 0,
    Rank.JACK: 2,
    Rank.QUEEN: 3,
    Rank.KING: 4,
    Rank.TEN: 10,
    Rank.ACE: 11,
}

# Rank order from lowest to highest for trick resolution.
RANK_ORDER: list[Rank] = [
    Rank.NINE,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.TEN,
    Rank.ACE,
]

RANK_STRENGTH: dict[Rank, int] = {rank: index for index, rank in enumerate(RANK_ORDER)}


# Meld (marriage) point values keyed by suit.
MARRIAGE_POINTS: dict[Suit, int] = {
    Suit.SPADES: 40,
    Suit.CLUBS: 60,
    Suit.DIAMONDS: 80,
    Suit.HEARTS: 100,
}


@dataclass(frozen=True, order=True)
class Card:
    """Immutable representation of a playing card."""

    rank: Rank
    suit: Suit

    def point_value(self) -> int:
        return CARD_POINTS[self.rank]


def card_strength(card: Card) -> int:
    """Return an integer strength used for ordering cards within a suit."""
    return RANK_STRENGTH[card.rank]


def is_complete_marriage(cards: Iterable[Card], suit: Suit) -> bool:
    """Return True if the iterable contains both K and Q of the given suit."""
    seen: Set[Rank] = {card.rank for card in cards if card.suit is suit}
    return Rank.KING in seen and Rank.QUEEN in seen


def marriage_points_for_suit(suit: Suit) -> int:
    return MARRIAGE_POINTS[suit]


def beats(candidate: Card, current: Card, led_suit: Suit, trump: Optional[Suit]) -> bool:
    """Return True if candidate wins over current within the trick context."""
    if candidate == current:
        return False

    candidate_trump = trump is not None and candidate.suit is trump
    current_trump = trump is not None and current.suit is trump

    if candidate_trump and not current_trump:
        return True
    if current_trump and not candidate_trump:
        return False

    if candidate.suit is current.suit:
        return card_strength(candidate) > card_strength(current)

    if candidate.suit is led_suit and current.suit is not led_suit:
        return True

    return False


def serialize_card(card: Card) -> dict[str, str]:
    return {"rank": card.rank.name.lower(), "suit": card.suit.name.lower()}


def deserialize_card(payload: Mapping[str, str]) -> Card:
    rank_name = payload["rank"].upper()
    suit_name = payload["suit"].upper()
    return Card(Rank[rank_name], Suit[suit_name])


def card_label(card: Card) -> str:
    return f"{card.rank.name.title()} of {card.suit.name.title()}"
