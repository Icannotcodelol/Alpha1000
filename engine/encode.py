"""Observation and action encoding helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .cards import Card, Rank, Suit


SUIT_INDEX = {Suit.SPADES: 0, Suit.CLUBS: 1, Suit.DIAMONDS: 2, Suit.HEARTS: 3}
RANK_INDEX = {
    Rank.NINE: 0,
    Rank.JACK: 1,
    Rank.QUEEN: 2,
    Rank.KING: 3,
    Rank.TEN: 4,
    Rank.ACE: 5,
}


def encode_hand_binary(hand: Iterable[Card]) -> List[int]:
    """Return a fixed-length binary vector indicating which cards are in hand."""
    vector = [0] * (len(SUIT_INDEX) * len(RANK_INDEX))
    for card in hand:
        idx = SUIT_INDEX[card.suit] * len(RANK_INDEX) + RANK_INDEX[card.rank]
        vector[idx] = 1
    return vector


def encode_trick(trick_cards: Sequence[Card]) -> List[int]:
    """Simple concatenation of card indices for current trick."""
    encoded: List[int] = []
    for card in trick_cards:
        encoded.append(SUIT_INDEX[card.suit])
        encoded.append(RANK_INDEX[card.rank])
    # Pad to two cards
    while len(encoded) < 4:
        encoded.append(-1)
    return encoded


def encode_card(card: Card) -> int:
    """Encode a card as a single index from 0..23."""
    return SUIT_INDEX[card.suit] * len(RANK_INDEX) + RANK_INDEX[card.rank]


def encode_seen_cards(cards: Iterable[Card]) -> List[int]:
    """Return binary indicators for cards that have been observed."""
    vector = [0] * (len(SUIT_INDEX) * len(RANK_INDEX))
    for card in cards:
        vector[encode_card(card)] = 1
    return vector
