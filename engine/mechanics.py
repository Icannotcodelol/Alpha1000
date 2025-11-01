"""Legal move generation for Tysiąc."""

from __future__ import annotations

from typing import Iterable, List, Optional

from .cards import Card, Suit, beats, card_strength
from .trick import Trick


def legal_moves(hand: Iterable[Card], trick: Trick, trump: Optional[Suit]) -> List[Card]:
    """Return the subset of cards that are legal to play given the current trick."""
    cards = list(hand)
    if trick.is_empty():
        return sorted(cards, key=card_strength)

    led = trick.led_suit()
    assert led is not None

    _, winning_card = trick.winning_play(trump)
    current_winning_suit = winning_card.suit

    in_led = [card for card in cards if card.suit is led]
    if in_led:
        winning_led = [card for card in in_led if beats(card, winning_card, led, trump)]
        return winning_led if winning_led else sorted(in_led, key=card_strength)

    if trump is not None:
        trump_cards = [card for card in cards if card.suit is trump]
        if trump_cards:
            if current_winning_suit is trump:
                winning_trump = [card for card in trump_cards if beats(card, winning_card, led, trump)]
                return winning_trump if winning_trump else sorted(trump_cards, key=card_strength)
            return sorted(trump_cards, key=card_strength)

    return sorted(cards, key=card_strength)
