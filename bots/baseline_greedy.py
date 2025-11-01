"""Baseline greedy bot."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from engine.cards import (
    CARD_POINTS,
    MARRIAGE_POINTS,
    Card,
    Rank,
    Suit,
    is_complete_marriage,
)
from engine.game import HandEngine
from engine.state import GameState

from .base import BotStrategy


def _current_hand(hand: HandEngine, player: int) -> Sequence[Card]:
    if hand.state is not None:
        return hand.state.hands[player]
    return hand.hands[player]


def _find_meld(hand_cards: Sequence[Card]) -> Optional[Suit]:
    best_suit: Optional[Suit] = None
    best_value = 0
    for suit, points in MARRIAGE_POINTS.items():
        if is_complete_marriage(hand_cards, suit) and points > best_value:
            best_suit = suit
            best_value = points
    return best_suit


class GreedyBot(BotStrategy):
    name = "Greedy"

    def offer_bid(self, hand: HandEngine, player: int):
        cards = _current_hand(hand, player)
        best_suit = _find_meld(cards)
        base = 100
        if best_suit:
            target = base + MARRIAGE_POINTS[best_suit]
        else:
            target = base if hand.auction.highest_bid is None else None

        highest = hand.auction.highest_bid or 0
        if target is None or target <= highest:
            return None

        proof = None
        if target > 120 and best_suit:
            marriage_cards = {card for card in cards if card.suit is best_suit and card.rank in {Rank.KING, Rank.QUEEN}}
            proof = {"suit": best_suit, "cards": marriage_cards}

        return {"bid": target, "proof": proof}

    def choose_musik(self, hand: HandEngine, player: int) -> int:
        pile_scores = [
            sum(CARD_POINTS[card.rank] for card in pile)
            for pile in hand.musik.piles
        ]
        return int(pile_scores[0] < pile_scores[1])

    def return_cards(self, hand: HandEngine, player: int) -> Sequence[Card]:
        cards = list(_current_hand(hand, player))
        cards.sort(key=lambda c: (CARD_POINTS[c.rank], c.rank.value))
        return cards[:2]

    def play_card(
        self, hand: HandEngine, player: int
    ) -> Tuple[Card, bool, Optional[Card]]:
        assert hand.state is not None
        legal = list(hand.state.available_moves(player))
        legal.sort(key=lambda c: (CARD_POINTS[c.rank], c.rank.value))
        chosen = legal[-1]

        declare = False
        partner: Optional[Card] = None
        if _should_meld(hand.state, player, chosen):
            partner_rank = Rank.KING if chosen.rank is Rank.QUEEN else Rank.QUEEN
            partner = next(
                (card for card in hand.state.hands[player] if card.suit is chosen.suit and card.rank is partner_rank),
                None,
            )
            declare = partner is not None

        return chosen, declare, partner


def _should_meld(state: GameState, player: int, card: Card) -> bool:
    if state.current_trick.is_empty() is False:
        return False
    if card.rank not in {Rank.KING, Rank.QUEEN}:
        return False
    if card.suit in state.melds_declared[player]:
        return False
    return True
