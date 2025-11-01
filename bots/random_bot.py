"""Random baseline bot for curriculum training."""

from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

from engine.cards import Card, Rank, Suit
from engine.game import HandEngine
from engine.bidding import BID_STEP, BASE_BID

from .base import BotStrategy


class RandomBot(BotStrategy):
    name = "Random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def offer_bid(self, hand: HandEngine, player: int):
        legal_bids = [bid for bid in range(BASE_BID, 201, BID_STEP)]
        highest = hand.auction.highest_bid or (BASE_BID - BID_STEP)
        legal_above = [bid for bid in legal_bids if bid > highest]
        if not legal_above or self._rng.random() < 0.5:
            return None
        hand_cards = hand.hands[player]
        meld_suits = {
            suit
            for suit in Suit
            if any(card.rank is Rank.KING and card.suit is suit for card in hand_cards)
            and any(card.rank is Rank.QUEEN and card.suit is suit for card in hand_cards)
        }
        candidates = []
        for bid in legal_above:
            required = bid - BASE_BID
            if required <= 20:
                candidates.append((bid, None))
            elif required == 40 and Suit.SPADES in meld_suits:
                candidates.append((bid, Suit.SPADES))
            elif required == 60 and Suit.CLUBS in meld_suits:
                candidates.append((bid, Suit.CLUBS))
            elif required == 80 and Suit.DIAMONDS in meld_suits:
                candidates.append((bid, Suit.DIAMONDS))
            elif required == 100 and Suit.HEARTS in meld_suits:
                candidates.append((bid, Suit.HEARTS))
        if not candidates:
            return None
        bid, suit = self._rng.choice(candidates)
        proof = None
        if suit is not None:
            proof_cards = {
                card
                for card in hand_cards
                if card.suit is suit and card.rank in {Rank.KING, Rank.QUEEN}
            }
            proof = {"suit": suit, "cards": proof_cards}
        return {"bid": bid, "proof": proof}

    def choose_musik(self, hand: HandEngine, player: int) -> int:
        return self._rng.choice([0, 1])

    def return_cards(self, hand: HandEngine, player: int) -> Sequence[Card]:
        cards = list(hand.hands[player])
        self._rng.shuffle(cards)
        return cards[:2]

    def choose_contract(self, hand: HandEngine, player: int) -> int:
        minimum = hand.winning_bid or 100
        bump = self._rng.choice([0, 10, 20])
        return minimum + bump

    def play_card(
        self, hand: HandEngine, player: int
    ) -> Tuple[Card, bool, Optional[Card]]:
        assert hand.state is not None
        legal = list(hand.state.available_moves(player))
        if not legal:
            raise RuntimeError("No legal plays available for bot.")
        choice = self._rng.choice(legal)
        return choice, False, None
