"""Bidding rules and meld-proof validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, Optional, Tuple, Union

from .cards import Card, MARRIAGE_POINTS, Rank, Suit, is_complete_marriage

BASE_BID = 100
BID_STEP = 10
PROOF_THRESHOLD = 120
VALID_MARRIAGE_POINTS = set(MARRIAGE_POINTS.values())


class BiddingError(ValueError):
    """Base class for bidding related errors."""


class BidNotAllowed(BiddingError):
    """Raised when the bid cannot legally be made (invalid step or unattainable)."""


class MissingMeldProof(BiddingError):
    """Raised when a required meld proof is not supplied."""


class InvalidMeldProof(BiddingError):
    """Raised when the provided meld proof does not satisfy the rules."""


@dataclass(frozen=True)
class MeldProof:
    suit: Suit
    cards: frozenset[Card]

    def required_points(self) -> int:
        return MARRIAGE_POINTS[self.suit]

    def is_valid_structure(self) -> bool:
        """Check that the proof is exactly K+Q of the given suit."""
        if len(self.cards) != 2:
            return False
        ranks = {card.rank for card in self.cards}
        suits = {card.suit for card in self.cards}
        return suits == {self.suit} and ranks == {Rank.KING, Rank.QUEEN}


ProofInput = Union[MeldProof, dict]


def validate_bid_with_meld(bid: int, hand: Iterable[Card], proof: Optional[ProofInput]) -> None:
    """Validate a bid against meld proof requirements.

    Raises:
        BidNotAllowed: illegal bid size or unattainable proof requirement.
        MissingMeldProof: proof required but absent.
        InvalidMeldProof: supplied proof invalid or not in hand.
    """
    if bid < BASE_BID or (bid - BASE_BID) % BID_STEP != 0:
        raise BidNotAllowed(f"Bid {bid} must start at {BASE_BID} and move in {BID_STEP} increments.")

    required_points = 0
    if bid > PROOF_THRESHOLD:
        required_points = bid - BASE_BID
        if required_points not in VALID_MARRIAGE_POINTS:
            raise BidNotAllowed(f"Bid {bid} demands {required_points} meld points which is impossible.")

    if required_points == 0:
        # No proof needed; ensure bid is otherwise legal.
        return

    if proof is None:
        raise MissingMeldProof(f"Bid {bid} requires proof of at least {required_points} meld points.")

    normalized = _normalize_proof(proof)
    if not normalized.is_valid_structure():
        raise InvalidMeldProof("Proof must show exactly the King and Queen of the chosen suit.")

    if normalized.required_points() < required_points:
        raise InvalidMeldProof(
            f"Proof worth {normalized.required_points()} points is insufficient for bid {bid}."
        )

    hand_cards = set(hand)
    if not normalized.cards.issubset(hand_cards):
        missing = normalized.cards.difference(hand_cards)
        raise InvalidMeldProof(f"Proof uses cards not present in hand: {sorted(map(str, missing))}")


def _normalize_proof(proof: ProofInput) -> MeldProof:
    if isinstance(proof, MeldProof):
        return proof

    if not isinstance(proof, dict):
        raise InvalidMeldProof("Proof must be a MeldProof or mapping with 'suit' and 'cards'.")

    try:
        suit = proof["suit"]
        cards = proof["cards"]
    except KeyError as exc:
        raise InvalidMeldProof("Proof mapping missing required keys 'suit' and 'cards'.") from exc

    if not isinstance(suit, Suit):
        raise InvalidMeldProof("Proof suit must be a Suit.")

    if not isinstance(cards, (set, frozenset)):
        raise InvalidMeldProof("Proof cards must be provided as a set.")

    if not all(isinstance(card, Card) for card in cards):
        raise InvalidMeldProof("Proof cards must all be Card instances.")

    return MeldProof(suit=suit, cards=frozenset(cards))


class AuctionPhase(Enum):
    ACTIVE = auto()
    COMPLETE = auto()


@dataclass
class Auction:
    """Two-player auction management for Tysiąc."""

    starting_player: int
    hands: List[List[Card]]
    phase: AuctionPhase = AuctionPhase.ACTIVE
    current_player: int = field(init=False)
    highest_bid: Optional[int] = None
    highest_bidder: Optional[int] = None
    passes: List[bool] = field(init=False)
    history: List[Tuple[int, str, Optional[int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_player = self.starting_player
        self.passes = [False, False]

    def bid(self, player: int, amount: int, proof: Optional[object] = None) -> None:
        self._ensure_active(player)

        validate_bid_with_meld(amount, self.hands[player], proof)
        if self.highest_bid is not None and amount <= self.highest_bid:
            raise BidNotAllowed("Bid must exceed the current highest bid.")

        self.highest_bid = amount
        self.highest_bidder = player
        self.passes[player] = False
        self.passes[1 - player] = False
        self.history.append((player, "bid", amount))

        self._advance_turn()

    def pass_bid(self, player: int) -> None:
        self._ensure_active(player)

        if self.highest_bidder is None:
            raise BidNotAllowed("At least one bid must be placed before passing.")
        if player == self.highest_bidder:
            raise BidNotAllowed("Current highest bidder cannot pass on their own turn.")

        self.passes[player] = True
        self.history.append((player, "pass", None))

        if self.highest_bidder is not None and player != self.highest_bidder:
            self.phase = AuctionPhase.COMPLETE
            self.current_player = -1
        else:
            self._advance_turn()

    def _advance_turn(self) -> None:
        if self.phase is AuctionPhase.COMPLETE:
            return
        next_player = 1 - self.current_player
        self.current_player = next_player
        if self.highest_bidder is not None and self.passes[next_player]:
            self.phase = AuctionPhase.COMPLETE
            self.current_player = -1

    def _ensure_active(self, player: int) -> None:
        if self.phase is AuctionPhase.COMPLETE:
            raise BiddingError("Auction already complete.")
        if player != self.current_player:
            raise BiddingError("Not this player's turn to act in the auction.")

    def is_complete(self) -> bool:
        return self.phase is AuctionPhase.COMPLETE and self.highest_bidder is not None

    def result(self) -> Tuple[int, int]:
        if not self.is_complete():
            raise BiddingError("Auction not yet complete.")
        assert self.highest_bidder is not None and self.highest_bid is not None
        return self.highest_bidder, self.highest_bid
