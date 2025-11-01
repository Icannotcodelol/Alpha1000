"""High-level game orchestration for Alpha-1000."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from random import Random
from typing import Iterable, List, Optional, Sequence, Tuple

from .bidding import Auction, BiddingError
from .cards import Card
from .deck import build_deck, deal_two_player
from .musik import InvalidMusikOperation, MusikPiles
from .scoring import HandScoreResult, InvalidContract, score_hand
from .state import GameState


class HandPhase(Enum):
    AUCTION = auto()
    MUSIK = auto()
    PLAY = auto()
    COMPLETE = auto()


@dataclass
class HandEngine:
    """Manage a single hand of Tysiąc."""

    starting_player: int
    rng: Optional[Random] = None
    deck: Optional[Sequence[Card]] = None

    phase: HandPhase = field(init=False, default=HandPhase.AUCTION)
    hands: List[List[Card]] = field(init=False)
    musik: MusikPiles = field(init=False)
    auction: Auction = field(init=False)
    playing_player: Optional[int] = field(init=False, default=None)
    winning_bid: Optional[int] = field(init=False, default=None)
    contract: Optional[int] = field(init=False, default=None)
    state: Optional[GameState] = field(init=False, default=None)
    _score_result: Optional[HandScoreResult] = field(default=None, init=False)

    def __post_init__(self) -> None:
        hands, musiki = deal_two_player(rng=self.rng, deck=self.deck)
        self.hands = [list(hand) for hand in hands]
        self.musik = MusikPiles([list(pile) for pile in musiki])
        self.auction = Auction(starting_player=self.starting_player, hands=self.hands)

    def bid(self, player: int, amount: int, proof=None) -> None:
        self._ensure_phase(HandPhase.AUCTION)
        self.auction.bid(player, amount, proof)
        if self.auction.is_complete():
            self.playing_player, self.winning_bid = self.auction.result()
            self.phase = HandPhase.MUSIK

    def pass_bid(self, player: int) -> None:
        self._ensure_phase(HandPhase.AUCTION)
        self.auction.pass_bid(player)
        if self.auction.is_complete():
            self.playing_player, self.winning_bid = self.auction.result()
            self.phase = HandPhase.MUSIK

    def choose_musik(self, player: int, index: int) -> None:
        self._ensure_phase(HandPhase.MUSIK)
        if player != self.playing_player:
            raise InvalidMusikOperation("Only the playing player may select a musik.")
        updated_hand = self.musik.take(self.hands[player], index=index)
        self.hands[player] = updated_hand

    def return_to_unused(self, player: int, cards: Sequence[Card]) -> None:
        self._ensure_phase(HandPhase.MUSIK)
        if player != self.playing_player:
            raise InvalidMusikOperation("Only the playing player may return cards.")
        new_hand = self.musik.return_to_unused(self.hands[player], cards_to_return=cards)
        self.hands[player] = new_hand

    def set_contract(self, value: int) -> None:
        if self.phase == HandPhase.AUCTION:
            raise InvalidContract("Contract cannot be set before the auction completes.")
        if self.winning_bid is None or self.playing_player is None:
            raise InvalidContract("Auction has not produced a winner.")
        if value % 10 != 0 or value < self.winning_bid:
            raise InvalidContract("Contract must be ≥ winning bid and a multiple of 10.")
        self.contract = value
        self._start_play()

    def play_card(self, player: int, card: Card, *, declare_meld: bool = False, meld_partner: Optional[Card] = None) -> None:
        self._ensure_phase(HandPhase.PLAY)
        assert self.state is not None
        self.state.play_card(player, card, declare_meld=declare_meld, meld_partner=meld_partner)
        if card in self.hands[player]:
            self.hands[player].remove(card)
        if self.state.is_finished():
            self.phase = HandPhase.COMPLETE

    def complete_scoring(self, prior_scores: Sequence[int]) -> HandScoreResult:
        self._ensure_phase(HandPhase.COMPLETE)
        if self.contract is None or self.winning_bid is None or self.playing_player is None:
            raise InvalidContract("Contract must be set before scoring.")
        assert self.state is not None
        result = score_hand(
            playing_player=self.playing_player,
            winning_bid=self.winning_bid,
            contract=self.contract,
            card_points=self.state.card_points,
            meld_points=self.state.meld_points,
            prior_scores=prior_scores,
        )
        self._score_result = result
        return result

    def _start_play(self) -> None:
        if self.phase != HandPhase.MUSIK or self.contract is None:
            return
        assert self.playing_player is not None
        defender = 1 - self.playing_player
        hands_for_state = [self.hands[0][:], self.hands[1][:]]
        self.state = GameState(hands=hands_for_state, playing_player=self.playing_player, trump=None)
        self.phase = HandPhase.PLAY

    def _ensure_phase(self, expected: HandPhase) -> None:
        if self.phase != expected:
            raise RuntimeError(f"Action not allowed in phase {self.phase}. Expected {expected}.")


@dataclass
class GameSession:
    """Track scores across multiple hands."""

    seed: Optional[int] = None
    scores: List[int] = field(default_factory=lambda: [0, 0])
    rng: Random = field(init=False)
    current_hand: Optional[HandEngine] = field(default=None, init=False)
    hand_history: List[HandScoreResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def start_hand(self, starting_player: int) -> HandEngine:
        self.current_hand = HandEngine(starting_player=starting_player, rng=self.rng)
        return self.current_hand

    def finish_hand(self) -> HandScoreResult:
        if self.current_hand is None:
            raise RuntimeError("No active hand.")
        if self.current_hand.phase != HandPhase.COMPLETE:
            raise RuntimeError("Cannot finish hand before play is complete.")
        result = self.current_hand.complete_scoring(self.scores)
        self.scores = list(result.new_scores)
        self.hand_history.append(result)
        self.current_hand = None
        return result
