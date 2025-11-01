"""Game state management for Tysiąc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

from .cards import Card, Rank, Suit, marriage_points_for_suit
from .mechanics import legal_moves
from .trick import Trick, TrickError


class InvalidPlay(RuntimeError):
    """Raised when an illegal card play is attempted."""


class InvalidMeld(RuntimeError):
    """Raised when meld declaration rules are violated."""


@dataclass
class GameState:
    hands: List[List[Card]]
    playing_player: int
    trump: Optional[Suit] = None
    current_player: int = field(init=False)
    current_trick: Trick = field(init=False)
    meld_points: List[int] = field(init=False)
    card_points: List[int] = field(init=False)
    melds_declared: List[Set[Suit]] = field(init=False)
    trick_history: List[Tuple[int, Card, int, Card, int]] = field(default_factory=list)
    last_trick_winner: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.hands) != 2:
            raise ValueError("GameState currently supports exactly two players.")
        self.hands = [list(hand) for hand in self.hands]
        self.current_player = self.playing_player
        self.current_trick = Trick(leader=self.current_player)
        self.meld_points = [0, 0]
        self.card_points = [0, 0]
        self.melds_declared = [set(), set()]
        self.last_trick_winner = None

    def opponent(self, player: int) -> int:
        return 1 - player

    def available_moves(self, player: int) -> List[Card]:
        if player != self.current_player:
            raise InvalidPlay("Not this player's turn.")
        return legal_moves(self.hands[player], self.current_trick, self.trump)

    def play_card(
        self,
        player: int,
        card: Card,
        *,
        declare_meld: bool = False,
        meld_partner: Optional[Card] = None,
    ) -> None:
        if player != self.current_player:
            raise InvalidPlay("Not this player's turn.")
        if card not in self.hands[player]:
            raise InvalidPlay("Card not present in hand.")

        legal = self.available_moves(player)
        if card not in legal:
            raise InvalidPlay(f"Card {card} is not legal in this context.")

        if declare_meld:
            self._handle_meld(player, card, meld_partner)

        self.hands[player].remove(card)
        try:
            self.current_trick.add_play(player, card)
        except TrickError as exc:
            raise InvalidPlay(str(exc)) from exc

        if self.current_trick.is_full():
            self._complete_trick()
        else:
            self.current_player = self.opponent(player)

    def _handle_meld(self, player: int, card: Card, meld_partner: Optional[Card]) -> None:
        if not self.current_trick.is_empty():
            raise InvalidMeld("Only the leader may declare a meld.")
        if card.rank not in {Rank.KING, Rank.QUEEN}:
            raise InvalidMeld("Meld can only be declared when playing a King or Queen.")
        if meld_partner is None:
            raise InvalidMeld("Must reveal the partner card when declaring a meld.")
        if meld_partner not in self.hands[player]:
            raise InvalidMeld("Meld partner must be in hand when declared.")
        if meld_partner.suit is not card.suit:
            raise InvalidMeld("Meld cards must share the same suit.")
        expected_partner_rank = Rank.KING if card.rank is Rank.QUEEN else Rank.QUEEN
        if meld_partner.rank is not expected_partner_rank:
            raise InvalidMeld("Meld must consist of the King and Queen.")
        if card.suit in self.melds_declared[player]:
            raise InvalidMeld("This suit has already been melded by the player.")

        self.trump = card.suit
        self.melds_declared[player].add(card.suit)
        self.meld_points[player] += marriage_points_for_suit(card.suit)

    def _complete_trick(self) -> None:
        leader = self.current_trick.leader
        follower = self.opponent(leader)
        lead_card = self.current_trick.plays[0][1]
        follow_card = self.current_trick.plays[1][1]
        winner, _ = self.current_trick.winning_play(self.trump)
        trick_points = lead_card.point_value() + follow_card.point_value()
        self.card_points[winner] += trick_points
        self.trick_history.append((leader, lead_card, follower, follow_card, winner))
        self.last_trick_winner = winner

        self.current_player = winner
        self.current_trick = Trick(leader=winner)

    def is_finished(self) -> bool:
        hands_empty = all(len(hand) == 0 for hand in self.hands)
        return hands_empty and self.current_trick.is_empty()

    def remaining_cards(self) -> Tuple[int, int]:
        return len(self.hands[0]), len(self.hands[1])
