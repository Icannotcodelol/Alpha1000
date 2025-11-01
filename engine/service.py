"""Convenience service layer for UI and agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .cards import Card, card_label, deserialize_card, serialize_card
from .game import GameSession, HandEngine, HandPhase


@dataclass
class TrickPlayView:
    player: int
    card: dict
    label: str


@dataclass
class TrickView:
    leader: int
    plays: list[TrickPlayView]


@dataclass
class HandView:
    phase: str
    playing_player: Optional[int]
    current_player: Optional[int]
    trump: Optional[str]
    winning_bid: Optional[int]
    contract: Optional[int]
    hand: list[dict]
    hand_labels: list[str]
    legal_moves: list[dict]
    legal_move_labels: list[str]
    meld_points: list[int]
    card_points: list[int]
    remaining_cards: list[int]
    trick: Optional[TrickView]
    trick_history: list[list[dict]]
    auction_history: list[dict]
    highest_bid: Optional[int]
    musik: dict


@dataclass
class SessionView:
    scores: list[int]
    hand: Optional[HandView]


class HandService:
    """Facade around GameSession for UI consumers."""

    def __init__(self, session: Optional[GameSession] = None) -> None:
        self.session = session or GameSession()

    # Session lifecycle -------------------------------------------------

    def start_new_hand(self, starting_player: int = 0) -> HandView:
        self.session.start_hand(starting_player=starting_player)
        return self.get_hand_view()

    def has_active_hand(self) -> bool:
        return self.session.current_hand is not None

    # Actions -----------------------------------------------------------

    def place_bid(self, player: int, amount: int, proof: Optional[dict] = None) -> HandView:
        hand = self._require_hand()
        hand.bid(player, amount, proof)
        return self.get_hand_view(player)

    def pass_bid(self, player: int) -> HandView:
        hand = self._require_hand()
        hand.pass_bid(player)
        return self.get_hand_view(player)

    def choose_musik(self, player: int, index: int) -> HandView:
        hand = self._require_hand()
        hand.choose_musik(player, index=index)
        return self.get_hand_view(player)

    def return_musik_cards(self, player: int, cards: Sequence[dict]) -> HandView:
        hand = self._require_hand()
        card_objs = [deserialize_card(card) for card in cards]
        hand.return_to_unused(player, cards=card_objs)
        return self.get_hand_view(player)

    def set_contract(self, value: int) -> HandView:
        hand = self._require_hand()
        hand.set_contract(value)
        return self.get_hand_view()

    def play_card(
        self,
        player: int,
        card_payload: dict,
        *,
        declare_meld: bool = False,
        meld_partner: Optional[dict] = None,
    ) -> HandView:
        hand = self._require_hand()
        card = deserialize_card(card_payload)
        partner = deserialize_card(meld_partner) if meld_partner else None
        hand.play_card(player, card, declare_meld=declare_meld, meld_partner=partner)
        return self.get_hand_view(player)

    def finish_hand(self) -> SessionView:
        result = self.session.finish_hand()
        return SessionView(scores=list(result.new_scores), hand=None)

    # Views -------------------------------------------------------------

    def get_session_view(self, perspective: int = 0) -> SessionView:
        return SessionView(scores=list(self.session.scores), hand=self.get_hand_view(perspective) if self.has_active_hand() else None)

    def get_hand_view(self, perspective: int = 0) -> HandView:
        hand = self._require_hand()
        phase = hand.phase.name.lower()
        playing_player = hand.playing_player
        current_player = None
        trump = None
        meld_points = [0, 0]
        card_points = [0, 0]
        remaining = [len(hand.hands[0]), len(hand.hands[1])]
        legal_moves: list[Card] = []
        trick_view: Optional[TrickView] = None
        trick_history: list[list[dict]] = []

        if hand.state is not None:
            trump = hand.state.trump.name.lower() if hand.state.trump else None
            meld_points = list(hand.state.meld_points)
            card_points = list(hand.state.card_points)
            remaining = [len(hand.state.hands[0]), len(hand.state.hands[1])]
            current_player = hand.state.current_player

            if not hand.state.current_trick.is_empty():
                trick_view = TrickView(
                    leader=hand.state.current_trick.leader,
                    plays=[
                        TrickPlayView(player=p, card=serialize_card(c), label=card_label(c))
                        for p, c in hand.state.current_trick.plays
                    ],
                )

            trick_history = []
            for leader, lead_card, follower, follow_card, winner in hand.state.trick_history:
                trick_history.append(
                    [
                        {"player": leader, "card": serialize_card(lead_card), "label": card_label(lead_card)},
                        {"player": follower, "card": serialize_card(follow_card), "label": card_label(follow_card)},
                        {"winner": winner},
                    ]
                )

            if hand.state.current_player == perspective:
                legal_moves = hand.state.available_moves(perspective)

        else:
            trump = None
            if hand.phase == HandPhase.AUCTION:
                current_player = hand.auction.current_player
            elif hand.phase == HandPhase.MUSIK:
                current_player = hand.playing_player
            else:
                current_player = None

        visible_hand = self._visible_hand_cards(hand, perspective)
        legal_move_payload = [serialize_card(card) for card in legal_moves]

        return HandView(
            phase=phase,
            playing_player=playing_player,
            current_player=current_player,
            trump=trump,
            winning_bid=hand.winning_bid,
            contract=hand.contract,
            hand=[serialize_card(card) for card in visible_hand],
            hand_labels=[card_label(card) for card in visible_hand],
            legal_moves=legal_move_payload,
            legal_move_labels=[card_label(card) for card in legal_moves],
            meld_points=meld_points,
            card_points=card_points,
            remaining_cards=remaining,
            trick=trick_view,
            trick_history=trick_history,
            auction_history=[
                {"player": player, "action": action, "amount": amount}
                for player, action, amount in hand.auction.history
            ],
            highest_bid=hand.auction.highest_bid,
            musik={
                "chosen_index": hand.musik.chosen_index,
                "pile_sizes": [len(pile) for pile in hand.musik.piles],
            },
        )

    # Helpers -----------------------------------------------------------

    def _visible_hand_cards(self, hand: HandEngine, perspective: int) -> list[Card]:
        if hand.state is not None:
            return list(hand.state.hands[perspective])
        return list(hand.hands[perspective])

    def _require_hand(self) -> HandEngine:
        if self.session.current_hand is None:
            raise RuntimeError("No active hand.")
        return self.session.current_hand
