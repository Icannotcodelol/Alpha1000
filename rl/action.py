"""Discrete action encoding for the Tysiąc environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

from engine.cards import Card
from engine.bidding import BASE_BID, BID_STEP
from engine.encode import encode_card


class ActionType(Enum):
    """Top-level action categories."""

    NO_OP = auto()
    BID = auto()
    PASS = auto()
    CHOOSE_MUSIK = auto()
    RETURN_CARDS = auto()
    SET_CONTRACT = auto()
    PLAY_CARD = auto()


# Only bids supported by rule-required proof thresholds.
BID_VALUES = [100, 110, 120, 140, 160, 180, 200]
CONTRACT_VALUES = list(range(100, 310, 10))
CARD_IDS = list(range(24))
CARD_PAIR_IDS = [tuple(sorted(pair)) for pair in combinations(CARD_IDS, 2)]


@dataclass(frozen=True)
class ActionSpec:
    action_type: ActionType
    payload: Tuple[int, ...]


class ActionCatalog:
    """Map between structured actions and discrete ids."""

    def __init__(self) -> None:
        self._specs: List[ActionSpec] = []
        self._spec_to_id: Dict[ActionSpec, int] = {}
        self._build()

    def _build(self) -> None:
        self._register(ActionSpec(ActionType.NO_OP, ()))
        for value in BID_VALUES:
            self._register(ActionSpec(ActionType.BID, (value,)))
        self._register(ActionSpec(ActionType.PASS, ()))

        for index in (0, 1):
            self._register(ActionSpec(ActionType.CHOOSE_MUSIK, (index,)))

        for pair in CARD_PAIR_IDS:
            self._register(ActionSpec(ActionType.RETURN_CARDS, pair))

        for value in CONTRACT_VALUES:
            self._register(ActionSpec(ActionType.SET_CONTRACT, (value,)))

        for card_id in CARD_IDS:
            self._register(ActionSpec(ActionType.PLAY_CARD, (card_id,)))

    def _register(self, spec: ActionSpec) -> None:
        self._spec_to_id[spec] = len(self._specs)
        self._specs.append(spec)

    @property
    def size(self) -> int:
        return len(self._specs)

    def spec(self, action_id: int) -> ActionSpec:
        return self._specs[action_id]

    def action_id(self, spec: ActionSpec) -> int:
        return self._spec_to_id[spec]

    def encode_card(self, card: Card) -> int:
        return encode_card(card)

    def encode_card_pair(self, cards: Sequence[Card]) -> int:
        encoded = tuple(sorted(encode_card(card) for card in cards))
        spec = ActionSpec(ActionType.RETURN_CARDS, encoded)
        return self.action_id(spec)

    def legal_mask(self, legal_specs: Iterable[ActionSpec]) -> List[int]:
        mask = [0] * len(self._specs)
        for spec in legal_specs:
            action_id = self._spec_to_id.get(spec)
            if action_id is not None:
                mask[action_id] = 1
        return mask

    def describe(self, action_id: int) -> str:
        spec = self.spec(action_id)
        if spec.action_type == ActionType.BID:
            return f"Bid {spec.payload[0]}"
        if spec.action_type == ActionType.PASS:
            return "Pass"
        if spec.action_type == ActionType.CHOOSE_MUSIK:
            return f"Choose musik {spec.payload[0]}"
        if spec.action_type == ActionType.SET_CONTRACT:
            return f"Set contract {spec.payload[0]}"
        if spec.action_type == ActionType.PLAY_CARD:
            return f"Play card #{spec.payload[0]}"
        if spec.action_type == ActionType.RETURN_CARDS:
            return f"Return cards {spec.payload}"
        return spec.action_type.name.title()
