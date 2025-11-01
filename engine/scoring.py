"""Hand scoring helpers for Tysiąc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


class ScoringError(ValueError):
    """Base class for scoring issues."""


class InvalidContract(ScoringError):
    """Raised when the declared contract is illegal."""


@dataclass(frozen=True)
class HandScoreResult:
    new_scores: Tuple[int, int]
    contract_success: bool
    defender_points_added: int


def round_up_to_ten(value: int) -> int:
    if value % 10 == 0:
        return value
    return value + (10 - value % 10)


def score_hand(
    *,
    playing_player: int,
    winning_bid: int,
    contract: int,
    card_points: Sequence[int],
    meld_points: Sequence[int],
    prior_scores: Sequence[int],
) -> HandScoreResult:
    if contract % 10 != 0:
        raise InvalidContract("Contract must be a multiple of 10.")
    if contract < winning_bid:
        raise InvalidContract("Contract must be at least the winning bid.")
    if len(card_points) != 2 or len(meld_points) != 2 or len(prior_scores) != 2:
        raise ScoringError("Exactly two players are supported.")

    defender = 1 - playing_player
    totals = [
        card_points[0] + meld_points[0],
        card_points[1] + meld_points[1],
    ]

    contract_success = totals[playing_player] >= contract
    new_scores = list(prior_scores)

    if new_scores[defender] >= 800:
        defender_add = 0
    else:
        desired = round_up_to_ten(totals[defender])
        max_allowed = max(0, 800 - new_scores[defender])
        defender_add = min(desired, max_allowed)
        new_scores[defender] += defender_add

    if contract_success:
        new_scores[playing_player] += contract
    else:
        new_scores[playing_player] -= contract

    return HandScoreResult(
        new_scores=(new_scores[0], new_scores[1]),
        contract_success=contract_success,
        defender_points_added=defender_add,
    )
