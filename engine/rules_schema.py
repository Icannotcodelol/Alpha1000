"""Validation schema for Alpha-1000 rules configuration."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, validator

SUIT_NAMES = ("spades", "clubs", "diamonds", "hearts")


def _validate_suit(value: str) -> str:
    normalized = value.lower()
    if normalized not in SUIT_NAMES:
        raise ValueError(f"Unknown suit: {value!r}")
    return normalized


class MeldConfig(BaseModel):
    trump_overrides: bool = Field(True, description="Whether declaring a meld sets/overrides trump immediately.")
    allow_one_per_suit: bool = Field(True, description="Prevent multiple meld declarations per player per suit.")


class MusikConfig(BaseModel):
    return_destination: Literal["unused_musik", "separate_discard"] = Field(
        "unused_musik",
        description="Where the two returned cards go after inspecting the musik.",
    )


class ScoringConfig(BaseModel):
    last_trick_bonus: int = Field(0, ge=0, description="Bonus points awarded to the last trick winner.")
    overshoot_behavior: Literal["none", "cap"] = Field(
        "none",
        description="Behavior when players exceed 1000 points.",
    )


class RuleSet(BaseModel):
    deck_ranks: list[str]
    deck_suits: list[str]
    card_points: dict[str, int]
    meld_points: dict[str, int]
    musik: MusikConfig = Field(default_factory=MusikConfig)
    melds: MeldConfig = Field(default_factory=MeldConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

    @validator("deck_ranks", "deck_suits")
    def ensure_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Deck configuration must not be empty.")
        return value

    @validator("deck_suits")
    def validate_suits(cls, value: list[str]) -> list[str]:
        return [_validate_suit(suit) for suit in value]

    @validator("card_points")
    def validate_card_points(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("Card point mapping cannot be empty.")
        for rank, points in value.items():
            if points < 0:
                raise ValueError(f"Card {rank} has negative points.")
        return value

    @validator("meld_points")
    def validate_meld_points(cls, value: dict[str, int]) -> dict[str, int]:
        for suit, points in value.items():
            _validate_suit(suit)
            if points <= 0:
                raise ValueError(f"Meld points for {suit} must be positive.")
        return value
