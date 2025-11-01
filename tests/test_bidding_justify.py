import pytest

from engine.bidding import BidNotAllowed, InvalidMeldProof, MissingMeldProof, validate_bid_with_meld
from engine.cards import Card, Rank, Suit


def hand_without_meld():
    """Ten-card hand with no complete marriage."""
    return [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TEN, Suit.SPADES),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.TEN, Suit.CLUBS),
        Card(Rank.NINE, Suit.CLUBS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.NINE, Suit.DIAMONDS),
        Card(Rank.ACE, Suit.HEARTS),
    ]


def hand_with_hearts_meld():
    """Ten-card hand that can prove the 100-point hearts meld."""
    return [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TEN, Suit.SPADES),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.TEN, Suit.CLUBS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.TEN, Suit.DIAMONDS),
        Card(Rank.NINE, Suit.CLUBS),
    ]


def hand_with_diamond_meld():
    """Ten-card hand that can prove the 80-point diamond meld."""
    return [
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TEN, Suit.SPADES),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.TEN, Suit.CLUBS),
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.NINE, Suit.CLUBS),
    ]


def test_bid_200_requires_meld_proof():
    hand = hand_with_hearts_meld()

    with pytest.raises(MissingMeldProof):
        validate_bid_with_meld(bid=200, hand=hand, proof=None)


def test_bid_200_accepts_hearts_meld_proof():
    hand = hand_with_hearts_meld()

    validate_bid_with_meld(
        bid=200,
        hand=hand,
        proof={"suit": Suit.HEARTS, "cards": {Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS)}},
    )


def test_bid_180_requires_80_point_meld_or_higher():
    hand = hand_with_diamond_meld()

    with pytest.raises(InvalidMeldProof):
        validate_bid_with_meld(
            bid=180,
            hand=hand,
            proof={"suit": Suit.SPADES, "cards": {Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.SPADES)}},
        )

    validate_bid_with_meld(
        bid=180,
        hand=hand,
        proof={"suit": Suit.DIAMONDS, "cards": {Card(Rank.KING, Suit.DIAMONDS), Card(Rank.QUEEN, Suit.DIAMONDS)}},
    )


def test_bid_130_is_not_allowed():
    hand = hand_with_hearts_meld()

    with pytest.raises(BidNotAllowed):
        validate_bid_with_meld(
            bid=130,
            hand=hand,
            proof={"suit": Suit.SPADES, "cards": {Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.SPADES)}},
        )
