import pytest

import pytest

from engine.bidding import Auction, BiddingError
from engine.cards import Card, Rank, Suit


def sample_hands():
    hand0 = [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TEN, Suit.SPADES),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.NINE, Suit.SPADES),
        Card(Rank.ACE, Suit.CLUBS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.CLUBS),
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.NINE, Suit.CLUBS),
    ]
    hand1 = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.DIAMONDS),
        Card(Rank.NINE, Suit.DIAMONDS),
    ]
    return [hand0, hand1]


def test_basic_auction_flow():
    auction = Auction(starting_player=0, hands=sample_hands())

    auction.bid(0, 100)
    auction.bid(1, 110)

    with pytest.raises(BiddingError):
        auction.pass_bid(1)

    auction.bid(0, 120)
    auction.pass_bid(1)

    assert auction.is_complete()
    bidder, bid = auction.result()
    assert bidder == 0
    assert bid == 120
