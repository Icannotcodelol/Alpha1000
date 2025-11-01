from engine.cards import Card, Rank, Suit
from engine.mechanics import legal_moves
from engine.trick import Trick


def test_must_follow_and_overtake_when_possible():
    trick = Trick(leader=0)
    trick.add_play(0, Card(Rank.TEN, Suit.CLUBS))

    hand = [
        Card(Rank.ACE, Suit.CLUBS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.NINE, Suit.HEARTS),
    ]

    moves = legal_moves(hand, trick, trump=Suit.HEARTS)
    assert moves == [Card(Rank.ACE, Suit.CLUBS)]


def test_must_play_trump_when_void_in_led_suit():
    trick = Trick(leader=0)
    trick.add_play(0, Card(Rank.JACK, Suit.SPADES))

    hand = [
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.TEN, Suit.CLUBS),
    ]

    moves = legal_moves(hand, trick, trump=Suit.HEARTS)
    assert moves == [
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
    ]


def test_must_overtrump_when_possible():
    trick = Trick(leader=0)
    trick.add_play(0, Card(Rank.TEN, Suit.HEARTS))

    hand = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.NINE, Suit.SPADES),
    ]

    moves = legal_moves(hand, trick, trump=Suit.HEARTS)
    assert moves == [Card(Rank.ACE, Suit.HEARTS)]
