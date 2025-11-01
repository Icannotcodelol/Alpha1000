import pytest

from engine.cards import Card, Rank, Suit
from engine.musik import InvalidMusikOperation, MusikPiles


def starting_hand():
    return [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TEN, Suit.SPADES),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.QUEEN, Suit.CLUBS),
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.TEN, Suit.CLUBS),
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.TEN, Suit.DIAMONDS),
    ]


def musiki():
    return MusikPiles(
        [
            [Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.NINE, Suit.HEARTS)],
            [Card(Rank.JACK, Suit.DIAMONDS), Card(Rank.NINE, Suit.DIAMONDS)],
        ]
    )


def test_take_musik_and_return_cards():
    hand = starting_hand()
    piles = musiki()

    updated_hand = piles.take(hand, index=0)
    assert len(updated_hand) == 12
    assert len(piles.pile(0)) == 0
    assert len(piles.pile(1)) == 2  # untouched unused musik

    to_return = [Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.TEN, Suit.DIAMONDS)]
    final_hand = piles.return_to_unused(updated_hand, cards_to_return=to_return)

    assert len(final_hand) == 10
    unused_pile = piles.pile(1)
    assert all(card in unused_pile for card in to_return)
    assert len(unused_pile) == 4

    with pytest.raises(InvalidMusikOperation):
        piles.return_to_unused(final_hand, cards_to_return=[Card(Rank.ACE, Suit.SPADES), Card(Rank.TEN, Suit.SPADES)])
