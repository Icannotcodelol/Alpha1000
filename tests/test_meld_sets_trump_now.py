from engine.cards import Card, Rank, Suit
from engine.state import GameState


def test_meld_sets_trump_now():
    state = GameState(
        hands=[
            [
                Card(Rank.QUEEN, Suit.DIAMONDS),
                Card(Rank.KING, Suit.DIAMONDS),
                Card(Rank.TEN, Suit.CLUBS),
            ],
            [
                Card(Rank.ACE, Suit.DIAMONDS),
                Card(Rank.NINE, Suit.HEARTS),
                Card(Rank.TEN, Suit.SPADES),
            ],
        ],
        playing_player=0,
    )

    state.play_card(
        0,
        Card(Rank.QUEEN, Suit.DIAMONDS),
        declare_meld=True,
        meld_partner=Card(Rank.KING, Suit.DIAMONDS),
    )

    assert state.trump == Suit.DIAMONDS
    assert state.meld_points[0] == 80

    # Opponent must still follow suit; trump already set before their move.
    legal_moves = state.available_moves(1)
    assert all(card.suit is Suit.DIAMONDS for card in legal_moves)
