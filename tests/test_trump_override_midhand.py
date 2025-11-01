from engine.cards import Card, Rank, Suit
from engine.state import GameState


def test_trump_override_midhand():
    state = GameState(
        hands=[
            [
                Card(Rank.QUEEN, Suit.DIAMONDS),
                Card(Rank.KING, Suit.DIAMONDS),
                Card(Rank.JACK, Suit.SPADES),
            ],
            [
                Card(Rank.ACE, Suit.DIAMONDS),
                Card(Rank.QUEEN, Suit.HEARTS),
                Card(Rank.KING, Suit.HEARTS),
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
    state.play_card(1, Card(Rank.ACE, Suit.DIAMONDS))

    assert state.current_player == 1
    assert state.trump == Suit.DIAMONDS

    state.play_card(
        1,
        Card(Rank.QUEEN, Suit.HEARTS),
        declare_meld=True,
        meld_partner=Card(Rank.KING, Suit.HEARTS),
    )

    assert state.trump == Suit.HEARTS
    assert state.meld_points[1] == 100
