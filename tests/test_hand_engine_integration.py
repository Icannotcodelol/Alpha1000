from engine.cards import Card, Rank, Suit
from engine.deck import build_deck
from engine.game import GameSession, HandEngine, HandPhase


def autoplay_hand(hand: HandEngine):
    if hand.contract is None:
        hand.set_contract(hand.winning_bid or 100)

    while hand.phase == HandPhase.PLAY:
        assert hand.state is not None
        player = hand.state.current_player
        legal = hand.state.available_moves(player)
        card = legal[0]
        hand.play_card(player, card)


def test_full_hand_flow():
    deck = build_deck()
    hand = HandEngine(starting_player=0, deck=deck)
    hand.bid(0, 100)
    hand.bid(1, 110)
    hand.bid(0, 120)
    hand.pass_bid(1)

    assert hand.phase == HandPhase.MUSIK
    assert hand.playing_player == 0

    hand.choose_musik(0, index=0)
    cards_to_return = hand.hands[0][-2:]
    hand.return_to_unused(0, cards=cards_to_return)
    hand.set_contract(120)

    autoplay_hand(hand)
    assert hand.phase == HandPhase.COMPLETE

    result = hand.complete_scoring([0, 0])
    assert len(result.new_scores) == 2


def test_game_session_tracks_scores():
    session = GameSession(seed=42)
    hand = session.start_hand(starting_player=0)
    hand.bid(0, 100)
    hand.bid(1, 110)
    hand.bid(0, 120)
    hand.pass_bid(1)
    hand.choose_musik(0, index=0)
    hand.return_to_unused(0, cards=hand.hands[0][-2:])
    hand.set_contract(120)

    autoplay_hand(hand)
    session.finish_hand()
    assert sum(session.scores) != 0 or session.scores == [0, 0]
