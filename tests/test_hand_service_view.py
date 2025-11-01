from engine.deck import build_deck
from engine.game import HandEngine
from engine.service import HandService


def test_hand_service_initial_view():
    service = HandService()
    service.session.current_hand = HandEngine(starting_player=0, deck=build_deck())

    view = service.get_hand_view(perspective=0)
    assert view.phase == "auction"
    assert len(view.hand) == 10
    assert view.highest_bid is None
    assert view.musik["pile_sizes"] == [2, 2]


def test_hand_service_updates_after_bid():
    service = HandService()
    service.session.current_hand = HandEngine(starting_player=0, deck=build_deck())

    service.place_bid(0, 100)
    view = service.get_hand_view(0)
    assert view.highest_bid == 100
    assert view.auction_history[-1]["player"] == 0
    assert view.auction_history[-1]["action"] == "bid"
