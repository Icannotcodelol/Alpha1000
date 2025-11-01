import pytest

from engine.cards import Card, Rank, Suit
from engine.game import HandEngine, HandPhase
from engine.state import GameState
from rl.action import ActionCatalog, ActionType
from rl.env import TysiacEnv
from rl.action import ActionSpec


def test_env_reset_provides_observation():
    env = TysiacEnv()
    obs = env.reset()
    assert len(obs.vector) > 0
    assert any(obs.legal_mask)
    assert obs.info["phase"] in {"auction", "musik", "play"}
    assert obs.info["agent_player"] == 0


def test_env_step_progresses():
    env = TysiacEnv()
    catalog = ActionCatalog()
    obs = env.reset()

    # pick first legal action
    action = next(idx for idx, allowed in enumerate(obs.legal_mask) if allowed)
    next_obs, reward, done, info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(next_obs.vector, list)
    assert isinstance(info, dict)


def test_episode_stats_provided_on_done():
    env = TysiacEnv()
    obs = env.reset()
    for _ in range(200):
        legal = [idx for idx, flag in enumerate(obs.legal_mask) if flag]
        assert legal, "Expected at least one legal action"
        action = legal[0]
        obs, reward, done, info = env.step(action)
        if done:
            assert "episode_stats" in info
            stats = info["episode_stats"]
            assert "reward_total" in stats
            assert "tricks_won" in stats
            break
    else:
        pytest.fail("Hand did not terminate within expected steps")


def test_play_card_auto_declares_meld_when_leading():
    env = TysiacEnv()
    hand = HandEngine(starting_player=0)

    king_hearts = Card(Rank.KING, Suit.HEARTS)
    queen_hearts = Card(Rank.QUEEN, Suit.HEARTS)
    filler = Card(Rank.TEN, Suit.CLUBS)
    defender_card = Card(Rank.NINE, Suit.CLUBS)

    agent_hand = [king_hearts, queen_hearts, filler]
    opponent_hand = [defender_card]

    hand.phase = HandPhase.PLAY
    hand.playing_player = 0
    hand.contract = 100
    hand.winning_bid = 100
    hand.hands = [agent_hand[:], opponent_hand[:]]
    hand.state = GameState(hands=[agent_hand[:], opponent_hand[:]], playing_player=0, trump=None)

    env.hand = hand

    catalog = env.catalog
    king_id = catalog.encode_card(king_hearts)
    spec = ActionSpec(ActionType.PLAY_CARD, (king_id,))

    env._apply_play_action(spec)

    assert hand.state.trump == Suit.HEARTS
    assert hand.state.meld_points[0] > 0
    assert Suit.HEARTS in hand.state.melds_declared[0]
