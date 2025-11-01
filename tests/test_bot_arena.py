from bots.baseline_greedy import GreedyBot
from bots.baseline_trump_manager import TrumpManagerBot
from bots.bot_arena import run_match


def test_run_match_executes():
    results = run_match(GreedyBot(), TrumpManagerBot(), n_hands=2, seed=7)
    assert "scores" in results
    assert len(results["scores"]) == 2
    assert len(results["history"]) == 2
