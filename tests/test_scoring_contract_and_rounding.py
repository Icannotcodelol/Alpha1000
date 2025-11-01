import pytest

from engine.scoring import HandScoreResult, InvalidContract, score_hand


def test_scoring_success_and_rounding():
    result = score_hand(
        playing_player=0,
        winning_bid=140,
        contract=140,
        card_points=[100, 20],
        meld_points=[40, 0],
        prior_scores=[0, 0],
    )

    assert isinstance(result, HandScoreResult)
    assert result.contract_success
    assert result.new_scores == (140, 20)
    assert result.defender_points_added == 20


def test_scoring_failure_subtracts_contract():
    result = score_hand(
        playing_player=1,
        winning_bid=120,
        contract=130,
        card_points=[0, 20],
        meld_points=[0, 0],
        prior_scores=[50, 60],
    )

    assert not result.contract_success
    assert result.new_scores == (50, -70)


def test_invalid_contract_rejected():
    with pytest.raises(InvalidContract):
        score_hand(
            playing_player=0,
            winning_bid=120,
            contract=110,
            card_points=[80, 20],
            meld_points=[40, 0],
            prior_scores=[0, 0],
        )
