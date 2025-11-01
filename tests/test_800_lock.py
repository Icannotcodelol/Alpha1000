from engine.scoring import score_hand


def test_defender_does_not_progress_past_800():
    result = score_hand(
        playing_player=1,
        winning_bid=120,
        contract=120,
        card_points=[20, 90],
        meld_points=[0, 40],
        prior_scores=[800, 780],
    )

    assert result.new_scores == (800, 900)
    assert result.defender_points_added == 0


def test_defender_clamped_to_800():
    result = score_hand(
        playing_player=0,
        winning_bid=120,
        contract=120,
        card_points=[80, 50],
        meld_points=[0, 25],
        prior_scores=[0, 790],
    )

    assert result.new_scores == (-120, 800)
    assert result.defender_points_added == 10
