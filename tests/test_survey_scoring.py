"""Tests for the survey scoring matrix and derived metrics."""

import pytest

from src.scoring.survey import (
    SURVEY_QUESTIONS,
    esg_interest_percent,
    max_pillar_scores,
    normalized_pillar_preferences,
    score_survey,
    tau_for_style,
)

FULL = "신경 쓴다."
HALF = "보통이다."
NONE = "신경 쓰지 않는다."


def test_all_positive_answers_reach_pillar_maxima():
    result = score_survey([FULL] * 15)
    maxima = max_pillar_scores()
    assert result.loc["E"].sum() == pytest.approx(maxima["E"])
    assert result.loc["S"].sum() == pytest.approx(maxima["S"])
    assert result.loc["G"].sum() == pytest.approx(maxima["G"])
    # Regression values from the original scoring table.
    assert maxima["E"] == pytest.approx(4.99)
    assert maxima["S"] == pytest.approx(4.79)
    assert maxima["G"] == pytest.approx(4.16)


def test_all_negative_answers_score_zero():
    result = score_survey([NONE] * 15)
    assert (result.values == 0).all()


def test_specific_cells_for_full_answers():
    result = score_survey([FULL] * 15)
    # Q1 contributes 1.0 to E/sustain; Q1 + Q5 give E/msci 0.5 + 0.5.
    assert result.at["E", "sustain"] == pytest.approx(1.0)
    assert result.at["E", "msci"] == pytest.approx(1.0)
    # Q2/Q3/Q4 each add 0.33 to E/iss.
    assert result.at["E", "iss"] == pytest.approx(0.99)


def test_half_answers_use_half_weights():
    result = score_survey([HALF] + [NONE] * 14)
    assert result.at["E", "sustain"] == pytest.approx(0.25)
    assert result.at["E", "msci"] == pytest.approx(0.125)


def test_normalized_preferences_are_within_unit_interval():
    result = score_survey([HALF] * 15)
    prefs = normalized_pillar_preferences(result)
    assert set(prefs) == {"environmental", "social", "governance"}
    for value in prefs.values():
        assert 0.0 < value < 1.0


def test_answer_count_is_validated():
    with pytest.raises(ValueError):
        score_survey([FULL] * 3)


def test_esg_interest_percent():
    assert esg_interest_percent("재무적인 요소를 중심적으로 고려한다.") == pytest.approx(40.0)
    assert esg_interest_percent("ESG 요소를 중심적으로 고려한다.") == pytest.approx(100 / 3)


def test_tau_matches_investment_style():
    # Larger tau = market-implied returns dominate, so the financial style
    # gets the largest tau and the ESG-centered style the smallest.
    assert tau_for_style("재무적인 요소를 중심적으로 고려한다.") == 2.0
    assert tau_for_style("ESG와 재무적인 요소를 모두 고려한다.") == 1.0
    assert tau_for_style("ESG 요소를 중심적으로 고려한다.") == 0.5
    assert tau_for_style("unknown") == 1.0


def test_every_question_has_positive_weights():
    for question in SURVEY_QUESTIONS:
        assert question.pillar in {"E", "S", "G"}
        for full, half in question.weights.values():
            assert full > 0
            assert half > 0
