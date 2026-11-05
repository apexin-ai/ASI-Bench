"""Machine-precision endpoint handling must not erase scientific deductions."""
import math

import pytest

from ai4sci_bench.core.scorer import normalize_task_score


@pytest.mark.parametrize('divisor', [1.0, 1.05, 2.0])
@pytest.mark.parametrize('direction', [0.0, math.inf])
def test_full_score_roundoff_is_normalized(divisor, direction):
    raw_max = 100.0 * divisor
    raw_score = math.nextafter(raw_max, direction)
    score, maximum = normalize_task_score({'score_divisor': divisor}, raw_score, raw_max)
    assert score == maximum


@pytest.mark.parametrize('score', [0.0, 98.57142857142857, 99.0, 99.999, 100 - 2e-12, 100 + 2e-12])
def test_actual_score_differences_are_preserved(score):
    assert normalize_task_score({}, score, 100.0) == (score, 100.0)


@pytest.mark.parametrize('score', [None, math.nan, math.inf, -math.inf])
def test_missing_and_nonfinite_scores_are_not_promoted(score):
    actual, maximum = normalize_task_score({}, score, 100.0)
    assert maximum == 100.0
    if score is None:
        assert actual is None
    elif math.isnan(score):
        assert math.isnan(actual)
    else:
        assert actual == score


@pytest.mark.parametrize('maximum', [0.0, -1.0, math.nan, math.inf])
def test_invalid_or_empty_maximum_does_not_promote_score(maximum):
    score = -1e-13
    actual, _ = normalize_task_score({}, score, maximum)
    assert actual == score


def test_small_score_scale_does_not_turn_zero_into_full_credit():
    assert normalize_task_score({}, 0.0, 5e-13) == (0.0, 5e-13)
