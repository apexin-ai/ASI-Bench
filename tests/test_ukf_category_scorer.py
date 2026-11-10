"""UKF scoring on generated instances without early ordinary updates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tasks.robotics.ukf_bearing_only_tracking.custom_scorer import (
    UKFBearingCategoryScorer,
)


@pytest.fixture
def instance(tmp_path):
    pred, ref = tmp_path / "pred", tmp_path / "reference"
    pred.mkdir()
    ref.mkdir()
    state = np.zeros((6, 4), dtype=np.float64)
    for root in (pred, ref):
        np.save(root / "state_history.npy", state)
    np.save(ref / "true_states_ref.npy", state)
    outliers = np.array([0, 1, 0, 1, 0], dtype=np.float64)
    np.save(pred / "measurement_outlier_prob.npy", outliers)
    np.save(ref / "outlier_mask.npy", outliers)
    masks = {"wrap_update": [1, 2], "early_normal": [], "late_normal": [4, 5]}
    (ref / "category_masks.json").write_text(json.dumps(masks), encoding="utf-8")
    task_dir = Path(__file__).parents[1] / "tasks/robotics/ukf_bearing_only_tracking"
    evaluation = yaml.safe_load((task_dir / "task_eval.yaml").read_text())["evaluation"]
    config = {**evaluation["scoring"][0]["config"], "weight": 100}
    return pred, ref, config


def test_perfect_prediction_can_earn_full_credit_without_early_samples(instance):
    pred, ref, config = instance
    result = UKFBearingCategoryScorer().score(pred, ref, config)
    assert result.score == 100
    assert result.details["total_available"] == 100
    categories = result.details["per_category"]
    assert categories["early_normal"]["available"] == 0
    assert categories["late_normal"]["available"] == 60


def test_transferred_points_still_require_accurate_late_tracking(instance):
    pred, ref, config = instance
    state = np.load(pred / "state_history.npy")
    state[4:, 0] = 2.0
    state[4:, 2] = 2.0  # combined error = 2 + 0.5*2 = 3; factor = 1 - 3/20
    np.save(pred / "state_history.npy", state)
    result = UKFBearingCategoryScorer().score(pred, ref, config)
    assert result.details["per_category"]["late_normal"]["earned"] == 51
    assert result.score == pytest.approx(35 + 60 * 0.85 + 5)


def test_populated_early_category_retains_original_weight_and_threshold(instance):
    pred, ref, config = instance
    path = ref / "category_masks.json"
    masks = json.loads(path.read_text())
    masks["early_normal"] = [3]
    path.write_text(json.dumps(masks), encoding="utf-8")
    scorer = UKFBearingCategoryScorer()
    assert scorer.score(pred, ref, config).score == 100
    state = np.load(pred / "state_history.npy")
    state[3, 0] = 1.26  # just above the original early threshold of 1.25
    np.save(pred / "state_history.npy", state)
    result = scorer.score(pred, ref, config)
    assert result.score == 95
    categories = result.details["per_category"]
    assert categories["early_normal"]["available"] == 5
    assert categories["late_normal"]["available"] == 55


def test_repeated_evaluations_do_not_accumulate_weight_transfers(instance):
    pred, ref, config = instance
    original = copy.deepcopy(config)
    scorer = UKFBearingCategoryScorer()
    for _ in range(2):
        result = scorer.score(pred, ref, config)
        assert result.details["per_category"]["late_normal"]["available"] == 60
        assert config == original


def test_empty_late_category_cannot_receive_transferred_credit(instance):
    pred, ref, config = instance
    path = ref / "category_masks.json"
    masks = json.loads(path.read_text())
    masks["late_normal"] = []
    path.write_text(json.dumps(masks), encoding="utf-8")
    with pytest.raises(ValueError, match="late_normal has no scoring evidence"):
        UKFBearingCategoryScorer().score(pred, ref, config)
