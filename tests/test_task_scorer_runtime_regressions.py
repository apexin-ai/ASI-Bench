"""Regression coverage for formal-task scorer runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path

from tasks.electrical_engineering.cmos_opamp_design import (
    custom_scorer as cmos_scorer,
)
from tasks.math.levin_context_grid_search import custom_scorer as levin_scorer
from tasks.math.levin_context_grid_search import lts_eval_runtime


def test_levin_scorer_reads_training_data_from_reference_bundle(tmp_path):
    pred_dir = tmp_path / "outputs"
    ref_dir = tmp_path / "reference"
    pred_dir.mkdir()
    (ref_dir / "data").mkdir(parents=True)
    (pred_dir / "analysis.py").write_text(
        "def make_policy(training_data, config):\n"
        "    assert training_data == {'levels': [{'id': 'train'}]}\n"
        "    return object()\n",
        encoding="utf-8",
    )
    (ref_dir / "data" / "training_levels.json").write_text(
        json.dumps({"levels": [{"id": "train"}]}),
        encoding="utf-8",
    )
    (ref_dir / "hidden_eval.json").write_text(
        json.dumps({"budget": 10, "levels": [], "baselines": {}}),
        encoding="utf-8",
    )

    result = levin_scorer.LevinContextGridSearchScorer().score(
        pred_dir,
        ref_dir,
        {"analysis_file": "analysis.py"},
    )

    assert list(pred_dir.iterdir()) == [pred_dir / "analysis.py"]
    assert result.details["hidden_level_count"] == 0
    assert not result.message.startswith("setup failed")


def test_levin_tree_search_returns_constructed_result():
    level = {
        "grid": [".."],
        "start": [0, 0],
        "goal": [0, 1],
        "max_depth": 2,
    }

    class RightPolicy:
        def action_probs(self, _level, _state, _legal):
            return {"R": 1.0}

    result = lts_eval_runtime.levin_tree_search(level, RightPolicy(), budget=2)

    assert result.solved is True
    assert result.solution == ["R"]
    assert result.expansions == 2


def test_cmos_scorer_loads_split_task_metadata():
    metadata = cmos_scorer._load_task_metadata_for_scoring()

    assert metadata["id"] == "electrical_engineering.cmos_opamp_design"
    assert metadata["runtime"]["dockerfile"] == "Dockerfile.os"
    assert "evaluation" in metadata
    assert Path(metadata["_task_dir"]) == Path(cmos_scorer.__file__).resolve().parent
