"""The accepted affine sl(4) task has the intended public split."""

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "math.sl4_affine_singular_vector"
TASK_DIR = ROOT / "tasks" / "math" / "sl4_affine_singular_vector"


def _load_scorer():
    spec = importlib.util.spec_from_file_location("accepted_sl4", TASK_DIR / "custom_scorer.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sl4_task_public_contract_and_scorer_import():
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text(encoding="utf-8"))
    evaluation = yaml.safe_load((TASK_DIR / "task_eval.yaml").read_text(encoding="utf-8"))
    assert meta["id"] == TASK_ID
    assert meta["status"] == "final"
    assert evaluation["task_id"] == TASK_ID
    assert "generation" not in evaluation
    assert sorted(path.name for path in TASK_DIR.iterdir() if path.is_file()) == [
        "custom_scorer.py",
        "task_eval.yaml",
        "task_meta.yaml",
    ]
    scorer = _load_scorer()
    from ai4sci_bench.core.scorer import list_scorers

    assert "affine_sl4_output_contract" in list_scorers()
    assert scorer.AffineSl4OutputContractScorer is not None


def test_sl4_public_scorer_rejects_missing_result(tmp_path):
    scorer = _load_scorer()
    result = scorer.AffineSl4OutputContractScorer().score(tmp_path, tmp_path, {})
    assert "error" in result.details
    assert result.passed is False
