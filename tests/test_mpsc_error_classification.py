"""MPSC scorer failures distinguish submissions from evaluator infrastructure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MPSC_DIR = ROOT / "tasks" / "math" / "mpsc_safety_filter"


@pytest.fixture
def mpsc_module():
    module_name = "mpsc_custom_scorer_error_test"
    sys.path.insert(0, str(MPSC_DIR))
    spec = importlib.util.spec_from_file_location(
        module_name,
        MPSC_DIR / "custom_scorer.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(MPSC_DIR))


@pytest.mark.parametrize(
    "scorer_class_name",
    ["MPSCInterfaceSmokeScorer", "MPSCSafetyFilterScorer"],
)
def test_missing_evaluator_inputs_are_internal_errors(
    monkeypatch, tmp_path, mpsc_module, scorer_class_name
):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "reference"
    pred_dir.mkdir()
    ref_dir.mkdir()
    (pred_dir / "analysis.py").write_text("def filter_control(*args): return [0.0]\n")

    def missing_inputs(*_args, **_kwargs):
        raise FileNotFoundError("hidden_cases.json is missing")

    monkeypatch.setattr(mpsc_module, "load_task_data", missing_inputs)
    monkeypatch.setattr(
        mpsc_module,
        "activate_declared_solver_dependencies",
        lambda: True,
    )

    scorer = getattr(mpsc_module, scorer_class_name)()
    detail = scorer.score(pred_dir, ref_dir, {})

    assert detail.score == 0.0
    assert detail.details["failure_kind"] == "scorer_internal_error"
    assert detail.details["scorer_internal_error"] is True
    assert "hidden_cases.json" in detail.details["setup_error"]


def test_submission_initialization_failure_is_a_scored_zero(
    monkeypatch, tmp_path, mpsc_module
):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "reference"
    pred_dir.mkdir()
    ref_dir.mkdir()
    (pred_dir / "analysis.py").write_text("raise RuntimeError('bad submission')\n")
    monkeypatch.setattr(
        mpsc_module,
        "load_task_data",
        lambda *_args: (
            {"system": {}, "public_cases": {"cases": []}},
            {"queries": [{}], "rollouts": [{}]},
            {},
        ),
    )

    class BrokenSubmissionController:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("could not import submitted analysis.py")

    monkeypatch.setattr(
        mpsc_module,
        "IsolatedSubmissionController",
        BrokenSubmissionController,
    )
    monkeypatch.setattr(
        mpsc_module,
        "activate_declared_solver_dependencies",
        lambda: True,
    )

    detail = mpsc_module.MPSCInterfaceSmokeScorer().score(pred_dir, ref_dir, {})

    assert detail.score == 0.0
    assert detail.details["failure_kind"] == "submission_error"
    assert detail.details["scorer_internal_error"] is False
    assert "submitted analysis.py" in detail.details["setup_error"]


def test_unavailable_declared_runtime_is_an_internal_error(
    monkeypatch, tmp_path, mpsc_module
):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "reference"
    pred_dir.mkdir()
    ref_dir.mkdir()
    (pred_dir / "analysis.py").write_text("def filter_control(*args): return [0.0]\n")
    monkeypatch.setattr(
        mpsc_module,
        "activate_declared_solver_dependencies",
        lambda: False,
    )

    detail = mpsc_module.MPSCSafetyFilterScorer().score(pred_dir, ref_dir, {})

    assert detail.score == 0.0
    assert detail.details["failure_kind"] == "scorer_internal_error"
    assert detail.details["scorer_internal_error"] is True
    assert "runtime" in detail.message.lower()


def test_missing_reference_certificate_is_internal_and_closes_controller(
    monkeypatch, tmp_path, mpsc_module
):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "reference"
    pred_dir.mkdir()
    ref_dir.mkdir()
    (pred_dir / "analysis.py").write_text("def filter_control(*args): return [0.0]\n")

    class Controller:
        closed = False

        def close(self):
            self.closed = True

    controller = Controller()
    monkeypatch.setattr(
        mpsc_module,
        "activate_declared_solver_dependencies",
        lambda: True,
    )
    monkeypatch.setattr(
        mpsc_module,
        "initialize_submission",
        lambda *_args: (
            {"system": {}},
            {"queries": [], "rollouts": []},
            controller,
            {"framework_gt_selfcheck": False},
        ),
    )
    monkeypatch.setattr(mpsc_module, "smoke_call_filter", lambda *_args: {})

    detail = mpsc_module.MPSCSafetyFilterScorer().score(pred_dir, ref_dir, {})

    assert controller.closed is True
    assert detail.score == 0.0
    assert detail.details["failure_kind"] == "scorer_internal_error"
    assert detail.details["scorer_internal_error"] is True
    assert "reference certificate" in detail.details["evaluation_error"]
