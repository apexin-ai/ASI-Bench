"""MPSC scorer failures distinguish submissions from evaluator infrastructure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ai4sci_bench.core.task import TaskLoader


ROOT = Path(__file__).resolve().parents[1]
MPSC_DIR = ROOT / "tasks" / "math" / "mpsc_safety_filter"


def test_mpsc_local_scoring_opts_into_its_declared_task_runtime():
    metadata = TaskLoader(ROOT / "tasks").load_task_by_id("math.mpsc_safety_filter")

    assert metadata["evaluation"]["runtime"] == "task"
    assert "cvxpy>=1.4" in metadata["_runtime_packages"]
    assert "clarabel>=0.7" in metadata["_runtime_packages"]
    assert "scs>=3.2" in metadata["_runtime_packages"]


def test_mpsc_workspace_launch_reuses_framework_runtime(
    monkeypatch, tmp_path, mpsc_module
):
    worker_path = tmp_path / "worker.py"
    worker_path.write_text("# worker\n", encoding="utf-8")
    trusted_path = tmp_path / "site-packages"
    trusted_path.mkdir()
    runtime_bin = Path(sys.executable).resolve().parent

    monkeypatch.setenv("AI4SCI_TASK_RUNTIME_ACTIVE", "1")
    monkeypatch.setenv("AI4SCI_TASK_RUNTIME_PYTHON", sys.executable)
    monkeypatch.setenv("AI4SCI_TASK_RUNTIME_BIN", str(runtime_bin))
    monkeypatch.setenv("AI4SCI_TRUSTED_SITE_PACKAGES", str(trusted_path))

    def unexpected_runtime_build():
        raise AssertionError("framework runtime must not be rebuilt by the scorer")

    monkeypatch.setattr(
        mpsc_module,
        "resolve_declared_runtime_environment",
        unexpected_runtime_build,
    )

    command, environment = mpsc_module._workspace_python_launch(
        tmp_path, worker_path
    )

    assert command == [str(Path(sys.executable).resolve()), str(worker_path)]
    assert environment["PYTHONPATH"] == str(trusted_path)
    assert environment["AI4SCI_TRUSTED_SITE_PACKAGES"] == str(trusted_path)
    assert environment["PATH"].split(":", 1)[0] == str(runtime_bin)


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
    assert detail.details["failure_kind"] == "evaluator_runtime_error"
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
    assert detail.details["failure_kind"] == "evaluator_runtime_error"
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
    assert detail.details["failure_kind"] == "evaluator_runtime_error"
    assert detail.details["scorer_internal_error"] is True
    assert "reference certificate" in detail.details["evaluation_error"]
