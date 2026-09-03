"""Regression tests for math.nnls_modulus_deblur input path resolution."""

import os
from pathlib import Path

import numpy as np

from ai4sci_bench.core.scorer import get_scorer
from ai4sci_bench.scorers.custom import load_custom_scorer


TASK_DIR = Path(__file__).parents[1] / "tasks/math/nnls_modulus_deblur"


def _write_instance(tmp_path: Path, *, pred_data: bool = False):
    instance = tmp_path / "instance"
    pred = tmp_path / "pred"
    ref = instance / "reference"
    data = instance / "data"
    pred.mkdir()
    ref.mkdir(parents=True)
    data.mkdir()

    x = np.ones((4, 4), dtype=np.float64)
    kernel = np.zeros_like(x)
    kernel[2, 2] = 1.0  # identity under ifftshift-based convolution
    np.save(pred / "reconstruction.npy", x)
    np.save(data / "observation.npy", x)
    np.save(data / "kernel.npy", kernel)
    (data / "measurement_info.json").write_text(
        '{"noise_std_estimate": 0.0}\n', encoding="utf-8"
    )
    if pred_data:
        pred_data_dir = pred / "data"
        pred_data_dir.mkdir()
        np.save(pred_data_dir / "observation.npy", x)
        np.save(pred_data_dir / "kernel.npy", kernel)
        (pred_data_dir / "measurement_info.json").write_text(
            '{"noise_std_estimate": 0.0}\n', encoding="utf-8"
        )
    return pred, ref, data


def _residual(pred: Path, ref: Path):
    load_custom_scorer(TASK_DIR)
    return get_scorer("nnls_modulus_gate_residual").score(
        pred,
        ref,
        {
            "reconstruction_file": "reconstruction.npy",
            "observation_file": "data/observation.npy",
            "kernel_file": "data/kernel.npy",
            "measurement_info_file": "data/measurement_info.json",
            "min_relative_residual": 0.12,
            "relative_residual_factor": 1.1,
            "relative_residual_slack": 0.02,
        },
    )


def test_residual_gate_reads_inputs_from_instance_data(tmp_path):
    pred, ref, _ = _write_instance(tmp_path)
    result = _residual(pred, ref)
    assert result.passed is True
    assert result.details["relative_residual"] == 0.0


def test_residual_gate_prefers_inputs_copied_into_prediction_dir(tmp_path):
    pred, ref, data = _write_instance(tmp_path, pred_data=True)
    # Make the fallback source invalid; the prediction workspace remains valid.
    np.save(data / "observation.npy", np.zeros((4, 4), dtype=np.float64))
    result = _residual(pred, ref)
    assert result.passed is True


def test_residual_gate_fails_closed_when_instance_inputs_are_missing(tmp_path):
    pred, ref, data = _write_instance(tmp_path)
    for path in data.iterdir():
        path.unlink()
    result = _residual(pred, ref)
    assert result.passed is False
    assert result.details["error"] == "missing or unreadable input"


def test_residual_gate_does_not_depend_on_process_cwd(tmp_path, monkeypatch):
    pred, ref, _ = _write_instance(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    result = _residual(pred, ref)
    assert result.passed is True
