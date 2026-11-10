"""Independent 0-100 scientific scorer for the robust Maxwell task."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import stat
from pathlib import Path

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


def _load_reference_module():
    path = Path(__file__).with_name("maxwell_eval_runtime.py")
    spec = importlib.util.spec_from_file_location("maxwell_scoring_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Maxwell scoring kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REQUIRED_OUTPUTS = ("fields.npy", "gradient.npy", "hvp.npy", "rho.npy", "report.json")
IMMUTABLE_INPUTS = ("problem.npz", "problem.json", "model_contract.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str, size_limit: int) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular, non-symlink file")
    if metadata.st_size > size_limit:
        raise ValueError(f"{label} exceeds the {size_limit}-byte size limit")


def _verify_immutable_inputs(pred_dir: Path, ref_dir: Path) -> Path:
    original_data = ref_dir.parent / "data"
    if not original_data.is_dir():
        raise ValueError("original immutable input directory is unavailable")
    for name in IMMUTABLE_INPUTS:
        original_path = original_data / name
        _regular_file(original_path, f"original immutable input {name}", 16 * 1024 * 1024)
        if pred_dir.resolve() == ref_dir.resolve():
            continue
        candidate_path = pred_dir / "data" / name
        _regular_file(candidate_path, f"candidate immutable input {name}", 16 * 1024 * 1024)
        if _sha256(candidate_path) != _sha256(original_path):
            raise ValueError(f"candidate immutable input was modified: {name}")
    return original_data


def _load_problem(data_dir: Path) -> dict:
    with np.load(data_dir / "problem.npz", allow_pickle=False) as arrays:
        problem = {
            "eps_background": arrays["eps_background"],
            "design_mask": arrays["design_mask"].astype(bool),
            "rho0": arrays["rho0"],
            "sources": arrays["sources"],
            "targets": arrays["targets"],
            "weights": arrays["weights"],
            "omegas": arrays["omegas"],
            "projection_etas": arrays["projection_etas"],
            "hvp_direction": arrays["hvp_direction"],
        }
    meta = json.loads((data_dir / "problem.json").read_text(encoding="utf-8"))
    if meta.get("model_contract_version") != "2.0.0":
        raise ValueError("unsupported model contract version")
    problem.update(
        {
            "dx": float(meta["grid"]["dx"]),
            "dy": float(meta["grid"]["dy"]),
            "npml": int(meta["pml"]["width_cells"]),
            "pml_strength": float(meta["pml"]["strength"]),
            "pml_power": int(meta["pml"]["power"]),
            "eps_min": float(meta["material"]["epsilon_min"]),
            "eps_max": float(meta["material"]["epsilon_max"]),
            "filter_radius": float(meta["fabrication_filter"]["radius_cells"]),
            "beta": float(meta["fabrication_filter"]["beta"]),
            "robust_order": float(meta["robust_objective"]["order"]),
            "robust_loss_offset": float(meta["robust_objective"]["loss_offset"]),
            "hvp_step": float(meta["hessian_vector_product"]["central_difference_step"]),
            "volume_fraction_max": float(meta["constraints"]["volume_fraction_max"]),
            "binary_tolerance": float(meta["constraints"]["binary_tolerance"]),
            "required_reduction": float(meta["design_full_credit_relative_reduction"]),
        }
    )
    return problem


@register_scorer("maxwell_scientific")
class MaxwellScientificScorer(Scorer):
    """Recompute fields, sensitivities, curvature, constraints, and performance."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        try:
            original_data = _verify_immutable_inputs(pred_dir, ref_dir)
            for name in REQUIRED_OUTPUTS:
                size_limit = 64 * 1024 if name == "report.json" else 16 * 1024 * 1024
                _regular_file(pred_dir / name, f"candidate output {name}", size_limit)
                if pred_dir.resolve() != ref_dir.resolve():
                    try:
                        if (pred_dir / name).samefile(ref_dir / name):
                            raise ValueError(f"candidate output aliases private reference: {name}")
                    except FileNotFoundError:
                        pass
            problem = _load_problem(original_data)
            ny, nx = problem["rho0"].shape
            nf = len(problem["omegas"])
            nb = len(problem["projection_etas"])
            fields_raw = np.load(pred_dir / "fields.npy", allow_pickle=False)
            gradient = np.load(pred_dir / "gradient.npy", allow_pickle=False)
            hvp = np.load(pred_dir / "hvp.npy", allow_pickle=False)
            rho = np.load(pred_dir / "rho.npy", allow_pickle=False)
            report = json.loads((pred_dir / "report.json").read_text(encoding="utf-8"))
            expected_specs = {
                "fields.npy": (fields_raw, (nb, nf, 2, ny, nx)),
                "gradient.npy": (gradient, (ny, nx)),
                "hvp.npy": (hvp, (ny, nx)),
                "rho.npy": (rho, (ny, nx)),
            }
            for name, (array, shape) in expected_specs.items():
                if array.dtype != np.float64 or array.shape != shape:
                    raise ValueError(f"{name} must be float64 with shape {shape}")
            if not all(np.all(np.isfinite(x)) for x in (fields_raw, gradient, hvp, rho)):
                raise ValueError("numeric artifacts contain NaN or infinity")

            kernel = _load_reference_module()
            initial_objective, expected_gradient, expected_fields, _, _ = kernel.evaluate_design(
                problem, problem["rho0"], need_gradient=True
            )
            expected_hvp = kernel.hessian_vector_product(problem, problem["rho0"])
            fields = fields_raw[:, :, 0] + 1j * fields_raw[:, :, 1]

            field_error = float(
                np.linalg.norm(fields - expected_fields)
                / max(np.linalg.norm(expected_fields), 1e-15)
            )
            field_points = 5.0 if field_error <= 1e-6 else 0.0

            gradient_error = float(
                np.linalg.norm(gradient - expected_gradient)
                / max(np.linalg.norm(expected_gradient), 1e-15)
            )
            gradient_points = 7.5 if gradient_error <= 2e-5 else 0.0

            hvp_error = float(
                np.linalg.norm(hvp - expected_hvp)
                / max(np.linalg.norm(expected_hvp), 1e-15)
            )
            hvp_points = 12.5 if hvp_error <= 3e-3 else 0.0

            residuals = kernel.field_residuals(problem, problem["rho0"], fields)
            max_residual = float(np.max(residuals))
            residual_points = 2.5 if max_residual <= 1e-9 else 0.0

            mask = problem["design_mask"]
            bounds_ok = bool(np.all((rho >= 0.0) & (rho <= 1.0)))
            frozen_ok = bool(np.max(np.abs(rho[~mask] - problem["rho0"][~mask])) <= 1e-12)
            volume_fraction = float(np.mean(rho[mask]))
            volume_ok = bool(volume_fraction <= problem["volume_fraction_max"] + 1e-12)
            binary_error = float(np.max(np.minimum(np.abs(rho[mask]), np.abs(1.0 - rho[mask]))))
            binary_ok = bool(binary_error <= problem["binary_tolerance"])
            if bounds_ok and frozen_ok and volume_ok and binary_ok:
                final_objective = kernel.evaluate_design(problem, rho, need_gradient=False)[0]
                reduction = float(1.0 - final_objective / max(initial_objective, 1e-30))
                design_points = 70.0 if reduction >= problem["required_reduction"] else 0.0
            else:
                final_objective = None
                reduction = None
                design_points = 0.0

            report_points = 0.0
            checks = [("initial_objective", initial_objective)]
            if final_objective is not None and reduction is not None:
                checks.extend(
                    [
                        ("final_objective", final_objective),
                        ("relative_reduction", reduction),
                        ("final_volume_fraction", volume_fraction),
                    ]
                )
            for key, expected in checks:
                actual = float(report.get(key, float("nan")))
                tolerance = 1e-6 * max(abs(expected), 1e-12)
                if np.isfinite(actual) and np.isfinite(expected) and abs(actual - expected) <= tolerance:
                    report_points += 2.5 / len(checks)

            raw_score = (
                field_points
                + gradient_points
                + hvp_points
                + residual_points
                + design_points
                + report_points
            )
            details = {
                "field_relative_l2": field_error,
                "field_points": field_points,
                "gradient_relative_l2": gradient_error,
                "gradient_points": gradient_points,
                "hvp_relative_l2": hvp_error,
                "hvp_points": hvp_points,
                "max_equation_residual": max_residual,
                "residual_points": residual_points,
                "initial_objective": float(initial_objective),
                "final_objective": None if final_objective is None else float(final_objective),
                "relative_reduction": None if reduction is None else float(reduction),
                "design_points": design_points,
                "bounds_ok": bounds_ok,
                "frozen_region_ok": frozen_ok,
                "volume_fraction": volume_fraction,
                "volume_constraint_ok": volume_ok,
                "binary_error": binary_error,
                "binary_topology_ok": binary_ok,
                "report_points": report_points,
                "raw_score_100": raw_score,
                "complete_public_model_contract": True,
                "independent_recomputation": True,
            }
            score = weight * raw_score / 100.0
            return ScoreDetail(
                scorer_name="maxwell_scientific",
                score=float(score),
                max_score=weight,
                passed=bool(raw_score >= 85.0),
                details=details,
                message=(
                    f"scientific score={raw_score:.2f}/100; field={field_error:.3e}; "
                    f"gradient={gradient_error:.3e}; hvp={hvp_error:.3e}; "
                    + (f"robust reduction={reduction:.3%}" if reduction is not None else "invalid design constraints")
                ),
            )
        except Exception as exc:
            return ScoreDetail(
                scorer_name="maxwell_scientific",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": f"{type(exc).__name__}: {exc}"},
                message=f"scientific scoring failed: {exc}",
            )
