"""Task-local scorer: re-execute the submitted entry point and time it.

Registered as `solution_runtime` (rerun validator) and `runtime_efficiency`
(accuracy-scaled runtime points). The harness loads this file automatically
(TaskLoader.load_custom_scorers / scorers.custom.load_custom_scorer) before any
gate referencing the name is resolved.

Three promises in the prompt's Constraints section are unenforceable by
inspecting artifacts alone, because a submission can simply write the arrays it
was asked for and never be asked to do it again:

  1. `python solution.py` regenerates every artifact from scratch;
  2. it finishes inside the stated wall-clock budget;
  3. it is deterministic — two runs produce the same numerical outputs, up to
     the floating-point noise that multithreaded linear algebra makes
     unavoidable between processes.

This scorer checks all three by copying the submission into a scratch directory,
running the entry point there with the same `data/` it was given, and comparing
the arrays it rewrites against the ones it submitted. Nothing is written back
into the submission directory.

Set `skip_in_gt_selfcheck: true` on the scoring entry that uses this: during GT
self-check the harness fills `solution.py` with a copy of generate_gt.py, which
takes `--output-dir` and writes reference/, so re-running it there proves
nothing.
"""

from __future__ import annotations

import shutil
import os
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

DEFAULT_ENTRY = "solution.py"
ARRAY_NAMES = ("pose", "pose_cov", "landmarks", "landmark_cov", "state",
               "covariance", "covariance_history", "motion_jacobian",
               "observation_jacobian")
DEFAULT_LIMIT_SECONDS = 60.0


def _runtime_image_from_metadata(pred_dir: Path) -> str | None:
    """Find the exact OS-sandbox image recorded for this evaluation run."""
    for parent in (pred_dir, *pred_dir.parents):
        metadata_path = parent / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sandbox = metadata.get("sandbox", {})
        task_runtime = metadata.get("task_runtime", {})
        candidates = [sandbox.get("image_identity")]
        if isinstance(task_runtime, dict):
            for runtime in task_runtime.values():
                if isinstance(runtime, dict):
                    candidates.append(runtime.get("docker_image_identity"))
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _docker_python_command(work: Path, entry: str, image: str) -> list[str]:
    """Run the submission with the Python and packages from the task image."""
    env = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/workspace/.runtime_mpl",
    }
    command = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{work}:/workspace:rw",
        "-w", "/workspace",
    ]
    for key, value in env.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend([image, "/opt/venv/bin/python", entry])
    return command


def runtime_credit(elapsed: float, full: float, limit: float) -> float:
    """Full credit through `full`, linearly decreasing to zero at `limit`."""
    if not (np.isfinite(full) and np.isfinite(limit) and 0 <= full < limit):
        raise ValueError("runtime thresholds must satisfy 0 <= full < limit")
    if not np.isfinite(elapsed) or elapsed < 0:
        return 0.0
    return float(np.clip((limit - elapsed) / (limit - full), 0, 1))


def moment_contract(pred_dir):
    """Validate all redundant views and every covariance, not selected samples."""
    a = {n: np.load(Path(pred_dir) / (n + ".npy"), allow_pickle=False) for n in ARRAY_NAMES}
    k = a["landmarks"].shape[0]
    t, n = a["pose"].shape[0], 3 + 2 * k
    expected = {"pose": (t, 3), "pose_cov": (t, 3, 3), "landmarks": (k, 2),
                "landmark_cov": (k, 2, 2), "state": (n,), "covariance": (n, n),
                "covariance_history": (t, n, n), "motion_jacobian": (t - 1, 3, 3),
                "observation_jacobian": (t - 1, k, 2, 5)}
    for name, arr in a.items():
        if arr.shape != expected[name] or arr.dtype != np.float64 or not np.isfinite(arr).all():
            raise ValueError(f"invalid shape, dtype or nonfinite values: {name}")
    history = a["covariance_history"]
    if np.max(np.abs(history - history.transpose(0, 2, 1))) > 1e-9:
        raise ValueError("joint history is not symmetric")
    if np.linalg.eigvalsh(history).min() < -1e-9:
        raise ValueError("joint history is not positive semidefinite")
    pairs = [(a["pose_cov"], history[:, :3, :3]),
             (a["covariance"], history[-1]), (a["pose"][-1], a["state"][:3]),
             (a["landmarks"].ravel(), a["state"][3:]),
             (a["pose"][0], np.zeros(3)), (a["pose_cov"][0], np.zeros((3, 3)))]
    pairs += [(a["landmark_cov"][i], history[-1, 3+2*i:5+2*i, 3+2*i:5+2*i])
              for i in range(k)]
    if any(not np.allclose(x, y, atol=1e-9, rtol=0) for x, y in pairs):
        raise ValueError("redundant output views disagree")
    if np.any(a["pose"][:, 2] < -np.pi) or np.any(a["pose"][:, 2] >= np.pi):
        raise ValueError("heading outside [-pi, pi)")
    # Decode the image, so an empty file called map.png cannot pass the gate.
    # Axis labels, geometric content and aspect remain an author/reviewer check.
    from PIL import Image
    with Image.open(Path(pred_dir) / "map.png") as figure:
        if figure.format != "PNG" or min(figure.size) < 1:
            raise ValueError("map.png is not a valid nonempty PNG")
        if min(figure.info.get("dpi", (0, 0))) < 149.9:
            raise ValueError("map.png must declare at least 150 dpi")
        figure.verify()
    return a


def _credit(errors, full, zero):
    """Shared linear error credit, using the worst window/block error."""
    if not (np.isfinite(full) and np.isfinite(zero) and 0 <= full < zero):
        raise ValueError("accuracy thresholds must satisfy 0 <= full < zero")
    worst = float(np.max(errors, initial=0.0))
    if not np.isfinite(worst):
        return 0.0
    return float(np.clip((zero - worst) / (zero - full), 0, 1))


def moment_accuracy(pred, ref, config):
    """Reference-only arithmetic: no private estimator or prompt-level input."""
    kind = config["kind"]
    full = float(config["full"])
    zero = float(config.get("zero", 1e-3))
    if kind in ("trajectory", "map"):
        name = "pose" if kind == "trajectory" else "landmarks"
        delta = pred[name] - ref[name]
        if kind == "trajectory":
            delta[:, 2] = (delta[:, 2] + np.pi) % (2*np.pi) - np.pi
            delta = delta / np.asarray(config["scales"])
            windows = np.array_split(delta, min(12, len(delta)))
        else:
            delta = delta / float(config["scale"])
            windows = [delta]
        errors = [float(np.sqrt(np.mean(w*w))) for w in windows]
        return _credit(errors, full, zero), {
            "window_errors": errors,
            "normalized_mse": float(np.mean(delta * delta)),
            "aggregation": "worst_window_normalized_rmse",
        }
    if kind == "jacobian":
        # Evaluate each timestamp/landmark independently: a sparse bad block
        # must not be diluted by thousands of correct linearizations.
        name = config["array"]
        target, actual = ref[name], pred[name]
        norms = np.linalg.norm(target, axis=(-2, -1))
        differences = np.linalg.norm(actual - target, axis=(-2, -1))
        observed = norms > 0
        relative = differences[observed] / norms[observed]
        max_relative = float(np.max(relative, initial=0))
        zero_error = float(np.max(differences[~observed], initial=0))
        zero_atol = float(config.get("zero_atol", 1e-9))
        if not np.isfinite(zero_atol) or zero_atol < 0:
            raise ValueError("zero_atol must be finite and nonnegative")
        credit = _credit([max_relative], full, zero)
        if zero_error > zero_atol:
            credit = 0.0
        return credit, {
            "max_block_relative_error": max_relative,
            "max_missing_block_absolute_error": zero_error,
            "incorrect_observed_blocks": int(np.count_nonzero(relative >= full)),
            "incorrect_missing_blocks": int(np.count_nonzero(differences[~observed] > zero_atol)),
            "aggregation": "every_timestamp_every_block",
        }
    if kind != "joint":
        raise ValueError(f"unknown moment metric {kind}")
    target, actual = ref["covariance_history"], pred["covariance_history"]
    n = target.shape[-1]
    k = (n-3)//2
    masks = {name: np.zeros((n, n), dtype=bool) for name in
             ("pose", "landmark_marginal", "pose_landmark", "distinct_landmarks")}
    masks["pose"][:3, :3] = True
    masks["pose_landmark"][:3, 3:] = True
    masks["distinct_landmarks"][3:, 3:] = True
    for i in range(k):
        s = slice(3+2*i, 5+2*i)
        masks["landmark_marginal"][s, s] = True
        masks["distinct_landmarks"][s, s] = False
    errors = {name: [] for name in masks}
    # Initial cross blocks are exactly zero; later timestamps define joint quality.
    for indices in np.array_split(np.arange(1, len(target)), min(12, len(target)-1)):
        r, p = target[indices], actual[indices]
        sd = np.sqrt(np.maximum(np.diagonal(r, axis1=1, axis2=2), 1e-14))
        denom = sd[:, :, None] * sd[:, None, :]
        r, difference = r / denom, (p-r) / denom
        for name, mask in masks.items():
            err = np.linalg.norm(difference[:, mask]) / max(np.linalg.norm(r[:, mask]), 1e-12)
            errors[name].append(float(err))
    # The joint estimate is only as reliable as its least accurate block group.
    # Equal windows prevent long stretches of easy observations masking outages.
    worst = np.max(np.asarray(list(errors.values())), axis=0)
    score = _credit(worst, full, zero)
    return score, {"block_window_errors": errors, "aggregation": "worst_window_worst_block"}


@register_scorer("moment_contract")
class MomentContractScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1))
        try:
            moment_contract(pred_dir)
        except Exception as exc:
            return ScoreDetail(scorer_name="moment_contract", score=0, max_score=weight,
                               passed=False, details={}, message=str(exc))
        return ScoreDetail(scorer_name="moment_contract", score=weight, max_score=weight,
                           passed=True, details={}, message="")


@register_scorer("moment_accuracy")
class MomentAccuracyScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1))
        try:
            names = {"trajectory": ["pose"], "map": ["landmarks"],
                     "joint": ["covariance_history"]}.get(config["kind"])
            if config["kind"] == "jacobian":
                names = [config["array"]]
            pred = {n: np.load(pred_dir / (n+".npy"), allow_pickle=False) for n in names}
            ref = {n: np.load(ref_dir / (n+".npy"), allow_pickle=False) for n in names}
            if any(pred[n].shape != ref[n].shape or not np.isfinite(pred[n]).all() for n in names):
                raise ValueError("shape mismatch or nonfinite prediction")
            credit, details = moment_accuracy(pred, ref, config)
        except Exception as exc:
            return ScoreDetail(scorer_name="moment_accuracy", score=0, max_score=weight,
                               passed=False, details={}, message=str(exc))
        return ScoreDetail(scorer_name="moment_accuracy", score=weight*credit,
                           max_score=weight,
                           passed=credit >= 1-1e-12, details=details, message="")


@register_scorer("accuracy_gate")
class AccuracyGateScorer(Scorer):
    """Award the accuracy tranche only when every numerical contract passes.

    EKF-SLAM outputs are a coupled scientific result: a trajectory with a
    wrong association, covariance, or Jacobian is not a partially useful
    realization of the reference filter.  The component checks remain
    independently reported in the details, while this composite score makes
    that contract explicit and prevents a few passing views from masking a
    failed one.
    """

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 80.0))
        components = config.get("components", [])
        if not components:
            return ScoreDetail(
                scorer_name="accuracy_gate", score=0.0, max_score=weight,
                passed=False, details={}, message="no accuracy components configured")
        try:
            contract = moment_contract(pred_dir)
            component_results = []
            passed = True
            for component in components:
                cfg = dict(component.get("config", {}))
                item_weight = float(component.get("weight", 1.0))
                result = MomentAccuracyScorer().score(
                    pred_dir, ref_dir, {**cfg, "weight": item_weight})
                component_results.append({
                    "kind": cfg.get("kind"),
                    "array": cfg.get("array"),
                    "score": result.score,
                    "max_score": result.max_score,
                    "passed": result.passed,
                    "details": result.details,
                    "message": result.message,
                })
                passed = passed and result.passed
        except Exception as exc:
            return ScoreDetail(
                scorer_name="accuracy_gate", score=0.0, max_score=weight,
                passed=False, details={}, message=str(exc))
        return ScoreDetail(
            scorer_name="accuracy_gate",
            score=weight if passed else 0.0,
            max_score=weight,
            passed=passed,
            details={
                "aggregation": "all_accuracy_components_required",
                "component_results": component_results,
                "contract_passed": True,
            },
            message="" if passed else "one or more numerical accuracy components failed",
        )


@register_scorer("solution_runtime")
class SolutionRuntimeScorer(Scorer):
    """Re-run the entry point; fail on error, on overtime, or on non-determinism."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        entry = config.get("entry", DEFAULT_ENTRY)
        limit = float(config.get("limit_seconds", DEFAULT_LIMIT_SECONDS))
        check_determinism = bool(config.get("check_determinism", True))
        # NOT zero: matmul and linear solves are multithreaded, so two runs of
        # the same correct code differ at the ~1e-14 level across processes.
        # The failure this catches — a seeded RNG, dict/set iteration order
        # leaking into the maths — moves the answer far more than 1e-9.
        atol = float(config.get("determinism_atol", 1e-9))
        rtol = float(config.get("determinism_rtol", 1e-9))
        weight = float(config.get("weight", 1.0))

        runtime_image = config.get("runtime_image") or _runtime_image_from_metadata(pred_dir)
        details: dict = {
            "entry": entry,
            "limit_seconds": limit,
            "runtime_mode": "docker_task_image" if runtime_image else "host_fallback",
        }
        if runtime_image:
            details["runtime_image"] = runtime_image

        if not (pred_dir / entry).exists():
            return self._fail(weight, details, f"{entry} not found")

        try:
            moment_contract(pred_dir)
        except Exception as exc:
            return self._fail(weight, details, f"invalid submitted artifacts: {exc}")

        with tempfile.TemporaryDirectory(prefix="solution_runtime_") as tmp:
            work = Path(tmp) / "work"
            shutil.copytree(pred_dir, work)
            input_dir = ref_dir.parent / "data"
            # Always use evaluator-provided inputs, not a modified submitted copy.
            if input_dir.is_dir():
                if (work / "data").exists():
                    shutil.rmtree(work / "data")
                shutil.copytree(input_dir, work / "data")
            # The rerun must produce the arrays itself, not inherit them.
            submitted = {p.name: p for p in sorted(pred_dir.glob("*.npy"))}
            for name in submitted:
                (work / name).unlink()
            (work / "map.png").unlink(missing_ok=True)

            t0 = time.perf_counter()
            try:
                command = (
                    _docker_python_command(work, entry, str(runtime_image))
                    if runtime_image
                    else [sys.executable, entry]
                )
                proc = subprocess.run(
                    command,
                    cwd=work,
                    capture_output=True,
                    text=True,
                    timeout=limit,
                    stdin=subprocess.DEVNULL,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1",
                         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                         "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                         "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(work / ".runtime_mpl")},
                )
            except subprocess.TimeoutExpired:
                details["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
                return self._fail(
                    weight, details,
                    f"{entry} was killed after {limit:g}s "
                    f"(limit {limit:g}s)")
            except OSError as exc:
                return self._fail(weight, details, f"cannot execute {entry}: {exc}")
            elapsed = time.perf_counter() - t0
            details["elapsed_seconds"] = elapsed

            if proc.returncode != 0:
                details["returncode"] = proc.returncode
                details["stderr_tail"] = proc.stderr[-2000:]
                return self._fail(weight, details,
                                  f"{entry} exited with code {proc.returncode}")

            if elapsed > limit:
                return self._fail(
                    weight, details,
                    f"{entry} took {elapsed:.1f}s, over the {limit:g}s limit")

            required = [n + ".npy" for n in ARRAY_NAMES] + ["map.png"]
            missing = [n for n in required if not (work / n).exists()]
            if missing:
                return self._fail(weight, details,
                                  f"{entry} did not regenerate: {', '.join(sorted(missing))}")

            try:
                regenerated = moment_contract(work)
                for name in ARRAY_NAMES:
                    target = np.load(ref_dir / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                    if regenerated[name].shape != target.shape:
                        raise ValueError(f"reference shape mismatch: {name}")
            except Exception as exc:
                return self._fail(weight, details, f"invalid regenerated artifacts: {exc}")

            if check_determinism:
                drifted = []
                for name, orig in submitted.items():
                    try:
                        a, b = np.load(orig), np.load(work / name)
                    except Exception as e:
                        return self._fail(weight, details, f"cannot compare {name}: {e}")
                    if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol,
                                                             equal_nan=False):
                        drifted.append(name)
                if drifted:
                    details["nondeterministic_files"] = drifted
                    return self._fail(
                        weight, details,
                        f"re-running {entry} did not reproduce: {', '.join(drifted)}")

        return ScoreDetail(
            scorer_name="solution_runtime",
            score=weight,
            max_score=weight,
            passed=True,
            details=details,
            message="",
        )

    @staticmethod
    def _fail(weight: float, details: dict, message: str) -> ScoreDetail:
        return ScoreDetail(
            scorer_name="solution_runtime",
            score=0.0,
            max_score=weight,
            passed=False,
            details=details,
            message=message,
        )


@register_scorer("runtime_efficiency")
class RuntimeEfficiencyScorer(Scorer):
    """Accuracy-scaled timing points, measured once by trusted scorer code."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 20.0))
        full = float(config.get("full_seconds", 30.0))
        limit = float(config.get("limit_seconds", DEFAULT_LIMIT_SECONDS))
        runtime_credit(0.0, full, limit)  # Reject invalid evaluator configuration.
        components = config["accuracy_components"]
        maximum = sum(float(c["weight"]) for c in components)
        if maximum <= 0:
            raise ValueError("accuracy component weights must sum to a positive value")
        runtime = SolutionRuntimeScorer().score(pred_dir, ref_dir, config)
        details = dict(runtime.details)
        details["full_seconds"] = full
        credit = 0.0
        if runtime.passed:
            earned = 0.0
            for component in components:
                result = MomentAccuracyScorer().score(
                    pred_dir, ref_dir,
                    {**component["config"], "weight": component["weight"]})
                earned += result.score
            quality = earned / maximum
            factor = runtime_credit(details["elapsed_seconds"], full, limit)
            credit = factor * quality
            details.update(time_factor=factor, accuracy_fraction=quality)
        else:
            details.update(time_factor=0.0, accuracy_fraction=None)
        return ScoreDetail(
            scorer_name="runtime_efficiency", score=weight * credit,
            max_score=weight, passed=runtime.passed and credit >= 1-1e-12,
            details=details, message=runtime.message)
