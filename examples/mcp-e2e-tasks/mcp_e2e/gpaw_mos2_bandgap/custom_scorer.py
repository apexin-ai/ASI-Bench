"""Scorers for the gpaw MoS2 band-structure MCP E2E fake task.

They compare the agent's ``result.json`` and the two figures it copied out of
the server's run directory with the pre-generated reference, using nothing but
the standard library: no GPAW, no ASE, no image library. Whether the numbers
actually came from the MCP tools is checked separately, from the run artefacts,
by scripts/mcp/e2e/verify_run.py.

The figures are the only part of the chain that exists nowhere but on disk —
every tool returns their *path*, never their bytes — so handing them back is
what proves the agent resolved the artefact listing to a real run directory.

Submission problems (missing or malformed files, wrong numbers, a figure that
is not a PNG) are ordinary zero scores. A missing or unreadable reference is an
evaluator failure, reported with ``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

# result.json's contract, by how it is compared.
ENERGIES = ("relax_total_energy_ev", "scf_total_energy_ev", "fermi_ev")
RELAXATION_NUMBERS = ("relax_max_force_ev_per_a",)
BAND_NUMBERS = ("band_gap_ev",)
NUMBERS = (*ENERGIES, *RELAXATION_NUMBERS, *BAND_NUMBERS, "recommended_kpts_density")
COUNTS = ("relax_n_steps", "recommended_ecut_ev", "artifact_count")
LABELS = ("gap_type", "vbm_label", "cbm_label", "verdict")
FLAGS = ("params_verified", "converged")

COPIED_PNG = ("bands.png", "dos.png")           # the task_eval.yaml default
PNG_MIN_WIDTH, PNG_MIN_HEIGHT, PNG_MIN_BYTES = 400, 300, 10_000


def figure_files(config: dict) -> tuple[str, ...]:
    """The figures to check, from the scorer config (names live in task_eval.yaml)."""
    names = config.get("png_files") or COPIED_PNG
    return tuple(str(name) for name in names)


class _PredictionError(ValueError):
    pass


class _ReferenceError(ValueError):
    pass


# --------------------------------------------------------------------------
# result.json
# --------------------------------------------------------------------------

def _number(value, what: str) -> float:
    if isinstance(value, bool):
        raise _PredictionError(f"{what} must be a number, got {value!r}")
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise _PredictionError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise _PredictionError(f"{what} is not finite: {number}")
    return number


def _count(value, what: str) -> int:
    number = _number(value, what)
    if number != int(number) or number < 0:
        raise _PredictionError(f"{what} must be a non-negative whole number, got {value!r}")
    return int(number)


def _label(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _PredictionError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _flag(value, what: str) -> bool:
    if not isinstance(value, bool):
        raise _PredictionError(f"{what} must be true or false, got {value!r}")
    return value


def _grid(value, what: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise _PredictionError(f"{what} must be a list of three integers, got {value!r}")
    return [_count(v, f"{what}[{i}]") for i, v in enumerate(value)]


def load_result(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must hold a JSON object")
    out: dict = {key: _number(data.get(key), key) for key in NUMBERS}
    out.update({key: _count(data.get(key), key) for key in COUNTS})
    out.update({key: _label(data.get(key), key) for key in LABELS})
    out.update({key: _flag(data.get(key), key) for key in FLAGS})
    out["relax_kpts"] = _grid(data.get("relax_kpts"), "relax_kpts")
    return out


def load_reference(path: Path) -> dict:
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        out = {key: float(ref[key]) for key in NUMBERS}
        out.update({key: int(ref[key]) for key in COUNTS})
        out.update({key: str(ref[key]) for key in LABELS})
        out.update({key: bool(ref[key]) for key in FLAGS})
        out["relax_kpts"] = [int(v) for v in ref["relax_kpts"]]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc
    if len(out["relax_kpts"]) != 3 or out["band_gap_ev"] <= 0:
        raise _ReferenceError("reference has no band structure")
    return out


# --------------------------------------------------------------------------
# The copied figures
# --------------------------------------------------------------------------

def png_size(blob: bytes) -> tuple[int, int]:
    """Width and height from the IHDR chunk; raises if this is not a PNG."""
    if len(blob) < 24 or blob[:8] != b"\x89PNG\r\n\x1a\n" or blob[12:16] != b"IHDR":
        raise _PredictionError("not a PNG (no signature or IHDR chunk)")
    return struct.unpack(">II", blob[16:24])


def png_problems(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path.name} not found"]
    try:
        blob = path.read_bytes()
    except OSError as exc:
        return [f"{path.name} is unreadable: {exc}"]
    try:
        width, height = png_size(blob)
    except _PredictionError as exc:
        return [f"{path.name}: {exc}"]
    problems = []
    if width < PNG_MIN_WIDTH or height < PNG_MIN_HEIGHT:
        problems.append(f"{path.name} is {width}x{height} px, smaller than "
                        f"{PNG_MIN_WIDTH}x{PNG_MIN_HEIGHT}")
    if len(blob) < PNG_MIN_BYTES:
        problems.append(f"{path.name} is {len(blob)} bytes, too small to be a rendered figure")
    return problems


# --------------------------------------------------------------------------
# Credit
# --------------------------------------------------------------------------

def credit(error: float, full_tol: float, zero_tol: float) -> float:
    """1 at <= full_tol, 0 at >= zero_tol, log-linear in between."""
    if not math.isfinite(error):
        return 0.0
    if error <= full_tol:
        return 1.0
    if error >= zero_tol:
        return 0.0
    span = math.log10(zero_tol) - math.log10(full_tol)
    return max(0.0, min(1.0, 1.0 - (math.log10(error) - math.log10(full_tol)) / span))


def relative(got: float, want: float) -> float:
    return abs(got - want) / abs(want) if want else abs(got - want)


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True,
                                "failure_kind": "missing_evaluator_input", "error": message},
                       message=message)


def _load(name: str, weight: float, pred_dir: Path, ref_dir: Path, config: dict):
    """``(reference, prediction, failure)`` — at most one of the first two is None."""
    try:
        ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
    except _ReferenceError as exc:
        return None, None, _evaluator_failure(name, weight, str(exc))
    try:
        pred = load_result(pred_dir / config.get("pred_file", "result.json"))
    except _PredictionError as exc:
        return ref, None, _zero(name, weight, str(exc))
    return ref, pred, None


def _numeric_detail(name: str, weight: float, keys, pred: dict, ref: dict,
                    config: dict, extra: dict | None = None) -> ScoreDetail:
    full_tol = float(config["full_score_tol"])
    zero_tol = float(config["zero_score_tol"])
    errors = {key: relative(pred[key], ref[key]) for key in keys}
    fractions = {key: credit(error, full_tol, zero_tol) for key, error in errors.items()}
    fractions.update(extra or {})
    fraction = sum(fractions.values()) / len(fractions)
    return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                       passed=all(value == 1.0 for value in fractions.values()),
                       details={"relative_errors": errors, "credit_fractions": fractions,
                                "predicted": {key: pred[key] for key in keys},
                                "reference": {key: ref[key] for key in keys}},
                       message="; ".join(f"{key}: {value:.2e}" for key, value in errors.items()))


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

@register_scorer("gpaw_e2e_schema")
class Schema(Scorer):
    """Gate: result.json is complete and well typed.

    Deliberately schema-only. The figures are scored by ``gpaw_e2e_artifacts``
    and not gated: the tools report artefact paths relative to the server's own
    working directory, so handing the figures back means finding that directory
    on the host, and an agent that drove the whole chain correctly but did not
    find it should lose those ten points rather than the instance.
    """

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "gpaw_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            load_result(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"pred_file": config.get("pred_file", "result.json")},
                           message="result.json complete and well typed")


@register_scorer("gpaw_e2e_energies")
class Energies(Scorer):
    """The cutoff-dependent numbers: the relaxed total energy, the ground-state
    total energy and the Fermi level. These cannot be known without running the
    calculation at this instance's parameters."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "gpaw_e2e_energies"
        weight = float(config.get("weight", 1.0))
        ref, pred, failure = _load(name, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure
        return _numeric_detail(name, weight, ENERGIES, pred, ref, config)


@register_scorer("gpaw_e2e_relaxation")
class Relaxation(Scorer):
    """What relax_structure reported: the residual force, the step count and the
    realized k-grid."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "gpaw_e2e_relaxation"
        weight = float(config.get("weight", 1.0))
        ref, pred, failure = _load(name, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure
        extra = {
            "relax_n_steps": 1.0 if pred["relax_n_steps"] == ref["relax_n_steps"] else 0.0,
            "relax_kpts": 1.0 if pred["relax_kpts"] == ref["relax_kpts"] else 0.0,
        }
        return _numeric_detail(name, weight, RELAXATION_NUMBERS, pred, ref, config, extra)


@register_scorer("gpaw_e2e_bands")
class Bands(Scorer):
    """The band gap and its character. Deliberately the lightest numeric scorer:
    a PBE MoS2 monolayer gap near 1.67 eV is a published number."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "gpaw_e2e_bands"
        weight = float(config.get("weight", 1.0))
        ref, pred, failure = _load(name, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure
        extra = {key: 1.0 if pred[key] == ref[key] else 0.0
                 for key in ("gap_type", "vbm_label", "cbm_label")}
        return _numeric_detail(name, weight, BAND_NUMBERS, pred, ref, config, extra)


@register_scorer("gpaw_e2e_convergence")
class Convergence(Scorer):
    """The convergence gate and the verification verdict: the recommendation the
    sweep produced for this tolerance, whether the run's own parameters cleared
    it, and what verify_run concluded at this gap tolerance."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "gpaw_e2e_convergence"
        weight = float(config.get("weight", 1.0))
        ref, pred, failure = _load(name, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure
        parts = {
            "recommended_ecut_ev": pred["recommended_ecut_ev"] == ref["recommended_ecut_ev"],
            "recommended_kpts_density": abs(pred["recommended_kpts_density"]
                                            - ref["recommended_kpts_density"]) <= 1e-9,
            "converged": pred["converged"] is ref["converged"],
            "params_verified": pred["params_verified"] is ref["params_verified"],
            "verdict": pred["verdict"] == ref["verdict"],
        }
        fractions = {key: 1.0 if ok else 0.0 for key, ok in parts.items()}
        fraction = sum(fractions.values()) / len(fractions)
        wrong = sorted(key for key, ok in parts.items() if not ok)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=not wrong,
                           details={"parts": fractions, "wrong": wrong,
                                    "predicted": {key: pred[key] for key in parts},
                                    "reference": {key: ref[key] for key in parts}},
                           message="gate and verdict match" if not wrong else f"wrong: {wrong}")


@register_scorer("gpaw_e2e_artifacts")
class Artifacts(Scorer):
    """What the run left on disk: how many artefacts the listing reported, and the
    two figures handed back. The figures exist only in the run directory — every
    tool returns their path, never their bytes — so producing them means the
    artefact listing was actually resolved to that directory."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "gpaw_e2e_artifacts"
        weight = float(config.get("weight", 1.0))
        ref, pred, failure = _load(name, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure

        parts: dict[str, float] = {}
        problems: list[str] = []
        for filename in figure_files(config):
            wrong = png_problems(pred_dir / filename)
            parts[filename] = 0.0 if wrong else 1.0
            problems += wrong
        if pred["artifact_count"] == ref["artifact_count"]:
            parts["artifact_count"] = 1.0
        else:
            parts["artifact_count"] = 0.0
            problems.append(f"artifact_count={pred['artifact_count']} != {ref['artifact_count']}")

        fraction = sum(parts.values()) / len(parts)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=not problems,
                           details={"parts": parts, "problems": problems[:20]},
                           message="both figures and the artefact count match" if not problems
                                   else "; ".join(problems[:4]))
