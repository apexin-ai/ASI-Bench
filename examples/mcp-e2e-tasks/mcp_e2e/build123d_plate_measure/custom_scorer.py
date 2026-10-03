"""Scorers for the build123d plate MCP E2E fake task.

They compare the agent's ``result.json`` and the exported ``part.step`` /
``part.stl`` with the pre-generated closed-form reference and never run
build123d, OpenCascade or a mesh library: the STL is parsed here with the
standard library (binary triangles, divergence-theorem volume, edge pairing and
connected components) and the STEP is checked as text. Whether the numbers came
from the MCP tools is checked separately from the run artefacts by
scripts/mcp/e2e/verify_run.py.

Submission problems (missing or malformed files, wrong numbers, a part that is
not one watertight solid) are ordinary zero scores. A missing or unreadable
reference is an evaluator failure, reported with ``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

MEASUREMENTS = ("volume_mm3", "surface_area_mm2", "mass_g", "izz_g_mm2",
                "reimported_volume_mm3")
NUMBERS = (*MEASUREMENTS, "bolt_hole_diameter_mm")
COUNTS = ("hole_count", "n_solids")


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
    bbox = data.get("bbox_mm")
    if not isinstance(bbox, list) or len(bbox) != 3:
        raise _PredictionError(f"bbox_mm must be a list of three numbers, got {bbox!r}")
    out["bbox_mm"] = [_number(v, f"bbox_mm[{i}]") for i, v in enumerate(bbox)]
    gate = data.get("passes_gate")
    if not isinstance(gate, bool):
        raise _PredictionError(f"passes_gate must be true or false, got {gate!r}")
    out["passes_gate"] = gate
    part = data.get("part")
    if not isinstance(part, str) or not part.strip():
        raise _PredictionError(f"part must be the part name, got {part!r}")
    out["part"] = part.strip()
    return out


def load_reference(path: Path) -> dict:
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        out = {key: float(ref[key]) for key in NUMBERS}
        out.update({key: int(ref[key]) for key in COUNTS})
        out["bbox_mm"] = [float(v) for v in ref["bbox_mm"]]
        out["part"] = str(ref["part"])
        out["passes_gate"] = bool(ref["passes_gate"])
        out["face_count"] = int(ref["face_count"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc
    if len(out["bbox_mm"]) != 3 or out["volume_mm3"] <= 0:
        raise _ReferenceError("reference has no plate geometry")
    return out


# --------------------------------------------------------------------------
# STL (binary) and STEP readers — standard library only
# --------------------------------------------------------------------------

Triangle = tuple[tuple[float, float, float], ...]
VERTEX_QUANTUM = 1e-6       # mm, for matching tessellation vertices


def parse_binary_stl(data: bytes) -> list[Triangle]:
    if len(data) < 84:
        raise _PredictionError(f"STL is too short to be binary ({len(data)} bytes)")
    if data[:5] == b"solid" and b"facet normal" in data[:512]:
        raise _PredictionError("STL is ASCII; the server writes binary STL")
    count = struct.unpack("<I", data[80:84])[0]
    if count == 0:
        raise _PredictionError("STL has no triangles")
    if len(data) != 84 + 50 * count:
        raise _PredictionError(f"STL declares {count} triangles but is {len(data)} bytes")
    triangles = []
    for index in range(count):
        base = 84 + 50 * index
        values = struct.unpack("<12f", data[base:base + 48])
        if not all(math.isfinite(v) for v in values):
            raise _PredictionError(f"STL triangle {index} has non-finite coordinates")
        triangles.append(tuple(tuple(values[3 * k:3 * k + 3]) for k in range(1, 4)))
    return triangles


def mesh_volume(triangles: list[Triangle]) -> float:
    total = 0.0
    for a, b, c in triangles:
        total += (a[0] * (b[1] * c[2] - b[2] * c[1])
                  - a[1] * (b[0] * c[2] - b[2] * c[0])
                  + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return abs(total) / 6.0


def mesh_bbox(triangles: list[Triangle]) -> list[float]:
    axes = [[v[axis] for t in triangles for v in t] for axis in range(3)]
    return [max(values) - min(values) for values in axes]


def _vertex_key(vertex) -> tuple[int, int, int]:
    return tuple(round(c / VERTEX_QUANTUM) for c in vertex)


def mesh_open_edges(triangles: list[Triangle]) -> int:
    counts: dict[tuple, int] = {}
    for triangle in triangles:
        keys = [_vertex_key(v) for v in triangle]
        for index in range(3):
            edge = tuple(sorted((keys[index], keys[(index + 1) % 3])))
            counts[edge] = counts.get(edge, 0) + 1
    return sum(1 for seen in counts.values() if seen != 2)


def mesh_components(triangles: list[Triangle]) -> int:
    """Connected components over shared vertices (a split part exports two shells)."""
    parent: dict[tuple, tuple] = {}

    def find(node):
        root = node
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[node] != root:          # path compression
            parent[node], node = root, parent[node]
        return root

    for triangle in triangles:
        keys = [_vertex_key(v) for v in triangle]
        for other in keys[1:]:
            a, b = find(keys[0]), find(other)
            if a != b:
                parent[a] = b
    return len({find(key) for key in parent})


_STEP_LABEL = re.compile(r"PRODUCT\s*\(\s*'([^']*)'")


def step_problems(text: str) -> list[str]:
    problems = []
    if not text.startswith("ISO-10303-21;"):
        problems.append("no ISO-10303-21 header")
    if not text.rstrip().endswith("END-ISO-10303-21;"):
        problems.append("no END-ISO-10303-21 terminator")
    if "FILE_SCHEMA" not in text:
        problems.append("no FILE_SCHEMA")
    if "MANIFOLD_SOLID_BREP" not in text and "ADVANCED_BREP_SHAPE_REPRESENTATION" not in text:
        problems.append("no solid B-rep entity")
    if text.count("CARTESIAN_POINT") < 8:
        problems.append(f"only {text.count('CARTESIAN_POINT')} CARTESIAN_POINTs")
    return problems


def step_labels(text: str) -> list[str]:
    return [name for name in _STEP_LABEL.findall(text) if name]


def read_file(path: Path, *, binary: bool = False):
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        return path.read_bytes() if binary else path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise _PredictionError(f"{path.name} is unreadable: {exc}") from None


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


def _reference(name: str, weight: float, ref_dir: Path, config: dict):
    try:
        return load_reference(ref_dir / config.get("ref_file", "reference.json")), None
    except _ReferenceError as exc:
        return None, _evaluator_failure(name, weight, str(exc))


def _mesh(pred_dir: Path, config: dict) -> list[Triangle]:
    return parse_binary_stl(read_file(pred_dir / config.get("stl_file", "part.stl"), binary=True))


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

@register_scorer("build123d_e2e_schema")
class PlateSchema(Scorer):
    """Gate: the three submitted files exist and the part is one watertight solid."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "build123d_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            load_result(pred_dir / config.get("pred_file", "result.json"))
            step = read_file(pred_dir / config.get("step_file", "part.step"))
            problems = step_problems(step)
            if problems:
                raise _PredictionError("STEP is not a solid model: " + "; ".join(problems))
            triangles = _mesh(pred_dir, config)
            open_edges = mesh_open_edges(triangles)
            if open_edges:
                raise _PredictionError(f"exported mesh is not watertight ({open_edges} unpaired edges)")
            components = mesh_components(triangles)
            if components != 1:
                raise _PredictionError(f"exported mesh has {components} disjoint bodies, expected one solid")
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"triangles": len(triangles), "step_labels": step_labels(step)},
                           message="result.json complete, STEP is a solid B-rep, STL is one watertight shell")


@register_scorer("build123d_e2e_measurements")
class Measurements(Scorer):
    """volume, surface area, mass and Izz from measure(), plus the volume the written STEP
    reports when it is imported back, all relative to the closed form."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "build123d_e2e_measurements"
        weight = float(config.get("weight", 1.0))
        ref, failure = _reference(name, weight, ref_dir, config)
        if failure is not None:
            return failure
        try:
            pred = load_result(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        full_tol = float(config["full_score_tol"])
        zero_tol = float(config["zero_score_tol"])
        errors = {key: relative(pred[key], ref[key]) for key in MEASUREMENTS}
        fractions = {key: credit(err, full_tol, zero_tol) for key, err in errors.items()}
        fraction = sum(fractions.values()) / len(fractions)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=all(err <= full_tol for err in errors.values()),
                           details={"relative_errors": errors, "credit_fractions": fractions,
                                    "predicted": {k: pred[k] for k in MEASUREMENTS},
                                    "reference": {k: ref[k] for k in MEASUREMENTS}},
                           message="; ".join(f"{k}: {v:.2e}" for k, v in errors.items()))


@register_scorer("build123d_e2e_features")
class Features(Scorer):
    """Hole table, bounding box and validity gate as reported in result.json."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "build123d_e2e_features"
        weight = float(config.get("weight", 1.0))
        ref, failure = _reference(name, weight, ref_dir, config)
        if failure is not None:
            return failure
        try:
            pred = load_result(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        dim_tol = float(config.get("dimension_tol", 1e-3))
        parts = {
            "hole_count": 1.0 if pred["hole_count"] == ref["hole_count"] else 0.0,
            "bolt_hole_diameter_mm": 1.0 if abs(pred["bolt_hole_diameter_mm"]
                                                - ref["bolt_hole_diameter_mm"]) <= dim_tol else 0.0,
            "bbox_mm": 1.0 if all(abs(p - r) <= dim_tol
                                  for p, r in zip(pred["bbox_mm"], ref["bbox_mm"])) else 0.0,
            "n_solids": 1.0 if pred["n_solids"] == ref["n_solids"] else 0.0,
            "passes_gate": 1.0 if pred["passes_gate"] is ref["passes_gate"] else 0.0,
        }
        fraction = sum(parts.values()) / len(parts)
        wrong = sorted(key for key, value in parts.items() if value < 1.0)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=not wrong,
                           details={"parts": parts, "wrong": wrong,
                                    "predicted": {k: pred[k] for k in parts},
                                    "reference": {k: ref[k] for k in parts}},
                           message="all features match" if not wrong else f"wrong: {wrong}")


@register_scorer("build123d_e2e_mesh")
class ExportedMesh(Scorer):
    """The STL the server wrote: tessellated volume and bounding box of the solid."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "build123d_e2e_mesh"
        weight = float(config.get("weight", 1.0))
        ref, failure = _reference(name, weight, ref_dir, config)
        if failure is not None:
            return failure
        try:
            triangles = _mesh(pred_dir, config)
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        volume_error = relative(mesh_volume(triangles), ref["volume_mm3"])
        volume_fraction = credit(volume_error, float(config["full_score_tol"]),
                                 float(config["zero_score_tol"]))
        box = mesh_bbox(triangles)
        dim_tol = float(config.get("dimension_tol", 1e-3))
        box_fraction = 1.0 if all(abs(p - r) <= dim_tol for p, r in zip(box, ref["bbox_mm"])) else 0.0
        fraction = 0.7 * volume_fraction + 0.3 * box_fraction
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=volume_fraction == 1.0 and box_fraction == 1.0,
                           details={"triangles": len(triangles), "mesh_volume_mm3": mesh_volume(triangles),
                                    "relative_volume_error": volume_error, "bbox_mm": box,
                                    "credit_fractions": {"volume": volume_fraction, "bbox": box_fraction}},
                           message=f"{len(triangles)} triangles, relative volume error "
                                   f"{volume_error:.2e}, bbox {'matches' if box_fraction else 'differs'}")


@register_scorer("build123d_e2e_step")
class ExportedStep(Scorer):
    """The STEP the server wrote carries the session name of the part."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "build123d_e2e_step"
        weight = float(config.get("weight", 1.0))
        ref, failure = _reference(name, weight, ref_dir, config)
        if failure is not None:
            return failure
        try:
            text = read_file(pred_dir / config.get("step_file", "part.step"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        labels = step_labels(text)
        ok = ref["part"] in labels
        message = (f"STEP product label {ref['part']!r} present" if ok
                   else f"STEP labels {labels[:4]} do not name the part {ref['part']!r} "
                        f"(export without object_name writes 'COMPOUND')")
        return ScoreDetail(scorer_name=name, score=weight if ok else 0.0, max_score=weight,
                           passed=ok, details={"labels": labels[:8], "expected": ref["part"]},
                           message=message)
