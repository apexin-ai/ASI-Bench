"""Scorers for the Trident shallow-Mogi plus deep-Yang inversion task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer, get_scorer
from ai4sci_bench.core.types import ScoreDetail


NU = 0.25
MU = 1.0e10


def _reported_time(pred_dir: Path) -> tuple[float, float]:
    payload = json.loads((pred_dir / "timing.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"elapsed_seconds", "timing_source"}:
        raise ValueError("timing.json requires exactly elapsed_seconds and timing_source")
    elapsed = payload["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise ValueError("elapsed_seconds must be a number, not a boolean")
    if not np.isfinite(elapsed) or elapsed < 0 or payload["timing_source"] != "agent_reported":
        raise ValueError("invalid agent-reported timing")
    penalty = min(20.0, 0.2 * max(0.0, float(elapsed) - 240.0))
    return float(elapsed), penalty


def _complexity(pred_dir: Path) -> tuple[int, float]:
    """Count declared independent fitted scalars, not quadrature nodes.

    The declaration is auditable, not a proof about arbitrary Python code.
    Reject counts inconsistent with the public scalar-parameter inventory.
    """
    payload = json.loads((pred_dir / "model.json").read_text())
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("missing model components")
    total = 0
    for component in components:
        params = component["parameters"]
        count = params.get("fitted_parameter_count", {})
        value = count.get("median")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value < 0 or value != int(value)
                or count.get("q05") != value or count.get("q95") != value
                or count.get("unit") != "1"):
            raise ValueError("each component requires an integer fitted_parameter_count summary")
        scalars = [v for k, v in params.items() if k != "fitted_parameter_count"]
        varying = sum(v["q05"] < v["q95"] for v in scalars)
        if not varying <= value <= len(scalars):
            raise ValueError("parameter count conflicts with scalar inventory; report all fitted coefficients")
        total += int(value)
    return total, min(80.0, 10.0 * max(0, total - 12))


@register_scorer("trident_reported_time")
class TridentReportedTimeGate(Scorer):
    """Validate self-reported timing; this does NOT authenticate host runtime."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        try:
            elapsed, penalty = _reported_time(pred_dir)
            passed = True
            return ScoreDetail(
                scorer_name=self.name, score=float(passed), max_score=1.0, passed=passed,
                details={"agent_reported_seconds": elapsed, "time_penalty_points": penalty,
                         "trusted_harness_measurement": False},
                message="Valid self-reported timing; additive penalty applies after 240 seconds",
            )
        except Exception as exc:
            return ScoreDetail(scorer_name=self.name, score=0.0, max_score=1.0,
                               passed=False, details={"error": str(exc)}, message=str(exc))


@register_scorer("trident_time_deducted")
class TridentTimeDeductedScorer(Scorer):
    """Allocate ONE total additive time deduction across the science components."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        elapsed, penalty = _reported_time(pred_dir)
        parameter_count, complexity_penalty = _complexity(pred_dir)
        inner_config = dict(config)
        all_components = inner_config.pop("all_components")
        base_name = inner_config.pop("base_scorer")
        if base_name not in {"trident_model_consistency", "trident_los_plateau", "trident_geometry", "trident_locations", "trident_finite_inference"}:
            raise ValueError("unsupported base scorer")
        base = get_scorer(base_name).score(pred_dir, ref_dir, inner_config)
        total = 0.0
        for component in all_components:
            part_config = dict(component["config"])
            part_config["weight"] = component["weight"]
            total += get_scorer(component["scorer"]).score(pred_dir, ref_dir, part_config).score
        # The runner sums per-component scores. Proportional allocation preserves
        # exactly max(0, TOTAL_SCIENCE - penalty), without deducting per component.
        final_total = max(0.0, total - penalty - complexity_penalty)
        allocation = final_total / total if total > 0 else 0.0
        return ScoreDetail(
            scorer_name=self.name, score=base.score * allocation, max_score=base.max_score,
            passed=base.passed,
            details={"base_scorer": base_name, "scientific_score": base.score,
                     "agent_reported_seconds": elapsed, "time_penalty_points": penalty,
                     "fitted_parameter_count": parameter_count,
                     "complexity_penalty_points": complexity_penalty,
                     "total_scientific_score": total, "final_total_score": final_total,
                     "allocated_penalty_points": base.score * (1.0 - allocation),
                     "trusted_harness_measurement": False, "scientific_details": base.details},
            message=f"{base.message}; time penalty={penalty:.6g}; complexity penalty={complexity_penalty:.6g}",
        )


@register_scorer("trident_completion")
class TridentCompletionScorer(Scorer):
    """A voluntary incomplete declaration always fails the common hard gate."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        marker = pred_dir / "failure.json"
        incomplete = marker.exists() or marker.is_symlink()
        return ScoreDetail(
            scorer_name=self.name,
            score=0.0 if incomplete else 1.0,
            max_score=1.0,
            passed=not incomplete,
            details={"incomplete_marker_present": incomplete},
            message="Explicitly incomplete attempt" if incomplete else "No abandonment marker",
        )


def _validate_summary(summary: np.ndarray) -> None:
    if summary.shape != (12, 3) or not np.all(np.isfinite(summary)):
        raise ValueError("source_summary.npy must be finite float64 with shape (12, 3)")
    if np.any(summary[:, 0] > summary[:, 1]) or np.any(summary[:, 1] > summary[:, 2]):
        raise ValueError("source quantiles are not ordered")
    s = summary[:, 1]
    if not (-20000.0 <= s[0] <= 10000.0 and -20000.0 <= s[1] <= 10000.0):
        raise ValueError("shallow source location is outside the declared bounds")
    if not (100.0 <= s[2] <= 8000.0 and 0.0 < s[3] <= 1.0e8):
        raise ValueError("invalid shallow source depth or volume-change rate")
    if not (-20000.0 <= s[4] <= 10000.0 and -20000.0 <= s[5] <= 10000.0):
        raise ValueError("deep source location is outside the declared bounds")
    if not (12000.0 <= s[6] <= 18000.0 and 0.0 < s[7] <= 1.0e8):
        raise ValueError("invalid deep source depth or pressure-change rate")
    if not (
        100.0 <= s[9] < s[8] <= 20000.0
        and s[9] <= 10000.0
        and 0.0 <= s[10] <= 180.0
        and 0.0 <= s[11] <= 90.0
    ):
        raise ValueError("invalid Yang spheroid axes or orientation")
    if not (np.hypot(s[8]*np.sin(np.radians(s[11])), s[9]*np.cos(np.radians(s[11]))) < s[6]):
        raise ValueError("deep spheroid intersects the free surface")


def _load_model(path: Path) -> np.ndarray:
    """Convert the public, variable-length model contract to the private 12-vector."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("model.json must contain a non-empty components array")

    by_type = {}
    for component in components:
        if not isinstance(component, dict):
            continue
        key = str(component.get("type", "")).strip().lower()
        by_type[key] = component.get("parameters", {})

    mogi = by_type.get("mogi")
    yang = by_type.get("yang_prolate_spheroid")
    if not isinstance(mogi, dict) or not isinstance(yang, dict):
        raise ValueError("a recognizable Mogi plus Yang prolate-spheroid model was not reported")

    names = (
        (mogi, "x_m"),
        (mogi, "y_m"),
        (mogi, "depth_m"),
        (mogi, "volume_change_rate_m3_per_year"),
        (yang, "x_m"),
        (yang, "y_m"),
        (yang, "depth_m"),
        (yang, "pressure_change_rate_pa_per_year"),
        (yang, "major_axis_m"),
        (yang, "minor_axis_m"),
        (yang, "strike_deg"),
        (yang, "dip_deg"),
    )
    summary = np.empty((12, 3), dtype=np.float64)
    for row, (parameters, name) in enumerate(names):
        value = parameters.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing parameter {name}")
        summary[row] = [value["q05"], value["median"], value["q95"]]
    _validate_summary(summary)
    return summary


def _finite_summary(path: Path) -> np.ndarray:
    """Read a finite prolate body's geometry without requiring another source."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for component in payload.get("components", []):
        p = component.get("parameters", {})
        if not all(k in p for k in ("major_axis_m", "minor_axis_m", "strike_deg", "dip_deg")):
            continue
        xyz = [_coordinate_summary(p, ns) for ns in (("x_m", "source_east"), ("y_m", "source_north"), ("depth_m", "depth"))]
        geom = np.array([[p[k][q] for q in ("q05", "median", "q95")] for k in
            ("major_axis_m", "minor_axis_m", "strike_deg", "dip_deg")], dtype=float)
        if not np.isfinite(geom).all() or np.any(geom[:, 0] > geom[:, 1]) or np.any(geom[:, 1] > geom[:, 2]):
            raise ValueError("invalid finite-body quantiles")
        depth = xyz[2][1]; a, b, strike, dip = geom[:, 1]
        vertical_extent = np.sqrt((a*np.sin(np.deg2rad(dip)))**2 + (b*np.cos(np.deg2rad(dip)))**2)
        if not a > b > 0:
            raise ValueError("finite body must have major > minor > 0")
        if not 0 <= dip <= 90:
            raise ValueError("dip must lie between 0 and 90 degrees")
        if not depth > vertical_extent:
            raise ValueError("finite body intersects the free surface")
        summary = np.zeros((12, 3))
        summary[4:7] = xyz
        summary[8:12] = geom
        return summary
    raise ValueError("no finite-body axis/orientation summary was reported")


def _mogi_surface(
    x: np.ndarray, y: np.ndarray, xcen: float, ycen: float, depth: float, dvolume: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = x - xcen
    dy = y - ycen
    radius = np.sqrt(dx * dx + dy * dy + depth * depth)
    coefficient = (1.0 - NU) * dvolume / np.pi
    common = coefficient / radius**3
    return common * dx, common * dy, common * depth


def _yang_spheroid(a: float, b: float, pressure: float) -> np.ndarray:
    c = np.sqrt(a * a - b * b)
    log_ac = np.log((a - c) / (a + c))
    iia = 2.0 / (a * c**2) + log_ac / c**3
    iiaa = 2.0 / (3.0 * a**3 * c**2) + 2.0 / (a * c**4) + log_ac / c**5
    coef1 = -2.0 * np.pi * a * b**2
    ia = coef1 * iia
    iaa = coef1 * iiaa
    q = 3.0 / (8.0 * np.pi * (1.0 - NU))
    r = (1.0 - 2.0 * NU) / (8.0 * np.pi * (1.0 - NU))
    a11 = 2.0 * r * (ia - 4.0 * np.pi)
    a12 = -2.0 * r * (ia + 4.0 * np.pi)
    a21 = q * a**2 * iaa + r * ia - 1.0
    a22 = -(q * a**2 * iaa + ia * (2.0 * r - q))
    lame = MU
    denom = 1.0 / (a11 * a22 - a12 * a21)
    e11 = (3.0 * a22 - a12) * pressure * denom / (3.0 * lame + 2.0 * MU)
    e22 = (a11 - 3.0 * a21) * pressure * denom / (3.0 * lame + 2.0 * MU)
    pdila = 2.0 * MU * (e11 - e22)
    pstar = lame * e11 + 2.0 * (lame + MU) * e22
    a1 = -2.0 * b**2 * pdila
    b1 = 3.0 * b**2 * pdila / c**2 + 2.0 * (1.0 - 2.0 * NU) * pstar
    return np.array([a, b, c, pdila, a1, b1], dtype=np.float64)


def _yang_half(
    sph: np.ndarray,
    xi: float,
    depth: float,
    x: np.ndarray,
    y: np.ndarray,
    sin_t: float,
    cos_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, pdila, a1, b1 = sph
    nu4 = 3.0 - 4.0 * NU
    coeff = a * b**2 / c**3 / (16.0 * MU * (1.0 - NU))
    xi2 = xi * cos_t
    xi3 = xi * sin_t
    x1 = x
    x2 = y
    x3 = -depth
    xbar3 = depth
    y1 = x1
    y2 = x2 - xi2
    y3 = x3 - xi3
    ybar3 = xbar3 + xi3
    r2 = x2 * sin_t - x3 * cos_t
    q2 = x2 * sin_t + xbar3 * cos_t
    r3 = x2 * cos_t + x3 * sin_t
    q3 = -x2 * cos_t + xbar3 * sin_t
    rbar3 = r3 - xi
    qbar3 = q3 + xi
    r1 = np.sqrt(y1**2 + y2**2 + y3**2)
    r_2 = np.sqrt(y1**2 + y2**2 + ybar3**2)
    c0 = depth / sin_t
    beta_top = cos_t * q2 + (1.0 + sin_t) * (r_2 + qbar3)
    beta_bottom = cos_t * y1
    atn_beta = np.pi / 2.0 * np.sign(beta_top)
    nz = np.abs(beta_bottom) != 0.0
    atn_beta[nz] = np.arctan(beta_top[nz] / beta_bottom[nz])
    rr = r1 + rbar3
    rq = r_2 + qbar3
    ry = r_2 + ybar3
    lrr = np.log(rr)
    lrq = np.log(rq)
    lry = np.log(ry)
    a1star = a1 / (r1 * rr) + b1 * (lrr + (r3 + xi) / rr)
    abar1star = -a1 / (r_2 * rq) - b1 * (lrq + (q3 - xi) / rq)
    a_1 = xi / r1 + lrr
    abar1 = xi / r_2 - lrq
    a2 = r1 - r3 * lrr
    abar2 = r_2 - q3 * lrq
    a3 = xi * rbar3 / r1 + r1
    abar3 = xi * qbar3 / r_2 - r_2
    bterm = xi * (xi + c0) / r_2 - abar2 - c0 * lrq
    bstar = a1 / r1 + 2.0 * b1 * a2 + nu4 * (a1 / r_2 + 2.0 * b1 * abar2)
    ff1 = (
        xi * y1 / ry
        + 3.0 / cos_t**2 * (y1 * lry * sin_t - y1 * lrq + 2.0 * q2 * atn_beta)
        + 2.0 * y1 * lrq
        - 4.0 * xbar3 * atn_beta / cos_t
    )
    ff2 = (
        xi * y2 / ry
        + 3.0
        / cos_t**2
        * (q2 * lrq * sin_t - q2 * lry + 2.0 * y1 * atn_beta * sin_t + cos_t * (r_2 - ybar3))
        - 2.0 * cos_t * abar2
        + 2.0 / cos_t * (xbar3 * lry - q3 * lrq)
    )
    ff3 = (q2 * lrq - q2 * lry * sin_t + 2.0 * y1 * atn_beta) / cos_t + 2.0 * sin_t * abar2 + q3 * lry - xi
    u1 = coeff * (a1star + nu4 * abar1star) * y1
    u2 = coeff * (sin_t * (a1star * r2 + nu4 * abar1star * q2) + cos_t * bstar)
    u3 = coeff * (-cos_t * (a1star * r2 + nu4 * abar1star * q2) + sin_t * bstar)
    u1 += 2.0 * coeff * pdila * ((a_1 + nu4 * abar1) * y1 - 4.0 * (1.0 - NU) * (1.0 - 2.0 * NU) * ff1)
    coeff2 = 4.0 * (1.0 - NU) * (1.0 - 2.0 * NU)
    u2 += 2.0 * coeff * pdila * (
        sin_t * (a_1 * r2 + nu4 * abar1 * q2)
        - coeff2 * ff2
        + 4.0 * (1.0 - NU) * cos_t * (a2 + abar2)
        + cos_t * (a3 - nu4 * abar3)
    )
    u3 += 2.0 * coeff * pdila * (
        cos_t * (-a_1 * r2 + nu4 * abar1 * q2)
        + coeff2 * ff3
        + 4.0 * (1.0 - NU) * sin_t * (a2 + abar2)
        + sin_t * (a3 + nu4 * abar3 - 2.0 * nu4 * bterm)
    )
    return u1, u2, u3


def _yang_surface(
    x: np.ndarray,
    y: np.ndarray,
    xcen: float,
    ycen: float,
    depth: float,
    pressure: float,
    a: float,
    b: float,
    strike_deg: float,
    dip_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dip_deg = 0.1 if dip_deg == 0.0 else 89.9 if dip_deg == 90.0 else dip_deg
    strike = np.radians(strike_deg)
    dip = np.radians(dip_deg)
    sin_t, cos_t = np.sin(dip), np.cos(dip)
    cos_p, sin_p = np.cos(strike), np.sin(strike)
    sph = _yang_spheroid(a, b, pressure)
    dx = x - xcen
    dy = y - ycen
    xp = dx * cos_p - dy * sin_p
    yp = dy * cos_p + dx * sin_p
    c = sph[2]
    plus = _yang_half(sph, c, depth, xp, yp, sin_t, cos_t)
    minus = _yang_half(sph, -c, depth, xp, yp, sin_t, cos_t)
    u1r = -plus[0] + minus[0]
    u2r = -plus[1] + minus[1]
    return (
        u1r * cos_p + u2r * sin_p,
        -u1r * sin_p + u2r * cos_p,
        plus[2] - minus[2],
    )


def _predict_los(summary: np.ndarray, geometry: np.lib.npyio.NpzFile) -> np.ndarray:
    s = summary[:, 1]
    x = geometry["x_m"]
    y = geometry["y_m"]
    m = _mogi_surface(x, y, *s[0:4])
    g = _yang_surface(x, y, *s[4:12])
    east = m[0] + g[0]
    north = m[1] + g[1]
    up = m[2] + g[2]
    incidence = np.radians(geometry["incidence_deg"])
    azimuth = np.radians(geometry["azimuth_deg"])
    los = -(east * np.sin(incidence) * np.cos(azimuth) - north * np.sin(incidence) * np.sin(azimuth) - up * np.cos(incidence))
    reference_index = int(geometry["reference_index"])
    return los - los[reference_index]


def _linear_score(value: float, full: float, zero: float) -> float:
    if not np.isfinite(value) or value >= zero:
        return 0.0
    if value <= full:
        return 1.0
    return 1.0 - (value - full) / (zero - full)


def _coordinate_summary(parameters: dict, names: tuple[str, ...]) -> np.ndarray:
    obj = next((parameters[n] for n in names if n in parameters), None)
    if not isinstance(obj, dict) or obj.get("unit") != "m":
        raise ValueError("coordinate requires quantiles and unit m")
    q = np.array([obj[k] for k in ("q05", "median", "q95")], dtype=float)
    if not np.isfinite(q).all() or not q[0] <= q[1] <= q[2]:
        raise ValueError("invalid coordinate quantiles")
    return q


@register_scorer("trident_locations")
class TridentLocationsScorer(Scorer):
    """One-to-one reference matching of fitted centers, independent of kernel name."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        from scipy.optimize import linear_sum_assignment
        weight = float(config.get("weight", 1.0))
        try:
            components = json.loads((pred_dir / config["model_file"]).read_text())["components"]
            candidates = []
            for component in components:
                try:
                    p = component["parameters"]
                    q = np.array([_coordinate_summary(p, names) for names in (
                        ("x_m", "source_east"), ("y_m", "source_north"), ("depth_m", "depth"))])
                    if q[2, 0] <= 0 or np.any(q[:, 0] >= q[:, 2]):
                        continue  # fixed coordinates/depth are not inferred centers
                    candidates.append(q[:, 1])
                except (KeyError, TypeError, ValueError):
                    continue
            target = np.load(ref_dir / config["target_file"], allow_pickle=False)
            references = target[[[0, 1, 2], [4, 5, 6]]]
            full = np.asarray(config["full_tolerances"], dtype=float)
            zero = np.asarray(config["zero_tolerances"], dtype=float)
            fractions = np.zeros(2)
            matches = []
            if candidates:
                errors = np.abs(references[:, None, :] - np.asarray(candidates)[None, :, :])
                cost = np.mean(errors / zero[:, None, :], axis=2)
                rows, cols = linear_sum_assignment(cost)
                for i, j in zip(rows, cols):
                    fractions[i] = np.mean([_linear_score(e, f, z) for e, f, z in zip(errors[i, j], full[i], zero[i])])
                    matches.append({"reference_index": int(i), "candidate_index": int(j), "absolute_errors_m": errors[i, j].tolist()})
            return ScoreDetail(scorer_name=self.name, score=weight * float(np.mean(fractions)),
                max_score=weight, passed=True, details={"reference_fractions": fractions.tolist(), "matches": matches},
                message="Missing or fixed centers receive no location/depth bonus")
        except Exception as exc:
            return ScoreDetail(scorer_name=self.name, score=0, max_score=weight, passed=False,
                details={"error": str(exc)}, message=str(exc))


@register_scorer("trident_finite_inference")
class TridentFiniteInferenceScorer(Scorer):
    """Extra credit for a valid finite-body summary, never a hard eligibility gate."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            summary = _finite_summary(pred_dir / config["model_file"])
            fitted = all(summary[i, 0] < summary[i, 2] for i in (6, 8, 9))
            return ScoreDetail(scorer_name=self.name, score=weight if fitted else 0,
                max_score=weight, passed=fitted, details={"nondegenerate_depth_and_size_intervals": fitted},
                message="Finite-source depth and axes must have genuine fitted uncertainty, not fixed grid values")
        except Exception as exc:
            return ScoreDetail(scorer_name=self.name, score=0, max_score=weight, passed=False,
                details={"error": str(exc)}, message=str(exc))


def _plateau_fraction(error: float, full: float, start: float, end: float, zero: float) -> float:
    """Continuous non-increasing score with a half-credit interval."""
    if not 0.0 <= full < start <= end < zero:
        raise ValueError("require 0 <= full < half_start <= half_end < zero")
    if not np.isfinite(error) or error >= zero:
        return 0.0
    if error <= full:
        return 1.0
    if error < start:
        return 1.0 - 0.5 * (error - full) / (start - full)
    if error <= end:
        return 0.5
    return 0.5 * (zero - error) / (zero - end)


@register_scorer("trident_los_plateau")
class TridentLosPlateauScorer(Scorer):
    """Same relative-L2 metric for all levels, with a declared score plateau."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred = np.load(pred_dir / config["pred_file"], allow_pickle=False)
            ref = np.load(ref_dir / config["ref_file"], allow_pickle=False)
            if pred.shape != (1321,) or ref.shape != pred.shape:
                raise ValueError("LOS arrays must both have shape (1321,)")
            if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(ref)):
                raise ValueError("LOS arrays must contain only finite values")
            denominator = float(np.linalg.norm(ref))
            error = float(np.linalg.norm(pred - ref))
            relative_l2 = error / denominator if denominator > 0.0 else (0.0 if error == 0.0 else np.inf)
            fraction = _plateau_fraction(
                relative_l2,
                float(config["full_score_threshold"]),
                float(config["half_score_start"]),
                float(config["half_score_end"]),
                float(config["zero_score_threshold"]),
            )
            return ScoreDetail(
                scorer_name=self.name, score=weight * fraction, max_score=weight,
                passed=True,
                details={"relative_l2": relative_l2 if np.isfinite(relative_l2) else None,
                         "raw_fraction": fraction},
                message=f"relative_l2={relative_l2:.6g}; score={weight*fraction:.2f}/{weight:.2f}",
            )
        except Exception as exc:
            return ScoreDetail(
                scorer_name=self.name, score=0.0, max_score=weight, passed=False,
                details={"error": str(exc)}, message=str(exc),
            )


@register_scorer("trident_model_consistency")
class TridentModelConsistencyScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        try:
            model_path = pred_dir / config["model_file"]
            components = json.loads(model_path.read_text())["components"]
            finite = _finite_summary(model_path)
            submitted = np.load(pred_dir / config["modeled_file"], allow_pickle=False)
            if submitted.shape != (1321,) or not np.all(np.isfinite(submitted)):
                raise ValueError("modeled_los.npy must be finite float64 with shape (1321,)")
            with np.load(ref_dir / config["scoring_data_file"], allow_pickle=False) as geometry:
                displacement = np.zeros((3, len(geometry["x_m"])))
                finite_count = 0
                for component in components:
                    p = component["parameters"]
                    xyz = [float(_coordinate_summary(p, names)[1]) for names in (
                        ("x_m", "source_east"), ("y_m", "source_north"), ("depth_m", "depth"))]
                    if xyz[2] <= 0:
                        raise ValueError("source depth must be positive")
                    if "major_axis_m" in p:
                        finite_count += 1
                        if finite_count > 1:
                            raise ValueError("finite-forward verifier currently supports one finite body")
                        strength = float(p["pressure_change_rate_pa_per_year"]["median"])
                        if not np.isfinite(strength) or strength <= 0:
                            raise ValueError("invalid pressure rate")
                        displacement += _yang_surface(geometry["x_m"], geometry["y_m"], *xyz, strength, *finite[8:12, 1])
                    else:
                        key = "volume_change_rate_m3_per_year" if "volume_change_rate_m3_per_year" in p else "volume_change_rate"
                        strength = float(p[key]["median"])
                        if not np.isfinite(strength) or strength <= 0:
                            raise ValueError("invalid volume rate")
                        displacement += _mogi_surface(geometry["x_m"], geometry["y_m"], *xyz, strength)
                inc = np.deg2rad(geometry["incidence_deg"])
                az = np.deg2rad(geometry["azimuth_deg"])
                reproduced = -(displacement[0]*np.sin(inc)*np.cos(az)-displacement[1]*np.sin(inc)*np.sin(az)-displacement[2]*np.cos(inc))
                reproduced -= reproduced[int(geometry["reference_index"])]
            denominator = float(np.linalg.norm(reproduced))
            error = float(np.linalg.norm(submitted - reproduced))
            relative_l2 = error / denominator if denominator > 0.0 else error

            if config.get("scoring") == "linear_interpolation":
                weight = float(config.get("weight", 1.0))
                full = float(config["full_score_threshold"])
                zero = float(config["zero_score_threshold"])
                fraction = _linear_score(relative_l2, full, zero)
                return ScoreDetail(
                    scorer_name=self.name,
                    score=weight * fraction,
                    max_score=weight,
                    passed=True,
                    details={"relative_l2": relative_l2, "raw_fraction": fraction},
                    message=f"relative_l2={relative_l2:.6g}; score={weight*fraction:.2f}/{weight:.2f}",
                )

            threshold = float(config.get("max_relative_l2", 0.03))
            passed = np.isfinite(relative_l2) and relative_l2 <= threshold
            return ScoreDetail(
                scorer_name=self.name,
                score=1.0 if passed else 0.0,
                max_score=1.0,
                passed=passed,
                details={"relative_l2": relative_l2, "threshold": threshold},
                message="" if passed else f"median parameters do not reproduce modeled_los.npy: {relative_l2:.6g} > {threshold:.6g}",
            )
        except Exception as exc:
            return ScoreDetail(
                scorer_name=self.name,
                score=0.0,
                max_score=float(config.get("weight", 1.0)),
                passed=False,
                details={"error": str(exc)},
                message=str(exc),
            )


@register_scorer("trident_geometry")
class TridentGeometryScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        try:
            summary = _finite_summary(pred_dir / config["model_file"])
            target = np.load(ref_dir / config["target_file"], allow_pickle=False)
            if target.shape != (12,) or not np.all(np.isfinite(target)):
                raise ValueError("invalid private source target")
            indices = [int(i) for i in config["indices"]]
            full = np.asarray(config["full_tolerances"], dtype=np.float64)
            zero = np.asarray(config["zero_tolerances"], dtype=np.float64)
            if len(indices) != len(full) or len(indices) != len(zero):
                raise ValueError("geometry scorer tolerance lengths differ")
            errors = np.abs(summary[indices, 1] - target[indices])
            periods = {int(k): float(v) for k, v in config.get("circular_periods", {}).items()}
            for j, index in enumerate(indices):
                if index in periods:
                    period = periods[index]
                    errors[j] = abs((summary[index, 1] - target[index] + period / 2.0) % period - period / 2.0)
            fractions = np.array([_linear_score(e, f, z) for e, f, z in zip(errors, full, zero)])
            fraction = float(np.mean(fractions))
            weight = float(config.get("weight", 1.0))
            return ScoreDetail(
                scorer_name=self.name,
                score=weight * fraction,
                max_score=weight,
                passed=True,
                details={"indices": indices, "absolute_errors": errors.tolist(), "parameter_fractions": fractions.tolist()},
                message=f"mean parameter fraction={fraction:.6g}; score={weight*fraction:.2f}/{weight:.2f}",
            )
        except Exception as exc:
            return ScoreDetail(
                scorer_name=self.name,
                score=0.0,
                max_score=float(config.get("weight", 1.0)),
                passed=False,
                details={"error": str(exc)},
                message=str(exc),
            )
