"""Scorer for earth_science.flood_peak_reconstruction.

Seven weighted components, all recomputed here:

===========================  ======  ==========================================
Component                    Weight  Compared with
===========================  ======  ==========================================
``gauge_hydrographs``            48  ``reference/results/gauge_hydrographs.csv``
``propagation_section``          22  ``reference/results/propagation_section.csv``
``peak_timing``                   6  ``reference/results/flood_summary.json``
``event_volume``                 10  ``reference/results/tributary_inflow.csv``
``tributary_inflow``              8  the stage record in ``reference/public_ref.json``
``report_consistency``            5  the submitted propagation-section curve
``figure``                        1  PNG contract
===========================  ======  ==========================================

``tributary_inflow`` is not compared with any hidden quantity. The submitted
inflow is spread over the published along-reach profiles, routed once through the
published channel, and the resulting stage is compared with the samples that
carry a reading; the residual is in units of the published stage noise.

``event_volume`` integrates the submitted tributary inflow over the published
submission times and compares the total event volume with the same integral of
the reference inflow. It is a pass/fail check, awarded in full when the relative
difference is at most ``2 f / (1 - f)`` with ``f`` the per-tributary forecast
bound stated in the forecast note; ``tributary_inflow`` and
``report_consistency`` are pass/fail on their previous zero-credit thresholds.

Channel routing here is an independent implementation that does not import the
generator module; the constants, scoring bands and finite-volume scheme are held
identical to it.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_banded

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

#: Component weights, summing to 100.
WEIGHTS: dict[str, float] = {
    "gauge_hydrographs": 48.0,
    "propagation_section": 22.0,
    "peak_timing": 6.0,
    "event_volume": 10.0,
    "tributary_inflow": 8.0,
    "report_consistency": 5.0,
    "figure": 1.0,
}

#: Scoring bands, ``(full-credit threshold, zero-credit threshold)``.
BANDS: dict[str, tuple[float, float]] = {
    "gauge_hydrographs": (0.00849618187695455, 0.020699144938555146),  # relative RMS
    "propagation_section": (0.006227997350621134, 0.014064815568536873),  # relative RMS
    "peak_timing": (0.0029184503097449085, 0.005684133646348131),  # relative error
    "tributary_inflow": (1.4, 6.0),  # multiples of the stage noise
    "report_consistency": (0.06, 0.25),  # relative deviation
}

#: Half-width of the per-tributary event-volume error of the published forecast,
#: as stated in the ``forecast_note`` of ``data/reach.json``.
FORECAST_VOLUME_ERR = 0.10

#: Largest relative error the total tributary event volume can show while the
#: submission still respects that published bound. Both the true total and any
#: submission that respects the note lie in ``[(1 - f) F, (1 + f) F]`` with ``F``
#: the published forecast total, so ``|S - A| / A <= 2 f / (1 - f)``, attained at
#: the corners. Derived from published quantities alone.
EVENT_VOLUME_MAX_REL = 2.0 * FORECAST_VOLUME_ERR / (1.0 - FORECAST_VOLUME_ERR)

#: Components scored pass/fail rather than on a band; the value is the threshold.
CHECKS: dict[str, float] = {
    "event_volume": EVENT_VOLUME_MAX_REL,
    "tributary_inflow": BANDS["tributary_inflow"][1],
    "report_consistency": BANDS["report_consistency"][1],
}

MAX_TOTAL = float(sum(WEIGHTS.values()))

_HYDROGRAPH_FILE = "results/gauge_hydrographs.csv"
_PROPAGATION_FILE = "results/propagation_section.csv"
_INFLOW_FILE = "results/tributary_inflow.csv"
_SUMMARY_FILE = "results/flood_summary.json"
_FIGURE_FILE = "results/diagnostics.png"

#: Channel-routing constants; identical to the generator.
_SIDE_SLOPE = 2.0
_DIFF_FLOOR_FRAC = 0.10
_Y_TABLE = np.geomspace(0.02, 20.0, 600)


# ── Readers ─────────────────────────────────────────────────────────────────
def _read_rows(path: Path) -> list[list[str]] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.reader(handle)
                    if row and any(c.strip() for c in row)]
    except OSError:
        return None
    return rows if len(rows) >= 2 else None


def _column_index(header: list[str], name: str, fallback: int) -> int:
    cleaned = [cell.strip() for cell in header]
    return cleaned.index(name) if name in cleaned else fallback


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _scalar(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ordered(values: list[float], keys: list[float] | None) -> list[float]:
    """Reorder by ``sample_index``; keep the file order when the index is unusable.

    The published grid is already ascending within a gauge, so this is the
    identity on a submission that follows the contract.
    """

    if keys is None or len(keys) != len(values):
        return values
    return [v for _, v in sorted(zip(keys, values), key=lambda kv: kv[0])]


def _read_hydrographs(path: Path, names: list[str], n_per_gauge: int
                      ) -> np.ndarray | None:
    """Group ``discharge_m3_s`` by the ``gauge`` column.

    Returns shape ``(n_gauge, n_per_gauge)``; rows of one gauge are ordered by
    ascending ``sample_index``, matching the published grid.
    """

    rows = _read_rows(path)
    if rows is None:
        return None
    header = rows[0]
    i_gauge = _column_index(header, "gauge", 0)
    i_index = _column_index(header, "sample_index", 1)
    i_value = _column_index(header, "discharge_m3_s", 2)
    grouped: dict[str, list[float]] = {name: [] for name in names}
    keys: dict[str, list[float] | None] = {name: [] for name in names}
    for row in rows[1:]:
        if len(row) <= max(i_gauge, i_index, i_value):
            return None
        key = row[i_gauge].strip()
        if key not in grouped:
            return None
        try:
            grouped[key].append(float(row[i_value].strip()))
        except ValueError:
            return None
        column = keys[key]
        if column is not None:
            try:
                column.append(float(row[i_index].strip()))
            except ValueError:
                keys[key] = None
    if any(len(grouped[name]) != n_per_gauge for name in names):
        return None
    out = np.asarray([_ordered(grouped[name], keys[name]) for name in names], float)
    return out if np.all(np.isfinite(out)) else None


def _read_series(path: Path, column: str, fallback: int, size: int,
                 key_column: str | None = None) -> np.ndarray | None:
    rows = _read_rows(path)
    if rows is None:
        return None
    index = _column_index(rows[0], column, fallback)
    i_key = None if key_column is None else _column_index(rows[0], key_column, 0)
    values: list[float] = []
    keys: list[float] | None = None if i_key is None else []
    for row in rows[1:]:
        if len(row) <= max(index, i_key or 0):
            return None
        try:
            values.append(float(row[index].strip()))
        except ValueError:
            return None
        if keys is not None:
            try:
                keys.append(float(row[i_key].strip()))
            except ValueError:
                keys = None
    arr = np.asarray(_ordered(values, keys), float)
    if arr.size != size or not np.all(np.isfinite(arr)):
        return None
    return arr


def _read_inflow(path: Path, n_trib: int, size: int) -> np.ndarray | None:
    rows = _read_rows(path)
    if rows is None:
        return None
    header = [cell.strip() for cell in rows[0]]
    idx = []
    for j in range(n_trib):
        name = f"T{j + 1}_m3_s"
        idx.append(header.index(name) if name in header else j + 1)
    table: list[list[float]] = [[] for _ in range(n_trib)]
    for row in rows[1:]:
        if len(row) <= max(idx):
            return None
        for j, index in enumerate(idx):
            try:
                table[j].append(float(row[index].strip()))
            except ValueError:
                return None
    arr = np.asarray(table, float)
    if arr.shape != (n_trib, size) or not np.all(np.isfinite(arr)):
        return None
    return arr


# ── Piecewise-linear credit ─────────────────────────────────────────────────
def _linear_desc(err: float, full: float, zero: float) -> float:
    if not math.isfinite(err):
        return 0.0
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return float((zero - err) / (zero - full))


def _relative_rms(pred: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, float)
    if np.any(truth == 0.0):
        return math.inf
    return float(np.sqrt(np.mean(((np.asarray(pred, float) - truth) / truth) ** 2)))


def _png_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) < 64 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    if data[12:16] != b"IHDR":
        return False
    return (int.from_bytes(data[16:20], "big") >= 400
            and int.from_bytes(data[20:24], "big") >= 300)


# ── Channel routing ─────────────────────────────────────────────────────────
def _rating(bottom_width: float, slope: float, manning: float
            ) -> dict[str, np.ndarray]:
    y = _Y_TABLE
    area = (bottom_width + _SIDE_SLOPE * y) * y
    perim = bottom_width + 2.0 * y * math.sqrt(1.0 + _SIDE_SLOPE ** 2)
    disch = (1.0 / manning) * area * (area / perim) ** (2.0 / 3.0) * math.sqrt(slope)
    top_w = bottom_width + 2.0 * _SIDE_SLOPE * y
    return {"y": y, "Q": disch, "c": np.gradient(disch, y) / top_w,
            "D": disch / (2.0 * top_w * slope)}


def _profiles(x: np.ndarray, centers: np.ndarray, length_scale: float,
              dx: float) -> np.ndarray:
    """Along-reach profile per tributary: cell-average of a Gaussian, exact total."""

    edges = np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])
    z = (edges[None, :] - centers[:, None]) / (length_scale * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z))
    prof = np.diff(cdf, axis=1) / np.maximum(np.diff(edges)[None, :], 1e-9)
    return prof / (prof.sum(axis=1, keepdims=True) * dx)


def _route(table: dict[str, np.ndarray], q_up: np.ndarray, q_lat: np.ndarray,
           q_base: float, dx: float, dt: float, nx: int) -> np.ndarray:
    """Backward Euler, upwind advection, centred diffusion.

    The diffusion coefficient used by the scheme has the numerical diffusion of
    the upwind term (``c dx / 2``) and of backward Euler (``c^2 dt / 2``)
    subtracted, so the effective diffusion of the scheme equals the physical one.
    """

    n_steps = q_up.size - 1
    q = np.full(nx, float(q_base))
    rec = np.empty((n_steps + 1, nx))
    rec[0] = q
    ab = np.zeros((3, nx))
    rhs = np.zeros(nx)
    for n in range(n_steps):
        cel = np.interp(q, table["Q"], table["c"])
        dif = np.interp(q, table["Q"], table["D"])
        dif = np.maximum(dif - 0.5 * cel * dx - 0.5 * cel ** 2 * dt,
                         _DIFF_FLOOR_FRAC * dif)
        lo = -cel / dx - dif / dx ** 2
        mid = 1.0 / dt + cel / dx + 2.0 * dif / dx ** 2
        up = -dif / dx ** 2
        rhs[:] = q / dt + cel * q_lat[n + 1]
        mid[0], rhs[0] = 1.0, q_up[n + 1]
        mid[-1] = 1.0 / dt + cel[-1] / dx
        lo[-1] = -cel[-1] / dx
        ab[0, 1:] = np.concatenate([[0.0], up[1:-1]])
        ab[1, :] = mid
        ab[2, :-1] = lo[1:]
        q = np.maximum(solve_banded((1, 1), ab, rhs, check_finite=False), 1e-3)
        rec[n + 1] = q
    return rec


def _sample_x(x: np.ndarray, rec: np.ndarray, targets: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(x, targets) - 1, 0, x.size - 2)
    wgt = (targets - x[i]) / (x[i + 1] - x[i])
    return rec[:, i] * (1.0 - wgt)[None, :] + rec[:, i + 1] * wgt[None, :]


def _field_residual(public: dict[str, Any], inflow: np.ndarray | None) -> float:
    """Route the submitted inflow and return the residual on the samples that
    carry a reading, in multiples of the published stage noise."""

    if inflow is None:
        return math.inf
    if np.any(inflow < -1e-9):
        return math.inf

    reach = public["reach"]
    nx = int(reach["n_cells"])
    length = float(reach["length_m"])
    x = np.linspace(0.0, length, nx)
    dx = float(x[1] - x[0])
    times = np.asarray(public["record"]["time_s"], float)
    dt = float(public["record"]["sample_interval_s"])
    q_up = np.asarray(public["upstream_release_m3_s"], float)
    table = _rating(float(reach["bottom_width_m"]), float(reach["bed_slope"]),
                    float(reach["manning_n"]))

    trib = public["tributaries"]
    centers = np.asarray([t["junction_x_m"] for t in trib], float)
    prof = _profiles(x, centers, float(trib[0]["inflow_length_m"]), dx)

    src = np.asarray(public["inflow_sample_times_s"], float)
    dense = np.stack([np.interp(times, src, row) for row in inflow])
    q_lat = dense.T @ prof

    rec = _route(table, q_up, q_lat, float(q_up[0]), dx, dt, nx)
    gauge_x = np.asarray([g["x_m"] for g in public["gauges"]], float)
    q_gauge = _sample_x(x, rec, gauge_x)
    slope = float(reach["bed_slope"])
    stage = (np.interp(q_gauge, table["Q"], table["y"])
             + (float(reach["bed_datum_m"]) - slope * gauge_x)[None, :])

    obs = np.asarray([[math.nan if v is None else float(v) for v in g["stage_m"]]
                      for g in public["gauges"]], float).T
    valid = np.asarray([[bool(v) for v in g["valid"]]
                        for g in public["gauges"]], bool).T
    sigma = max(float(public["record"]["stage_sigma_m"]), 1e-9)
    resid = (stage[valid] - obs[valid]) / sigma
    if not np.all(np.isfinite(resid)):
        return math.inf
    return float(np.sqrt(np.mean(resid ** 2)))


def _grid_peak_time(grid: np.ndarray, seg: np.ndarray, prop: np.ndarray,
                    dt: float) -> float:
    """Peak time of the submitted propagation curve: the vertex of a three-point
    parabola through points lying in the same no-reading interval."""

    i = int(np.argmax(prop))
    if 0 < i < prop.size - 1 and seg[i - 1] == seg[i] == seg[i + 1]:
        coef = np.polyfit(grid[i - 1:i + 2], prop[i - 1:i + 2], 2)
        if coef[0] < 0:
            return float(-coef[1] / (2.0 * coef[0]) * dt)
    return float(grid[i] * dt)


# ── Scorer ──────────────────────────────────────────────────────────────────
@register_scorer("flood_peak_reconstruction_score")
class FloodPeakReconstructionScorer(Scorer):
    """Weighted six-component scorer; every component is recomputed here."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", MAX_TOTAL))
        components = {name: 0.0 for name in WEIGHTS}
        detail: dict[str, Any] = {}

        hydrograph_name = str(config.get("hydrograph_file", _HYDROGRAPH_FILE))
        propagation_name = str(config.get("propagation_file", _PROPAGATION_FILE))
        inflow_name = str(config.get("inflow_file", _INFLOW_FILE))
        summary_name = str(config.get("summary_file", _SUMMARY_FILE))
        figure_name = str(config.get("figure_file", _FIGURE_FILE))
        public_name = str(config.get("public_file", "public_ref.json"))

        public = _read_json(ref_dir / public_name)
        truth_summary = _read_json(ref_dir / _SUMMARY_FILE)
        if public is None or truth_summary is None:
            return ScoreDetail(
                scorer_name=self.name, score=0.0, max_score=weight, passed=False,
                details={"reference_incomplete": True},
                message="reference files missing or unreadable")

        grid = public["scored_grid"]
        names = [str(n) for n in grid["gauge_names"]]
        n_per_gauge = len(grid["gauge_sample_index"][0])
        prop_grid = np.asarray(grid["propagation_sample_index"], float)
        prop_seg = np.asarray(grid["propagation_segment_index"], int)
        n_trib = len(public["tributaries"])
        n_inflow = len(public["inflow_sample_times_s"])
        dt = float(public["record"]["sample_interval_s"])

        hyd_true = _read_hydrographs(ref_dir / _HYDROGRAPH_FILE, names, n_per_gauge)
        prop_true = _read_series(ref_dir / _PROPAGATION_FILE, "discharge_m3_s", 1,
                                 prop_grid.size, key_column="sample_index")
        if hyd_true is None or prop_true is None:
            return ScoreDetail(
                scorer_name=self.name, score=0.0, max_score=weight, passed=False,
                details={"reference_incomplete": True},
                message="reference tables missing or unreadable")

        hyd = _read_hydrographs(pred_dir / hydrograph_name, names, n_per_gauge)
        prop = _read_series(pred_dir / propagation_name, "discharge_m3_s", 1,
                            prop_grid.size, key_column="sample_index")
        inflow = _read_inflow(pred_dir / inflow_name, n_trib, n_inflow)
        summary = _read_json(pred_dir / summary_name)
        detail["parsed"] = {
            "gauge_hydrographs_csv": hyd is not None,
            "propagation_section_csv": prop is not None,
            "tributary_inflow_csv": inflow is not None,
            "flood_summary_json": summary is not None,
        }

        if hyd is not None:
            err = _relative_rms(hyd, hyd_true)
            detail["hydrograph_rel_rms"] = err
            components["gauge_hydrographs"] = WEIGHTS["gauge_hydrographs"] * _linear_desc(
                err, *BANDS["gauge_hydrographs"])

        if prop is not None:
            err = _relative_rms(prop, prop_true)
            detail["propagation_rel_rms"] = err
            components["propagation_section"] = WEIGHTS["propagation_section"] * _linear_desc(
                err, *BANDS["propagation_section"])

        residual = _field_residual(public, inflow)
        detail["stage_residual_in_noise_units"] = residual
        components["tributary_inflow"] = WEIGHTS["tributary_inflow"] * float(
            residual <= CHECKS["tributary_inflow"])

        inflow_true = _read_inflow(ref_dir / _INFLOW_FILE, n_trib, n_inflow)
        vol_err = math.inf
        if inflow is not None and inflow_true is not None:
            times = np.asarray(public["inflow_sample_times_s"], float)
            v_ref = float(np.trapezoid(inflow_true, times, axis=1).sum())
            v_sub = float(np.trapezoid(inflow, times, axis=1).sum())
            if abs(v_ref) > 0.0:
                vol_err = abs(v_sub - v_ref) / abs(v_ref)
        detail["event_volume_rel_rms"] = vol_err
        components["event_volume"] = WEIGHTS["event_volume"] * float(
            vol_err <= CHECKS["event_volume"])

        if summary is not None:
            reference_peak = _scalar(truth_summary.get("peak_time_s"))
            reported_time = _scalar(summary.get("peak_time_s"))
            if reference_peak is not None and reported_time is not None and reported_time > 0.0:
                err = abs(reported_time - reference_peak) / max(abs(reference_peak), 1e-9)
                detail["peak_timing_relative_error"] = err
                components["peak_timing"] = WEIGHTS["peak_timing"] * _linear_desc(
                    err, *BANDS["peak_timing"])

            if prop is not None:
                reported_q = _scalar(summary.get("peak_discharge_m3_s"))
                q_max = float(np.max(prop))
                t_at_max = _grid_peak_time(prop_grid, prop_seg, prop, dt)
                gaps = [
                    math.inf if reported_q is None
                    else abs(reported_q - q_max) / max(q_max, 1e-12),
                    math.inf if reported_time is None
                    else abs(reported_time - t_at_max) / max(t_at_max, 1e-12),
                ]
                detail["report_gaps"] = gaps
                components["report_consistency"] = WEIGHTS["report_consistency"] * float(
                    max(gaps) <= CHECKS["report_consistency"])

        if _png_ok(pred_dir / figure_name):
            components["figure"] = WEIGHTS["figure"]

        total = float(sum(components.values()))
        detail["components"] = components
        detail["component_max"] = dict(WEIGHTS)
        detail["total_out_of_100"] = total
        scaled = total * weight / MAX_TOTAL
        return ScoreDetail(
            scorer_name=self.name,
            score=scaled,
            max_score=weight,
            passed=abs(scaled - weight) < 1.0e-9,
            details=detail,
            message="" if abs(scaled - weight) < 1.0e-9 else _message(components),
        )


def _message(components: dict[str, float]) -> str:
    short = [f"{name} {value:.2f}/{WEIGHTS[name]:.0f}"
             for name, value in components.items() if value < WEIGHTS[name]]
    return "; ".join(short)
