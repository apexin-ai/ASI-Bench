"""Custom scorer for computer_science.heterogeneous_click_log_counterfactuals.

Five components (fractions of the configured weight):

  contract        0.05  every required artifact parses and covers every requested row
  counterfactual  0.15  MAE of predicted CTR on counterfactual (query, doc, position, device) cells
  conditional     0.25  KL excess of next-click predictions on held-out sessions, normalised by the
                        KL of the best possible *history-blind* predictor (true device-marginal CTRs)
  policy          0.25  expected clicks (relative error), zero-click and multi-click rates (absolute
                        error normalised by the error of the *independence oracle*, i.e. true marginal
                        CTRs combined under the assumption that clicks in a session are independent)
  ranking         0.30  realised success rate of the submitted device-specific rankings on new queries,
                        normalised between a segment-blind plug-in ranking (0) and the plug-in optimum (1)

Normalisers are computed from the hidden simulator by generate_gt.py and stored in reference/.
A structure-blind click model (any single-population PBM / trust-bias / cascade-style model,
however well fitted) can at best match the normalising baselines and therefore earns 0 on the
session-level components (a submission must beat those baselines by 10 % before any credit).

Marginal components (counterfactual CTR, expected clicks) use tolerant full-credit thresholds
that any sound click model reaches, so solutions that model positions well but miss the latent
user structure all receive the same marginal credit (25 points in total with the contract).
Session-level full-credit thresholds are set so that the reference two-segment model saturates
on every calibration seed.
"""

from __future__ import annotations

import csv
import json
import math
import stat
from pathlib import Path
from typing import Any

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

SCORER_NAME = "heterogeneous_click_log_score"
SCORER_REVISION = "v3-2026-10-13"
MAX_TABLE_BYTES = 64 * 1024 * 1024
EPS = 1e-6

COMPONENT_FRACTIONS = {
    "contract": 0.05,
    "counterfactual": 0.15,
    "conditional": 0.25,
    "policy": 0.25,
    "ranking": 0.30,
}
POLICY_SPLIT = {"expected_clicks": 0.20, "zero_click_rate": 0.40, "multi_click_rate": 0.40}

DEFAULTS = {
    "ctr_mae_full": 0.022,
    "ctr_mae_zero": 0.05,
    "conditional_ratio_full": 0.40,
    "conditional_ratio_zero": 0.90,
    "ec_rel_full": 0.08,
    "ec_rel_zero": 0.20,
    "session_ratio_full": 0.40,
    "session_ratio_zero": 0.90,
    "ranking_gain_full": 0.70,
    "ranking_gain_zero": 0.00,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lin_desc(err: float, full: float, zero: float) -> float:
    if not math.isfinite(err):
        return 0.0
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return float((zero - err) / max(zero - full, 1e-12))


def _lin_asc(val: float, zero: float, full: float) -> float:
    if not math.isfinite(val):
        return 0.0
    if val >= full:
        return 1.0
    if val <= zero:
        return 0.0
    return float((val - zero) / max(full - zero, 1e-12))


def _check_regular(path: Path) -> None:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{path.name} must be a regular file")
    if st.st_size > MAX_TABLE_BYTES:
        raise ValueError(f"{path.name} exceeds the scorer size limit")


def _read_rows(path: Path, required: list[str]) -> list[dict[str, str]]:
    _check_regular(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(f"{path.name} missing columns {missing}")
        rows = []
        for raw in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
    return rows


class ReferenceInputError(RuntimeError):
    def __init__(self, cause):
        super().__init__(str(cause))
        self.kind = 'missing_evaluator_input' if isinstance(cause, OSError) else 'evaluator_runtime_error'


def _read_reference_rows(path, required):
    try:
        rows = _read_rows(path, required)
        if not rows:
            raise ValueError(f'Empty reference table: {path.name}')
        numeric = {'position','theta','eps_plus','eps_minus','share_a','query_id','doc_id','rho_a','rho_b','request_id','ctr','session_id','p_click','expected_clicks','zero_click_rate','multi_click_rate','floor_success','top_success'}
        for row in rows:
            for key in numeric.intersection(required):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f'Invalid reference number: {path.name}:{key}')
        return rows
    except (OSError, ValueError, UnicodeError, KeyError) as exc:
        raise ReferenceInputError(exc) from exc


def _num(v: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _int(v: str) -> int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or abs(f - round(f)) > 1e-9:
        return None
    return int(round(f))


def _bern_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# ---------------------------------------------------------------------------
# hidden-model utilities (reference side)
# ---------------------------------------------------------------------------

class _World:
    def __init__(self, ref_dir: Path):
        rows = _read_reference_rows(ref_dir / "click_model_ref.csv", ["segment", "position", "theta", "eps_plus", "eps_minus"])
        K = max(int(r["position"]) for r in rows)
        self.K = K
        self.theta = np.zeros((2, K))
        self.ep = np.zeros((2, K))
        self.em = np.zeros((2, K))
        for r in rows:
            m = 0 if r["segment"] == "A" else 1
            k = int(r["position"]) - 1
            self.theta[m, k] = float(r["theta"])
            self.ep[m, k] = float(r["eps_plus"])
            self.em[m, k] = float(r["eps_minus"])
        mix = _read_reference_rows(ref_dir / "segment_mix_ref.csv", ["device", "share_a"])
        self.pi = {r["device"]: np.array([float(r["share_a"]), 1.0 - float(r["share_a"])]) for r in mix}
        rel = _read_reference_rows(ref_dir / "new_query_relevance_ref.csv", ["query_id", "doc_id", "rho_a", "rho_b"])
        self.rho: dict[int, dict[int, np.ndarray]] = {}
        for r in rel:
            self.rho.setdefault(int(r["query_id"]), {})[int(r["doc_id"])] = np.array([float(r["rho_a"]), float(r["rho_b"])])

    def success(self, qid: int, device: str, ranking: list[int]) -> float:
        pi = self.pi[device]
        val = 0.0
        for m in range(2):
            p_none = 1.0
            for k, d in enumerate(ranking):
                rho = self.rho[qid][d][m]
                c = self.theta[m, k] * (self.ep[m, k] * rho + self.em[m, k] * (1 - rho))
                p_none *= 1.0 - c
            val += pi[m] * (1.0 - p_none)
        return float(val)


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------

def _score_counterfactual(pred_dir, ref_dir, cfg, details):
    ref = _read_reference_rows(ref_dir / "counterfactual_ctr_ref.csv", ["request_id", "ctr"])
    truth = {int(r["request_id"]): float(r["ctr"]) for r in ref}
    pred_rows = _read_rows(pred_dir / "results/counterfactual_ctr.csv", ["request_id", "ctr"])
    pred: dict[int, float] = {}
    for r in pred_rows:
        rid = _int(r["request_id"])
        v = _num(r["ctr"])
        if rid is not None and math.isfinite(v) and 0.0 <= v <= 1.0:
            pred[rid] = v
    errs = []
    for rid, t in truth.items():
        errs.append(abs(pred[rid] - t) if rid in pred else max(t, 1 - t))
    mae = float(np.mean(errs))
    coverage = float(np.mean([rid in pred for rid in truth]))
    details["counterfactual"] = {"mae": mae, "coverage": coverage}
    return _lin_desc(mae, cfg["ctr_mae_full"], cfg["ctr_mae_zero"]), coverage


def _score_conditional(pred_dir, ref_dir, cfg, details):
    ref = _read_reference_rows(ref_dir / "conditional_clicks_ref.csv", ["session_id", "position", "p_click"])
    base = _read_reference_rows(ref_dir / "conditional_clicks_marginal_ref.csv", ["session_id", "position", "p_click"])
    keys = [(int(r["session_id"]), int(r["position"])) for r in ref]
    p_true = np.array([float(r["p_click"]) for r in ref])
    base_map = {(int(r["session_id"]), int(r["position"])): float(r["p_click"]) for r in base}
    p_base = np.array([base_map[k] for k in keys])
    pred_rows = _read_rows(pred_dir / "results/conditional_clicks.csv", ["session_id", "position", "p_click"])
    pred: dict[tuple[int, int], float] = {}
    for r in pred_rows:
        s, k, v = _int(r["session_id"]), _int(r["position"]), _num(r["p_click"])
        if s is not None and k is not None and math.isfinite(v) and 0.0 <= v <= 1.0:
            pred[(s, k)] = v
    q = np.array([pred.get(k, 0.5) for k in keys])
    coverage = float(np.mean([k in pred for k in keys]))
    kl_agent = float(np.mean(_bern_kl(p_true, q)))
    kl_base = float(np.mean(_bern_kl(p_true, p_base)))
    ratio = kl_agent / max(kl_base, 1e-12)
    details["conditional"] = {"kl_agent": kl_agent, "kl_history_blind_oracle": kl_base,
                              "ratio": ratio, "coverage": coverage}
    return _lin_desc(ratio, cfg["conditional_ratio_full"], cfg["conditional_ratio_zero"]), coverage


def _score_policy(pred_dir, ref_dir, cfg, details):
    cols = ["policy_id", "device", "expected_clicks", "zero_click_rate", "multi_click_rate"]
    ref = _read_reference_rows(ref_dir / "policy_values_ref.csv", cols)
    ind = _read_reference_rows(ref_dir / "policy_values_independent_ref.csv", cols)
    ind_map = {(r["policy_id"], r["device"]): r for r in ind}
    pred_rows = _read_rows(pred_dir / "results/policy_values.csv", cols)
    pred = {}
    for r in pred_rows:
        vals = [_num(r[c]) for c in cols[2:]]
        if all(math.isfinite(v) for v in vals) and vals[0] >= 0 and 0 <= vals[1] <= 1 and 0 <= vals[2] <= 1:
            pred[(r["policy_id"], r["device"])] = vals
    ec_err, zc_err, mc_err, zc_ind, mc_ind = [], [], [], [], []
    for r in ref:
        key = (r["policy_id"], r["device"])
        t = [float(r[c]) for c in cols[2:]]
        i = [float(ind_map[key][c]) for c in cols[2:]]
        p = pred.get(key)
        if p is None:
            ec_err.append(1.0)
            zc_err.append(1.0)
            mc_err.append(1.0)
        else:
            ec_err.append(abs(p[0] - t[0]) / max(t[0], 1e-9))
            zc_err.append(abs(p[1] - t[1]))
            mc_err.append(abs(p[2] - t[2]))
        zc_ind.append(abs(i[1] - t[1]))
        mc_ind.append(abs(i[2] - t[2]))
    coverage = float(np.mean([(r["policy_id"], r["device"]) in pred for r in ref]))
    ec = float(np.mean(ec_err))
    zc_ratio = float(np.mean(zc_err)) / max(float(np.mean(zc_ind)), 1e-12)
    mc_ratio = float(np.mean(mc_err)) / max(float(np.mean(mc_ind)), 1e-12)
    s_ec = _lin_desc(ec, cfg["ec_rel_full"], cfg["ec_rel_zero"])
    s_zc = _lin_desc(zc_ratio, cfg["session_ratio_full"], cfg["session_ratio_zero"])
    s_mc = _lin_desc(mc_ratio, cfg["session_ratio_full"], cfg["session_ratio_zero"])
    details["policy"] = {"expected_clicks_rel_err": ec, "zero_click_err_ratio_vs_independence": zc_ratio,
                         "multi_click_err_ratio_vs_independence": mc_ratio,
                         "independence_oracle_zero_click_mae": float(np.mean(zc_ind)),
                         "independence_oracle_multi_click_mae": float(np.mean(mc_ind)),
                         "sub_scores": {"expected_clicks": s_ec, "zero_click_rate": s_zc, "multi_click_rate": s_mc},
                         "coverage": coverage}
    score = (POLICY_SPLIT["expected_clicks"] * s_ec + POLICY_SPLIT["zero_click_rate"] * s_zc
             + POLICY_SPLIT["multi_click_rate"] * s_mc)
    return score, coverage


def _score_ranking(pred_dir, ref_dir, cfg, details):
    world = _World(ref_dir)
    bounds = _read_reference_rows(ref_dir / "new_query_bounds_ref.csv", ["query_id", "device", "floor_success", "top_success"])
    K = world.K
    pred_rows = _read_rows(pred_dir / "results/new_query_rankings.csv", ["query_id", "device", "rank", "doc_id"])
    lists: dict[tuple[int, str], dict[int, int]] = {}
    bad_keys: set[tuple[int, str]] = set()
    for r in pred_rows:
        q, rk, d = _int(r["query_id"]), _int(r["rank"]), _int(r["doc_id"])
        key = (q, r["device"])
        if q is None or rk is None or d is None:
            continue
        slot = lists.setdefault(key, {})
        if rk in slot:
            bad_keys.add(key)
        slot[rk] = d
    num = den = 0.0
    n_valid = 0
    for b in bounds:
        qid, dev = int(b["query_id"]), b["device"]
        floor, top = float(b["floor_success"]), float(b["top_success"])
        key = (qid, dev)
        ranking = None
        if key in lists and key not in bad_keys:
            slot = lists[key]
            cand = [slot.get(k) for k in range(1, K + 1)]
            valid_docs = world.rho.get(qid, {})
            if (len(slot) == K and all(c is not None and c in valid_docs for c in cand)
                    and len(set(cand)) == K):
                ranking = cand
        if ranking is None:
            value = floor - abs(top - floor)
        else:
            value = world.success(qid, dev, ranking)
            n_valid += 1
        num += value - floor
        den += top - floor
    gain = num / den if den > 1e-12 else 0.0
    coverage = n_valid / max(len(bounds), 1)
    details["ranking"] = {"normalised_gain": gain, "valid_lists": n_valid, "required_lists": len(bounds),
                          "total_gap": den}
    return _lin_asc(gain, cfg["ranking_gain_zero"], cfg["ranking_gain_full"]), coverage


def _score_contract(pred_dir, coverages, details):
    ok = []
    summary_path = pred_dir / "results/model_summary.json"
    try:
        _check_regular(summary_path)
        js = json.loads(summary_path.read_text(encoding="utf-8"))
        ok.append(float(all(k in js for k in ("model_description", "n_parameters", "heldout_validation"))))
    except Exception:  # noqa: BLE001
        ok.append(0.0)
    fig = pred_dir / "results/diagnostics.png"
    try:
        _check_regular(fig)
        ok.append(float(fig.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"))
    except Exception:  # noqa: BLE001
        ok.append(0.0)
    ok.extend(coverages)
    details["contract"] = {"checks": ok}
    return float(np.mean(ok))


@register_scorer(SCORER_NAME)
class HeterogeneousClickLogScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict[str, Any]) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        cfg = {k: float(config.get(k, v)) for k, v in DEFAULTS.items()}
        pred_dir, ref_dir = Path(pred_dir), Path(ref_dir)
        details: dict[str, Any] = {"revision": SCORER_REVISION}
        comp: dict[str, float] = {}
        coverages: list[float] = []
        for name, fn in (("counterfactual", _score_counterfactual), ("conditional", _score_conditional),
                         ("policy", _score_policy), ("ranking", _score_ranking)):
            try:
                s, cov = fn(pred_dir, ref_dir, cfg, details)
            except ReferenceInputError as exc:
                return ScoreDetail(SCORER_NAME,0,weight,False,
                    {'scorer_internal_error':True,'failure_kind':exc.kind,'error':str(exc)},'Invalid evaluator input')
            except FileNotFoundError as exc:
                missing = Path(exc.filename) if exc.filename else None
                if missing is not None and ref_dir.resolve() in missing.resolve().parents:
                    raise  # reference-side problem: infrastructure error, not the agent's fault
                s, cov = 0.0, 0.0
                details.setdefault("errors", {})[name] = f"missing file: {exc}"
            except Exception as exc:  # noqa: BLE001
                s, cov = 0.0, 0.0
                details.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"
            comp[name] = s
            coverages.append(cov)
        comp["contract"] = _score_contract(pred_dir, coverages, details)
        total = sum(COMPONENT_FRACTIONS[k] * comp[k] for k in COMPONENT_FRACTIONS)
        details["components"] = comp
        details["component_points"] = {k: round(100 * COMPONENT_FRACTIONS[k] * comp[k], 3) for k in comp}
        score = float(weight * total)
        return ScoreDetail(
            scorer_name=SCORER_NAME,
            score=score,
            max_score=weight,
            passed=total >= 0.5,
            details=details,
            message="; ".join(f"{k}={v:.3f}" for k, v in comp.items()),
        )
