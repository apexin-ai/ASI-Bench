"""Predeclared signed-network and prospective-transfer scorer."""

from pathlib import Path

import numpy as np
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


N_GENES = 25
N_INTERVENTIONS = 24
N_TIMES = 12
EDGE_EPS = 1e-6
OUTPUT_SHAPES = {
    "weights": (N_GENES, N_GENES),
    "future": (N_INTERVENTIONS, N_TIMES, N_GENES),
    "hubs": (2,),
}


def _failure(reason: str, *, reference: bool = False) -> dict:
    return {
        "score": None if reference else 0.0,
        "scorer_internal_error": bool(reference),
        "reason": reason,
    }


def _load_reference(ref_dir: Path) -> tuple[dict | None, str | None]:
    path = Path(ref_dir) / "ground_truth.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = ("weights", "future", "wildtype", "hubs")
            missing = [key for key in required if key not in archive]
            if missing:
                return None, "Reference error: missing keys " + ", ".join(missing)
            truth = {key: np.asarray(archive[key]) for key in required}
    except Exception as exc:
        return None, "Reference error: " + type(exc).__name__

    expected = {"weights": OUTPUT_SHAPES["weights"],
                "future": OUTPUT_SHAPES["future"], "hubs": OUTPUT_SHAPES["hubs"]}
    for key, shape in expected.items():
        if truth[key].shape != shape:
            return None, f"Reference error: {key} shape {truth[key].shape}, expected {shape}"
    if truth["weights"].dtype.kind not in "fiu" or truth["future"].dtype.kind not in "fiu":
        return None, "Reference error: weights and future must be numeric"
    if truth["hubs"].dtype.kind not in "iu":
        return None, "Reference error: hubs must contain integer gene IDs"
    if truth["wildtype"].dtype.kind not in "fiu" or truth["wildtype"].ndim < 1 or truth["wildtype"].shape[-1] != N_GENES:
        return None, "Reference error: wildtype must be numeric with genes on its final axis"
    if any(not np.isfinite(truth[key]).all() for key in required):
        return None, "Reference error: nonfinite scientific array"
    if np.any(np.abs(np.diag(truth["weights"])) > EDGE_EPS):
        return None, "Reference error: true weight diagonal must be zero"
    if np.any((truth["hubs"] < 0) | (truth["hubs"] >= N_GENES)):
        return None, "Reference error: hub ID outside 0..24"
    return truth, None


def _load_prediction(pred_dir: Path) -> tuple[dict | None, str | None]:
    path = Path(pred_dir) / "predictions.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in OUTPUT_SHAPES if key not in archive]
            if missing:
                return None, "Prediction error: missing keys " + ", ".join(missing)
            pred = {key: np.asarray(archive[key]) for key in OUTPUT_SHAPES}
    except FileNotFoundError:
        return None, "Prediction error: predictions.npz is missing"
    except Exception as exc:
        return None, "Prediction error: " + type(exc).__name__

    for key, shape in OUTPUT_SHAPES.items():
        if pred[key].shape != shape:
            return None, f"Prediction error: {key} shape {pred[key].shape}, expected {shape}"
    if pred["weights"].dtype != np.dtype(np.float64):
        return None, "Prediction error: weights must have dtype float64"
    if pred["future"].dtype != np.dtype(np.float64):
        return None, "Prediction error: future must have dtype float64"
    if pred["hubs"].dtype != np.dtype(np.int64):
        return None, "Prediction error: hubs must have dtype int64"
    if not np.isfinite(pred["weights"]).all() or not np.isfinite(pred["future"]).all():
        return None, "Prediction error: weights or future contains nonfinite values"
    if np.any(np.abs(np.diag(pred["weights"])) > EDGE_EPS):
        return None, "Prediction error: weights diagonal must be zero (tolerance 1e-6)"
    if np.any((pred["hubs"] < 0) | (pred["hubs"] >= N_GENES)):
        return None, "Prediction error: hub ID outside 0..24"
    if len(np.unique(pred["hubs"])) != 2:
        return None, "Prediction error: hubs must contain two distinct gene IDs"
    return pred, None


def _average_precision(confidence: np.ndarray, labels: np.ndarray) -> float:
    """Average precision with equal-score groups kept tied."""
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="mergesort")
    sorted_confidence = confidence[order]
    sorted_labels = labels[order].astype(np.int64)
    group_ends = np.r_[np.flatnonzero(sorted_confidence[1:] != sorted_confidence[:-1]) + 1,
                       len(sorted_confidence)]
    cumulative_tp = np.cumsum(sorted_labels)[group_ends - 1]
    previous_tp = np.r_[0, cumulative_tp[:-1]]
    group_tp = cumulative_tp - previous_tp
    precision = cumulative_tp / group_ends
    return float(np.sum((group_tp / positives) * precision))


def evaluate(pred_dir, ref_dir) -> dict:
    """Score a prediction directory against reference/ground_truth.npz."""
    truth, error = _load_reference(Path(ref_dir))
    if error:
        return _failure(error, reference=True)
    pred, error = _load_prediction(Path(pred_dir))
    if error:
        return _failure(error)

    truth_w = np.asarray(truth["weights"], dtype=np.float64)
    pred_w = np.asarray(pred["weights"], dtype=np.float64).copy()
    pred_w[np.abs(pred_w) <= EDGE_EPS] = 0.0
    truth_edge = np.abs(truth_w) > EDGE_EPS
    pred_edge = pred_w != 0.0
    correct_sign = pred_edge & truth_edge & (np.sign(pred_w) == np.sign(truth_w))
    wrong_sign = pred_edge & truth_edge & ~correct_sign
    tp = int(correct_sign.sum())
    fp = int((pred_edge & ~truth_edge).sum() + wrong_sign.sum())
    fn = int((truth_edge & ~correct_sign).sum())
    f1_denominator = 2 * tp + fp + fn
    signed_f1 = (2.0 * tp / f1_denominator) if f1_denominator else 1.0
    edge_points = 60.0 * signed_f1

    wildtype = np.asarray(truth["wildtype"], dtype=np.float64)
    wildtype_mean = wildtype.reshape(-1, N_GENES).mean(axis=0)
    scale = 0.2 + wildtype_mean
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        return _failure("Reference error: 0.2 + wildtype gene mean must be positive", reference=True)
    normalized_error = ((np.asarray(pred["future"], dtype=np.float64) -
                         np.asarray(truth["future"], dtype=np.float64)) /
                        scale[None, None, :])
    intervention_rmse = np.sqrt(np.mean(normalized_error ** 2, axis=(1, 2)))
    intervention_success = intervention_rmse <= 0.10
    forecast_points = 20.0 * float(intervention_success.mean())

    true_hubs = set(int(x) for x in truth["hubs"].tolist())
    pred_hubs = set(int(x) for x in pred["hubs"].tolist())
    hub_overlap = int(len(true_hubs & pred_hubs))
    hub_score = hub_overlap / 2.0
    hub_points = 10.0 * hub_score

    off_diagonal = ~np.eye(N_GENES, dtype=bool)
    ranking_labels = truth_edge[off_diagonal]
    ranking_confidence = np.abs(pred_w[off_diagonal])
    ranking_pairs = int(ranking_labels.size)
    prevalence = float(ranking_labels.mean()) if ranking_pairs else 0.0
    average_precision = _average_precision(ranking_confidence, ranking_labels)
    if 0.0 < prevalence < 1.0:
        normalized_ap = float(np.clip((average_precision - prevalence) /
                                      (1.0 - prevalence), 0.0, 1.0))
    else:
        normalized_ap = 0.0
    ranking_points = 10.0 * normalized_ap

    points = {
        "signed_direct_edge_f1": edge_points,
        "prospective_prediction": forecast_points,
        "hub_score": hub_points,
        "edge_ranking": ranking_points,
    }
    raw_score = float(sum(points.values()))
    # Both research claims must be reliable; exactly the same rule applies at every level.
    strict_error = np.max(np.abs(normalized_error), axis=(1, 2))
    strict_fraction = float(np.mean(strict_error <= 0.03))
    reliable_network = signed_f1 >= 0.90
    reliable_transfer = strict_fraction >= 0.90
    if reliable_network and reliable_transfer:
        applied_cap = None
        cap_reason = "both_reliability_criteria_met"
    elif reliable_network or reliable_transfer:
        applied_cap = 35.0
        cap_reason = "only_network_reliable" if reliable_network else "only_transfer_reliable"
    else:
        applied_cap = 30.0
        cap_reason = "neither_reliability_criterion_met"
    total = raw_score if applied_cap is None else min(raw_score, applied_cap)
    return {
        "score": round(total, 10),
        "raw_score": round(raw_score, 10),
        "reliability": {
            "signed_f1": float(signed_f1),
            "signed_f1_threshold": 0.90,
            "strict_forecast_fraction": strict_fraction,
            "strict_forecast_fraction_threshold": 0.90,
            "strict_component_normalized_error_threshold": 0.03,
            "strict_errors_by_protocol": strict_error.tolist(),
            "reliable_network": bool(reliable_network),
            "reliable_transfer": bool(reliable_transfer),
            "applied_cap": applied_cap,
            "cap_reason": cap_reason,
        },
        "scorer_internal_error": False,
        "points": points,
        "familyN": {
            "signed_direct_edge_pairs": N_GENES * (N_GENES - 1),
            "prospective_interventions": N_INTERVENTIONS,
            "hub_ids": 2,
            "edge_ranking_pairs": ranking_pairs,
        },
        "details": {
            "signed_direct_edge_f1": {
                "f1": float(signed_f1), "tp": tp, "fp": fp, "fn": fn,
                "true_edges": int(truth_edge.sum()),
                "predicted_edges": int(pred_edge.sum()),
                "wrong_sign_predictions": int(wrong_sign.sum()),
                "n": N_GENES * (N_GENES - 1),
            },
            "prospective_prediction": {
                "threshold_normalized_rmse": 0.10,
                "successful_interventions": int(intervention_success.sum()),
                "intervention_n": N_INTERVENTIONS,
                "per_intervention_normalized_rmse": intervention_rmse.tolist(),
                "familyN": {"prospective_interventions": N_INTERVENTIONS},
            },
            "hub_score": {
                "overlap": hub_overlap, "n": 2,
                "predicted_hubs": sorted(pred_hubs),
                "reference_hubs": sorted(true_hubs),
            },
            "edge_ranking": {
                "average_precision": float(average_precision),
                "prevalence": prevalence,
                "normalized_score": normalized_ap,
                "ranked_pairs": ranking_pairs,
            },
        },
    }


@register_scorer("mixed_perturbation_gene_transfer_v1")
class MixedPerturbationGeneTransferScorer(Scorer):
    def score(self, pred_dir, ref_dir, config):
        result = evaluate(pred_dir, ref_dir)
        value = result["score"]
        return ScoreDetail(
            scorer_name=self.name,
            score=0.0 if value is None else value,
            max_score=100.0,
            passed=value is not None and value >= 90.0,
            details=result,
            message=result.get("reason", "Signed network and prospective intervention transfer"),
        )
