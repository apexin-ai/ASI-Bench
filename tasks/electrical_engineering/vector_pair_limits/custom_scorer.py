"""Scorer for electrical_engineering.vector_pair_limits.

The submission reports, per instance, either an exact largest total rate or
`null` to say the quantity is not determined. Two different capabilities are
being measured and the score is the product of how well each is done:

    acc_value  = correct numbers / instances whose value is determined
    acc_call   = correct nulls   / instances whose value is not determined
    score      = 100 * acc_value * acc_call

The product, rather than a weighted sum, is deliberate. With most of the set
determined, an additive rule rewards a solver that answers every instance and
never abstains — the exact failure this task is built to expose. Under the
product that strategy scores zero, and so does abstaining everywhere.

This file does no science. It parses a submission, compares it against a
private reference, and reports. Nothing here discloses how the reference was
produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

SUBMISSION = "results/solution.json"
REFERENCE = "labels.json"
DEFAULT_REL_TOL = 1.0e-6


def _load_reference(ref_dir: Path) -> dict:
    for candidate in (ref_dir / REFERENCE, ref_dir.parent / REFERENCE):
        if candidate.is_file():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f"reference {REFERENCE} not found near {ref_dir}")


@register_scorer("custom")
class VectorPairLimitsScorer(Scorer):
    """Grades the per-instance verdict and, where a value is called for, the value."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        rel_tol = float(config.get("rel_tol", DEFAULT_REL_TOL))

        def fail(msg: str, details: dict | None = None) -> ScoreDetail:
            return ScoreDetail(
                scorer_name="custom:vector_pair_limits",
                score=0.0,
                max_score=weight,
                passed=False,
                details=details or {},
                message=msg,
            )

        try:
            reference = _load_reference(ref_dir)
        except Exception as exc:  # evaluator-side problem, not the agent's
            return fail(f"reference unavailable: {exc}")

        path = pred_dir / SUBMISSION
        if not path.is_file():
            return fail(f"{SUBMISSION} not found")
        try:
            submission = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            return fail(f"{SUBMISSION} is not valid JSON: {exc}")
        if not isinstance(submission, dict):
            return fail(f"{SUBMISSION} must be a JSON object keyed by instance id")

        ids = sorted(reference)
        missing = [i for i in ids if i not in submission]
        extra = [k for k in submission if k not in reference]

        n_det = n_und = 0
        ok_det = ok_und = 0
        wrong_value = []       # number given, number wrong
        over_certified = []    # number given where the answer is not determined
        under_certified = []   # null given where a value is determined
        malformed = []

        for iid in ids:
            want = reference[iid]["total_rate"]
            determined = want is not None
            n_det += determined
            n_und += not determined

            entry = submission.get(iid)
            if not isinstance(entry, dict) or "total_rate" not in entry:
                malformed.append(iid)
                continue
            got = entry["total_rate"]

            if got is None:
                if determined:
                    under_certified.append(iid)
                else:
                    ok_und += 1
                continue

            if isinstance(got, bool) or not isinstance(got, (int, float)):
                malformed.append(iid)
                continue
            got = float(got)
            if not math.isfinite(got):
                malformed.append(iid)
                continue

            if not determined:
                over_certified.append(iid)
                continue
            if abs(got - want) <= rel_tol * abs(want):
                ok_det += 1
            else:
                wrong_value.append(iid)

        acc_value = ok_det / n_det if n_det else 1.0
        acc_call = ok_und / n_und if n_und else 1.0
        score = weight * acc_value * acc_call

        details = {
            "n_instances": len(ids),
            "n_determined": n_det,
            "n_undetermined": n_und,
            "correct_values": ok_det,
            "correct_nulls": ok_und,
            "acc_value": round(acc_value, 6),
            "acc_call": round(acc_call, 6),
            "rel_tol": rel_tol,
            "counts": {
                "wrong_value": len(wrong_value),
                "over_certified": len(over_certified),
                "under_certified": len(under_certified),
                "malformed": len(malformed),
                "missing": len(missing),
                "unrecognised_keys": len(extra),
            },
            "ids": {
                "wrong_value": wrong_value[:20],
                "over_certified": over_certified[:20],
                "under_certified": under_certified[:20],
                "malformed": malformed[:20],
                "missing": missing[:20],
            },
        }

        message = (
            f"values {ok_det}/{n_det} exact to {rel_tol:g} relative; "
            f"not-determined called correctly {ok_und}/{n_und}; "
            f"over-certified {len(over_certified)}, under-certified "
            f"{len(under_certified)}, wrong value {len(wrong_value)}, "
            f"malformed {len(malformed)}, missing {len(missing)}; "
            f"score {score:.2f}/{weight:.2f} "
            f"(= {weight:.0f} x {acc_value:.4f} x {acc_call:.4f})"
        )

        return ScoreDetail(
            scorer_name="custom:vector_pair_limits",
            score=score,
            max_score=weight,
            passed=score >= weight,
            details=details,
            message=message,
        )
