"""Composite scorer — wraps multiple sub-scorers into a single scorer entry."""

from __future__ import annotations

from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer, get_scorer
from ai4sci_bench.core.types import ScoreDetail


@register_scorer("composite")
class CompositeScorer(Scorer):
    """Bundle multiple sub-scorers into a single named scorer entry.

    Does NOT implement the Gate/Scoring two-layer pattern — that orchestration
    is the responsibility of BenchmarkOrchestrator._evaluate().
    """

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = config.get("weight", 1.0)
        sub_scorers = config.get("sub_scorers", [])

        if not sub_scorers:
            return ScoreDetail(
                scorer_name="composite",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": "No sub_scorers defined"},
            )

        sub_results = []
        total_weight = sum(s.get("weight", 1.0) for s in sub_scorers)
        weighted_score = 0.0

        for sub_cfg in sub_scorers:
            scorer_name = sub_cfg["scorer"]
            sub_weight = sub_cfg.get("weight", 1.0)
            sub_config = sub_cfg.get("config", {})
            sub_config["weight"] = 1.0  # normalize for sub-scorer

            scorer = get_scorer(scorer_name)
            result = scorer.score(pred_dir, ref_dir, sub_config)
            sub_results.append({
                "scorer": scorer_name,
                "score": result.score,
                "max_score": result.max_score,
                "passed": result.passed,
                "details": result.details,
            })
            weighted_score += (result.score / result.max_score) * sub_weight if result.max_score > 0 else 0.0

        final_score = (weighted_score / total_weight) * weight if total_weight > 0 else 0.0
        all_passed = all(r["passed"] for r in sub_results)

        return ScoreDetail(
            scorer_name="composite",
            score=final_score,
            max_score=weight,
            passed=all_passed,
            details={"sub_results": sub_results},
        )
