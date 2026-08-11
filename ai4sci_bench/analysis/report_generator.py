"""Human-readable error report generation."""

from __future__ import annotations

from typing import Any

from ai4sci_bench.core.types import AnalysisReport, EvalResult


class ReportGenerator:
    """Generate human-readable error analysis reports."""

    def generate_instance_report(self, eval_result: EvalResult) -> str:
        """Generate a report for a single instance."""
        lines = [
            f"# Instance Report: {eval_result.instance_id}",
            f"Task: {eval_result.task_id}",
            f"Prompt Level: {eval_result.prompt_level.value}",
            f"Score: {eval_result.final_score:.1f}/{eval_result.max_possible_score:.1f}",
            f"Status: {eval_result.status.value}",
            "",
        ]

        if eval_result.gate_results:
            lines.append("## Gates")
            for g in eval_result.gate_results:
                status = "PASS" if g.passed else "FAIL"
                lines.append(f"  [{status}] {g.scorer_name}: {g.score}/{g.max_score}")
            lines.append("")

        if eval_result.score_results:
            lines.append("## Scores")
            for s in eval_result.score_results:
                lines.append(f"  {s.scorer_name}: {s.score:.1f}/{s.max_score:.1f}")
            lines.append("")

        if eval_result.error_analysis:
            a = eval_result.error_analysis
            lines.extend([
                "## Error Analysis",
                f"  Category: {a.error_category}/{a.error_subcategory}",
                f"  Root Cause: {a.root_cause}",
                f"  Confidence: {a.confidence:.0%}",
            ])

        return "\n".join(lines)

    def generate_summary_report(self, results: list[EvalResult]) -> str:
        """Generate a summary report for multiple instances."""
        if not results:
            return "No results to report."

        total = len(results)
        scores = [r.final_score for r in results]
        mean_score = sum(scores) / total

        lines = [
            f"# Evaluation Summary",
            f"Total instances: {total}",
            f"Mean score: {mean_score:.1f}/100",
            f"Min/Max: {min(scores):.1f}/{max(scores):.1f}",
            "",
        ]

        # By task
        tasks: dict[str, list[float]] = {}
        for r in results:
            tasks.setdefault(r.task_id, []).append(r.final_score)

        if len(tasks) > 1:
            lines.append("## By Task")
            for tid, task_scores in sorted(tasks.items()):
                lines.append(f"  {tid}: {sum(task_scores)/len(task_scores):.1f} ({len(task_scores)} instances)")
            lines.append("")

        # Error distribution
        categories: dict[str, int] = {}
        for r in results:
            if r.error_analysis:
                cat = r.error_analysis.error_category
                categories[cat] = categories.get(cat, 0) + 1

        if categories:
            lines.append("## Error Distribution")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                lines.append(f"  {cat}: {count} ({pct:.0f}%)")

        return "\n".join(lines)
