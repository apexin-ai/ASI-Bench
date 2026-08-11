"""Result data classes for benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai4sci_bench.core.types import EvalResult


def is_unscored_submission(result: EvalResult) -> bool:
    """Return whether a result awaits scoring through the ASI-Bench website."""
    return any(
        detail.scorer_name == "unscored_submission"
        or bool((detail.details or {}).get("unscored_submission"))
        for detail in result.score_results
    )


@dataclass
class TaskSummary:
    """Aggregated results for a single task."""
    task_id: str
    n_instances: int
    mean_score: float
    min_score: float
    max_score: float
    std_score: float
    gates_pass_rate: float
    scores_by_level: dict[str, float] = field(default_factory=dict)


@dataclass
class DomainSummary:
    """Aggregated results for a domain."""
    domain: str
    n_tasks: int
    mean_score: float
    task_summaries: list[TaskSummary] = field(default_factory=list)


@dataclass
class BehaviorSummary:
    """Aggregated agent behavior statistics from trajectory summaries."""
    avg_turns: float = 0.0
    avg_tool_calls: float = 0.0
    tool_call_distribution: dict[str, int] = field(default_factory=dict)
    avg_thinking_chars: float = 0.0
    avg_code_execution_failures: float = 0.0


@dataclass
class RunReport:
    """Complete benchmark run report."""
    agent_name: str
    n_tasks: int
    n_instances: int
    overall_mean_score: float

    # Breakdowns
    by_task: list[TaskSummary] = field(default_factory=list)
    by_domain: list[DomainSummary] = field(default_factory=list)
    by_prompt_level: dict[str, float] = field(default_factory=dict)

    # Error distribution
    error_distribution: dict[str, int] = field(default_factory=dict)

    # Outcome breakdown
    gate_failed_instances: int = 0
    soft_gate_warning_instances: int = 0
    zero_score_after_gates_instances: int = 0
    nonzero_score_instances: int = 0
    unscored_submission_instances: int = 0

    # Cost tracking
    total_execution_time: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    # Trajectory behavior statistics
    behavior_summary: BehaviorSummary | None = None

    # Raw results
    results: list[EvalResult] = field(default_factory=list)

    def _fallback_reason_for_score(self, score_result) -> str:
        """Build a useful explanation from details when message is absent."""
        details = score_result.details or {}
        score = score_result.score
        max_score = score_result.max_score

        if score_result.scorer_name == "numerical:per_frame_relative_l2":
            frame_items = [
                (name, info)
                for name, info in details.items()
                if name.startswith("frame_") and isinstance(info, dict)
            ]
            if frame_items:
                awarded = sum(1 for _, info in frame_items if info.get("pts", 0.0) > 0)
                worst_name, worst_info = max(
                    frame_items,
                    key=lambda item: item[1].get("rel_err", float("-inf")),
                )
                worst_err = worst_info.get("rel_err")
                if isinstance(worst_err, (int, float)):
                    return (
                        f"{awarded}/{len(frame_items)} frames awarded; "
                        f"worst {worst_name} rel_err={worst_err:.6f}; "
                        f"score={score:.2f}/{max_score:.2f}"
                    )
            return f"score={score:.2f}/{max_score:.2f}"

        for metric_key in [
            "mean_relative_l2",
            "relative_l2",
            "rmse",
            "mape",
            "max_relative_error",
            "max_absolute_error",
        ]:
            metric_value = details.get(metric_key)
            if isinstance(metric_value, (int, float)):
                return f"{metric_key}={metric_value:.6f}; score={score:.2f}/{max_score:.2f}"

        return f"score={score:.2f}/{max_score:.2f}"

    def _reason_lines_for_result(self, result: EvalResult, max_reasons: int = 3) -> list[str]:
        """Summarize why an instance failed gates or scored below perfect."""
        reasons: list[str] = []

        if result.agent_output:
            status = result.agent_output.status.value
            if status != "completed":
                reasons.append(f"agent_status: {status}")
            if result.agent_output.error_message:
                reasons.append(f"agent_error: {result.agent_output.error_message}")

        if not result.gates_passed:
            for gate in result.gate_results:
                if gate.passed or gate.severity == "soft":
                    continue
                message = gate.message or f"{gate.scorer_name} gate failed"
                reasons.append(f"{gate.scorer_name}: {message}")
                if len(reasons) >= max_reasons:
                    break
            return reasons[:max_reasons]

        if result.soft_gate_failures:
            for gate in result.gate_results:
                if gate.passed or gate.severity != "soft":
                    continue
                message = gate.message or f"{gate.scorer_name} soft check failed"
                reasons.append(f"{gate.scorer_name} (soft): {message}")
                if len(reasons) >= max_reasons:
                    return reasons[:max_reasons]

        for score_result in result.score_results:
            if score_result.score >= score_result.max_score:
                continue
            message = score_result.message or self._fallback_reason_for_score(score_result)
            reasons.append(f"{score_result.scorer_name}: {message}")
            if len(reasons) >= max_reasons:
                break

        return reasons[:max_reasons]

    def _format_low_score_section(self, max_instances: int = 5) -> list[str]:
        """Format a compact per-instance summary for non-perfect runs."""
        interesting = [
            r for r in sorted(self.results, key=lambda x: (x.final_score, x.instance_id))
            if not is_unscored_submission(r)
            if (not r.gates_passed) or r.soft_gate_failures > 0 or r.final_score < r.max_possible_score
        ]
        if not interesting:
            return []

        lines = ["  Low-Score Instances:"]
        for result in interesting[:max_instances]:
            if not result.gates_passed:
                outcome = "hard gate failure"
            elif result.soft_gate_failures > 0:
                outcome = f"soft warning; score {result.final_score:.1f}/{result.max_possible_score:.0f}"
            else:
                outcome = f"score {result.final_score:.1f}/{result.max_possible_score:.0f}"
            lines.append(f"    {result.instance_id}: {outcome}")
            reason_lines = self._reason_lines_for_result(result)
            for reason in reason_lines:
                lines.append(f"      - {reason}")

            traj_summary = self._get_trajectory_summary(result)
            if traj_summary:
                code_execs = traj_summary.get("total_code_executions", 0)
                code_fails = traj_summary.get("code_execution_failures", 0)
                if code_execs > 0:
                    lines.append(f"      - {code_execs} code executions ({code_fails} failures)")

        remaining = len(interesting) - max_instances
        if remaining > 0:
            lines.append(f"    ... and {remaining} more")
        lines.append("")
        return lines

    @staticmethod
    def _get_trajectory_summary(result: EvalResult) -> dict | None:
        """Extract trajectory_summary from the result's agent_output metadata."""
        if not result.agent_output:
            return None
        ao = result.agent_output
        if hasattr(ao, "_trajectory_summary"):
            return ao._trajectory_summary
        return None

    def format_summary(self) -> str:
        """Format as a human-readable summary string."""
        all_unscored = (
            self.n_instances > 0
            and self.unscored_submission_instances == self.n_instances
        )
        lines = [
            "=" * 60,
            "  ASI-Bench Produce-Only Summary" if all_unscored else "  ASI-Bench Results Summary",
            f"  Agent: {self.agent_name}",
            f"  Tasks: {self.n_tasks}  |  Instances: {self.n_instances}",
            "=" * 60,
        ]

        if all_unscored:
            lines.extend([
                "  Scoring: submit to https://asibench.apexin.ai/",
                "",
            ])
        else:
            lines.extend([f"  Overall: {self.overall_mean_score:.1f} / 100", ""])

        if self.by_prompt_level and not all_unscored:
            lines.append("  By Prompt Level:")
            for level, score in sorted(self.by_prompt_level.items()):
                lines.append(f"    {level.upper()}: {score:.1f} / 100")
            lines.append("")

        if self.by_domain and not all_unscored:
            lines.append("  By Domain:")
            for ds in self.by_domain:
                lines.append(f"    {ds.domain}: {ds.mean_score:.1f}")
            lines.append("")

        if self.unscored_submission_instances:
            lines.append(
                "  Unscored submissions (submit mode): "
                f"{self.unscored_submission_instances}"
            )
            lines.append("")

        if self.n_instances > 0 and not all_unscored:
            lines.append("  Outcome Breakdown:")
            lines.append(f"    Hard gate failures: {self.gate_failed_instances}")
            lines.append(f"    Soft gate warnings with score: {self.soft_gate_warning_instances}")
            lines.append(
                f"    Passed hard gates but scored 0: {self.zero_score_after_gates_instances}"
            )
            lines.append(f"    Non-zero scored runs: {self.nonzero_score_instances}")
            lines.append("")

        if self.error_distribution:
            total_errors = sum(self.error_distribution.values())
            lines.append(f"  Error Distribution ({total_errors} failures analyzed):")
            for cat, count in sorted(self.error_distribution.items(), key=lambda x: -x[1]):
                pct = count / total_errors * 100 if total_errors > 0 else 0
                lines.append(f"    {cat}: {count} ({pct:.0f}%)")
            lines.append("")

        if self.behavior_summary:
            bs = self.behavior_summary
            lines.append("  Agent Behavior:")
            lines.append(f"    Avg turns: {bs.avg_turns:.1f}")
            tool_parts = ", ".join(
                f"{name}: {count}" for name, count in sorted(bs.tool_call_distribution.items())
            )
            lines.append(f"    Avg tool calls: {bs.avg_tool_calls:.1f}" + (f" ({tool_parts})" if tool_parts else ""))
            lines.append(f"    Avg thinking length: {bs.avg_thinking_chars:.0f} chars")
            lines.append("")

        lines.extend(self._format_low_score_section())

        retry_lines = [] if all_unscored else self._format_retry_diff_section()
        if retry_lines:
            lines.extend(retry_lines)

        lines.append(f"  Total execution time: {self.total_execution_time:.0f}s")
        if self.total_cost_usd > 0:
            lines.append(f"  Total cost: ${self.total_cost_usd:.2f} ({self.total_tokens:,} tokens)")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_retry_diff_section(self) -> list[str]:
        """Format comparison of multiple retry attempts."""
        by_instance: dict[str, list[EvalResult]] = {}
        for r in self.results:
            by_instance.setdefault(r.instance_id, []).append(r)

        multi_attempt = {
            iid: sorted(attempts, key=lambda x: x.attempt)
            for iid, attempts in by_instance.items()
            if len(attempts) > 1
        }
        if not multi_attempt:
            return []

        lines = ["  Retry Comparison:"]
        for iid, attempts in sorted(multi_attempt.items()):
            lines.append(f"    {iid}:")
            for a in attempts:
                status = a.status.value if hasattr(a.status, 'value') else str(a.status)
                lines.append(
                    f"      Attempt {a.attempt}: score {a.final_score:.1f}/{a.max_possible_score:.0f}, {status}"
                )
        lines.append("")
        return lines

    def format_csv(self, *, include_header: bool = True) -> str:
        """Export per-task x level scores as CSV."""
        rows: list[str] = []
        if include_header:
            rows.append("task_id,domain,level,n_instances,mean_score,min_score,max_score")
        for ts in sorted(self.by_task, key=lambda t: t.task_id):
            domain = ts.task_id.split(".")[0] if "." in ts.task_id else "unknown"
            if ts.scores_by_level:
                for level, score in sorted(ts.scores_by_level.items()):
                    by_level = [
                        r for r in self.results
                        if r.task_id == ts.task_id and r.prompt_level.value == level
                    ]
                    n = len(by_level)
                    scores = [r.final_score for r in by_level]
                    rows.append(
                        f"{ts.task_id},{domain},{level},{n},"
                        f"{sum(scores)/n:.1f},{min(scores):.1f},{max(scores):.1f}"
                    )
            else:
                rows.append(
                    f"{ts.task_id},{domain},all,{ts.n_instances},"
                    f"{ts.mean_score:.1f},{ts.min_score:.1f},{ts.max_score:.1f}"
                )
        return "\n".join(rows)

    def format_task_table(self) -> str:
        """Format a per-task score table for terminal display."""
        if not self.by_task:
            return ""
        levels = sorted({
            lv for ts in self.by_task for lv in ts.scores_by_level
        })
        header = f"  {'Task':<50s}" + "".join(f" {lv.upper():>6s}" for lv in levels) + "   Mean"
        sep = "  " + "-" * (50 + 7 * len(levels) + 7)
        lines = ["", "  Per-Task Scores:", header, sep]
        for ts in sorted(self.by_task, key=lambda t: t.task_id):
            row = f"  {ts.task_id:<50s}"
            for lv in levels:
                s = ts.scores_by_level.get(lv)
                row += f" {s:6.1f}" if s is not None else "      -"
            row += f"  {ts.mean_score:5.1f}"
            lines.append(row)
        lines.append(sep)
        lines.append(
            f"  {'OVERALL':<50s}"
            + "".join(
                f" {self.by_prompt_level.get(lv, 0):6.1f}" for lv in levels
            )
            + f"  {self.overall_mean_score:5.1f}"
        )
        lines.append("")
        return "\n".join(lines)
