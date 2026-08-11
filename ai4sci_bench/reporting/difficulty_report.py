"""Build difficulty-check reports (terminal table + JSON + Markdown).

A "verdict detail" row describes one (agent, prompt_level) outcome:

```
{
  "agent": "claude-opus-4-6",           # human-readable label
  "agent_name": "direct_llm",           # adapter name
  "agent_config": {"model": "..."},
  "prompt_level": "b3",
  "mean_score": 12.0,
  "max_score": 15.5,
  "min_score": 8.5,
  "instances": 1,
  "passed": True,
}
```

A task PASSES the difficulty check when every row's ``mean_score`` is strictly
below the threshold.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import click

from ai4sci_bench.core.types import EvalResult


@dataclass
class VerdictRow:
    """One agent × prompt-level outcome."""

    agent: str
    agent_name: str
    agent_config: dict[str, Any]
    prompt_level: str
    mean_score: float
    max_score: float
    min_score: float
    instances: int
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "agent_name": self.agent_name,
            "agent_config": dict(self.agent_config),
            "prompt_level": self.prompt_level,
            "mean_score": round(self.mean_score, 2),
            "max_score": round(self.max_score, 2),
            "min_score": round(self.min_score, 2),
            "instances": self.instances,
            "passed": self.passed,
        }


@dataclass
class DifficultyReport:
    """Aggregated difficulty-check result for one task."""

    task_id: str
    threshold: int
    overall_pass: bool
    rows: list[VerdictRow] = field(default_factory=list)
    tool_mode: str = "restricted"
    sandbox: str = "none"
    instances_per_task: int = 1
    seeds: list[int] = field(default_factory=list)
    task_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "threshold": self.threshold,
            "verdict": "pass" if self.overall_pass else "fail",
            "tool_mode": self.tool_mode,
            "sandbox": self.sandbox,
            "instances_per_task": self.instances_per_task,
            "seeds": list(self.seeds),
            "task_version": self.task_version,
            "results": [row.as_dict() for row in self.rows],
        }

    def to_per_agent_results(self) -> list[dict[str, Any]]:
        """Reshape rows into the ``scores/`` file structure (one block per agent)."""
        by_agent: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.rows:
            key = (row.agent_name, json.dumps(row.agent_config, sort_keys=True))
            block = by_agent.setdefault(key, {
                "agent": row.agent_name,
                "agent_config": dict(row.agent_config),
                "scores": {},
            })
            block["scores"][row.prompt_level] = {
                "mean": round(row.mean_score, 2),
                "max": round(row.max_score, 2),
                "min": round(row.min_score, 2),
                "n": row.instances,
            }
        return list(by_agent.values())


def build_report(
    task_id: str,
    agent_runs: Sequence[tuple[str, str, dict[str, Any], list[EvalResult]]],
    threshold: int,
    *,
    tool_mode: str = "restricted",
    sandbox: str = "none",
    instances_per_task: int = 1,
    seeds: Sequence[int] | None = None,
    task_version: str = "",
) -> DifficultyReport:
    """Aggregate raw results into a DifficultyReport.

    ``agent_runs`` is a sequence of ``(agent_label, agent_name, agent_config, results)``
    tuples. Each tuple's EvalResults may span multiple prompt levels — we re-extract
    the level from each EvalResult and aggregate per (agent, level).
    """
    rows: list[VerdictRow] = []
    overall = True
    for agent_label, agent_name, agent_config, results in agent_runs:
        by_level: dict[str, list[float]] = {}
        for r in results:
            level = r.prompt_level.value if hasattr(r.prompt_level, "value") else str(r.prompt_level)
            by_level.setdefault(level, []).append(r.final_score)
        for level, scores in sorted(by_level.items()):
            mean_s = statistics.mean(scores)
            passed = mean_s < threshold
            if not passed:
                overall = False
            rows.append(VerdictRow(
                agent=agent_label,
                agent_name=agent_name,
                agent_config=agent_config,
                prompt_level=level,
                mean_score=mean_s,
                max_score=max(scores),
                min_score=min(scores),
                instances=len(scores),
                passed=passed,
            ))

    rows.sort(key=lambda r: (r.agent, r.prompt_level))
    return DifficultyReport(
        task_id=task_id,
        threshold=int(threshold),
        overall_pass=overall,
        rows=rows,
        tool_mode=tool_mode,
        sandbox=sandbox,
        instances_per_task=int(instances_per_task),
        seeds=list(seeds) if seeds is not None else [],
        task_version=task_version,
    )


def format_terminal(report: DifficultyReport, *, use_color: bool = True) -> str:
    """Render a difficulty report as a colored ASCII table."""
    title_bar = "=" * 63
    lines = [
        title_bar,
        f"  Difficulty Check: {report.task_id}",
        f"  Threshold: < {report.threshold} (tool_mode={report.tool_mode}, sandbox={report.sandbox})",
        title_bar,
        "",
    ]

    agent_w = max([len("Agent")] + [len(r.agent) for r in report.rows]) if report.rows else len("Agent")
    header = (
        f"  {'Agent'.ljust(agent_w)} | Level | Mean  | Max   | Min   | Verdict"
    )
    sep = (
        f"  {'-' * agent_w}-+-------+-------+-------+-------+--------"
    )
    lines.append(header)
    lines.append(sep)

    for row in report.rows:
        verdict_word = "PASS" if row.passed else "FAIL"
        if use_color:
            verdict_word = click.style(verdict_word, fg="green" if row.passed else "red", bold=True)
        lines.append(
            f"  {row.agent.ljust(agent_w)} | {row.prompt_level.upper():<5} | "
            f"{row.mean_score:5.1f} | {row.max_score:5.1f} | {row.min_score:5.1f} | {verdict_word}"
        )

    lines.append("")
    overall_word = "PASS" if report.overall_pass else "FAIL"
    if use_color:
        overall_word = click.style(overall_word, fg="green" if report.overall_pass else "red", bold=True)
    lines.append(f"  Overall: {overall_word}")
    lines.append("")

    if report.overall_pass:
        lines.append("  This task meets the difficulty requirement.")
        lines.append("  Recommended next step: change status to 'test' and submit PR.")
    else:
        offenders = [r for r in report.rows if not r.passed]
        for r in offenders:
            lines.append(
                f"  {r.agent} scores {r.mean_score:.1f} on {r.prompt_level.upper()} "
                f"(threshold: {report.threshold})"
            )
        lines.append("")
        lines.append("  Suggestions:")
        lines.append("  - Review difficulty-engineering.md for hardening strategies")
        lines.append("  - Consider: tighter eval thresholds, more implementation steps,")
        lines.append("    wider method space, or anti-shortcut gates")

    lines.append("")
    lines.append(title_bar)
    return "\n".join(lines)


def format_markdown(report: DifficultyReport) -> str:
    """Render a difficulty report as Markdown (PR-comment friendly)."""
    icon = ":white_check_mark:" if report.overall_pass else ":x:"
    verdict = "PASS" if report.overall_pass else "FAIL"
    lines = [
        f"## {icon} Difficulty Check: `{report.task_id}`",
        "",
        f"**Verdict:** {verdict} | **Threshold:** `< {report.threshold}` | "
        f"**Tool mode:** `{report.tool_mode}` | **Sandbox:** `{report.sandbox}`",
        "",
        "| Agent | Level | Mean | Max | Min | Verdict |",
        "|-------|-------|------|-----|-----|---------|",
    ]
    for row in report.rows:
        v = "PASS" if row.passed else "**FAIL**"
        lines.append(
            f"| `{row.agent}` | {row.prompt_level.upper()} | "
            f"{row.mean_score:.1f} | {row.max_score:.1f} | {row.min_score:.1f} | {v} |"
        )
    lines.append("")
    if not report.overall_pass:
        lines.append("> One or more (agent, level) combinations scored at or above the threshold.")
        lines.append("> Consider hardening the task before promoting to `final`.")
        lines.append("")
    return "\n".join(lines)


def write_json(report: DifficultyReport, path: Path | str) -> Path:
    """Write the report as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def write_markdown(report: DifficultyReport, path: Path | str) -> Path:
    """Write the report as Markdown."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_markdown(report) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Batch CSV (T9) — summary for `difficulty-check --status <status>` runs
# ---------------------------------------------------------------------------

BATCH_CSV_FIELDS: tuple[str, ...] = (
    "task_id",
    "task_version",
    "agent",
    "agent_name",
    "model",
    "prompt_level",
    "mean_score",
    "max_score",
    "min_score",
    "instances",
    "threshold",
    "row_verdict",
    "overall_verdict",
    "tool_mode",
    "sandbox",
)


def _model_from_config(cfg: dict[str, Any]) -> str:
    """Extract a printable model name from an agent config; fall back to empty string."""
    return str(cfg.get("model", "") or "")


def format_batch_csv(reports: Sequence[DifficultyReport]) -> str:
    """Render batch difficulty-check reports as a CSV string.

    One row per (task, agent, prompt_level). `overall_verdict` is the per-task
    pass/fail; `row_verdict` is the per-row pass/fail.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=BATCH_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for report in reports:
        overall = "pass" if report.overall_pass else "fail"
        for row in report.rows:
            writer.writerow({
                "task_id": report.task_id,
                "task_version": report.task_version,
                "agent": row.agent,
                "agent_name": row.agent_name,
                "model": _model_from_config(row.agent_config),
                "prompt_level": row.prompt_level,
                "mean_score": round(row.mean_score, 2),
                "max_score": round(row.max_score, 2),
                "min_score": round(row.min_score, 2),
                "instances": row.instances,
                "threshold": report.threshold,
                "row_verdict": "pass" if row.passed else "fail",
                "overall_verdict": overall,
                "tool_mode": report.tool_mode,
                "sandbox": report.sandbox,
            })
    return buf.getvalue()


def write_batch_csv(reports: Sequence[DifficultyReport], path: Path | str) -> Path:
    """Write a batch CSV summary to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_batch_csv(reports), encoding="utf-8")
    return out_path


@dataclass
class FlaggedTaskEntry:
    """A task that failed the batch threshold; surfaced in the re-eval summary."""

    task_id: str
    worst_mean: float
    worst_agent: str
    worst_level: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "worst_mean_score": round(self.worst_mean, 2),
            "worst_agent": self.worst_agent,
            "worst_prompt_level": self.worst_level,
        }


def collect_flagged(reports: Sequence[DifficultyReport]) -> list[FlaggedTaskEntry]:
    """For each failing report, return the (agent, level) row with the worst (highest) mean score."""
    flagged: list[FlaggedTaskEntry] = []
    for report in reports:
        if report.overall_pass:
            continue
        bad_rows = [r for r in report.rows if not r.passed]
        if not bad_rows:
            continue
        worst = max(bad_rows, key=lambda r: r.mean_score)
        flagged.append(FlaggedTaskEntry(
            task_id=report.task_id,
            worst_mean=worst.mean_score,
            worst_agent=worst.agent,
            worst_level=worst.prompt_level,
        ))
    flagged.sort(key=lambda e: (-e.worst_mean, e.task_id))
    return flagged


def format_batch_summary(
    reports: Sequence[DifficultyReport],
    *,
    status_filter: str | None,
    threshold: int,
    use_color: bool = True,
    next_step_hint: str = "Run `asibench catalog --flagged` for details.",
    scores_dir: str | None = None,
) -> str:
    """Render the re-evaluation summary block printed after a batch run.

    Format follows the design in research/task_submission_pipeline_design.md §9:
    counts of evaluated / still-passing / flagged tasks, then a list of the
    flagged tasks sorted by worst mean score, then a "next step" hint.
    """
    n_tasks = len(reports)
    flagged = collect_flagged(reports)
    n_flagged = len(flagged)
    n_passing = n_tasks - n_flagged

    title_bar = "=" * 63
    status_label = status_filter or "batch"
    lines = [
        title_bar,
        f"  Re-evaluation Summary ({status_label}, threshold={threshold}, "
        f"{n_tasks} tasks)",
        title_bar,
        "",
        f"  {n_tasks} tasks evaluated",
    ]

    passing_word = f"{n_passing} still passing"
    if use_color and n_passing > 0:
        passing_word = click.style(passing_word, fg="green")
    lines.append(f"  {passing_word}")

    if n_flagged == 0:
        flagged_word = "0 flagged"
        if use_color:
            flagged_word = click.style(flagged_word, fg="green")
        lines.append(f"  {flagged_word}")
    else:
        flagged_word = f"{n_flagged} flagged (any score >= {threshold}):"
        if use_color:
            flagged_word = click.style(flagged_word, fg="red", bold=True)
        lines.append(f"  {flagged_word}")
        for entry in flagged:
            lines.append(
                f"    - {entry.task_id}: {entry.worst_mean:.1f} "
                f"({entry.worst_agent}, {entry.worst_level.upper()})"
            )

    lines.append("")
    if scores_dir:
        lines.append(f"  Scores saved to {scores_dir}")
    lines.append(f"  {next_step_hint}")
    lines.append(title_bar)
    return "\n".join(lines)


def group_results_by_agent(
    results: Iterable[EvalResult],
    agent_label_by_name: dict[str, str] | None = None,
    agent_config_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, str, dict[str, Any], list[EvalResult]]]:
    """Group EvalResults into ``(agent_label, agent_name, agent_config, results)`` tuples.

    ``agent_label_by_name`` / ``agent_config_by_name`` allow callers to override
    the label/config for known agent names; fallback uses ``EvalResult.agent_name``.
    """
    label_map = agent_label_by_name or {}
    config_map = agent_config_by_name or {}
    # Use canonical JSON form of (label, name, config) as the bucket key
    buckets: dict[str, list[EvalResult]] = {}
    meta: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for r in results:
        name = r.agent_name
        label = label_map.get(name, name)
        config = config_map.get(name, {})
        key = json.dumps([label, name, config], sort_keys=True)
        buckets.setdefault(key, []).append(r)
        meta.setdefault(key, (label, name, config))
    return [(*meta[k], buckets[k]) for k in buckets]
