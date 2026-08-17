"""Derived multi-run record artifacts built from saved result JSON files."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ai4sci_bench.reporting.aggregator import aggregate_results
from ai4sci_bench.reporting.result_loader import AgentProvenance, load_grouped_results
from ai4sci_bench.reporting.results import is_unscored_submission

PROMPT_LEVELS = ("b1", "b2", "b3", "b4")


def refresh_batch_records(results_root: Path) -> list[Path]:
    """Regenerate derived batch record artifacts under ``results_root/batch_records``."""
    groups = load_grouped_results(results_root)
    if not groups:
        return []

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    batch_dir = results_root / "batch_records"
    batch_dir.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []

    for group in groups:
        run_report = aggregate_results(group.results)
        run_report.agent_name = group.label
        provenance = group.provenance
        has_scored_results = any(
            not is_unscored_submission(result)
            for result in group.results
        )
        overview_rows.append({
            "agent_label": provenance.agent_label,
            "agent_name": provenance.agent_name,
            "adapter_class": provenance.adapter_class,
            "model_name": provenance.model_name,
            "method_group": provenance.method_group,
            "n_tasks": run_report.n_tasks,
            "n_instances": run_report.n_instances,
            "overall_mean_score": (
                f"{run_report.overall_mean_score:.1f}"
                if has_scored_results else ""
            ),
            "hard_gate_failures": run_report.gate_failed_instances,
            "soft_gate_warnings": run_report.soft_gate_warning_instances,
            "results_root": str(group.group_root or results_root / group.label),
            "generated_at": generated_at,
            **_format_level_columns(run_report.by_prompt_level),
        })
        for task_summary in run_report.by_task:
            task_has_scored_results = any(
                result.task_id == task_summary.task_id
                and not is_unscored_submission(result)
                for result in group.results
            )
            task_rows.append({
                "agent_label": provenance.agent_label,
                "agent_name": provenance.agent_name,
                "adapter_class": provenance.adapter_class,
                "model_name": provenance.model_name,
                "method_group": provenance.method_group,
                "task_id": task_summary.task_id,
                "n_instances": task_summary.n_instances,
                "mean_score": (
                    f"{task_summary.mean_score:.1f}"
                    if task_has_scored_results else ""
                ),
                "min_score": (
                    f"{task_summary.min_score:.1f}"
                    if task_has_scored_results else ""
                ),
                "max_score": (
                    f"{task_summary.max_score:.1f}"
                    if task_has_scored_results else ""
                ),
                "std_score": (
                    f"{task_summary.std_score:.1f}"
                    if task_has_scored_results else ""
                ),
                "gates_pass_rate": (
                    f"{task_summary.gates_pass_rate:.3f}"
                    if task_has_scored_results else ""
                ),
                "generated_at": generated_at,
                **_format_level_columns(task_summary.scores_by_level),
            })
        level_rows.extend(
            _build_task_level_rows(
                provenance=provenance,
                results=group.results,
                generated_at=generated_at,
            )
        )

    overview_path = batch_dir / "batch_overview.csv"
    _write_csv(
        overview_path,
        overview_rows,
        fieldnames=[
            "agent_label",
            "agent_name",
            "adapter_class",
            "model_name",
            "method_group",
            "n_tasks",
            "n_instances",
            "overall_mean_score",
            "hard_gate_failures",
            "soft_gate_warnings",
            *PROMPT_LEVELS,
            "results_root",
            "generated_at",
        ],
    )
    overview_json_path = batch_dir / "batch_overview.json"
    _write_json(overview_json_path, overview_rows)

    scoreboard_path = batch_dir / "task_scoreboard.csv"
    _write_csv(
        scoreboard_path,
        task_rows,
        fieldnames=[
            "agent_label",
            "agent_name",
            "adapter_class",
            "model_name",
            "method_group",
            "task_id",
            "n_instances",
            "mean_score",
            "min_score",
            "max_score",
            "std_score",
            "gates_pass_rate",
            *PROMPT_LEVELS,
            "generated_at",
        ],
    )
    scoreboard_json_path = batch_dir / "task_scoreboard.json"
    _write_json(scoreboard_json_path, task_rows)

    task_level_long_path = batch_dir / "task_level_long.csv"
    _write_csv(
        task_level_long_path,
        level_rows,
        fieldnames=[
            "agent_label",
            "agent_name",
            "adapter_class",
            "model_name",
            "method_group",
            "task_id",
            "prompt_level",
            "n_instances",
            "mean_score",
            "gates_pass_rate",
            "generated_at",
        ],
    )
    task_level_long_json_path = batch_dir / "task_level_long.json"
    _write_json(task_level_long_json_path, level_rows)

    markdown_path = batch_dir / "batch_overview.md"
    _write_text_atomic(
        markdown_path,
        _render_markdown(results_root, generated_at, overview_rows, task_rows, level_rows),
    )

    return [
        overview_path,
        overview_json_path,
        scoreboard_path,
        scoreboard_json_path,
        task_level_long_path,
        task_level_long_json_path,
        markdown_path,
    ]


def write_batch_records(results_root: Path) -> list[Path]:
    """Backward-compatible alias for refreshing derived batch record artifacts."""
    return refresh_batch_records(results_root)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(path, buffer.getvalue(), newline="")


def _write_json(path: Path, payload: object) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_text_atomic(path: Path, content: str, *, newline: str | None = None) -> None:
    """Write a file via temp-path + rename to avoid partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        tmp_path.write_text(content, encoding="utf-8", newline=newline)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _render_markdown(
    results_root: Path,
    generated_at: str,
    overview_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
    level_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Batch Run Scoreboard",
        "",
        f"- Results root: `{results_root}`",
        f"- Generated at: `{generated_at}`",
        "",
        "## Run Overview",
        "",
        "| label | method | model | agent_name | adapter_class | tasks | instances | overall | b1 | b2 | b3 | b4 | hard gate fail | soft warn |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overview_rows:
        lines.append(
            f"| {row['agent_label']} | {row['method_group']} | {_display_level_value(row['model_name'])} | "
            f"{row['agent_name']} | {row['adapter_class']} | {row['n_tasks']} | {row['n_instances']} | "
            f"{row['overall_mean_score']} | {_display_level_value(row['b1'])} | {_display_level_value(row['b2'])} | "
            f"{_display_level_value(row['b3'])} | {_display_level_value(row['b4'])} | "
            f"{row['hard_gate_failures']} | {row['soft_gate_warnings']} |"
        )

    lines.extend([
        "",
        "## Task Scoreboard",
        "",
        "| label | method | model | agent_name | adapter_class | task_id | instances | mean | min | max | std | b1 | b2 | b3 | b4 | gates pass rate |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in task_rows:
        lines.append(
            f"| {row['agent_label']} | {row['method_group']} | {_display_level_value(row['model_name'])} | "
            f"{row['agent_name']} | {row['adapter_class']} | {row['task_id']} | {row['n_instances']} | "
            f"{row['mean_score']} | {row['min_score']} | {row['max_score']} | "
            f"{row['std_score']} | {_display_level_value(row['b1'])} | {_display_level_value(row['b2'])} | "
            f"{_display_level_value(row['b3'])} | {_display_level_value(row['b4'])} | {row['gates_pass_rate']} |"
        )

    lines.extend([
        "",
        "## Task Level Long",
        "",
        "| label | method | model | agent_name | adapter_class | task_id | prompt_level | instances | mean | gates pass rate |",
        "|---|---|---|---|---|---|---|---:|---:|---:|",
    ])
    for row in level_rows:
        lines.append(
            f"| {row['agent_label']} | {row['method_group']} | {_display_level_value(row['model_name'])} | "
            f"{row['agent_name']} | {row['adapter_class']} | {row['task_id']} | {row['prompt_level']} | "
            f"{row['n_instances']} | {row['mean_score']} | {row['gates_pass_rate']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _format_level_columns(scores_by_level: dict[str, float]) -> dict[str, object]:
    return {
        level: f"{scores_by_level[level]:.1f}" if level in scores_by_level else ""
        for level in PROMPT_LEVELS
    }


def _display_level_value(value: object) -> str:
    return str(value) if value not in ("", None) else "NA"


def _build_task_level_rows(
    *,
    provenance: AgentProvenance,
    results,
    generated_at: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list] = {}
    for result in results:
        grouped.setdefault((result.task_id, result.prompt_level.value), []).append(result)

    rows: list[dict[str, object]] = []
    for (task_id, prompt_level), level_results in sorted(grouped.items()):
        n_instances = len(level_results)
        scored_level_results = [
            result for result in level_results
            if not is_unscored_submission(result)
        ]
        if scored_level_results:
            mean_score = (
                sum(result.final_score for result in scored_level_results)
                / len(scored_level_results)
            )
            gates_pass_rate = (
                sum(1 for result in scored_level_results if result.gates_passed)
                / len(scored_level_results)
            )
        else:
            mean_score = None
            gates_pass_rate = None
        rows.append({
            "agent_label": provenance.agent_label,
            "agent_name": provenance.agent_name,
            "adapter_class": provenance.adapter_class,
            "model_name": provenance.model_name,
            "method_group": provenance.method_group,
            "task_id": task_id,
            "prompt_level": prompt_level,
            "n_instances": n_instances,
            "mean_score": f"{mean_score:.1f}" if mean_score is not None else "",
            "gates_pass_rate": (
                f"{gates_pass_rate:.3f}"
                if gates_pass_rate is not None else ""
            ),
            "generated_at": generated_at,
        })
    return rows
