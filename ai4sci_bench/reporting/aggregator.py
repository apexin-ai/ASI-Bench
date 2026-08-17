"""Multi-task / multi-model result aggregation."""

from __future__ import annotations

import math
from typing import Any

from ai4sci_bench.core.types import EvalResult
from ai4sci_bench.reporting.results import (
    BehaviorSummary,
    DomainSummary,
    RunReport,
    TaskSummary,
    is_unscored_submission,
)


def aggregate_results(results: list[EvalResult], config: Any = None) -> RunReport:
    """Aggregate evaluation results into a RunReport."""
    if not results:
        return RunReport(
            agent_name="unknown",
            n_tasks=0,
            n_instances=0,
            overall_mean_score=0.0,
        )

    agent_name = results[0].agent_name
    scored_results = [
        result for result in results
        if not is_unscored_submission(result)
    ]

    # Group by task
    by_task: dict[str, list[EvalResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    task_summaries = []
    scored_task_ids: set[str] = set()
    for task_id, task_results in sorted(by_task.items()):
        scored_task_results = [
            result for result in task_results
            if not is_unscored_submission(result)
        ]
        scores = [r.final_score for r in scored_task_results]
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / n) if n > 1 else 0.0
        gates_passed = sum(1 for r in scored_task_results if r.gates_passed)
        if scored_task_results:
            scored_task_ids.add(task_id)

        # By prompt level
        by_level: dict[str, list[float]] = {}
        for r in scored_task_results:
            by_level.setdefault(r.prompt_level.value, []).append(r.final_score)
        scores_by_level = {
            level: sum(s) / len(s) for level, s in by_level.items()
        }

        task_summaries.append(TaskSummary(
            task_id=task_id,
            n_instances=len(task_results),
            mean_score=mean,
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            std_score=std,
            gates_pass_rate=gates_passed / n if n else 0.0,
            scores_by_level=scores_by_level,
        ))

    # Group by domain
    by_domain: dict[str, list[TaskSummary]] = {}
    for ts in task_summaries:
        if ts.task_id not in scored_task_ids:
            continue
        domain = ts.task_id.split(".")[0] if "." in ts.task_id else "unknown"
        by_domain.setdefault(domain, []).append(ts)

    domain_summaries = []
    for domain, domain_tasks in sorted(by_domain.items()):
        domain_mean = sum(ts.mean_score for ts in domain_tasks) / len(domain_tasks)
        domain_summaries.append(DomainSummary(
            domain=domain,
            n_tasks=len(domain_tasks),
            mean_score=domain_mean,
            task_summaries=domain_tasks,
        ))

    # By prompt level (overall)
    by_level_overall: dict[str, list[float]] = {}
    for r in scored_results:
        by_level_overall.setdefault(r.prompt_level.value, []).append(r.final_score)
    prompt_level_scores = {
        level: sum(s) / len(s) for level, s in by_level_overall.items()
    }

    # Error distribution
    error_dist: dict[str, int] = {}
    for r in results:
        if r.error_analysis:
            cat = r.error_analysis.error_category
            error_dist[cat] = error_dist.get(cat, 0) + 1

    all_scores = [r.final_score for r in scored_results]
    total_time = sum(r.execution_time_seconds for r in results)
    unscored_submission_instances = sum(is_unscored_submission(r) for r in results)
    gate_failed_instances = sum(
        1 for r in results if not is_unscored_submission(r) and not r.gates_passed
    )
    soft_gate_warning_instances = sum(
        1 for r in results
        if not is_unscored_submission(r)
        and r.gates_passed and r.soft_gate_failures > 0 and r.final_score > 0.0
    )
    zero_score_after_gates_instances = sum(
        1 for r in results
        if not is_unscored_submission(r) and r.gates_passed and r.final_score == 0.0
    )
    nonzero_score_instances = sum(
        1 for r in results
        if not is_unscored_submission(r)
        and r.gates_passed and r.soft_gate_failures == 0 and r.final_score > 0.0
    )

    # Cost aggregation
    total_cost = sum(r.cost.estimated_cost_usd for r in results if r.cost)
    total_tokens = sum(r.cost.total_tokens for r in results if r.cost)

    behavior = _compute_behavior_summary(results)

    return RunReport(
        agent_name=agent_name,
        n_tasks=len(by_task),
        n_instances=len(results),
        overall_mean_score=sum(all_scores) / len(all_scores) if all_scores else 0.0,
        by_task=task_summaries,
        by_domain=domain_summaries,
        by_prompt_level=prompt_level_scores,
        error_distribution=error_dist,
        gate_failed_instances=gate_failed_instances,
        soft_gate_warning_instances=soft_gate_warning_instances,
        zero_score_after_gates_instances=zero_score_after_gates_instances,
        nonzero_score_instances=nonzero_score_instances,
        unscored_submission_instances=unscored_submission_instances,
        total_execution_time=total_time,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        behavior_summary=behavior,
        results=results,
    )


def _compute_behavior_summary(results: list[EvalResult]) -> BehaviorSummary | None:
    """Aggregate trajectory summaries from results into a BehaviorSummary."""
    traj_summaries = []
    for r in results:
        if not r.agent_output:
            continue
        ts = getattr(r.agent_output, "_trajectory_summary", None)
        if ts and isinstance(ts, dict):
            traj_summaries.append(ts)

    if not traj_summaries:
        return None

    n = len(traj_summaries)
    total_turns = sum(ts.get("total_turns", 0) for ts in traj_summaries)
    total_tool_calls = sum(ts.get("total_tool_calls", 0) for ts in traj_summaries)
    total_thinking = sum(ts.get("thinking_total_chars", 0) for ts in traj_summaries)
    total_code_failures = sum(ts.get("code_execution_failures", 0) for ts in traj_summaries)

    tool_dist: dict[str, int] = {}
    for ts in traj_summaries:
        for name, count in ts.get("tool_call_distribution", {}).items():
            tool_dist[name] = tool_dist.get(name, 0) + count

    return BehaviorSummary(
        avg_turns=total_turns / n,
        avg_tool_calls=total_tool_calls / n,
        tool_call_distribution=tool_dist,
        avg_thinking_chars=total_thinking / n,
        avg_code_execution_failures=total_code_failures / n,
    )
