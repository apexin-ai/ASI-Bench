"""Parallel execution of benchmark instances using concurrent.futures."""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Callable

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    RunStatus,
    ScoreDetail,
    TaskInstance,
)

logger = get_logger(__name__)


def _make_failed_eval_result(instance: TaskInstance, exc: Exception) -> EvalResult:
    """Build a zero-score EvalResult when an instance run fails unexpectedly."""
    return EvalResult(
        instance_id=instance.instance_id,
        task_id=instance.task_id,
        prompt_level=instance.prompt_level,
        agent_name="unknown",
        parameters=instance.parameters,
        gate_results=[],
        gates_passed=False,
        hard_gates_passed=False,
        score_results=[ScoreDetail(
            scorer_name="_runner",
            score=0.0,
            max_score=100.0,
            passed=False,
            message=f"Instance run crashed: {type(exc).__name__}: {exc}",
            details={"runner_internal_error": True},
        )],
        final_score=0.0,
        status=RunStatus.FAILED,
    )


def auto_limit_workers(
    requested: int,
    *,
    sandbox: str = "none",
    memory_limit_gb: float = 8.0,
) -> int:
    """Clamp parallel workers based on host resources when using --sandbox os.

    For ``os`` sandbox, each container reserves ``memory_limit_gb`` of RAM.
    The function limits concurrency so total reservation does not exceed 80%
    of available physical memory.  For other sandbox modes, the requested
    value is returned unchanged (clamped to >= 1).

    CPU-based limiting: each container defaults to 4.0 CPUs, so we also
    limit to (cpu_count / 4) when that is smaller.
    """
    requested = max(1, requested)
    if sandbox != "os":
        return requested

    import shutil

    # Memory-based limit
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mem_gb = mem_bytes / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        # os.sysconf not available on all platforms (e.g. macOS may raise)
        mem_gb = 16.0  # conservative default

    usable_gb = mem_gb * 0.8
    mem_limit = max(1, int(usable_gb / memory_limit_gb))

    # CPU-based limit (4 CPUs per container default)
    cpu_count = os.cpu_count() or 4
    cpu_limit = max(1, cpu_count // 4)

    effective = min(requested, mem_limit, cpu_limit)
    if effective < requested:
        logger.info(
            "Clamping --parallel from %d to %d for --sandbox os "
            "(host: %.1f GB RAM, %d CPUs; per-container: %.1f GB, 4 CPUs)",
            requested,
            effective,
            mem_gb,
            cpu_count,
            memory_limit_gb,
        )
    return effective


class ParallelRunner:
    """Run benchmark instances concurrently using a thread pool.

    Uses ThreadPoolExecutor because agent adapters are typically I/O-bound
    (subprocess calls, API requests). For CPU-bound workloads, consider
    ProcessPoolExecutor, but agent execution is rarely CPU-bound in the
    orchestrator process itself.
    """

    def __init__(self, max_workers: int = 1):
        self.max_workers = max(1, max_workers)

    def run_instances(
        self,
        instances: list[TaskInstance],
        run_fn: Callable[[TaskInstance], EvalResult],
        completed_ids: set[str] | None = None,
    ) -> list[EvalResult]:
        """Run instances through run_fn, optionally in parallel.

        Args:
            instances: task instances to evaluate
            run_fn: function that takes a TaskInstance and returns an EvalResult
                     (should handle agent execution, evaluation, analysis, and saving)
            completed_ids: set of instance IDs to skip (for resume support)

        Returns:
            list of EvalResult for all completed instances
        """
        completed_ids = completed_ids or set()
        pending = [i for i in instances if i.run_key not in completed_ids]

        if not pending:
            logger.info("No pending instances to run.")
            return []

        if self.max_workers <= 1:
            return self._run_sequential(pending, run_fn)
        return self._run_parallel(pending, run_fn)

    def _run_sequential(
        self,
        instances: list[TaskInstance],
        run_fn: Callable[[TaskInstance], EvalResult],
    ) -> list[EvalResult]:
        """Run instances sequentially (max_workers=1)."""
        results = []
        for i, instance in enumerate(instances, 1):
            logger.info(f"[{i}/{len(instances)}] Running {instance.instance_id}")
            try:
                result = run_fn(instance)
            except Exception as e:
                logger.error(
                    "Unexpected error for %s: %s", instance.instance_id, e,
                )
                result = _make_failed_eval_result(instance, e)
            results.append(result)
        return results

    def _run_parallel(
        self,
        instances: list[TaskInstance],
        run_fn: Callable[[TaskInstance], EvalResult],
    ) -> list[EvalResult]:
        """Run instances in parallel using ThreadPoolExecutor."""
        results = []
        total = len(instances)
        logger.info(f"Running {total} instances with {self.max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_instance = {
                executor.submit(run_fn, instance): instance
                for instance in instances
            }

            for future in concurrent.futures.as_completed(future_to_instance):
                instance = future_to_instance[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"[{len(results)}/{total}] Completed {instance.instance_id}: "
                        f"{result.final_score:.1f}/100"
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error for {instance.instance_id}: {e}"
                    )
                    result = _make_failed_eval_result(instance, e)
                    results.append(result)

        return results
