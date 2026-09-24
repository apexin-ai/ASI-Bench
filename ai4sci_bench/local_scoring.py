"""Local scoring for the public-reference seed31415 benchmark contract.

The seed42 contract is intentionally excluded: its references stay on the
ASI-Bench website and results must be submitted for scoring there.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import signal
import tempfile
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from ai4sci_bench.core.judge_api import (
    JudgeAPIOverride,
    get_judge_api_override,
    use_judge_api_override,
)
from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.core.types import ScoreDetail

PUBLIC_LOCAL_SCORING_REPO = "seed31415"
PRIVATE_SCORING_REPO = "seed42"
DEFAULT_LOCAL_SCORE_REPORT = "local_score_seed31415.json"

logger = logging.getLogger(__name__)

ScoreProgressCallback = Callable[[int, int, dict[str, Any]], None]
_EVALUATOR_FAILURE_KINDS = frozenset(
    {
        "evaluator_unavailable",
        "evaluator_runtime_error",
        "missing_evaluator_input",
    }
)


class LocalScoringError(RuntimeError):
    """Raised when a public local-scoring input is incomplete or inconsistent."""


class _MissingEvaluatorInputError(LocalScoringError):
    """Raised when immutable instance inputs cannot be materialized for scoring."""

    failure_kind = "missing_evaluator_input"


@dataclass(frozen=True)
class _ScoreJob:
    index: int
    source_result: str
    task_id: str
    instance_id: str
    prompt_level: str
    attempt: int
    task_dir: str
    output_dir: str
    reference_dir: str
    data_dir: str
    requires_instance_data: bool
    evaluation: dict[str, Any]
    parameters: dict[str, Any]
    max_score: float


@dataclass
class _ActiveScoreWorker:
    job: _ScoreJob
    process: multiprocessing.Process
    connection: Connection


def _iter_result_files(results_dir: Path):
    for path in sorted(results_dir.glob("*/*.json")):
        if ".trajectory." in path.name or ".agent_model_output." in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("task_id") and data.get("instance_id"):
            yield path, data


def _load_parameters(result: dict[str, Any], instance_dir: Path) -> dict[str, Any]:
    parameters = result.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    meta_path = instance_dir / "instance_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        params_used = meta.get("params_used")
        if isinstance(params_used, dict):
            return params_used
    return {}


def _detail_dict(detail: ScoreDetail) -> dict[str, Any]:
    return asdict(detail)


def _has_internal_error(details: list[ScoreDetail]) -> bool:
    return any(
        isinstance(detail.details, dict)
        and detail.details.get("scorer_internal_error") is True
        for detail in details
    )


def _failure_kind(details: list[ScoreDetail]) -> str | None:
    """Return the first canonical evaluator failure kind, if any."""
    for detail in details:
        if not isinstance(detail.details, dict):
            continue
        if detail.details.get("scorer_internal_error") is not True:
            continue
        kind = detail.details.get("failure_kind")
        if kind in _EVALUATOR_FAILURE_KINDS:
            return kind
        return "evaluator_runtime_error"
    return None


def _copy_tree_without_symlinks(source: Path, destination: Path, *, label: str) -> None:
    """Copy a scorer input tree while rejecting symlinks and special files."""
    if source.is_symlink() or not source.is_dir():
        raise _MissingEvaluatorInputError(f"{label} is not a real directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise _MissingEvaluatorInputError(
                f"{label} contains a symlink: {relative.as_posix()}"
            )
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        else:
            raise _MissingEvaluatorInputError(
                f"{label} contains unsupported input: {relative.as_posix()}"
            )


def _copy_output_tree(source: Path, destination: Path) -> None:
    """Merge persisted outputs into a fresh staging directory."""
    if source.is_symlink() or not source.is_dir():
        raise _MissingEvaluatorInputError(
            f"Persisted output directory is not a real directory: {source}"
        )
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise _MissingEvaluatorInputError(
                f"Persisted outputs contain a symlink: {relative.as_posix()}"
            )
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise _MissingEvaluatorInputError(
                f"Persisted outputs contain unsupported artifact: {relative.as_posix()}"
            )
        if target.exists():
            # ``data/`` is immutable evaluator input. An agent artifact must
            # never replace it while constructing the scorer workspace.
            if target.is_dir() or target.is_symlink():
                raise _MissingEvaluatorInputError(
                    f"Persisted output conflicts with evaluator input: {relative.as_posix()}"
                )
            raise _MissingEvaluatorInputError(
                f"Persisted output would overwrite evaluator input: {relative.as_posix()}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


@contextmanager
def _staged_prediction_dir(job: _ScoreJob) -> Iterator[Path]:
    """Materialize immutable instance data plus persisted outputs for one job."""
    with tempfile.TemporaryDirectory(prefix="asibench-score-") as temporary:
        pred_dir = Path(temporary)
        data_dir = Path(job.data_dir)
        if data_dir.exists():
            _copy_tree_without_symlinks(data_dir, pred_dir / "data", label="Instance data")
        elif job.requires_instance_data:
            raise _MissingEvaluatorInputError(
                f"Required instance data directory is missing: {data_dir}"
            )

        _copy_output_tree(Path(job.output_dir), pred_dir)
        yield pred_dir


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _validate_parallel(parallel: int) -> int:
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
        raise LocalScoringError("parallel must be an integer greater than or equal to 1")
    return parallel


def _prepare_score_jobs(
    result_files: list[tuple[Path, dict[str, Any]]],
    *,
    results_root: Path,
    instances_root: Path,
    tasks_root: Path,
) -> list[_ScoreJob]:
    """Validate the complete batch before any scorer or Judge call starts."""
    try:
        import ai4sci_bench.scorers  # noqa: F401
    except ModuleNotFoundError as exc:
        raise LocalScoringError(
            "Local scoring requires scientific dependencies. Install "
            "`asibench[full]` and retry."
        ) from exc

    loader = TaskLoader(tasks_root)
    task_cache: dict[str, tuple[dict[str, Any], Path, bool]] = {}
    jobs: list[_ScoreJob] = []
    for index, (result_path, source) in enumerate(result_files):
        task_id = str(source["task_id"])
        instance_id = str(source["instance_id"])
        if not instance_id.endswith("__seed31415"):
            raise LocalScoringError(
                f"Result {result_path} is not a seed31415 instance: {instance_id}"
            )

        instance_dir = instances_root / instance_id
        reference_dir = instance_dir / "reference"
        if not reference_dir.is_dir() or not any(reference_dir.iterdir()):
            raise LocalScoringError(
                f"Public seed31415 reference directory is missing or empty: "
                f"{reference_dir}. Re-run `asibench task pull --repo seed31415`."
            )

        output_dir = result_path.parent / f"{result_path.stem}.outputs"
        if not output_dir.is_dir():
            raise LocalScoringError(
                f"Persisted output directory not found for {result_path.name}: {output_dir}"
            )

        cached = task_cache.get(task_id)
        if cached is None:
            try:
                metadata = loader.load_task_by_id(task_id)
            except ValueError as exc:
                raise LocalScoringError(str(exc)) from exc
            evaluation = metadata.get("evaluation")
            if not isinstance(evaluation, dict):
                raise LocalScoringError(
                    f"Task {task_id} has no public evaluation contract under {tasks_root}"
                )
            task_dir = Path(metadata["_task_dir"])
            input_config = metadata.get("input", {})
            input_files = (
                input_config.get("files", [])
                if isinstance(input_config, dict)
                else []
            )
            # InstanceGenerator materializes the complete data tree. Input
            # names may contain expansion templates (for example
            # ``pairs/<ii>/source.npy``), so local scoring validates the
            # declared data root and stages it wholesale instead of treating
            # declarations as literal paths.
            requires_instance_data = bool(input_files)
            cached = (evaluation, task_dir, requires_instance_data)
            task_cache[task_id] = cached
        evaluation, task_dir, requires_instance_data = cached

        prompt_level = str(source.get("prompt_level") or "")
        max_score = float(
            sum(
                float(config.get("weight", 1.0))
                for config in evaluation.get("scoring", [])
            )
        )
        jobs.append(
            _ScoreJob(
                index=index,
                source_result=str(result_path.relative_to(results_root)),
                task_id=task_id,
                instance_id=instance_id,
                prompt_level=prompt_level,
                attempt=int(source.get("attempt", 1)),
                task_dir=str(task_dir.resolve()),
                output_dir=str(output_dir.absolute()),
                reference_dir=str(reference_dir.resolve()),
                data_dir=str((instance_dir / "data").absolute()),
                requires_instance_data=requires_instance_data,
                evaluation=evaluation,
                parameters=_load_parameters(source, instance_dir),
                max_score=max_score,
            )
        )
    return jobs


def _evaluate_score_job(
    job: _ScoreJob,
    judge_api_override: JudgeAPIOverride | None,
    *,
    preserve_ambient_override: bool = False,
) -> dict[str, Any]:
    import ai4sci_bench.scorers  # noqa: F401
    from ai4sci_bench.runner.orchestrator import _evaluate_gates_and_scores
    from ai4sci_bench.scorers.custom import load_custom_scorer

    load_custom_scorer(Path(job.task_dir))
    judge_scope = (
        nullcontext()
        if preserve_ambient_override and judge_api_override is None
        else use_judge_api_override(judge_api_override)
    )
    with judge_scope:
        with _staged_prediction_dir(job) as pred_dir:
            gates, hard_ok, soft_failures, scores, final_score = (
                _evaluate_gates_and_scores(
                    job.evaluation,
                    pred_dir,
                    Path(job.reference_dir),
                    job.parameters,
                    prompt_level=job.prompt_level or None,
                )
            )
    all_details = [*gates, *scores]
    internal_error = _has_internal_error(all_details)
    failure_kind = _failure_kind(all_details)
    result = {
        "source_result": job.source_result,
        "task_id": job.task_id,
        "instance_id": job.instance_id,
        "prompt_level": job.prompt_level,
        "attempt": job.attempt,
        "hard_gates_passed": hard_ok,
        "soft_gate_failures": soft_failures,
        "gate_results": [_detail_dict(item) for item in gates],
        "score_results": [_detail_dict(item) for item in scores],
        "evaluation_status": "evaluation_invalid" if internal_error else "completed",
        "final_score": None if internal_error else float(final_score),
        "max_score": job.max_score,
        "scorer_internal_error": internal_error,
    }
    if failure_kind is not None:
        result["failure_kind"] = failure_kind
    return result


def _redact_secret(value: Any, secret: str | None) -> Any:
    """Remove a resolved Judge secret from worker-owned diagnostics."""
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "<redacted>")
    if isinstance(value, dict):
        return {
            _redact_secret(key, secret): _redact_secret(inner, secret)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret(inner, secret) for inner in value]
    if isinstance(value, tuple):
        return tuple(_redact_secret(inner, secret) for inner in value)
    return value


@contextmanager
def _redact_scoring_logs(secret: str | None) -> Iterator[None]:
    """Redact the Judge secret from log records created during evaluation."""
    if not secret:
        yield
        return

    previous_factory = logging.getLogRecordFactory()

    def redacting_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        try:
            record.msg = _redact_secret(record.getMessage(), secret)
            record.args = ()
            if record.exc_info:
                formatted = "".join(traceback.format_exception(*record.exc_info))
                record.exc_text = _redact_secret(formatted, secret)
        except Exception:
            # Redaction must never turn an evaluator diagnostic into a crash.
            record.msg = "<unavailable worker log message>"
            record.args = ()
        return record

    logging.setLogRecordFactory(redacting_factory)
    try:
        yield
    finally:
        logging.setLogRecordFactory(previous_factory)


def _worker_failure_result(
    job: _ScoreJob,
    *,
    error_type: str,
    error: str,
    failure_kind: str = "evaluator_runtime_error",
    exit_code: int | None = None,
    scorer_name: str = "_parallel_scoring_worker",
) -> dict[str, Any]:
    if failure_kind not in _EVALUATOR_FAILURE_KINDS:
        failure_kind = "evaluator_runtime_error"
    details: dict[str, Any] = {
        "failure_kind": failure_kind,
        "scorer_internal_error": True,
        "exception_type": error_type,
    }
    if exit_code is not None:
        details["worker_exit_code"] = exit_code
    detail = ScoreDetail(
        scorer_name=scorer_name,
        score=0.0,
        max_score=job.max_score,
        passed=False,
        message=f"Scoring evaluator failed: {error_type}: {error}",
        details=details,
    )
    return {
        "source_result": job.source_result,
        "task_id": job.task_id,
        "instance_id": job.instance_id,
        "prompt_level": job.prompt_level,
        "attempt": job.attempt,
        "hard_gates_passed": False,
        "soft_gate_failures": 0,
        "gate_results": [],
        "score_results": [_detail_dict(detail)],
        "evaluation_status": "evaluation_invalid",
        "final_score": None,
        "max_score": job.max_score,
        "scorer_internal_error": True,
        "failure_kind": failure_kind,
    }


def _score_job_process_entry(
    connection: Connection,
    job: _ScoreJob,
    judge_api_override: JudgeAPIOverride | None,
) -> None:
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            # Parent cleanup still falls back to terminating this process.
            pass
    secret = None
    if judge_api_override is not None:
        try:
            secret = judge_api_override.resolve_api_key()
        except Exception:
            # The coordinator validates this before workers start. Keep the
            # worker path defensive if its environment changes later.
            pass
    try:
        with _redact_scoring_logs(secret):
            try:
                result = _evaluate_score_job(job, judge_api_override)
                connection.send(("ok", _redact_secret(result, secret)))
            except BaseException as exc:
                connection.send(
                    (
                        "error",
                        type(exc).__name__,
                        _redact_secret(str(exc), secret),
                        getattr(exc, "failure_kind", "evaluator_runtime_error"),
                    )
                )
    except (BrokenPipeError, EOFError, OSError):
        # The coordinator classifies a closed channel as an evaluator failure.
        # There may be no receiver while cancellation is in progress.
        pass
    finally:
        connection.close()


def _terminate_worker_process_group(process: multiprocessing.Process) -> None:
    """Best-effort termination of a worker and evaluator children it launched."""
    if os.name == "posix" and process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            pass
        except OSError:
            pass
    if process.is_alive():
        process.terminate()


def _kill_worker_process_group(process: multiprocessing.Process) -> None:
    if os.name == "posix" and process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            pass
        except OSError:
            pass
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()


def _stop_worker(worker: _ActiveScoreWorker) -> None:
    worker.connection.close()
    _terminate_worker_process_group(worker.process)
    worker.process.join(timeout=5)
    if worker.process.is_alive():
        _kill_worker_process_group(worker.process)
        worker.process.join(timeout=5)


def _worker_start_failure(
    job: _ScoreJob,
    exc: Exception,
    judge_api_override: JudgeAPIOverride | None,
) -> dict[str, Any]:
    secret = None
    if judge_api_override is not None:
        try:
            secret = judge_api_override.resolve_api_key()
        except Exception:
            pass
    return _worker_failure_result(
        job,
        error_type="WorkerStartError",
        error=_redact_secret(f"{type(exc).__name__}: {exc}", secret),
    )


def _notify_progress(
    callback: ScoreProgressCallback | None,
    completed: int,
    total: int,
    result: dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(completed, total, result)
    except Exception as exc:
        logger.warning("Local scoring progress callback failed: %s", exc)


def _score_jobs_parallel(
    jobs: list[_ScoreJob],
    *,
    parallel: int,
    judge_api_override: JudgeAPIOverride | None,
    progress_callback: ScoreProgressCallback | None,
) -> list[dict[str, Any]]:
    """Evaluate jobs in fresh spawned processes with a strict active-worker cap."""
    context = multiprocessing.get_context("spawn")
    pending = iter(jobs)
    active: list[_ActiveScoreWorker] = []
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    completed = 0

    def fill_worker_slots() -> None:
        nonlocal completed
        while len(active) < parallel:
            try:
                job = next(pending)
            except StopIteration:
                return
            receive_connection: Connection | None = None
            send_connection: Connection | None = None
            process: multiprocessing.Process | None = None
            try:
                receive_connection, send_connection = context.Pipe(duplex=False)
                process = context.Process(
                    target=_score_job_process_entry,
                    args=(send_connection, job, judge_api_override),
                    name=f"asibench-score-{job.index}",
                )
                process.start()
            except Exception as exc:
                if receive_connection is not None:
                    receive_connection.close()
                if send_connection is not None:
                    send_connection.close()
                if process is not None and process.pid is not None:
                    _terminate_worker_process_group(process)
                    process.join(timeout=1)
                    if process.is_alive():
                        _kill_worker_process_group(process)
                        process.join(timeout=1)
                result = _worker_start_failure(job, exc, judge_api_override)
                results[job.index] = result
                completed += 1
                _notify_progress(progress_callback, completed, len(jobs), result)
                continue
            send_connection.close()
            active.append(
                _ActiveScoreWorker(
                    job=job,
                    process=process,
                    connection=receive_connection,
                )
            )

    try:
        fill_worker_slots()
        while active:
            handles: list[Any] = []
            for worker in active:
                handles.extend((worker.connection, worker.process.sentinel))
            ready = set(wait(handles))
            finished: list[tuple[_ActiveScoreWorker, dict[str, Any]]] = []
            for worker in list(active):
                connection_ready = worker.connection in ready or worker.connection.poll()
                process_ready = worker.process.sentinel in ready
                if not connection_ready and not process_ready:
                    continue

                result: dict[str, Any]
                if connection_ready:
                    try:
                        message = worker.connection.recv()
                    except EOFError:
                        message = None
                    if message and message[0] == "ok":
                        result = message[1]
                    elif message and message[0] == "error":
                        result = _worker_failure_result(
                            worker.job,
                            error_type=str(message[1]),
                            error=str(message[2]),
                            failure_kind=str(message[3]),
                        )
                    else:
                        worker.process.join(timeout=1)
                        result = _worker_failure_result(
                            worker.job,
                            error_type="WorkerProtocolError",
                            error="worker closed its result channel without a valid payload",
                            exit_code=worker.process.exitcode,
                        )
                else:
                    worker.process.join(timeout=1)
                    if worker.connection.poll(0.1):
                        try:
                            message = worker.connection.recv()
                        except EOFError:
                            message = None
                        if message and message[0] == "ok":
                            result = message[1]
                        elif message and message[0] == "error":
                            result = _worker_failure_result(
                                worker.job,
                                error_type=str(message[1]),
                                error=str(message[2]),
                                failure_kind=str(message[3]),
                            )
                        else:
                            result = _worker_failure_result(
                                worker.job,
                                error_type="WorkerProtocolError",
                                error=(
                                    "worker closed its result channel without "
                                    "a valid payload"
                                ),
                                exit_code=worker.process.exitcode,
                            )
                        finished.append((worker, result))
                        continue
                    # A clean process must still send a result. Treat all silent
                    # exits as evaluator failures rather than scored zeros.
                    result = _worker_failure_result(
                        worker.job,
                        error_type="WorkerProcessExit",
                        error="worker exited before returning a score result",
                        exit_code=worker.process.exitcode,
                    )
                finished.append((worker, result))

            for worker, result in finished:
                active.remove(worker)
                worker.connection.close()
                worker.process.join(timeout=5)
                if worker.process.is_alive():
                    _stop_worker(worker)
                results[worker.job.index] = result
                completed += 1
                _notify_progress(progress_callback, completed, len(jobs), result)
            fill_worker_slots()
    except BaseException:
        for worker in active:
            _stop_worker(worker)
        raise

    for job in jobs:
        if results[job.index] is None:
            result = _worker_failure_result(
                job,
                error_type="MissingWorkerResult",
                error="parallel scoring finished without a result for this job",
            )
            results[job.index] = result
            completed += 1
            _notify_progress(progress_callback, completed, len(jobs), result)
    final_results: list[dict[str, Any]] = []
    for result in results:
        if result is None:
            raise LocalScoringError(
                "parallel scoring ended with an unclassified missing result"
            )
        final_results.append(result)
    return final_results


def _write_report_atomic(destination: Path, report: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def score_seed31415_results(
    results_dir: str | Path,
    instances_dir: str | Path,
    tasks_dir: str | Path = "tasks",
    *,
    output_path: str | Path | None = None,
    judge_api_override: JudgeAPIOverride | None = None,
    parallel: int = 1,
    progress_callback: ScoreProgressCallback | None = None,
) -> tuple[dict[str, Any], Path]:
    """Score produce-only results against public seed31415 references.

    The source result JSON files are never modified. A separate report records
    local, non-official scores and full scorer details.
    """
    parallel = _validate_parallel(parallel)
    results_root = Path(results_dir).resolve()
    instances_root = Path(instances_dir).resolve()
    tasks_root = Path(tasks_dir).resolve()
    for label, path in (
        ("results", results_root),
        ("instances", instances_root),
        ("tasks", tasks_root),
    ):
        if not path.is_dir():
            raise LocalScoringError(f"{label} directory not found: {path}")

    result_files = list(_iter_result_files(results_root))
    if not result_files:
        raise LocalScoringError(
            f"No per-instance result JSON found under {results_root}. "
            "Run `asibench run ...` first."
        )

    effective_override = (
        judge_api_override
        if judge_api_override is not None
        else get_judge_api_override()
    )
    if effective_override is not None:
        # Validate the named credential once in the coordinator, before custom
        # scorers or network-backed Judges can start.
        effective_override.resolve_api_key()

    jobs = _prepare_score_jobs(
        result_files,
        results_root=results_root,
        instances_root=instances_root,
        tasks_root=tasks_root,
    )
    if parallel == 1:
        scored_results = []
        secret = (
            effective_override.resolve_api_key()
            if effective_override is not None
            else None
        )
        for completed, job in enumerate(jobs, start=1):
            with _redact_scoring_logs(secret):
                try:
                    result = _evaluate_score_job(
                        job,
                        effective_override,
                        preserve_ambient_override=True,
                    )
                except Exception as exc:
                    result = _worker_failure_result(
                        job,
                        error_type=type(exc).__name__,
                        error=_redact_secret(str(exc), secret),
                        failure_kind=getattr(
                            exc, "failure_kind", "evaluator_runtime_error"
                        ),
                        scorer_name="_local_scoring_runtime",
                    )
            result = _redact_secret(result, secret)
            scored_results.append(result)
            _notify_progress(progress_callback, completed, len(jobs), result)
    else:
        # Context variables do not propagate to spawned processes. Pass only
        # the credential-safe override metadata and scope it inside workers.
        scored_results = _score_jobs_parallel(
            jobs,
            parallel=parallel,
            judge_api_override=effective_override,
            progress_callback=progress_callback,
        )

    scorer_error_count = sum(
        int(item["evaluation_status"] == "evaluation_invalid")
        for item in scored_results
    )

    valid_results = [
        item for item in scored_results if item["evaluation_status"] == "completed"
    ]
    total_score = sum(float(item["final_score"]) for item in valid_results)
    total_max = sum(item["max_score"] for item in valid_results)
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": PUBLIC_LOCAL_SCORING_REPO,
        "official": False,
        "score_scope": "public_local",
        "instance_count": len(scored_results),
        "scored_instance_count": len(valid_results),
        "scorer_error_count": scorer_error_count,
        "total_score": total_score,
        "total_max_score": total_max,
        "mean_percent": (100.0 * total_score / total_max) if total_max else None,
        "results": scored_results,
    }
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else results_root / DEFAULT_LOCAL_SCORE_REPORT
    )
    _write_report_atomic(destination, report)
    return report, destination
