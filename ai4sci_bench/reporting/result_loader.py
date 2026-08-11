"""Shared helpers for loading and grouping saved evaluation results."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4sci_bench.core.result_schema import ensure_supported_result_schema, looks_like_eval_result_json
from ai4sci_bench.core.types import AgentOutput, AnalysisReport, EvalResult, PromptLevel, RunStatus, ScoreDetail
from ai4sci_bench.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResultGroup:
    """A display-labeled group of evaluation results."""

    key: str
    label: str
    results: list[EvalResult]
    provenance: AgentProvenance
    group_root: Path | None = None


@dataclass(frozen=True)
class AgentProvenance:
    """Normalized agent/model provenance used by reports and batch records."""

    agent_label: str
    agent_name: str
    adapter_class: str
    model_name: str
    method_group: str


_ADAPTER_TO_AGENT_NAME = {
    "DirectLLMAdapter": "direct_llm",
    "CodexCLIAdapter": "codex_cli",
    "ClaudeCodeCLIAdapter": "claude_code_cli",
    "CLIAgentAdapter": "agent_cmd",
}


def derive_agent_label(agent_name: str, agent_config: dict[str, Any]) -> str:
    """Derive a short directory-safe label from agent name + config."""
    if agent_name == "direct_llm":
        model = agent_config.get("model", "unknown")
        parts = model.split("/")
        if len(parts) >= 3 and parts[0] == "openrouter":
            return f"openrouter_{parts[-1]}"
        return parts[-1]
    if agent_name in ("codex_cli", "claude_code_cli"):
        parts = [agent_name]
        model = agent_config.get("model", "")
        if model:
            parts.append(model)
        effort = agent_config.get("effort", "")
        if effort:
            parts.append(effort)
        return "_".join(parts)
    return agent_name


def extract_agent_provenance(
    agent_metadata: dict[str, Any],
    *,
    agent_label_override: str | None = None,
) -> AgentProvenance:
    """Normalize persisted agent metadata into stable comparison fields."""
    raw_agent_name = agent_metadata.get("agent_name")
    raw_adapter_class = agent_metadata.get("adapter_class")
    agent_config = agent_metadata.get("config")

    agent_name = raw_agent_name.strip() if isinstance(raw_agent_name, str) else ""
    adapter_class = raw_adapter_class.strip() if isinstance(raw_adapter_class, str) else ""
    if not adapter_class and agent_name in _ADAPTER_TO_AGENT_NAME:
        adapter_class = agent_name
        agent_name = _ADAPTER_TO_AGENT_NAME[adapter_class]
    elif not agent_name and adapter_class in _ADAPTER_TO_AGENT_NAME:
        agent_name = _ADAPTER_TO_AGENT_NAME[adapter_class]

    if not agent_name:
        agent_name = "unknown"
    if not adapter_class:
        adapter_class = "unknown"

    normalized_config = agent_config if isinstance(agent_config, dict) else {}
    model_name = normalized_config.get("model")
    if not isinstance(model_name, str):
        model_name = ""

    derived_label = (
        derive_agent_label(agent_name, normalized_config)
        if agent_name != "unknown"
        else (derive_agent_label("direct_llm", {"model": model_name}) if model_name else adapter_class)
    )
    if not derived_label:
        derived_label = "unknown"

    method_group = agent_name
    if method_group == "unknown" and adapter_class in _ADAPTER_TO_AGENT_NAME:
        method_group = _ADAPTER_TO_AGENT_NAME[adapter_class]

    return AgentProvenance(
        agent_label=agent_label_override or derived_label,
        agent_name=agent_name,
        adapter_class=adapter_class,
        model_name=model_name,
        method_group=method_group,
    )


def is_better_result(candidate: EvalResult, existing: EvalResult) -> bool:
    """Return True if candidate should replace existing for the same run key."""
    return (
        candidate.status == RunStatus.COMPLETED,
        candidate.final_score,
        -candidate.execution_time_seconds,
    ) > (
        existing.status == RunStatus.COMPLETED,
        existing.final_score,
        -existing.execution_time_seconds,
    )


def dedupe_results(results: list[EvalResult]) -> list[EvalResult]:
    """Keep the best retry attempt per (instance_id, prompt_level)."""
    deduped: dict[str, EvalResult] = {}
    for result in results:
        key = f"{result.instance_id}__{result.prompt_level.value}"
        existing = deduped.get(key)
        if existing is None or is_better_result(result, existing):
            deduped[key] = result
    return list(deduped.values())


def is_eval_result_json(data: dict[str, Any]) -> bool:
    """Return whether a JSON blob matches the saved eval-result schema."""
    return looks_like_eval_result_json(data)


def parse_eval_result(data: dict[str, Any]) -> EvalResult:
    """Parse an EvalResult from a JSON dict."""
    ensure_supported_result_schema(data)

    gate_results = [
        ScoreDetail(
            scorer_name=r["scorer_name"],
            score=r["score"],
            max_score=r["max_score"],
            passed=r["passed"],
            details=r.get("details", {}),
            message=r.get("message", ""),
            severity=r.get("severity"),
        )
        for r in data.get("gate_results", [])
    ]
    score_results = [
        ScoreDetail(
            scorer_name=r["scorer_name"],
            score=r["score"],
            max_score=r["max_score"],
            passed=r["passed"],
            details=r.get("details", {}),
            message=r.get("message", ""),
        )
        for r in data.get("score_results", [])
    ]

    error_analysis = None
    if "error_analysis" in data and data["error_analysis"]:
        ea = data["error_analysis"]
        error_analysis = AnalysisReport(
            instance_id=data["instance_id"],
            error_category=ea.get("error_category", "unknown"),
            error_subcategory=ea.get("error_subcategory", "unknown"),
            root_cause=ea.get("root_cause", ""),
            evidence=ea.get("evidence", []),
            fix_suggestions=ea.get("fix_suggestions", []),
            raw_analysis="",
            confidence=ea.get("confidence", 0.0),
        )

    agent_output = None
    if "agent_output" in data and data["agent_output"]:
        ao = data["agent_output"]
        agent_output = AgentOutput(
            instance_id=data["instance_id"],
            output_dir=Path("."),
            code_files=ao.get("code_files", []),
            data_files=ao.get("data_files", []),
            log=ao.get("log", ""),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            status=RunStatus(ao.get("status", data.get("status", "completed"))),
            error_message=ao.get("error_message"),
            raw_stdout_format=ao.get("raw_stdout_format"),
            raw_stdout_file=ao.get("raw_stdout_file"),
            raw_stderr_file=ao.get("raw_stderr_file"),
            raw_model_output_format=ao.get("raw_model_output_format"),
            raw_model_output_file=ao.get("raw_model_output_file"),
        )
        if ao.get("trajectory_summary"):
            agent_output._trajectory_summary = ao["trajectory_summary"]

    return EvalResult(
        instance_id=data["instance_id"],
        task_id=data["task_id"],
        prompt_level=PromptLevel(data.get("prompt_level", "b2")),
        agent_name=data.get("agent_name", "unknown"),
        parameters=data.get("parameters", {}),
        gate_results=gate_results,
        gates_passed=data.get("gates_passed", False),
        hard_gates_passed=data.get("hard_gates_passed"),
        soft_gate_failures=data.get("soft_gate_failures", 0),
        score_results=score_results,
        final_score=data.get("final_score", 0.0),
        max_possible_score=data.get("max_possible_score", 100.0),
        execution_time_seconds=data.get("execution_time_seconds", 0.0),
        status=RunStatus(data.get("status", "completed")),
        error_analysis=error_analysis,
        agent_output=agent_output,
    )


_GROUP_KEY_EXCLUDE = frozenset({
    "api_key", "api_key_env", "api_base", "api_protocol", "provider",
})


def _strip_runtime_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only API credential and routing fields for stable grouping."""
    cleaned = {}
    for k, v in metadata.items():
        if k in _GROUP_KEY_EXCLUDE:
            continue
        if k == "config" and isinstance(v, dict):
            cleaned[k] = {ck: cv for ck, cv in v.items() if ck not in _GROUP_KEY_EXCLUDE}
        else:
            cleaned[k] = v
    return cleaned


def _report_group_from_agent_metadata(agent_metadata: dict[str, Any]) -> tuple[str, str]:
    """Build a stable grouping key and display label from saved agent provenance."""
    stable = _strip_runtime_config(agent_metadata)
    key = f"provenance:{json.dumps(stable, sort_keys=True, separators=(',', ':'))}"
    provenance = extract_agent_provenance(agent_metadata)
    return key, provenance.agent_label


def _result_provenance_from_data(
    data: dict[str, Any],
    result: EvalResult,
    *,
    agent_label_override: str | None = None,
) -> AgentProvenance:
    """Recover normalized agent provenance from result JSON or legacy fields."""
    provenance = data.get("provenance")
    if isinstance(provenance, dict):
        agent_metadata = provenance.get("agent")
        if isinstance(agent_metadata, dict):
            return extract_agent_provenance(
                agent_metadata,
                agent_label_override=agent_label_override,
            )

    return extract_agent_provenance(
        {
            "agent_name": result.agent_name,
            "adapter_class": result.agent_name if result.agent_name in _ADAPTER_TO_AGENT_NAME else None,
        },
        agent_label_override=agent_label_override,
    )


def _run_group_root(results_path: Path, json_file: Path) -> Path | None:
    """Return the top-level run root when scanning a batch parent directory."""
    try:
        relative_parts = json_file.relative_to(results_path).parts
    except ValueError:
        relative_parts = json_file.parts

    if len(relative_parts) < 2:
        return None

    candidate = results_path / relative_parts[0]
    return candidate if (candidate / "run_metadata.json").exists() else None


def _load_run_group_provenance(run_root: Path, *, agent_label: str) -> AgentProvenance | None:
    """Read run-level provenance from ``run_metadata.json`` when available."""
    run_metadata_path = run_root / "run_metadata.json"
    try:
        data = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    agent_metadata = data.get("agent_config")
    if not isinstance(agent_metadata, dict):
        return None
    return extract_agent_provenance(agent_metadata, agent_label_override=agent_label)


def report_group_for_result(
    *,
    results_path: Path,
    json_file: Path,
    data: dict[str, Any],
    result: EvalResult,
) -> tuple[str, str]:
    """Return a stable group key and display label for report bucketing."""
    run_root = _run_group_root(results_path, json_file)
    if run_root is not None:
        label = run_root.name
        return f"path:{label}", label

    provenance = data.get("provenance")
    if isinstance(provenance, dict):
        agent_metadata = provenance.get("agent")
        if isinstance(agent_metadata, dict):
            return _report_group_from_agent_metadata(agent_metadata)

    provenance = _result_provenance_from_data(data, result)
    return f"agent_name:{provenance.agent_name}", provenance.agent_label


_SKIP_DIR_NAMES = frozenset({
    "batch_records",
    "_workspaces",
    "reference",
    "__pycache__",
    "agent_api_cassettes",
    "scorer_api_cassettes",
    "recovery_archive",
    "reproducibility",
})


def _iter_result_json_files(results_path: Path) -> Iterator[Path]:
    """Yield JSON candidates without descending into non-result directories."""
    for dirpath, dirnames, filenames in os.walk(results_path, topdown=True):
        dirnames[:] = [
            dirname for dirname in dirnames if dirname not in _SKIP_DIR_NAMES
        ]
        for filename in filenames:
            if filename.endswith(".json"):
                yield Path(dirpath) / filename


def load_grouped_results(results_path: Path) -> list[ResultGroup]:
    """Load and group eval results using the same semantics as `report`."""
    grouped_results: dict[str, list[EvalResult]] = {}
    group_labels: dict[str, str] = {}
    group_provenance: dict[str, AgentProvenance] = {}
    group_roots: dict[str, Path | None] = {}
    run_provenance_cache: dict[Path, AgentProvenance | None] = {}

    for json_file in _iter_result_json_files(results_path):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if not is_eval_result_json(data):
                continue
            ensure_supported_result_schema(data)
            result = parse_eval_result(data)
            group_key, display_label = report_group_for_result(
                results_path=results_path,
                json_file=json_file,
                data=data,
                result=result,
            )
            grouped_results.setdefault(group_key, []).append(result)
            group_labels.setdefault(group_key, display_label)
            if group_key not in group_provenance:
                group_root = _run_group_root(results_path, json_file)
                group_roots[group_key] = group_root
                if group_root is not None:
                    if group_root not in run_provenance_cache:
                        run_provenance_cache[group_root] = _load_run_group_provenance(
                            group_root,
                            agent_label=display_label,
                        )
                    provenance = run_provenance_cache[group_root]
                else:
                    provenance = None

                if provenance is None:
                    provenance = _result_provenance_from_data(
                        data,
                        result,
                        agent_label_override=display_label,
                    )
                group_provenance[group_key] = provenance
        except ValueError as exc:
            logger.warning("Skipping unsupported result file %s: %s", json_file, exc)
            continue
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSON file %s: %s", json_file, exc)
            continue
        except Exception as exc:
            logger.warning("Unexpected error reading result file %s: %s", json_file, exc)
            continue

    ordered_groups = sorted(grouped_results.items(), key=lambda item: group_labels[item[0]])
    return [
        ResultGroup(
            key=group_key,
            label=group_labels[group_key],
            results=dedupe_results(results),
            provenance=group_provenance[group_key],
            group_root=group_roots.get(group_key),
        )
        for group_key, results in ordered_groups
    ]


def merge_result_groups(groups_list: list[list[ResultGroup]]) -> list[ResultGroup]:
    """Merge ResultGroups from multiple directories, deduping across dirs.

    Results with the same (instance_id, prompt_level) across directories
    are deduplicated keeping the highest-scoring entry.
    """
    merged: dict[str, list[EvalResult]] = {}
    labels: dict[str, str] = {}
    provenances: dict[str, AgentProvenance] = {}
    roots: dict[str, Path | None] = {}

    for groups in groups_list:
        for group in groups:
            merged.setdefault(group.key, []).extend(group.results)
            labels.setdefault(group.key, group.label)
            if group.key not in provenances:
                provenances[group.key] = group.provenance
                roots[group.key] = group.group_root

    ordered = sorted(merged.items(), key=lambda item: labels[item[0]])
    return [
        ResultGroup(
            key=key,
            label=labels[key],
            results=dedupe_results(results),
            provenance=provenances[key],
            group_root=roots.get(key),
        )
        for key, results in ordered
    ]
