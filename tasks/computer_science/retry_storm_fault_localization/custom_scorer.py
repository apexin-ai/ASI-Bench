"""Deterministic artifact gate and counterfactual scorer for this task."""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from retry_storm_eval_runtime import (  # noqa: E402
    SimulationError,
    apply_remediation,
    read_workload,
    read_yaml,
    simulate,
)


REQUIRED_FILES = {
    "diagnosis.json": 300_000,
    "remediation.yaml": 100_000,
    "counterfactuals.json": 500_000,
    "analysis.py": 300_000,
    "results/summary.json": 300_000,
    "results/counterfactual_metrics.csv": 4_000_000,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def _instance_data_dir(pred_dir: Path, ref_dir: Path) -> Path:
    # During GT self-check data/ is copied into pred_dir.  Normal evaluation
    # likewise scores an agent workspace containing data/.
    candidate = pred_dir / "data"
    if candidate.is_dir():
        return candidate
    fallback = ref_dir.parent / "data"
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError("generated instance data directory is unavailable")


def _load_candidate(pred_dir: Path, ref_dir: Path) -> dict[str, Any]:
    diagnosis = _load_json(pred_dir / "diagnosis.json")
    counterfactuals = _load_json(pred_dir / "counterfactuals.json")
    remediation = yaml.safe_load((pred_dir / "remediation.yaml").read_text(encoding="utf-8"))
    submitted_summary = _load_json(pred_dir / "results" / "summary.json")
    if not all(
        isinstance(item, dict)
        for item in (diagnosis, remediation, counterfactuals, submitted_summary)
    ):
        raise ValueError("diagnosis, remediation, counterfactuals, and summary must be mappings")
    if not all(
        _finite_tree(item)
        for item in (diagnosis, remediation, counterfactuals, submitted_summary)
    ):
        raise ValueError("structured outputs contain non-finite or unsupported values")
    data_dir = _instance_data_dir(pred_dir, ref_dir)
    topology = read_yaml(data_dir / "topology.yaml")
    runtime = read_yaml(data_dir / "configs" / "runtime.yaml")
    workload = read_workload(data_dir / "workload.csv")
    updated_runtime, intervention_cost, normalized_changes = apply_remediation(runtime, remediation)
    return {
        "diagnosis": diagnosis,
        "counterfactuals": counterfactuals,
        "remediation": remediation,
        "submitted_summary": submitted_summary,
        "topology": topology,
        "runtime": runtime,
        "updated_runtime": updated_runtime,
        "workload": workload,
        "data_dir": data_dir,
        "intervention_cost": intervention_cost,
        "normalized_changes": normalized_changes,
    }


@register_scorer("retry_storm_artifact_gate")
class RetryStormArtifactGate(Scorer):
    """Reject missing, unsafe-sized, unparsable, or out-of-contract artifacts."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        failures: list[str] = []
        for relative, size_limit in REQUIRED_FILES.items():
            path = pred_dir / relative
            if not path.is_file():
                failures.append(f"missing {relative}")
            elif path.stat().st_size == 0:
                failures.append(f"empty {relative}")
            elif path.stat().st_size > size_limit:
                failures.append(f"oversized {relative}")
        if not failures:
            try:
                candidate = _load_candidate(pred_dir, ref_dir)
                diagnosis = candidate["diagnosis"]
                required_diagnosis = {
                    "schema_version", "faults", "causal_graph", "evidence", "baseline",
                }
                missing = required_diagnosis - set(diagnosis)
                if missing:
                    failures.append(f"diagnosis missing fields {sorted(missing)}")
                if diagnosis.get("schema_version") != 2:
                    failures.append("diagnosis.schema_version must equal 2")
                faults = diagnosis.get("faults")
                if not isinstance(faults, list) or not 1 <= len(faults) <= 4:
                    failures.append("diagnosis.faults must contain one to four records")
                else:
                    fault_fields = {
                        "fault_id", "fault_type", "culprit_service", "culprit_edge",
                        "culprit_rpc_policy", "parameter",
                    }
                    for index, fault in enumerate(faults):
                        if not isinstance(fault, dict) or not fault_fields.issubset(fault):
                            failures.append(f"diagnosis.faults[{index}] has the wrong schema")
                if not isinstance(diagnosis.get("causal_graph"), list):
                    failures.append("diagnosis.causal_graph must be a list")
                if not isinstance(diagnosis.get("evidence"), list):
                    failures.append("diagnosis.evidence must be a list")
                if not isinstance(diagnosis.get("baseline"), dict):
                    failures.append("diagnosis.baseline must be a mapping")
                counterfactuals = candidate["counterfactuals"]
                probes = counterfactuals.get("probes")
                if counterfactuals.get("schema_version") != 1:
                    failures.append("counterfactuals.schema_version must equal 1")
                if not isinstance(probes, list) or not 1 <= len(probes) <= 12:
                    failures.append("counterfactuals.probes must contain one to twelve records")
                else:
                    probe_fields = {"probe_id", "kind", "seed", "scenario", "summary"}
                    for index, probe in enumerate(probes):
                        if not isinstance(probe, dict) or not probe_fields.issubset(probe):
                            failures.append(f"counterfactuals.probes[{index}] has the wrong schema")
                with (pred_dir / "results" / "counterfactual_metrics.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    reader = csv.DictReader(handle)
                    expected_columns = {
                        "window_start_ms", "window_end_ms", "service", "workers",
                        "arrivals", "starts", "completions", "drops", "timeouts",
                        "retries", "hedges", "late_completions", "cancelled_attempts",
                        "mean_queue_depth", "max_queue_depth", "utilization",
                        "service_p50_ms", "service_p95_ms", "queue_wait_p95_ms",
                    }
                    if reader.fieldnames is None or not expected_columns.issubset(reader.fieldnames):
                        failures.append("counterfactual_metrics.csv has the wrong columns")
                    elif next(reader, None) is None:
                        failures.append("counterfactual_metrics.csv has no data rows")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, yaml.YAMLError, SimulationError) as exc:
                failures.append(f"artifact contract error: {type(exc).__name__}: {exc}")
        passed = not failures
        return ScoreDetail(
            scorer_name="retry_storm_artifact_gate",
            score=1.0 if passed else 0.0,
            max_score=1.0,
            passed=passed,
            details={"failures": failures},
            message="" if passed else "; ".join(failures[:5]),
        )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _improvement_utility(
    candidate: float, baseline: float, target: float, *, higher_is_better: bool
) -> float:
    """Measure continuous recovery from the faulty baseline to the public target."""
    if higher_is_better:
        if baseline < target:
            return _clip01((candidate - baseline) / max(target - baseline, 1e-9))
        return 1.0 if candidate >= target else _clip01(candidate / max(target, 1e-9))
    if baseline > target:
        return _clip01((baseline - candidate) / max(baseline - target, 1e-9))
    return 1.0 if candidate <= target else _clip01(target / max(candidate, 1e-9))


def _scenario_observables(summary: dict[str, Any]) -> dict[str, float]:
    queues = summary["peak_queue_depth"]
    return {
        "success_rate": float(summary["success_rate"]),
        "e2e_p95_ms": float(summary["e2e_p95_ms"]),
        "retry_amplification": float(summary["retry_amplification"]),
        "max_tool_route_retry_amplification": float(
            summary["max_tool_route_retry_amplification"]
        ),
        "peak_tool_queue_depth": max(
            float(queues[f"{route}_tool"])
            for route in ("search", "database", "code")
        ),
        "peak_model_queue_depth": float(queues["model_backend"]),
        "post_deadline_completions": float(summary["post_deadline_completions"]),
    }


def _scenario_quality(
    summary: dict[str, Any], baseline: dict[str, Any], slo: dict[str, float]
) -> tuple[float, bool, dict[str, float]]:
    actual = _scenario_observables(summary)
    starting = _scenario_observables(baseline)
    targets = {
        "success_rate": float(slo["success_rate_min"]),
        "e2e_p95_ms": float(slo["e2e_p95_ms_max"]),
        "retry_amplification": float(slo["retry_amplification_max"]),
        "max_tool_route_retry_amplification": float(
            slo["max_tool_route_retry_amplification_max"]
        ),
        "peak_tool_queue_depth": float(slo["peak_tool_queue_depth_max"]),
        "peak_model_queue_depth": float(slo["peak_model_queue_depth_max"]),
        "post_deadline_completions": float(slo["post_deadline_completions_max"]),
    }
    weights = {
        "success_rate": 0.32,
        "e2e_p95_ms": 0.10,
        "retry_amplification": 0.08,
        "max_tool_route_retry_amplification": 0.12,
        "peak_tool_queue_depth": 0.11,
        "peak_model_queue_depth": 0.12,
        "post_deadline_completions": 0.15,
    }
    utilities = {
        name: _improvement_utility(
            actual[name], starting[name], targets[name],
            higher_is_better=(name == "success_rate"),
        )
        for name in targets
    }
    quality = sum(weights[name] * utilities[name] for name in weights)
    passed = (
        actual["success_rate"] >= targets["success_rate"]
        and actual["e2e_p95_ms"] <= targets["e2e_p95_ms"]
        and actual["retry_amplification"] <= targets["retry_amplification"]
        and actual["max_tool_route_retry_amplification"]
        <= targets["max_tool_route_retry_amplification"]
        and actual["peak_tool_queue_depth"] <= targets["peak_tool_queue_depth"]
        and actual["peak_model_queue_depth"] <= targets["peak_model_queue_depth"]
        and actual["post_deadline_completions"]
        <= targets["post_deadline_completions"]
    )
    return quality, passed, utilities


def _lcs_length(left: list[str], right: list[str]) -> int:
    dp = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for j, expected in enumerate(right, start=1):
            saved = dp[j]
            if item == expected:
                dp[j] = previous + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            previous = saved
    return dp[-1]


def _fault_signature(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item.get(field, ""))
        for field in (
            "fault_type", "culprit_service", "culprit_edge",
            "culprit_rpc_policy", "parameter",
        )
    )


def _score_localization(
    diagnosis: dict[str, Any], truth: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    actual_faults = diagnosis.get("faults", [])
    if not isinstance(actual_faults, list):
        actual_faults = []
    truth_items = list(truth["faults"])
    actual_items = [
        item for item in actual_faults[:len(truth_items)]
        if isinstance(item, dict)
    ]
    fields = (
        ("fault_type", 0.30),
        ("culprit_service", 0.10),
        ("culprit_edge", 0.10),
        ("culprit_rpc_policy", 0.30),
        ("parameter", 0.20),
    )
    best_accuracy = 0.0
    best_exact = 0
    padded: list[dict[str, Any] | None] = actual_items + [None] * (
        len(truth_items) - len(actual_items)
    )
    for assignment in itertools.permutations(padded, len(truth_items)):
        accuracy = 0.0
        exact = 0
        for predicted, expected in zip(assignment, truth_items):
            if predicted is None:
                continue
            pair_accuracy = sum(
                weight for field, weight in fields
                if str(predicted.get(field, "")) == str(expected[field])
            )
            accuracy += pair_accuracy
            exact += int(pair_accuracy >= 1.0 - 1e-9)
        if (accuracy, exact) > (best_accuracy, best_exact):
            best_accuracy, best_exact = accuracy, exact
    field_accuracy = best_accuracy / len(truth_items)
    exact_fraction = best_exact / len(truth_items)
    # Exact compound identities carry most of the scientific localization
    # credit.  Field-level similarity remains continuous, but cannot make a
    # set of plausible-looking partial guesses equivalent to root-cause
    # recovery.
    exact_identity_score = 20.0 * exact_fraction**2
    partial_field_score = 3.0 * field_accuracy

    numeric_score = 0.0
    actual_baseline = diagnosis.get("baseline", {})
    truth_baseline = truth["baseline"]
    if isinstance(actual_baseline, dict):
        if actual_baseline.get("bottleneck_service") == truth_baseline["bottleneck_service"]:
            numeric_score += 0.5
        for field, tolerance, weight in (
            ("retry_amplification", 0.02, 0.5),
            ("peak_bottleneck_utilization", 0.035, 0.5),
            ("peak_bottleneck_queue_depth", 8.0, 0.5),
        ):
            try:
                error = abs(float(actual_baseline[field]) - float(truth_baseline[field]))
                numeric_score += weight * _clip01(1.0 - error / tolerance)
            except (KeyError, TypeError, ValueError):
                pass
    return exact_identity_score + partial_field_score + numeric_score, {
        "matched_faults": best_exact,
        "required_faults": len(truth_items),
        "exact_fault_fraction": exact_fraction,
        "fault_field_accuracy": field_accuracy,
        "exact_identity_score": exact_identity_score,
        "partial_field_score": partial_field_score,
        "baseline_score": numeric_score,
    }


def _score_mechanism(
    diagnosis: dict[str, Any], truth: dict[str, Any], anchors: list[dict[str, Any]],
    data_dir: Path,
) -> tuple[float, dict[str, float | int]]:
    graph = diagnosis.get("causal_graph", [])
    if not isinstance(graph, list):
        graph = []
    actual_edges = {
        (str(item.get("source", "")), str(item.get("target", "")))
        for item in graph[:16] if isinstance(item, dict)
    }
    expected_edges = {
        (str(item["source"]), str(item["target"]))
        for item in truth["causal_graph"]
    }
    matched_edges = len(actual_edges & expected_edges)
    graph_precision = matched_edges / len(actual_edges) if actual_edges else 0.0
    graph_recall = matched_edges / len(expected_edges) if expected_edges else 0.0
    graph_f1 = (
        2.0 * graph_precision * graph_recall / (graph_precision + graph_recall)
        if graph_precision + graph_recall else 0.0
    )
    graph_score = 12.0 * graph_f1

    evidence = diagnosis.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    public_rows: dict[str, list[dict[str, str]]] = {}
    for source, relative in (
        ("service_metrics", "metrics.csv"),
        ("rpc_metrics", "rpc_metrics.csv"),
    ):
        with (data_dir / relative).open(newline="", encoding="utf-8") as handle:
            public_rows[source] = list(csv.DictReader(handle))

    def is_grounded(item: dict[str, Any]) -> bool:
        source = str(item.get("source", ""))
        metric = str(item.get("metric", ""))
        target = str(item.get("target", ""))
        target_field = "service" if source == "service_metrics" else "rpc_policy"
        if source not in public_rows or not metric:
            return False
        try:
            window = int(item.get("window_start_ms"))
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return False
        for row in public_rows[source]:
            if row.get(target_field) != target or int(float(row["window_start_ms"])) != window:
                continue
            try:
                if abs(float(row[metric]) - value) <= 1e-6:
                    return True
            except (KeyError, TypeError, ValueError):
                return False
        return False

    submitted = [item for item in evidence[:12] if isinstance(item, dict)]
    grounded = [item for item in submitted if is_grounded(item)]
    grounding_precision = len(grounded) / len(submitted) if submitted else 0.0
    covered_requirements = 0
    for anchor in anchors:
        if any(
            item.get("source") == anchor["source"]
            and item.get("metric") == anchor["metric"]
            and item.get("target") == anchor["target"]
            for item in grounded
        ):
            covered_requirements += 1
    evidence_coverage = covered_requirements / len(anchors) if anchors else 0.0
    # Correctly copied rows are useful only to the extent that they cover the
    # mechanisms under investigation. This prevents irrelevant-but-grounded
    # telemetry from substituting for evidential coverage.
    evidence_score = (
        3.0 * grounding_precision * evidence_coverage
        + 5.0 * evidence_coverage
    )
    return graph_score + evidence_score, {
        "matched_causal_edges": matched_edges,
        "required_causal_edges": len(expected_edges),
        "causal_graph_precision": graph_precision,
        "causal_graph_recall": graph_recall,
        "causal_graph_score": graph_score,
        "grounded_evidence_rows": len(grounded),
        "submitted_evidence_rows": len(submitted),
        "evidence_grounding_precision": grounding_precision,
        "matched_evidence_anchors": covered_requirements,
        "required_evidence_anchors": len(anchors),
        "grounded_evidence_score": evidence_score,
    }


def _compare_public_replay(
    pred_dir: Path, submitted: dict[str, Any], replay: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    summary_fields = {
        "total_requests": 0.0,
        "completed_requests": 0.0,
        "success_rate": 1e-7,
        "e2e_p95_ms": 1e-4,
        "retry_amplification": 1e-7,
        "max_tool_route_retry_amplification": 1e-7,
        "post_deadline_completions": 0.0,
        "intervention_cost": 1e-7,
    }
    matched_summary = 0
    for field, tolerance in summary_fields.items():
        try:
            expected = float(replay["summary"][field])
            actual = float(submitted[field])
            if abs(actual - expected) <= tolerance:
                matched_summary += 1
        except (KeyError, TypeError, ValueError):
            pass
    summary_score = 3.0 * matched_summary / len(summary_fields)

    metrics_score = 0.0
    matched_metric_fields = 0
    total_metric_fields = 0
    actual_path = pred_dir / "results" / "counterfactual_metrics.csv"
    try:
        with actual_path.open(newline="", encoding="utf-8") as handle:
            actual_rows = list(csv.DictReader(handle))
        expected_rows = replay["metrics"]
        if len(actual_rows) == len(expected_rows):
            for actual_row, expected_row in zip(actual_rows, expected_rows):
                for field, expected in expected_row.items():
                    total_metric_fields += 1
                    actual = actual_row.get(field)
                    if field == "service":
                        matched_metric_fields += int(actual == str(expected))
                    else:
                        try:
                            matched_metric_fields += int(
                                abs(float(actual) - float(expected)) <= 1e-5
                            )
                        except (TypeError, ValueError):
                            pass
            metrics_score = 2.0 * matched_metric_fields / total_metric_fields
    except (OSError, csv.Error):
        pass
    return summary_score + metrics_score, {
        "summary_fields_matched": matched_summary,
        "summary_fields_total": len(summary_fields),
        "metric_fields_matched": matched_metric_fields,
        "metric_fields_total": total_metric_fields,
        "metrics_score": metrics_score,
    }


def _score_counterfactual_necessity(
    candidate: dict[str, Any], private: dict[str, Any], root_cause_accuracy: float,
) -> tuple[float, dict[str, Any]]:
    """Reward only agent-declared, reproducible intervention probes."""
    changes = list(candidate["normalized_changes"])
    if not changes:
        return 0.0, {
            "validated_probes": [], "probe_replay_precision": 0.0,
            "necessary_fraction": 0.0, "full_repair_probe_valid": False,
            "covered_full_repair_shift_families": [],
            "required_full_repair_shift_families": [],
            "full_repair_shift_coverage_complete": False,
        }

    contract = _load_json(candidate["data_dir"] / "evaluation_contract.json")
    envelope = contract["hidden_shift_envelope"]
    probe_requirements = contract["counterfactual_probe_requirements"]
    required_shift_families = set(
        probe_requirements["full_repair_shift_families"]
    )

    def in_range(value: Any, bounds: list[float]) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return float(bounds[0]) <= number <= float(bounds[1])

    def valid_scenario(scenario: Any) -> bool:
        if not isinstance(scenario, dict):
            return False
        allowed = {
            "rate_scale", "root_deadline_scale", "route_fractions",
            "service_time_scale", "failure_probability",
            "service_worker_scale", "service_tail_probability",
            "service_tail_multiplier",
        }
        if not set(scenario).issubset(allowed):
            return False
        if "rate_scale" in scenario and not in_range(scenario["rate_scale"], envelope["rate_scale"]):
            return False
        if "root_deadline_scale" in scenario and not in_range(
            scenario["root_deadline_scale"], envelope["root_deadline_scale"]
        ):
            return False
        fractions = scenario.get("route_fractions")
        if fractions is not None:
            if not isinstance(fractions, dict) or set(fractions) != {"search", "database", "code"}:
                return False
            if not all(in_range(value, envelope["route_fraction"]) for value in fractions.values()):
                return False
            if not math.isclose(sum(float(value) for value in fractions.values()), 1.0, abs_tol=1e-6):
                return False
        scales = scenario.get("service_time_scale", {})
        if not isinstance(scales, dict):
            return False
        for service, value in scales.items():
            if service.endswith("_tool"):
                bounds = envelope["tool_service_time_scale"]
            elif service == "planner":
                bounds = envelope["planner_service_time_scale"]
            elif service == "model_backend":
                bounds = envelope["model_service_time_scale"]
            else:
                return False
            if not in_range(value, bounds):
                return False
        failures = scenario.get("failure_probability", {})
        if not isinstance(failures, dict):
            return False
        if any(
            not service.endswith("_tool")
            or not in_range(value, envelope["tool_failure_probability"])
            for service, value in failures.items()
        ):
            return False
        worker_values = scenario.get("service_worker_scale", {})
        if not isinstance(worker_values, dict):
            return False
        for service, value in worker_values.items():
            if service == "model_backend":
                bounds = envelope["model_worker_scale"]
            elif service.endswith("_tool"):
                bounds = envelope["tool_worker_scale"]
            else:
                return False
            if not in_range(value, bounds):
                return False
        for scenario_key, envelope_key in (
            ("service_tail_probability", "model_tail_probability"),
            ("service_tail_multiplier", "model_tail_multiplier"),
        ):
            values = scenario.get(scenario_key, {})
            if not isinstance(values, dict):
                return False
            if any(
                service != "model_backend"
                or not in_range(value, envelope[envelope_key])
                for service, value in values.items()
            ):
                return False
        return True

    def shift_families(scenario: dict[str, Any]) -> set[str]:
        families: set[str] = set()
        service_scales = scenario.get("service_time_scale", {})
        failures = scenario.get("failure_probability", {})
        if (
            "route_fractions" in scenario
            or any(str(service).endswith("_tool") for service in service_scales)
            or any(str(service).endswith("_tool") for service in failures)
        ):
            families.add("tool_route")
        if all(
            isinstance(scenario.get(field), dict)
            and "model_backend" in scenario[field]
            for field in (
                "service_worker_scale",
                "service_tail_probability",
                "service_tail_multiplier",
            )
        ) and (
            float(scenario["service_worker_scale"]["model_backend"])
            > float(envelope["model_worker_scale"][0])
            and float(scenario["service_tail_probability"]["model_backend"])
            > float(envelope["model_tail_probability"][0])
            and float(scenario["service_tail_multiplier"]["model_backend"])
            > float(envelope["model_tail_multiplier"][0])
        ):
            families.add("model_tail_capacity")
        try:
            if float(scenario.get("root_deadline_scale", 1.0)) < 1.0:
                families.add("deadline")
        except (TypeError, ValueError):
            pass
        return families

    submitted = candidate["counterfactuals"].get("probes", [])
    best_by_change = [0.0] * len(changes)
    full_probe_valid = False
    covered_full_repair_shift_families: set[str] = set()
    entries: list[dict[str, Any]] = []
    validated_count = 0
    for raw_probe in submitted[:12]:
        entry: dict[str, Any] = {
            "probe_id": raw_probe.get("probe_id") if isinstance(raw_probe, dict) else None,
            "replayed": False,
            "necessity": 0.0,
        }
        entries.append(entry)
        if not isinstance(raw_probe, dict) or not valid_scenario(raw_probe.get("scenario")):
            continue
        seed = raw_probe.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
            continue
        kind = raw_probe.get("kind")
        if kind == "full_repair":
            probe_runtime = candidate["updated_runtime"]
            omitted_index = None
        elif kind == "leave_one_out":
            omitted_index = raw_probe.get("omitted_change_index")
            if (
                isinstance(omitted_index, bool)
                or not isinstance(omitted_index, int)
                or not 0 <= omitted_index < len(changes)
            ):
                continue
            ablation = {
                "schema_version": 1,
                "changes": [
                    change for index, change in enumerate(changes)
                    if index != omitted_index
                ],
            }
            probe_runtime, _, _ = apply_remediation(candidate["runtime"], ablation)
        else:
            continue
        scenario = dict(raw_probe["scenario"])
        replay = simulate(
            candidate["topology"], probe_runtime, candidate["workload"],
            seed=seed, scenario=scenario,
        )["summary"]
        actual = _scenario_observables(replay)
        claimed = raw_probe.get("summary")
        if not isinstance(claimed, dict):
            continue
        tolerances = {
            "success_rate": 1e-7, "e2e_p95_ms": 1e-4,
            "retry_amplification": 1e-7,
            "max_tool_route_retry_amplification": 1e-7,
            "peak_tool_queue_depth": 0.0,
            "peak_model_queue_depth": 0.0, "post_deadline_completions": 0.0,
        }
        try:
            replayed = all(
                abs(float(claimed[name]) - actual[name]) <= tolerance
                for name, tolerance in tolerances.items()
            )
        except (KeyError, TypeError, ValueError):
            replayed = False
        if not replayed:
            continue
        validated_count += 1
        entry["replayed"] = True
        baseline = simulate(
            candidate["topology"], candidate["runtime"], candidate["workload"],
            seed=seed, scenario=scenario,
        )["summary"]
        probe_quality, probe_passed, _ = _scenario_quality(
            replay, baseline, private["slo"]
        )
        entry["probe_slo_passed"] = probe_passed
        if kind == "full_repair":
            full_probe_valid = full_probe_valid or probe_passed
            if probe_passed:
                families = shift_families(scenario)
                entry["covered_shift_families"] = sorted(families)
                covered_full_repair_shift_families.update(families)
            continue
        full = simulate(
            candidate["topology"], candidate["updated_runtime"], candidate["workload"],
            seed=seed, scenario=scenario,
        )["summary"]
        full_quality, full_passed, _ = _scenario_quality(full, baseline, private["slo"])
        delta = full_quality - probe_quality
        loses_validity = full_passed and not probe_passed
        necessity = 1.0 if loses_validity else _clip01(delta / 0.15)
        entry.update({
            "omitted_change_index": omitted_index,
            "quality_drop_when_omitted": delta,
            "full_repair_slo_passed": full_passed,
            "loses_validity": loses_validity,
            "necessity": necessity,
        })
        best_by_change[omitted_index] = max(best_by_change[omitted_index], necessity)

    precision = validated_count / len(submitted) if submitted else 0.0
    necessary_fraction = sum(best_by_change) / len(best_by_change)
    completeness = 0.85 * necessary_fraction + 0.15 * float(full_probe_valid)
    shift_coverage_complete = required_shift_families.issubset(
        covered_full_repair_shift_families
    )
    # Reproducible ablations demonstrate that submitted changes matter, but
    # they do not by themselves prove the claimed initiating faults.  Split
    # that credit between intervention necessity and attribution accuracy.
    attribution_weight = _clip01(root_cause_accuracy) ** 2
    score = (
        10.0 * completeness * precision * attribution_weight
        * float(shift_coverage_complete)
    )
    return score, {
        "validated_probes": entries,
        "probe_replay_precision": precision,
        "necessary_fraction": necessary_fraction,
        "full_repair_probe_valid": full_probe_valid,
        "covered_full_repair_shift_families": sorted(
            covered_full_repair_shift_families
        ),
        "required_full_repair_shift_families": sorted(required_shift_families),
        "full_repair_shift_coverage_complete": shift_coverage_complete,
        "root_cause_attribution_weight": attribution_weight,
        "counterfactual_probe_score": score,
    }


@register_scorer("retry_storm_score")
class RetryStormScorer(Scorer):
    """Score diagnosis and the submitted intervention by independent replay."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        try:
            candidate = _load_candidate(pred_dir, ref_dir)
            private = _load_json(ref_dir / "evaluation_private.json")
            if not isinstance(private, dict):
                raise ValueError("invalid evaluator reference")

            scenario_results: list[dict[str, Any]] = []
            qualities: list[float] = []
            passes: list[bool] = []
            for index, raw_scenario in enumerate(private["hidden_scenarios"]):
                scenario = dict(raw_scenario)
                scenario_seed = int(scenario.pop("seed"))
                result = simulate(
                    candidate["topology"], candidate["updated_runtime"],
                    candidate["workload"], seed=scenario_seed, scenario=scenario,
                )
                quality, passed, metric_utilities = _scenario_quality(
                    result["summary"], private["hidden_baselines"][index], private["slo"]
                )
                qualities.append(quality)
                passes.append(passed)
                observables = _scenario_observables(result["summary"])
                scenario_results.append({
                    "name": result["summary"]["scenario"],
                    **observables,
                    "slo_passed": passed,
                    "quality": quality,
                    "metric_utilities": metric_utilities,
                })
            localization_score, localization_parts = _score_localization(
                candidate["diagnosis"], private["truth"]
            )
            observational_score, mechanism_parts = _score_mechanism(
                candidate["diagnosis"], private["truth"],
                private["evidence_anchors"], candidate["data_dir"],
            )

            exact_fraction = float(localization_parts["exact_fault_fraction"])
            field_accuracy = float(localization_parts["fault_field_accuracy"])
            root_cause_accuracy = _clip01(
                0.75 * exact_fraction + 0.25 * field_accuracy
            )
            graph_recall = float(mechanism_parts["causal_graph_recall"])
            evidence_coverage = float(
                mechanism_parts["matched_evidence_anchors"]
            ) / max(float(mechanism_parts["required_evidence_anchors"]), 1.0)
            causal_attribution = root_cause_accuracy * math.sqrt(
                _clip01(graph_recall) * _clip01(evidence_coverage)
            )
            scenario_quality_by_name = {
                result["name"]: result["quality"] for result in scenario_results
            }
            scenario_pass_by_name = {
                result["name"]: result["slo_passed"] for result in scenario_results
            }
            capability_groups = []
            assigned_scenarios: set[str] = set()
            for group in private["hidden_capability_groups"]:
                members = list(group["scenarios"])
                if not members or any(
                    member not in scenario_quality_by_name for member in members
                ):
                    raise ValueError("invalid hidden capability group")
                if assigned_scenarios.intersection(members):
                    raise ValueError("hidden scenarios appear in multiple capability groups")
                assigned_scenarios.update(members)
                capability_groups.append({
                    "name": group["name"],
                    "scenarios": members,
                    "mean_quality": sum(
                        scenario_quality_by_name[member] for member in members
                    ) / len(members),
                    "all_slos_passed": all(
                        scenario_pass_by_name[member] for member in members
                    ),
                })
            if assigned_scenarios != set(scenario_quality_by_name):
                raise ValueError("hidden capability groups do not partition scenarios")
            mean_quality = sum(
                group["mean_quality"] for group in capability_groups
            ) / len(capability_groups)
            average_recovery_credit = 10.0 * mean_quality
            full_envelope_credit = 20.0 * float(all(passes))
            raw_behavior_score = average_recovery_credit + full_envelope_credit
            performance_score = raw_behavior_score * causal_attribution

            necessity_score, necessity_parts = _score_counterfactual_necessity(
                candidate, private, root_cause_accuracy
            )
            mechanism_score = observational_score + necessity_score
            mechanism_parts["counterfactual_necessity_score"] = necessity_score
            mechanism_parts.update(necessity_parts)

            behavioral_validity = _clip01((mean_quality - 0.55) / 0.45)
            oracle_cost = max(float(private["oracle_intervention_cost"]), 1e-9)
            excess_ratio = max(
                0.0, float(candidate["intervention_cost"]) / oracle_cost - 1.0
            )
            cost_efficiency = math.exp(-2.5 * excess_ratio)
            minimality_score = 10.0 * behavioral_validity * cost_efficiency

            public_scenario = dict(private["public_replay_scenario"])
            public = simulate(
                candidate["topology"], candidate["updated_runtime"],
                candidate["workload"], seed=int(private["public_replay_seed"]),
                scenario=public_scenario,
            )
            public["summary"]["intervention_cost"] = candidate["intervention_cost"]
            public["summary"]["applied_changes"] = candidate["normalized_changes"]
            replay_score, replay_parts = _compare_public_replay(
                pred_dir, candidate["submitted_summary"], public
            )

            raw_score = (
                performance_score + localization_score + mechanism_score
                + minimality_score + replay_score
            )
            score = raw_score * weight / 100.0
            content_status = {
                "all_hidden_slos_passed": all(passes),
                "all_faults_exact": (
                    localization_parts["matched_faults"]
                    == localization_parts["required_faults"]
                ),
                "causal_graph_recall_sufficient": (
                    mechanism_parts["causal_graph_recall"] >= 0.8
                ),
                "evidence_coverage_sufficient": (
                    mechanism_parts["matched_evidence_anchors"]
                    >= mechanism_parts["required_evidence_anchors"]
                ),
                "all_changes_causally_necessary": (
                    mechanism_parts["necessary_fraction"] >= 0.95
                ),
                "counterfactual_probes_replay_complete": (
                    mechanism_parts["probe_replay_precision"] >= 0.95
                    and mechanism_parts["full_repair_probe_valid"]
                    and mechanism_parts["full_repair_shift_coverage_complete"]
                ),
                "intervention_cost_reasonable": (
                    candidate["intervention_cost"] <= 1.5 * oracle_cost
                ),
                "public_replay_complete": replay_score >= 5.0 - 1e-6,
            }
            solution_valid = all(content_status.values())
            details = {
                "components": {
                    "causally_attributed_recovery": performance_score,
                    "compound_fault_localization": localization_score,
                    "causal_graph_and_evidence": mechanism_score,
                    "minimal_intervention": minimality_score,
                    "artifact_replay_consistency": replay_score,
                },
                "localization": localization_parts,
                "mechanism": mechanism_parts,
                "replay": replay_parts,
                "intervention_cost": candidate["intervention_cost"],
                "oracle_intervention_cost": private["oracle_intervention_cost"],
                "behavioral_validity": behavioral_validity,
                "cost_efficiency": cost_efficiency,
                "raw_behavior_score_before_causal_integration": raw_behavior_score,
                "average_recovery_credit_before_causal_integration": (
                    average_recovery_credit
                ),
                "full_envelope_credit_before_causal_integration": (
                    full_envelope_credit
                ),
                "mean_capability_group_quality": mean_quality,
                "root_cause_accuracy": root_cause_accuracy,
                "causal_attribution_factor": causal_attribution,
                "hidden_scenarios": scenario_results,
                "hidden_capability_groups": capability_groups,
                "component_maxima": {
                    "causally_attributed_recovery": 30.0,
                    "compound_fault_localization": 25.0,
                    "causal_graph_evidence_and_probes": 30.0,
                    "minimal_intervention": 10.0,
                    "artifact_replay_consistency": 5.0,
                },
                "scoring_mode": "robust_envelope_10_20_compound_causal_no_caps",
                "content_status": content_status,
                "solution_valid": solution_valid,
            }
            return ScoreDetail(
                scorer_name="retry_storm_score",
                score=score,
                max_score=weight,
                passed=solution_valid,
                details=details,
                message=(
                    f"{sum(passes)}/{len(passes)} held-out scenarios satisfy all SLOs; "
                    f"intervention cost={candidate['intervention_cost']:.3f}"
                ),
            )
        except Exception as exc:
            return ScoreDetail(
                scorer_name="retry_storm_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"candidate_error": f"{type(exc).__name__}: {exc}"},
                message=f"candidate could not be evaluated: {type(exc).__name__}: {exc}",
            )
