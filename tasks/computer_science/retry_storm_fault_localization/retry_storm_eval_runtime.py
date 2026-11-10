"""Deterministic discrete-event simulator for the retry-storm task.

This module intentionally contains only generic queueing and retry dynamics.
Instance-specific fault labels, oracle changes, and held-out scenarios belong in
the generated reference directory, not here.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import heapq
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = 1
EVENT_PRIORITY = {
    "service_complete": 0,
    "rpc_hedge": 1,
    "rpc_timeout": 2,
    "root_deadline": 3,
    "attempt_arrival": 4,
    "root_arrival": 5,
}


class SimulationError(ValueError):
    """Raised for invalid simulation inputs or interventions."""


@dataclass
class Attempt:
    attempt_id: int
    root_id: int
    stage_index: int
    retry_index: int
    service: str
    policy: str | None
    launch_kind: str
    submitted_ms: float
    arrived_ms: float | None = None
    started_ms: float | None = None
    completed_ms: float | None = None
    state: str = "scheduled"


def stable_uniform(seed: int, *parts: Any) -> float:
    """Return a stable value in [0, 1), independent of process hash state."""
    payload = "|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def read_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SimulationError(f"{path} must contain a YAML mapping")
    return obj


def read_workload(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "start_ms", "end_ms", "rate_rps", "search_fraction",
        "database_fraction", "code_fraction",
    }
    if not rows or not required.issubset(rows[0]):
        raise SimulationError(f"workload is empty or missing columns: {sorted(required)}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        item = {key: float(row[key]) for key in required}
        if item["end_ms"] <= item["start_ms"] or item["rate_rps"] < 0:
            raise SimulationError("workload windows require end > start and rate >= 0")
        fraction_sum = sum(item[f"{name}_fraction"] for name in ("search", "database", "code"))
        if not math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise SimulationError("route fractions must sum to 1")
        parsed.append(item)
    return parsed


def _resolve_stage(stage: dict[str, Any], route: str) -> tuple[str, str | None]:
    service = stage.get("service")
    policy = stage.get("policy")
    if service is None:
        template = stage.get("service_template")
        if not template:
            raise SimulationError(f"workflow stage lacks service: {stage}")
        service = str(template).format(route=route)
    if policy is None and stage.get("policy_template"):
        policy = str(stage["policy_template"]).format(route=route)
    return str(service), (str(policy) if policy is not None else None)


def _service_time_ms(
    seed: int,
    service: str,
    root_id: int,
    stage_index: int,
    retry_index: int,
    service_cfg: dict[str, Any],
    scenario: dict[str, Any],
) -> float:
    median = float(service_cfg["service_time_ms"])
    sigma = float(service_cfg.get("variability", 0.20))
    # A deterministic Irwin-Hall approximation gives a smooth, bounded normal
    # variate while preserving exact reproducibility on every platform.
    z = sum(
        stable_uniform(seed, "svc", service, root_id, stage_index, retry_index, i)
        for i in range(12)
    ) - 6.0
    multiplier = math.exp(sigma * z)
    tail_p = float(service_cfg.get("tail_probability", 0.0))
    tail_probability_spec = scenario.get("service_tail_probability", {})
    if isinstance(tail_probability_spec, dict) and service in tail_probability_spec:
        tail_p = float(tail_probability_spec[service])
    if not 0.0 <= tail_p <= 1.0:
        raise SimulationError("service tail probabilities must be in [0, 1]")
    if stable_uniform(seed, "tail", service, root_id, stage_index, retry_index) < tail_p:
        tail_multiplier = float(service_cfg.get("tail_multiplier", 1.0))
        tail_multiplier_spec = scenario.get("service_tail_multiplier", {})
        if isinstance(tail_multiplier_spec, dict) and service in tail_multiplier_spec:
            tail_multiplier = float(tail_multiplier_spec[service])
        if tail_multiplier < 1.0:
            raise SimulationError("service tail multipliers must be at least 1")
        multiplier *= tail_multiplier
    scale_spec = scenario.get("service_time_scale", 1.0)
    if isinstance(scale_spec, dict):
        scale = float(scale_spec.get(service, 1.0))
    else:
        scale = float(scale_spec)
    return max(0.01, median * multiplier * scale)


def _service_fails(
    seed: int,
    service: str,
    root_id: int,
    stage_index: int,
    retry_index: int,
    service_cfg: dict[str, Any],
    scenario: dict[str, Any],
) -> bool:
    """Return a deterministic transient-failure outcome for one attempt."""
    probability = float(service_cfg.get("failure_probability", 0.0))
    scenario_spec = scenario.get("failure_probability")
    if isinstance(scenario_spec, dict) and service in scenario_spec:
        probability = float(scenario_spec[service])
    probability *= float(scenario.get("failure_probability_scale", 1.0))
    if not 0.0 <= probability <= 1.0:
        raise SimulationError("failure probabilities must be in [0, 1]")
    return stable_uniform(
        seed, "failure", service, root_id, stage_index, retry_index
    ) < probability


def expand_workload(
    rows: list[dict[str, Any]], seed: int, scenario: dict[str, Any]
) -> list[tuple[float, str]]:
    """Expand windowed workload rows into deterministic request arrivals."""
    arrivals: list[tuple[float, str]] = []
    rate_scale = float(scenario.get("rate_scale", 1.0))
    mix_override = scenario.get("route_fractions")
    for window_index, row in enumerate(rows):
        duration_s = (row["end_ms"] - row["start_ms"]) / 1000.0
        count = int(round(duration_s * row["rate_rps"] * rate_scale))
        if count <= 0:
            continue
        spacing = (row["end_ms"] - row["start_ms"]) / count
        if isinstance(mix_override, dict):
            fractions = {name: float(mix_override[name]) for name in ("search", "database", "code")}
        else:
            fractions = {name: float(row[f"{name}_fraction"]) for name in ("search", "database", "code")}
        if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-6):
            raise SimulationError("scenario route fractions must sum to 1")
        for local_index in range(count):
            jitter = (stable_uniform(seed, "arrival", window_index, local_index) - 0.5) * spacing * 0.60
            at = row["start_ms"] + (local_index + 0.5) * spacing + jitter
            selector = stable_uniform(seed, "route", window_index, local_index)
            if selector < fractions["search"]:
                route = "search"
            elif selector < fractions["search"] + fractions["database"]:
                route = "database"
            else:
                route = "code"
            arrivals.append((round(at, 6), route))
    arrivals.sort(key=lambda item: (item[0], item[1]))
    return arrivals


def validate_model(topology: dict[str, Any], runtime: dict[str, Any]) -> None:
    if int(topology.get("schema_version", -1)) != SCHEMA_VERSION:
        raise SimulationError("unsupported topology schema_version")
    services = runtime.get("services")
    policies = runtime.get("rpc_policies")
    stages = topology.get("workflow", {}).get("stages")
    if not isinstance(services, dict) or not services:
        raise SimulationError("runtime.services must be a non-empty mapping")
    if not isinstance(policies, dict) or not policies:
        raise SimulationError("runtime.rpc_policies must be a non-empty mapping")
    if not isinstance(stages, list) or not stages:
        raise SimulationError("topology.workflow.stages must be a non-empty list")
    for name, cfg in services.items():
        if int(cfg.get("workers", 0)) <= 0 or float(cfg.get("service_time_ms", 0)) <= 0:
            raise SimulationError(f"invalid worker/time configuration for {name}")
    for name, cfg in policies.items():
        if float(cfg.get("timeout_ms", 0)) <= 0:
            raise SimulationError(f"invalid timeout for {name}")
        if not 1 <= int(cfg.get("max_attempts", 0)) <= 8:
            raise SimulationError(f"invalid max_attempts for {name}")
        hedge_delay = cfg.get("hedge_delay_ms")
        if hedge_delay is not None and not 10.0 <= float(hedge_delay) <= 1500.0:
            raise SimulationError(f"invalid hedge_delay_ms for {name}")
        retry_budget_ratio = float(cfg.get("retry_budget_ratio", 7.0))
        if not 0.0 <= retry_budget_ratio <= 7.0:
            raise SimulationError(f"invalid retry_budget_ratio for {name}")
        retry_budget_min = cfg.get("retry_budget_min_retries", 0)
        if (
            isinstance(retry_budget_min, bool)
            or int(retry_budget_min) != retry_budget_min
            or not 0 <= int(retry_budget_min) <= 512
        ):
            raise SimulationError(f"invalid retry_budget_min_retries for {name}")
    for route in ("search", "database", "code"):
        for stage in stages:
            service, policy = _resolve_stage(stage, route)
            if service not in services:
                raise SimulationError(f"stage resolves to unknown service {service}")
            if policy is not None and policy not in policies:
                raise SimulationError(f"stage resolves to unknown policy {policy}")


def apply_remediation(
    runtime: dict[str, Any], remediation: dict[str, Any] | None
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Validate and apply the public intervention schema.

    Returns a deep-copied runtime, a normalized intervention cost, and the
    normalized list of changes.  The cost is intentionally generic; oracle cost
    and scoring thresholds remain evaluator-held.
    """
    updated = copy.deepcopy(runtime)
    if remediation is None:
        return updated, 0.0, []
    if not isinstance(remediation, dict) or int(remediation.get("schema_version", -1)) != 1:
        raise SimulationError("remediation requires schema_version: 1")
    changes = remediation.get("changes")
    if not isinstance(changes, list) or len(changes) > 4:
        raise SimulationError("remediation.changes must be a list with at most four entries")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    cost = 0.0
    for raw in changes:
        if not isinstance(raw, dict):
            raise SimulationError("every remediation change must be a mapping")
        target_type = str(raw.get("target_type", ""))
        target = str(raw.get("target", ""))
        parameter = str(raw.get("parameter", ""))
        value = raw.get("value")
        key = (target_type, target, parameter)
        if key in seen:
            raise SimulationError(f"duplicate remediation target: {key}")
        seen.add(key)
        if target_type == "rpc_policy":
            if target not in updated["rpc_policies"]:
                raise SimulationError(f"unknown RPC policy {target}")
            cfg = updated["rpc_policies"][target]
            old = cfg.get(parameter)
            if parameter == "timeout_ms":
                value = float(value)
                if not 25.0 <= value <= 1500.0:
                    raise SimulationError("timeout_ms must be in [25, 1500]")
                cost += 0.35 * abs(math.log(value / float(old)))
            elif parameter == "max_attempts":
                if isinstance(value, bool) or int(value) != value or not 1 <= int(value) <= 8:
                    raise SimulationError("max_attempts must be an integer in [1, 8]")
                value = int(value)
                cost += 0.22 * abs(value - int(old))
            elif parameter in {"backoff_ms", "jitter_ms"}:
                value = float(value)
                if not 0.0 <= value <= 750.0:
                    raise SimulationError(f"{parameter} must be in [0, 750]")
                cost += 0.002 * abs(value - float(old))
            elif parameter == "hedge_delay_ms":
                value = float(value)
                if not 10.0 <= value <= 1500.0:
                    raise SimulationError("hedge_delay_ms must be in [10, 1500]")
                if old is None:
                    cost += 0.75
                else:
                    cost += 0.003 * abs(value - float(old))
            elif parameter in {"cancel_on_success", "propagate_deadline"}:
                if not isinstance(value, bool):
                    raise SimulationError(f"{parameter} must be boolean")
                cost += 0.18 if value != bool(old) else 0.0
            else:
                raise SimulationError(f"parameter {parameter} is not allowed for RPC policies")
            cfg[parameter] = value
        elif target_type == "service":
            if target not in updated["services"] or parameter != "workers":
                raise SimulationError("service changes may only alter workers on known services")
            old = int(updated["services"][target]["workers"])
            if isinstance(value, bool) or int(value) != value:
                raise SimulationError("workers must be an integer")
            value = int(value)
            if not max(1, math.ceil(old * 0.5)) <= value <= math.floor(old * 2.0):
                raise SimulationError("workers must remain within [0.5x, 2x] of the observed value")
            # Capacity is valid but deliberately expensive: the scientific goal
            # is to find causal policy repairs rather than buy away every queue.
            cost += 2.5 * abs(value - old) / old
            updated["services"][target]["workers"] = value
        else:
            raise SimulationError("target_type must be rpc_policy or service")
        normalized.append({
            "target_type": target_type,
            "target": target,
            "parameter": parameter,
            "value": value,
        })
    return updated, round(cost, 9), normalized


def simulate(
    topology: dict[str, Any],
    runtime: dict[str, Any],
    workload: list[dict[str, Any]],
    *,
    seed: int,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the deterministic simulation and return summary and telemetry."""
    scenario = dict(scenario or {})
    validate_model(topology, runtime)
    arrivals = expand_workload(workload, seed, scenario)
    stages = topology["workflow"]["stages"]
    services = {
        name: dict(config) for name, config in runtime["services"].items()
    }
    worker_scale_spec = scenario.get("service_worker_scale", {})
    if worker_scale_spec is not None and not isinstance(worker_scale_spec, dict):
        raise SimulationError("service_worker_scale must be a mapping")
    for service, scale in (worker_scale_spec or {}).items():
        if service not in services:
            raise SimulationError(f"unknown service in service_worker_scale: {service}")
        scale = float(scale)
        if scale <= 0.0:
            raise SimulationError("service worker scales must be positive")
        services[service]["workers"] = max(
            1, int(round(float(services[service]["workers"]) * scale))
        )
    policies = runtime["rpc_policies"]
    sim_cfg = runtime.get("simulation", {})
    root_deadline = float(sim_cfg.get("root_deadline_ms", 2500.0)) * float(
        scenario.get("root_deadline_scale", 1.0)
    )
    bucket_ms = int(sim_cfg.get("metrics_bucket_ms", 2000))
    queue_limit = int(sim_cfg.get("queue_limit", 1024))

    service_state: dict[str, dict[str, Any]] = {
        name: {
            "busy": 0,
            "queue": deque(),
            "queue_last_ms": 0.0,
            "queue_last_depth": 0,
        }
        for name in services
    }
    roots: dict[int, dict[str, Any]] = {}
    attempts: dict[int, Attempt] = {}
    stage_resolved: set[tuple[int, int]] = set()
    submitted_stage_retries: set[tuple[int, int, int]] = set()
    event_heap: list[tuple[Any, ...]] = []
    event_seq = 0
    next_attempt_id = 0
    policy_retry_budgets: dict[str, dict[str, int]] = {
        name: {"initial_calls": 0, "submitted_retries": 0, "denied_retries": 0}
        for name in policies
    }
    service_stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "arrivals": 0, "starts": 0, "completions": 0, "errors": 0, "drops": 0,
            "timeouts": 0, "retries": 0, "hedges": 0, "late_completions": 0,
            "cancelled_attempts": 0, "busy_ms": 0.0, "queue_area": 0.0,
            "max_queue_depth": 0, "latencies": [], "queue_waits": [],
        }
    )
    rpc_stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "attempts": 0, "successes": 0, "failures": 0, "timeouts": 0,
            "retries": 0, "hedges": 0, "retry_budget_denials": 0,
            "late_completions": 0, "cancelled_attempts": 0, "latencies": [],
        }
    )
    traces: list[dict[str, Any]] = []
    logs: dict[str, list[str]] = defaultdict(list)

    def stat_for(service: str, at_ms: float) -> dict[str, Any]:
        return service_stats[(int(at_ms // bucket_ms), service)]

    def rpc_stat_for(policy: str, at_ms: float) -> dict[str, Any]:
        return rpc_stats[(int(at_ms // bucket_ms), policy)]

    def schedule(
        at_ms: float, kind: str, root_id: int, stage_index: int,
        retry_index: int, payload: Any,
    ) -> None:
        nonlocal event_seq
        event_seq += 1
        heapq.heappush(event_heap, (
            round(float(at_ms), 9), EVENT_PRIORITY[kind], int(root_id),
            int(stage_index), int(retry_index), event_seq, kind, payload,
        ))

    def add_interval(
        service: str, field: str, start_ms: float, end_ms: float, value: float
    ) -> None:
        cursor = max(0.0, start_ms)
        end_ms = max(cursor, end_ms)
        while cursor < end_ms - 1e-12:
            bucket = int(cursor // bucket_ms)
            boundary = min(end_ms, (bucket + 1) * bucket_ms)
            service_stats[(bucket, service)][field] += (boundary - cursor) * value
            cursor = boundary

    def accrue_queue(service: str, at_ms: float) -> None:
        state = service_state[service]
        add_interval(
            service, "queue_area", state["queue_last_ms"], at_ms,
            float(state["queue_last_depth"]),
        )
        state["queue_last_ms"] = at_ms
        state["queue_last_depth"] = len(state["queue"])

    def record_queue_depth(service: str, at_ms: float) -> None:
        accrue_queue(service, at_ms)
        stat_for(service, at_ms)["max_queue_depth"] = max(
            stat_for(service, at_ms)["max_queue_depth"], len(service_state[service]["queue"])
        )
        service_state[service]["queue_last_depth"] = len(service_state[service]["queue"])

    def submit_attempt(
        root_id: int, stage_index: int, retry_index: int, at_ms: float,
        *, launch_kind: str,
    ) -> None:
        nonlocal next_attempt_id
        if launch_kind not in {"initial", "retry", "hedge"}:
            raise RuntimeError(f"invalid attempt launch_kind: {launch_kind}")
        submission_key = (root_id, stage_index, retry_index)
        if submission_key in submitted_stage_retries:
            return
        root = roots[root_id]
        service, policy_name = _resolve_stage(stages[stage_index], root["route"])
        if launch_kind == "retry" and policy_name is not None:
            policy_cfg = policies[policy_name]
            budget = policy_retry_budgets[policy_name]
            ratio = float(policy_cfg.get("retry_budget_ratio", 7.0))
            minimum = int(policy_cfg.get("retry_budget_min_retries", 0))
            allowed = max(minimum, int(math.floor(budget["initial_calls"] * ratio)))
            if budget["submitted_retries"] >= allowed:
                submitted_stage_retries.add(submission_key)
                budget["denied_retries"] += 1
                rpc_stat_for(policy_name, at_ms)["retry_budget_denials"] += 1
                logs[service].append(
                    f"{at_ms:012.3f} WARN root={root_id} stage={stage_index} "
                    f"rpc={policy_name} event=retry_budget_exhausted "
                    f"allowed={allowed} submitted={budget['submitted_retries']}"
                )
                return
            budget["submitted_retries"] += 1
        submitted_stage_retries.add(submission_key)
        next_attempt_id += 1
        attempt = Attempt(
            attempt_id=next_attempt_id,
            root_id=root_id,
            stage_index=stage_index,
            retry_index=retry_index,
            service=service,
            policy=policy_name,
            launch_kind=launch_kind,
            submitted_ms=at_ms,
        )
        attempts[attempt.attempt_id] = attempt
        schedule(at_ms, "attempt_arrival", root_id, stage_index, retry_index, attempt.attempt_id)
        if policy_name is not None:
            policy_cfg = policies[policy_name]
            timeout_at = at_ms + float(policy_cfg["timeout_ms"])
            schedule(timeout_at, "rpc_timeout", root_id, stage_index, retry_index, attempt.attempt_id)
            hedge_delay = policy_cfg.get("hedge_delay_ms")
            if retry_index == 0 and hedge_delay is not None and int(policy_cfg["max_attempts"]) > 1:
                schedule(
                    at_ms + float(hedge_delay), "rpc_hedge", root_id,
                    stage_index, retry_index, attempt.attempt_id,
                )

    def try_start(service: str, at_ms: float) -> None:
        state = service_state[service]
        workers = int(services[service]["workers"])
        while state["busy"] < workers and state["queue"]:
            accrue_queue(service, at_ms)
            attempt_id = state["queue"].popleft()
            state["queue_last_depth"] = len(state["queue"])
            attempt = attempts[attempt_id]
            key = (attempt.root_id, attempt.stage_index)
            policy_cfg = policies.get(attempt.policy, {})
            root = roots[attempt.root_id]
            if attempt.state == "cancelled":
                continue
            if root["failed"] and bool(policy_cfg.get("propagate_deadline", True)):
                attempt.state = "cancelled"
                stat_for(service, at_ms)["cancelled_attempts"] += 1
                if attempt.policy:
                    rpc_stat_for(attempt.policy, at_ms)["cancelled_attempts"] += 1
                continue
            if key in stage_resolved and bool(policy_cfg.get("cancel_on_success", True)):
                attempt.state = "cancelled"
                stat_for(service, at_ms)["cancelled_attempts"] += 1
                if attempt.policy:
                    rpc_stat_for(attempt.policy, at_ms)["cancelled_attempts"] += 1
                continue
            attempt.state = "running"
            attempt.started_ms = at_ms
            state["busy"] += 1
            stat_for(service, at_ms)["starts"] += 1
            stat_for(service, at_ms)["queue_waits"].append(at_ms - float(attempt.arrived_ms))
            duration = _service_time_ms(
                seed, service, attempt.root_id, attempt.stage_index,
                attempt.retry_index, services[service], scenario,
            )
            schedule(
                at_ms + duration, "service_complete", attempt.root_id,
                attempt.stage_index, attempt.retry_index, attempt.attempt_id,
            )

    for root_id, (at_ms, route) in enumerate(arrivals):
        schedule(at_ms, "root_arrival", root_id, 0, 0, route)

    while event_heap:
        at_ms, _, root_id, stage_index, retry_index, _, kind, payload = heapq.heappop(event_heap)

        if kind == "root_arrival":
            route = str(payload)
            roots[root_id] = {
                "arrival_ms": at_ms,
                "deadline_ms": at_ms + root_deadline,
                "route": route,
                "completed": False,
                "failed": False,
                "completion_ms": None,
            }
            schedule(at_ms + root_deadline, "root_deadline", root_id, 0, 0, None)
            submit_attempt(root_id, 0, 0, at_ms, launch_kind="initial")
            continue

        root = roots[root_id]

        if kind == "root_deadline":
            if not root["completed"]:
                root["failed"] = True
                logs["gateway"].append(
                    f"{at_ms:012.3f} ERROR root={root_id} route={root['route']} "
                    f"event=deadline_exceeded deadline_ms={root_deadline:.1f}"
                )
                # A propagated root deadline cancels both queued and in-flight
                # downstream work. Running attempts release their worker at
                # the cancellation timestamp; their already-scheduled
                # completion event becomes a harmless tombstone. Policies
                # with propagation disabled deliberately keep consuming the
                # pool and can therefore create an orphan-work cascade.
                touched_services: set[str] = set()
                for pending in attempts.values():
                    if pending.root_id != root_id or pending.state not in {"queued", "running"}:
                        continue
                    pending_policy = policies.get(pending.policy, {})
                    if not bool(pending_policy.get("propagate_deadline", True)):
                        continue
                    pending_service = pending.service
                    if pending.state == "running":
                        if pending.started_ms is None:
                            raise RuntimeError("running attempt has no start time")
                        service_state[pending_service]["busy"] -= 1
                        add_interval(
                            pending_service, "busy_ms", pending.started_ms, at_ms, 1.0
                        )
                    pending.state = "cancelled"
                    pending.completed_ms = at_ms
                    stat_for(pending_service, at_ms)["cancelled_attempts"] += 1
                    if pending.policy:
                        rpc_stat_for(pending.policy, at_ms)["cancelled_attempts"] += 1
                    touched_services.add(pending_service)
                for touched_service in sorted(touched_services):
                    try_start(touched_service, at_ms)
            continue

        attempt = attempts[int(payload)]
        service = attempt.service
        key = (root_id, stage_index)
        policy_cfg = policies.get(attempt.policy, {})

        if kind == "rpc_hedge":
            if key in stage_resolved or root["failed"] or attempt.state in {"failed", "completed", "cancelled"}:
                continue
            assert attempt.policy is not None
            stat_for(service, at_ms)["hedges"] += 1
            rpc_stat_for(attempt.policy, at_ms)["hedges"] += 1
            logs[service].append(
                f"{at_ms:012.3f} INFO root={root_id} stage={stage_index} "
                f"rpc={attempt.policy} event=hedge_scheduled next_retry=1"
            )
            submit_attempt(root_id, stage_index, 1, at_ms, launch_kind="hedge")
            continue

        if kind == "attempt_arrival":
            attempt.arrived_ms = at_ms
            if key in stage_resolved and bool(policy_cfg.get("cancel_on_success", True)):
                attempt.state = "cancelled"
                stat_for(service, at_ms)["cancelled_attempts"] += 1
                if attempt.policy:
                    rpc_stat_for(attempt.policy, at_ms)["cancelled_attempts"] += 1
                continue
            if root["failed"] and bool(policy_cfg.get("propagate_deadline", True)):
                attempt.state = "cancelled"
                stat_for(service, at_ms)["cancelled_attempts"] += 1
                continue
            attempt.state = "queued"
            stat_for(service, at_ms)["arrivals"] += 1
            if attempt.policy:
                rpc_stat_for(attempt.policy, at_ms)["attempts"] += 1
                if attempt.launch_kind == "initial":
                    policy_retry_budgets[attempt.policy]["initial_calls"] += 1
                elif attempt.launch_kind == "retry":
                    rpc_stat_for(attempt.policy, at_ms)["retries"] += 1
            if attempt.launch_kind == "retry":
                stat_for(service, at_ms)["retries"] += 1
            state = service_state[service]
            if len(state["queue"]) >= queue_limit:
                attempt.state = "dropped"
                stat_for(service, at_ms)["drops"] += 1
                logs[service].append(
                    f"{at_ms:012.3f} ERROR root={root_id} stage={stage_index} "
                    f"retry={retry_index} event=queue_overflow depth={len(state['queue'])}"
                )
            else:
                accrue_queue(service, at_ms)
                state["queue"].append(attempt.attempt_id)
                state["queue_last_depth"] = len(state["queue"])
                stat_for(service, at_ms)["max_queue_depth"] = max(
                    stat_for(service, at_ms)["max_queue_depth"], len(state["queue"])
                )
                try_start(service, at_ms)
            continue

        if kind == "rpc_timeout":
            if key in stage_resolved or attempt.state in {"cancelled", "failed", "completed"}:
                continue
            stat_for(service, at_ms)["timeouts"] += 1
            assert attempt.policy is not None
            rpc_stat_for(attempt.policy, at_ms)["timeouts"] += 1
            logs[service].append(
                f"{at_ms:012.3f} WARN root={root_id} stage={stage_index} "
                f"rpc={attempt.policy} retry={retry_index} event=rpc_timeout "
                f"attempt_state={attempt.state} queue_depth={len(service_state[service]['queue'])}"
            )
            max_attempts = int(policy_cfg["max_attempts"])
            if retry_index + 1 < max_attempts and not root["failed"]:
                backoff = float(policy_cfg.get("backoff_ms", 0.0))
                jitter_ms = float(policy_cfg.get("jitter_ms", 0.0))
                jitter = stable_uniform(seed, "retry", root_id, stage_index, retry_index) * jitter_ms
                retry_at = at_ms + backoff + jitter
                logs[service].append(
                    f"{at_ms:012.3f} INFO root={root_id} stage={stage_index} "
                    f"rpc={attempt.policy} event=retry_scheduled next_retry={retry_index + 1} "
                    f"delay_ms={backoff + jitter:.3f}"
                )
                submit_attempt(
                    root_id, stage_index, retry_index + 1, retry_at,
                    launch_kind="retry",
                )
            continue

        if kind == "service_complete":
            if attempt.state == "cancelled":
                continue
            if attempt.state != "running" or attempt.started_ms is None:
                raise RuntimeError("completion event for non-running attempt")
            attempt.state = "completed"
            attempt.completed_ms = at_ms
            state = service_state[service]
            state["busy"] -= 1
            duration = at_ms - attempt.started_ms
            add_interval(service, "busy_ms", attempt.started_ms, at_ms, 1.0)
            stat_for(service, at_ms)["completions"] += 1
            stat_for(service, at_ms)["latencies"].append(duration)
            late = key in stage_resolved or root["failed"]
            failed = (
                not late
                and _service_fails(
                    seed, service, root_id, stage_index, retry_index,
                    services[service], scenario,
                )
            )
            status = (
                "post_deadline"
                if root["failed"]
                else "late_duplicate"
                if key in stage_resolved
                else "transient_error"
                if failed
                else "ok"
            )
            if late:
                stat_for(service, at_ms)["late_completions"] += 1
                if attempt.policy:
                    rpc_stat_for(attempt.policy, at_ms)["late_completions"] += 1
            elif failed:
                attempt.state = "failed"
                stat_for(service, at_ms)["errors"] += 1
                if attempt.policy:
                    rpc_stat_for(attempt.policy, at_ms)["failures"] += 1
                logs[service].append(
                    f"{at_ms:012.3f} ERROR root={root_id} stage={stage_index} "
                    f"retry={retry_index} event=transient_service_error"
                )
            elif attempt.policy:
                rpc_stat_for(attempt.policy, at_ms)["successes"] += 1
                rpc_stat_for(attempt.policy, at_ms)["latencies"].append(at_ms - attempt.submitted_ms)
            traces.append({
                "trace_id": f"root-{root_id:06d}",
                "span_id": f"attempt-{attempt.attempt_id:08d}",
                "root_id": root_id,
                "route": root["route"],
                "stage_index": stage_index,
                "stage": stages[stage_index]["name"],
                "service": service,
                "rpc_policy": attempt.policy,
                "retry_index": retry_index,
                "launch_kind": attempt.launch_kind,
                "submitted_ms": round(attempt.submitted_ms, 6),
                "start_ms": round(attempt.started_ms, 6),
                "end_ms": round(at_ms, 6),
                "queue_wait_ms": round(attempt.started_ms - float(attempt.arrived_ms), 6),
                "service_time_ms": round(duration, 6),
                "status": status,
            })
            if failed and attempt.policy and not root["failed"]:
                max_attempts = int(policy_cfg["max_attempts"])
                if retry_index + 1 < max_attempts:
                    backoff = float(policy_cfg.get("backoff_ms", 0.0))
                    jitter_ms = float(policy_cfg.get("jitter_ms", 0.0))
                    jitter = stable_uniform(
                        seed, "failure_retry", root_id, stage_index, retry_index
                    ) * jitter_ms
                    submit_attempt(
                        root_id, stage_index, retry_index + 1,
                        at_ms + backoff + jitter,
                        launch_kind="retry",
                    )
            elif not late:
                stage_resolved.add(key)
                if stage_index + 1 < len(stages):
                    submit_attempt(
                        root_id, stage_index + 1, 0, at_ms,
                        launch_kind="initial",
                    )
                else:
                    root["completed"] = True
                    root["completion_ms"] = at_ms
            try_start(service, at_ms)
            continue

        raise RuntimeError(f"unhandled event kind {kind}")

    final_ms = max(
        [float(row["end_ms"]) for row in workload] +
        [float(root["deadline_ms"]) for root in roots.values()] + [0.0]
    )
    for service in services:
        accrue_queue(service, final_ms)

    metric_rows: list[dict[str, Any]] = []
    max_bucket = int(math.ceil(final_ms / bucket_ms))
    for bucket in range(max_bucket):
        for service in sorted(services):
            stat = service_stats[(bucket, service)]
            workers = int(services[service]["workers"])
            row = {
                "window_start_ms": bucket * bucket_ms,
                "window_end_ms": (bucket + 1) * bucket_ms,
                "service": service,
                "workers": workers,
                "arrivals": stat["arrivals"],
                "starts": stat["starts"],
                "completions": stat["completions"],
                "errors": stat["errors"],
                "drops": stat["drops"],
                "timeouts": stat["timeouts"],
                "retries": stat["retries"],
                "hedges": stat["hedges"],
                "late_completions": stat["late_completions"],
                "cancelled_attempts": stat["cancelled_attempts"],
                "mean_queue_depth": round(stat["queue_area"] / bucket_ms, 6),
                "max_queue_depth": stat["max_queue_depth"],
                "utilization": round(min(1.0, stat["busy_ms"] / (workers * bucket_ms)), 6),
                "service_p50_ms": round(percentile(stat["latencies"], 0.50), 6),
                "service_p95_ms": round(percentile(stat["latencies"], 0.95), 6),
                "queue_wait_p95_ms": round(percentile(stat["queue_waits"], 0.95), 6),
            }
            metric_rows.append(row)

    rpc_rows: list[dict[str, Any]] = []
    for bucket in range(max_bucket):
        for policy in sorted(policies):
            stat = rpc_stats[(bucket, policy)]
            attempts_count = stat["attempts"]
            rpc_rows.append({
                "window_start_ms": bucket * bucket_ms,
                "window_end_ms": (bucket + 1) * bucket_ms,
                "rpc_policy": policy,
                "source": policies[policy]["source"],
                "destination": policies[policy]["destination"],
                "attempts": attempts_count,
                "successes": stat["successes"],
                "failures": stat["failures"],
                "timeouts": stat["timeouts"],
                "retries": stat["retries"],
                "hedges": stat["hedges"],
                "retry_budget_denials": stat["retry_budget_denials"],
                "late_completions": stat["late_completions"],
                "cancelled_attempts": stat["cancelled_attempts"],
                "timeout_rate": round(stat["timeouts"] / attempts_count, 6) if attempts_count else 0.0,
                "rpc_p50_ms": round(percentile(stat["latencies"], 0.50), 6),
                "rpc_p95_ms": round(percentile(stat["latencies"], 0.95), 6),
            })

    completed_latencies = [
        root["completion_ms"] - root["arrival_ms"]
        for root in roots.values() if root["completed"] and root["completion_ms"] is not None
    ]
    completed = sum(1 for root in roots.values() if root["completed"])
    total = len(roots)
    total_attempts = len(attempts)
    logical_stage_attempts = len({(item.root_id, item.stage_index) for item in attempts.values()})
    retry_count = sum(row["retries"] for row in rpc_rows)
    hedge_count = sum(row["hedges"] for row in rpc_rows)
    tool_route_retry_amplification: dict[str, float] = {}
    for route in ("search", "database", "code"):
        policy = f"router_to_{route}"
        selected = [row for row in rpc_rows if row["rpc_policy"] == policy]
        route_attempts = sum(int(row["attempts"]) for row in selected)
        route_retries = sum(int(row["retries"]) for row in selected)
        route_hedges = sum(int(row["hedges"]) for row in selected)
        # Retries and hedges are extra physical attempts for existing logical
        # calls.  Per-route measurement prevents unrelated reductions in model
        # work from hiding a local retry storm in the global average.
        logical_route_calls = route_attempts - route_retries - route_hedges
        route_amplification = (
            1.0 + route_retries / logical_route_calls
            if logical_route_calls > 0 else 1.0
        )
        tool_route_retry_amplification[route] = round(route_amplification, 9)
    post_deadline_completions = sum(
        item["status"] == "post_deadline" for item in traces
    )
    late_duplicate_completions = sum(
        item["status"] == "late_duplicate" for item in traces
    )
    peak_utilization: dict[str, float] = {}
    peak_queue: dict[str, int] = {}
    for service in services:
        selected = [row for row in metric_rows if row["service"] == service]
        peak_utilization[service] = max((float(row["utilization"]) for row in selected), default=0.0)
        peak_queue[service] = max((int(row["max_queue_depth"]) for row in selected), default=0)
    summary = {
        "schema_version": 1,
        "scenario": str(scenario.get("name", "public_observed")),
        "seed": int(seed),
        "total_requests": total,
        "completed_requests": completed,
        "deadline_misses": total - completed,
        "success_rate": round(completed / total, 9) if total else 1.0,
        "e2e_p50_ms": round(percentile(completed_latencies, 0.50), 6),
        "e2e_p95_ms": round(percentile(completed_latencies, 0.95), 6),
        "e2e_p99_ms": round(percentile(completed_latencies, 0.99), 6),
        "total_attempts": total_attempts,
        "logical_stage_attempts": logical_stage_attempts,
        "total_retries": retry_count,
        "total_hedges": hedge_count,
        "total_retry_budget_denials": sum(
            row["retry_budget_denials"] for row in rpc_rows
        ),
        "total_transient_failures": sum(row["failures"] for row in rpc_rows),
        "post_deadline_completions": post_deadline_completions,
        "late_duplicate_completions": late_duplicate_completions,
        "retry_amplification": round(total_attempts / logical_stage_attempts, 9) if logical_stage_attempts else 1.0,
        "tool_route_retry_amplification": tool_route_retry_amplification,
        "max_tool_route_retry_amplification": max(
            tool_route_retry_amplification.values(), default=1.0
        ),
        "peak_service_utilization": peak_utilization,
        "peak_queue_depth": peak_queue,
    }
    return {
        "summary": summary,
        "metrics": metric_rows,
        "rpc_metrics": rpc_rows,
        "traces": sorted(traces, key=lambda item: (item["start_ms"], item["root_id"], item["stage_index"], item["retry_index"])),
        "logs": {name: sorted(lines) for name, lines in sorted(logs.items())},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_simulation(result: dict[str, Any], output_dir: Path, *, full_telemetry: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "metrics.csv", result["metrics"])
    _write_csv(output_dir / "rpc_metrics.csv", result["rpc_metrics"])
    if full_telemetry:
        with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
            for span in result["traces"]:
                handle.write(json.dumps(span, sort_keys=True) + "\n")
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        for service, lines in result["logs"].items():
            (logs_dir / f"{service}.log").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_from_files(
    *,
    topology_path: Path,
    runtime_path: Path,
    workload_path: Path,
    output_dir: Path,
    seed: int,
    scenario: dict[str, Any] | None = None,
    remediation_path: Path | None = None,
    full_telemetry: bool = False,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    topology = read_yaml(topology_path)
    runtime = read_yaml(runtime_path)
    workload = read_workload(workload_path)
    remediation = read_yaml(remediation_path) if remediation_path is not None else None
    runtime, cost, changes = apply_remediation(runtime, remediation)
    result = simulate(topology, runtime, workload, seed=seed, scenario=scenario)
    result["summary"]["intervention_cost"] = cost
    result["summary"]["applied_changes"] = changes
    write_simulation(result, output_dir, full_telemetry=full_telemetry)
    return result, cost, changes
