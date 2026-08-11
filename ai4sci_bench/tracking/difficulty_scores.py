"""Persistence helpers for `scores/<task_id>.json` difficulty-check history.

Schema (per file):

```
{
  "task_id": "physics.new_task",
  "task_version": "1.0",
  "evaluations": [
    {
      "date": "2026-05-29T14:30:00Z",
      "trigger": "manual" | "ci" | "re-evaluation",
      "trigger_ref": "pr#142",
      "threshold": 50,
      "verdict": "pass" | "fail",
      "tool_mode": "restricted",
      "sandbox": "task",
      "instances_per_task": 1,
      "seeds": [42],
      "results": [
        {
          "agent": "direct_llm",
          "agent_config": {"model": "claude-opus-4-6"},
          "scores": {
            "b1": {"mean": 35.0, "max": 38.2, "min": 31.8, "n": 1},
            "b3": {"mean": 12.0, "max": 15.5, "min": 8.5, "n": 1}
          }
        }
      ]
    }
  ]
}
```

The latest evaluation is always the last element of `evaluations`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_VALID_VERDICTS = {"pass", "fail"}
_VALID_TRIGGERS = {"manual", "ci", "re-evaluation"}


def score_file_path(scores_dir: Path | str, task_id: str) -> Path:
    """Return the canonical path for a task's score file."""
    return Path(scores_dir) / f"{task_id}.json"


def load_scores(scores_dir: Path | str, task_id: str) -> dict | None:
    """Load score history for a task. Returns ``None`` if no record exists."""
    path = score_file_path(scores_dir, task_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt score file at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Score file {path} must contain a JSON object, got {type(data).__name__}")
    return data


def append_evaluation(
    scores_dir: Path | str,
    task_id: str,
    task_version: str,
    results: list[dict[str, Any]],
    threshold: int,
    verdict: str,
    trigger: str = "manual",
    trigger_ref: str | None = None,
    tool_mode: str = "restricted",
    sandbox: str = "none",
    instances_per_task: int = 1,
    seeds: Iterable[int] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Append a new evaluation record to the task's score file.

    Creates the file (and parent directory) if missing. Returns the file path.
    """
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"Invalid verdict {verdict!r}; expected one of {sorted(_VALID_VERDICTS)}")
    if trigger not in _VALID_TRIGGERS:
        raise ValueError(f"Invalid trigger {trigger!r}; expected one of {sorted(_VALID_TRIGGERS)}")

    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    path = score_file_path(scores_dir, task_id)
    if path.exists():
        data = load_scores(scores_dir, task_id) or {}
    else:
        data = {}

    data.setdefault("task_id", task_id)
    data["task_version"] = task_version
    evaluations = data.setdefault("evaluations", [])

    record = {
        "date": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trigger": trigger,
        "trigger_ref": trigger_ref or "",
        "threshold": int(threshold),
        "verdict": verdict,
        "tool_mode": tool_mode,
        "sandbox": sandbox,
        "instances_per_task": int(instances_per_task),
        "seeds": list(seeds) if seeds is not None else [],
        "results": list(results),
    }
    evaluations.append(record)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def get_latest_verdict(scores_dir: Path | str, task_id: str) -> str | None:
    """Return the most recent verdict (``pass`` / ``fail``) or ``None`` if never evaluated."""
    data = load_scores(scores_dir, task_id)
    if not data:
        return None
    evaluations = data.get("evaluations") or []
    if not evaluations:
        return None
    return evaluations[-1].get("verdict")


def get_all_task_scores(scores_dir: Path | str) -> dict[str, dict]:
    """Load every ``*.json`` file in the scores directory keyed by task_id."""
    scores_dir = Path(scores_dir)
    if not scores_dir.exists():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(scores_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        task_id = data.get("task_id") or path.stem
        out[task_id] = data
    return out


def find_flagged_tasks(
    scores_dir: Path | str,
    tasks_dir: Path | str | None = None,
    threshold: int = 50,
    only_final: bool = True,
) -> list[dict[str, Any]]:
    """Identify tasks whose latest evaluation has any score >= threshold.

    When ``only_final`` is True and ``tasks_dir`` is provided, only tasks whose
    ``task.yaml`` declares ``status: final`` are considered. Tasks without a
    discoverable status are still included (defensive: we'd rather over-flag).
    """
    flagged: list[dict[str, Any]] = []
    statuses = _load_task_statuses(tasks_dir) if (only_final and tasks_dir) else None

    for task_id, data in get_all_task_scores(scores_dir).items():
        if statuses is not None:
            status = statuses.get(task_id)
            if status is not None and status != "final":
                continue

        evaluations = data.get("evaluations") or []
        if not evaluations:
            continue
        latest = evaluations[-1]
        for agent_block in latest.get("results", []):
            agent_label = agent_block.get("agent_config", {}).get("model") or agent_block.get("agent", "unknown")
            for level, scores in (agent_block.get("scores") or {}).items():
                mean = _coerce_float(scores.get("mean"))
                if mean is None:
                    continue
                if mean >= threshold:
                    flagged.append({
                        "task_id": task_id,
                        "model": agent_label,
                        "agent": agent_block.get("agent"),
                        "prompt_level": level,
                        "mean_score": mean,
                        "threshold": int(latest.get("threshold", threshold)),
                        "date": latest.get("date"),
                    })
    flagged.sort(key=lambda row: (-row["mean_score"], row["task_id"]))
    return flagged


def _load_task_statuses(tasks_dir: Path | str) -> dict[str, str]:
    """Best-effort scan of ``tasks_dir/*/*/task.yaml`` for {task_id: status}."""
    try:
        import yaml  # local import keeps tracking package importable without PyYAML
    except ImportError:
        return {}

    out: dict[str, str] = {}
    base = Path(tasks_dir)
    if not base.exists():
        return out
    for yaml_path in base.glob("*/*/task.yaml"):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        task_id = data.get("id")
        status = data.get("status")
        if task_id and status:
            out[str(task_id)] = str(status)
    return out


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
