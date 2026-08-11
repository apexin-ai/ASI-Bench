"""Repository boundary for official benchmark prompts."""

from __future__ import annotations

import json
from pathlib import Path


def test_repository_contains_only_declared_full_sample_prompts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tasks_root = repo_root / "tasks"
    policy = json.loads(
        (repo_root / "config" / "public_examples.json").read_text(encoding="utf-8")
    )
    full_task_ids = {
        entry["task_id"] for entry in policy["examples"] if entry["kind"] == "full"
    }
    expected_paths = {
        tasks_root / task_id.replace(".", "/") / f"prompt_{band}.md"
        for task_id in full_task_ids
        for band in ("b1", "b2", "b3", "b4")
    }
    assert set(tasks_root.rglob("prompt_*.md")) == expected_paths
