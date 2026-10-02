"""Conventions every MCP E2E fake task meets, parametrized over examples/mcp-e2e-tasks: a new task
is covered by these without writing them again (its own test_task_<name>.py adds the specifics)."""
import importlib
import json

import pytest
import yaml

from ai4sci_bench.core.task import TaskLoader

from . import support
from .support import E2E_TASKS, ROOT

TASKS = support.all_tasks()
IDS = [task.name for task in TASKS]
MANIFEST = support.setup.load_manifest()
verify = support.verify
# Topics each task's own test module must cover (matched against its test function names).
REQUIRED_TOPICS = {"generator determinism": "deterministic", "submission failures score zero": "submission",
                   "evaluator failures are flagged": "evaluator_failure", "a genuine run passes": "genuine"}


def test_tasks_are_test_status_and_only_discovered_from_their_own_dir():
    assert TASKS
    loader = TaskLoader(E2E_TASKS)
    assert sorted(t["id"] for t in loader.discover_tasks(include_test=True)) == sorted(t.task_id for t in TASKS)
    assert loader.discover_tasks() == []
    formal = TaskLoader(ROOT / "tasks").discover_tasks(include_test=True, include_sample=True, include_dev=True)
    assert not {t["id"] for t in formal} & {t.task_id for t in TASKS}


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_meta_is_test_status(task):
    assert yaml.safe_load((task.dir / "task_meta.yaml").read_text())["status"] == "test"


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_prompts_are_agent_neutral(task):
    # Harness-specific tool names (Claude's mcp__<server>__<tool>) primed agents to look for a
    # tool-loading step and give up; prompts name the server and the tool only.
    for level in ("b1", "b2", "b3", "b4"):
        text = (task.dir / f"prompt_{level}.md").read_text()
        assert "result.json" in text
        assert not [word for word in ("mcp__", "Claude", "Codex") if word in text], level


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_e2e_check_parses_and_matches_the_manifest(task):
    path = task.dir / "e2e_check.json"
    spec = verify.parse_spec(json.loads(path.read_text()), str(path))
    assert spec.server and spec.calls and spec.answers
    assert all(src.call in {cs.name for cs in spec.calls} for a in spec.answers for src in a.sources)
    expected = MANIFEST[spec.server]["expected_tools"]
    assert {cs.tool for cs in spec.calls} <= set(expected)
    if spec.server_tools is not None:
        assert list(spec.server_tools) == expected
    for name in [*spec.bypass_tools, *spec.suspicious_tools]:
        if name.startswith("mcp__"):
            assert name.startswith(f"mcp__{spec.server}__") and name.split("__", 2)[2] in expected, name


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_task_has_its_own_test_module(task):
    module = importlib.import_module(f"tests.mcp_e2e.test_task_{task.name}")
    names = [name for name in dir(module) if name.startswith("test_")]
    missing = [topic for topic, word in REQUIRED_TOPICS.items() if not any(word in name for name in names)]
    assert not missing, f"test_task_{task.name}.py lacks tests for: {missing}"



