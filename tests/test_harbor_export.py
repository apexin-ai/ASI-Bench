"""Contract tests for a single materialized public Harbor export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4sci_bench.harbor_export import (
    HarborExportError,
    audit_harbor_tasks,
    export_harbor_dataset,
    export_harbor_task,
)


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    task = root / "task"
    task.mkdir()
    (task / "task_meta.yaml").write_text(
        "id: math.example\n"
        "runtime:\n  packages: ['numpy>=2.0']\n"
        "input:\n  files:\n  - name: data/input.json\n"
        "output:\n  files:\n  - name: result.json\n",
        encoding="utf-8",
    )
    (task / "task_eval.yaml").write_text(
        "task_id: math.example\nevaluation:\n  scoring: []\n", encoding="utf-8"
    )
    (task / "custom_scorer.py").write_text("# public scorer\n", encoding="utf-8")
    instance = root / "math.example__seed31415"
    (instance / "data").mkdir(parents=True)
    (instance / "data" / "input.json").write_text("{}\n", encoding="utf-8")
    (instance / "reference").mkdir()
    (instance / "reference" / "answer.json").write_text("{}\n", encoding="utf-8")
    (instance / "prompt_b1.md").write_text("Solve the task.\n", encoding="utf-8")
    wheel = root / "asibench-0.1.6-py3-none-any.whl"
    wheel.write_bytes(b"test wheel")
    return task, instance, wheel


def test_export_separates_agent_input_from_verifier_reference(tmp_path: Path) -> None:
    task, instance, wheel = _inputs(tmp_path)
    output = tmp_path / "export"
    exported = export_harbor_task(
        task_dir=task, instance_dir=instance, level="b1", wheel=wheel,
        output_dir=output,
    )
    assert (exported / "environment/workspace/data/input.json").exists()
    assert not (exported / "environment/workspace/reference").exists()
    assert (exported / "tests/instance/math.example__seed31415/reference/answer.json").exists()
    assert (exported / "tests/task_bundle/custom_scorer.py").exists()
    assert " litellm " in (exported / "tests/Dockerfile").read_text()
    assert "'numpy>=1.26'" in (exported / "tests/Dockerfile").read_text()
    assert 'org.asibench.task_id="math.example"' in (exported / "tests/Dockerfile").read_text()
    assert 'org.asibench.instance_id="math.example__seed31415"' in (
        exported / "environment/Dockerfile"
    ).read_text()
    assert "--prompt-level b1" in (exported / "tests/test.sh").read_text()
    assert 'artifacts = ["/workspace/result.json"]' in (exported / "task.toml").read_text()
    registry = json.loads((output / "registry.json").read_text())
    assert registry[0]["tasks"] == [{
        "name": "asi-bench/math-example-b1", "path": "tasks/math-example-b1",
    }]
    assert json.loads((output / "export_manifest.json").read_text())["official"] is False


def test_export_rejects_non_public_and_unsafe_inputs(tmp_path: Path) -> None:
    task, instance, wheel = _inputs(tmp_path)
    seed42 = tmp_path / "math.example__seed42"
    seed42.mkdir()
    with pytest.raises(HarborExportError, match="seed31415"):
        export_harbor_task(
            task_dir=task, instance_dir=seed42,
            level="b1", wheel=wheel, output_dir=tmp_path / "bad",
        )
    (instance / "data" / "link").symlink_to(tmp_path)
    with pytest.raises(HarborExportError, match="Symlink"):
        export_harbor_task(
            task_dir=task, instance_dir=instance, level="b1", wheel=wheel,
            output_dir=tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()


def test_export_rejects_path_escape(tmp_path: Path) -> None:
    task, instance, wheel = _inputs(tmp_path)
    (task / "task_meta.yaml").write_text(
        (task / "task_meta.yaml").read_text().replace("result.json", "../result.json"),
        encoding="utf-8",
    )
    with pytest.raises(HarborExportError, match="Unsafe output path"):
        export_harbor_task(
            task_dir=task, instance_dir=instance, level="b1", wheel=wheel,
            output_dir=tmp_path / "bad",
        )


def test_export_task_runtime_uses_dedicated_verifier_image(tmp_path: Path) -> None:
    task, instance, wheel = _inputs(tmp_path)
    (task / "task_eval.yaml").write_text(
        "task_id: math.example\nevaluation:\n  runtime: task\n  scoring: []\n",
        encoding="utf-8",
    )
    exported = export_harbor_task(
        task_dir=task, instance_dir=instance, level="b1", wheel=wheel,
        output_dir=tmp_path / "export",
    )
    dockerfile = (exported / "tests/Dockerfile").read_text()
    assert "numpy>=2.0" in dockerfile
    assert "ASIBENCH_HARBOR_TASK_RUNTIME_PREINSTALLED=1" in dockerfile
    assert "--task-runtime-preinstalled" in (exported / "tests/test.sh").read_text()
    assert "ASIBENCH_HARBOR_TASK_RUNTIME_PREINSTALLED" not in (
        exported / "environment/Dockerfile"
    ).read_text()


def _write_batch_task(root: Path, task_id: str, *, scorer: str = "file_match") -> tuple[Path, Path]:
    domain, name = task_id.split(".")
    task = root / "tasks" / domain / name
    task.mkdir(parents=True)
    (task / "task_meta.yaml").write_text(
        f"id: {task_id}\nruntime:\n  python: '>=3.11'\n  packages: []\n"
        "input:\n  files:\n  - name: data/input.json\n"
        "output:\n  files:\n  - name: result.json\n",
        encoding="utf-8",
    )
    (task / "task_eval.yaml").write_text(
        f"task_id: {task_id}\nevaluation:\n  scoring:\n  - scorer: {scorer}\n",
        encoding="utf-8",
    )
    instance = root / "instances" / f"{task_id}__seed31415"
    (instance / "data").mkdir(parents=True)
    (instance / "data" / "input.json").write_text("{}\n", encoding="utf-8")
    (instance / "reference").mkdir()
    (instance / "reference" / "answer.json").write_text("{}\n", encoding="utf-8")
    (instance / "prompt_b1.md").write_text("Solve.\n", encoding="utf-8")
    return task, instance


def test_batch_export_creates_one_registry_and_records_skipped_cells(tmp_path: Path) -> None:
    _write_batch_task(tmp_path, "math.alpha")
    _write_batch_task(tmp_path, "math.beta", scorer="tsp_solver_total_score")
    wheel = tmp_path / "asibench-0.1.6-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    report = audit_harbor_tasks(
        tasks_dir=tmp_path / "tasks",
        instances_dir=tmp_path / "instances",
        levels=("b1",),
    )
    assert report["task_count"] == 2
    assert report["ready_count"] == 1
    assert report["blocked_count"] == 1

    with pytest.raises(HarborExportError, match="1 of 2 cells are blocked"):
        export_harbor_dataset(
            tasks_dir=tmp_path / "tasks", instances_dir=tmp_path / "instances",
            wheel=wheel, output_dir=tmp_path / "strict", levels=("b1",),
        )
    assert not (tmp_path / "strict").exists()

    result = export_harbor_dataset(
        tasks_dir=tmp_path / "tasks", instances_dir=tmp_path / "instances",
        wheel=wheel, output_dir=tmp_path / "partial", levels=("b1",),
        skip_unsupported=True,
    )
    registry = json.loads((tmp_path / "partial/registry.json").read_text())
    assert result["exported_count"] == 1
    assert result["skipped_count"] == 1
    assert len(registry[0]["tasks"]) == 1
    assert (tmp_path / "partial/compatibility_report.json").is_file()
    assert len(list((tmp_path / "partial/tasks").iterdir())) == 1


def test_audit_identifies_python_runtime_and_nested_docker_blockers(tmp_path: Path) -> None:
    task, _ = _write_batch_task(tmp_path, "math.runtime", scorer="tsp_solver_total_score")
    meta = (task / "task_meta.yaml").read_text().replace(">=3.11", ">=3.13")
    (task / "task_meta.yaml").write_text(meta, encoding="utf-8")
    report = audit_harbor_tasks(tasks_dir=tmp_path / "tasks", levels=("b1",))
    assert report["blocked_count"] == 1
    assert any("Python 3.12" in item for item in report["cells"][0]["blockers"])
    assert any("nested Docker" in item for item in report["cells"][0]["blockers"])


def test_audit_reports_unsupported_python_specifier_as_blocker(tmp_path: Path) -> None:
    task, _ = _write_batch_task(tmp_path, "math.bad_runtime")
    meta = (task / "task_meta.yaml").read_text().replace(">=3.11", "~=3.11")
    (task / "task_meta.yaml").write_text(meta, encoding="utf-8")
    report = audit_harbor_tasks(tasks_dir=tmp_path / "tasks", levels=("b1",))
    assert report["blocked_count"] == 1
    assert "unsupported runtime.python" in report["cells"][0]["blockers"][0]


def test_batch_export_refuses_all_blocked_dataset(tmp_path: Path) -> None:
    _write_batch_task(tmp_path, "math.blocked", scorer="tsp_solver_total_score")
    wheel = tmp_path / "asibench-0.1.6-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(HarborExportError, match="No compatible task-level cells"):
        export_harbor_dataset(
            tasks_dir=tmp_path / "tasks", instances_dir=tmp_path / "instances",
            wheel=wheel, output_dir=tmp_path / "empty", levels=("b1",),
            skip_unsupported=True,
        )
    assert not (tmp_path / "empty").exists()


def test_export_uses_declared_task_dockerfile_for_agent_environment(tmp_path: Path) -> None:
    task, instance, wheel = _inputs(tmp_path)
    custom = task / "Dockerfile.agent"
    custom.write_text("FROM python:3.12-slim\nUSER agent\n", encoding="utf-8")
    meta = (task / "task_meta.yaml").read_text().replace(
        "runtime:\n", "runtime:\n  dockerfile: Dockerfile.agent\n"
    )
    (task / "task_meta.yaml").write_text(meta, encoding="utf-8")
    exported = export_harbor_task(
        task_dir=task, instance_dir=instance, level="b1", wheel=wheel,
        output_dir=tmp_path / "export-custom",
    )
    dockerfile = (exported / "environment/Dockerfile").read_text()
    assert dockerfile.startswith("FROM python:3.12-slim\nUSER agent")
    assert "COPY --chown=1000:1000 workspace/ /workspace/" in dockerfile


def test_audit_blocks_custom_dockerfile_context_files(tmp_path: Path) -> None:
    task, _ = _write_batch_task(tmp_path, "math.custom")
    (task / "Dockerfile.agent").write_text("FROM python:3.12-slim\nCOPY app/ /app/\n", encoding="utf-8")
    meta = (task / "task_meta.yaml").read_text().replace(
        "runtime:\n", "runtime:\n  dockerfile: Dockerfile.agent\n"
    )
    (task / "task_meta.yaml").write_text(meta, encoding="utf-8")
    report = audit_harbor_tasks(tasks_dir=tmp_path / "tasks", levels=("b1",))
    assert report["blocked_count"] == 1
    assert "COPY/ADD" in report["cells"][0]["blockers"][0]


def test_judge_export_forwards_key_by_name_without_embedding_secret(tmp_path: Path) -> None:
    task, instance, wheel = _inputs(tmp_path)
    (task / "task_eval.yaml").write_text(
        "task_id: math.example\nevaluation:\n  scoring:\n"
        "  - scorer: multimodal\n    mode: vlm_judge\n"
        "    model: openai/gpt-5.5\n",
        encoding="utf-8",
    )
    exported = export_harbor_task(
        task_dir=task, instance_dir=instance, level="b1", wheel=wheel,
        output_dir=tmp_path / "judge-export",
    )
    task_toml = (exported / "task.toml").read_text()
    assert "ASIBENCH_JUDGE_API_KEY_ENV = \"OPENAI_API_KEY\"" in task_toml
    assert 'OPENAI_API_KEY = "${OPENAI_API_KEY}"' in task_toml
    assert "test-secret" not in task_toml
