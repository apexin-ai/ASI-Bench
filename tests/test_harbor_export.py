"""Contract tests for a single materialized public Harbor export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4sci_bench.harbor_export import HarborExportError, export_harbor_task


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
