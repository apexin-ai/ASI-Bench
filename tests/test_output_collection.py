"""Opt-in helper collection, safety boundaries and persisted scoring inputs."""

import os
import shutil
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ai4sci_bench.adapters.subprocess_base import collect_output_files
from ai4sci_bench.core.types import AgentOutput, RunStatus
from ai4sci_bench.local_scoring import _staged_prediction_dir
from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator


def task_with_output(task, *, enabled=True, files=None, inputs=()):
    return replace(task, metadata={
        "output": {"collect_extra_files": enabled, "files": files or [{"name": "solver.py"}]},
        "input": {"files": [{"name": name} for name in inputs]},
    })


def write(root, name, text="helper"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_extra_collection_requires_explicit_boolean(sample_task_instance, tmp_path, flag):
    write(tmp_path, "solver.py")
    write(tmp_path, "helper")
    task = task_with_output(sample_task_instance, enabled=flag,
                            files=[{"name": "solver.py"}, {"name": "solver.py"}])
    assert collect_output_files(tmp_path, task) == ["solver.py", "solver.py"]


def test_helpers_are_sorted_relative_deduplicated_and_not_required(sample_task_instance, tmp_path):
    task = task_with_output(sample_task_instance, files=[{"name": "solver.py"},
                                                        {"name": "./solver.py"}])
    for name in ["solver.py", "lib/module.py", "config.json", "native", "helper.cc"]:
        write(tmp_path, name)
    assert collect_output_files(tmp_path, task) == [
        "config.json", "helper.cc", "lib/module.py", "native", "solver.py"]
    (tmp_path / "solver.py").unlink()
    assert "solver.py" not in collect_output_files(tmp_path, task)
    assert "native" in collect_output_files(tmp_path, task)


def test_framework_inputs_and_caches_are_not_submissions(sample_task_instance, tmp_path):
    excluded = ["prompt.md", "task_info.json", "framework_task_info.json",
                "_ai4sci_task_python", "_ai4sci_task_python.cmd",
                "data/x", "reference/x", ".ai4sci-bench/home/x", ".git/config",
                "__pycache__/x.pyc", "lib/__pycache__/x.pyc", ".cache/x",
                ".pytest_cache/x", ".mypy_cache/x", ".ruff_cache/x", ".hypothesis/x",
                "provided.txt", "inputs/nested/x", "case_7.dat", "unknown_99.dat", "frame_02.npy"]
    for name in excluded + ["solver.py", "lib/helper.py", "inputs_extra.txt"]:
        write(tmp_path, name)
    task = task_with_output(sample_task_instance, inputs=(
        "sub/../provided.txt", "./inputs", "case_{size}.dat", "unknown_{unbound}.dat", "frame_<ii>.npy"))
    task.parameters = {"size": 7}
    assert collect_output_files(tmp_path, task) == ["inputs_extra.txt", "lib/helper.py", "solver.py"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX special files and symlinks")
def test_links_special_files_and_escaped_declarations_are_rejected(sample_task_instance, tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = write(tmp_path, "outside", "private")
    write(root, "solver.py")
    write(root, "lib/ok.py")
    (root / "external").symlink_to(outside)
    (root / "internal").symlink_to(root / "solver.py")
    (root / "linked_dir").symlink_to(root / "lib", target_is_directory=True)
    (root / "outside_dir").symlink_to(tmp_path, target_is_directory=True)
    (root / "broken").symlink_to(root / "missing")
    (root / "loop").symlink_to(root, target_is_directory=True)
    os.mkfifo(root / "fifo")
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(root / "socket"))
        task = task_with_output(sample_task_instance, files=[{"name": name} for name in [
            "solver.py", "../outside", str(outside), "linked_dir/ok.py", "internal", "fifo"]])
        assert collect_output_files(root, task) == ["lib/ok.py", "solver.py"]
    # The framework itself exposes live workspaces through a trusted root symlink.
    alias = tmp_path / "workspace_alias"
    alias.symlink_to(root, target_is_directory=True)
    assert collect_output_files(alias, task) == ["lib/ok.py", "solver.py"]


def test_missing_root_entry_is_not_repaired(sample_task_instance, tmp_path):
    write(tmp_path, "nested/analysis.py")
    task = task_with_output(sample_task_instance, files=[{"name": "analysis.py"}])
    assert collect_output_files(tmp_path, task) == ["nested/analysis.py"]
    assert not (tmp_path / "analysis.py").exists()
    from ai4sci_bench.scorers.file_match import FileMatchScorer

    detail = FileMatchScorer().score(tmp_path, tmp_path, {"checks": [{"file": "analysis.py"}]})
    assert detail.passed is False
    assert detail.score == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX native executable fixture")
def test_max3sat_helpers_survive_persistence_and_fresh_scoring(sample_task_instance, tmp_path):
    meta = yaml.safe_load(Path("tasks/computer_science/max3sat_assignment_optimization/task_meta.yaml").read_text())
    assert meta["output"]["collect_extra_files"] is True
    assert [f["name"] for f in meta["output"]["files"]] == ["solver.py"]
    workspace = tmp_path / "workspace"
    write(workspace, "solver.py", "from lib.helper import run\nprint(run())\n")
    write(workspace, "lib/helper.py", 'import json, subprocess\ndef run():\n'
          '    value = json.load(open("config.json"))["value"]\n'
          '    return value + int(subprocess.check_output(["./echo", "2"]))\n')
    write(workspace, "config.json", '{"value": 40}')
    # Copy a real extensionless executable, preserving its executable mode.
    shutil.copy2(shutil.which("echo"), workspace / "echo")
    task = replace(sample_task_instance, metadata=meta)
    files = collect_output_files(workspace, task)
    output = AgentOutput(
        task.instance_id, workspace,
        [name for name in files if name.endswith(".py")],
        [name for name in files if not name.endswith(".py")],
        "", 0.0, RunStatus.COMPLETED,
    )
    results = tmp_path / "results"
    manifest = BenchmarkOrchestrator._persist_output_artifacts(None, results, "run", output)
    assert {entry["path"] for entry in manifest["files"]} == {
        "solver.py", "lib/helper.py", "config.json", "echo"}
    shutil.rmtree(workspace)
    job = SimpleNamespace(data_dir=str(tmp_path / "no-data"), requires_instance_data=False,
                          output_dir=str(results / manifest["dir"]), outputs_recorded_empty=False)
    with _staged_prediction_dir(job) as fresh:
        assert (fresh / "echo").stat().st_mode & 0o111
        result = subprocess.run([sys.executable, "solver.py"], cwd=fresh, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "42"
