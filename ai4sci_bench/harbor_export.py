"""Export one materialized seed31415 instance as a Harbor task."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path, PurePosixPath

import yaml


class HarborExportError(ValueError):
    """The public instance or task bundle cannot be exported safely."""


def _copy_public_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise HarborExportError(f"Required directory is missing or a symlink: {source}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise HarborExportError(f"Symlink in public input: {path}")
    shutil.copytree(source, destination)


def _safe_output_name(value: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise HarborExportError(f"Unsafe output path: {value!r}")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise HarborExportError(f"Unsafe output path: {value!r}")
    if path.parts[0] in {"data", "reference", "task_bundle", "tests"}:
        raise HarborExportError(f"Output overlaps an input or verifier path: {value!r}")
    return path.as_posix()


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def export_harbor_task(
    *,
    task_dir: str | Path,
    instance_dir: str | Path,
    level: str,
    wheel: str | Path,
    output_dir: str | Path,
) -> Path:
    """Generate a local Harbor dataset containing one public B1–B4 task."""
    if level not in {"b1", "b2", "b3", "b4"}:
        raise HarborExportError("level must be b1, b2, b3, or b4")
    task = Path(task_dir).absolute()
    instance = Path(instance_dir).absolute()
    wheel_path = Path(wheel).absolute()
    destination = Path(output_dir).absolute()
    if not task.is_dir() or task.is_symlink():
        raise HarborExportError("task_dir must be a real public task directory")
    if instance.is_symlink() or not instance.is_dir():
        raise HarborExportError("instance_dir must be a real materialized instance")
    if (
        not wheel_path.is_file()
        or wheel_path.is_symlink()
        or not re.fullmatch(r"asibench-[A-Za-z0-9_.+-]+\.whl", wheel_path.name)
    ):
        raise HarborExportError("wheel must be a real asibench wheel file")
    if destination.exists():
        raise HarborExportError(f"output_dir already exists: {destination}")
    for source in (task, instance, wheel_path):
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise HarborExportError("output_dir must be separate from all inputs")

    meta_path = task / "task_meta.yaml"
    eval_path = task / "task_eval.yaml"
    if any(path.is_symlink() or not path.is_file() for path in (meta_path, eval_path)):
        raise HarborExportError("task_dir needs real task_meta.yaml and task_eval.yaml")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    task_id = meta.get("id") if isinstance(meta, dict) else None
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", task_id):
        raise HarborExportError("task_meta.yaml needs a valid task id")
    if not isinstance(evaluation, dict) or evaluation.get("task_id") != task_id:
        raise HarborExportError("task_eval.yaml task_id does not match task_meta.yaml")
    if instance.name != f"{task_id}__seed31415":
        raise HarborExportError("Only the matching materialized seed31415 instance is supported")
    if evaluation.get("evaluation", {}).get("runtime") == "task":
        raise HarborExportError("Task evaluator runtimes need a dedicated verifier image")
    reference = instance / "reference"
    if not reference.is_dir() or not any(reference.iterdir()):
        raise HarborExportError("Public seed31415 reference/ is missing or empty")
    prompt = instance / f"prompt_{level}.md"
    if prompt.is_symlink() or not prompt.is_file():
        raise HarborExportError(f"Materialized prompt is missing: {prompt}")
    output_files = meta.get("output", {}).get("files", [])
    if not isinstance(output_files, list) or not output_files:
        raise HarborExportError("task_meta.yaml needs output.files")
    if not all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in output_files):
        raise HarborExportError("output.files entries need string names")
    names = [_safe_output_name(item["name"]) for item in output_files]
    if len(names) != len(set(names)):
        raise HarborExportError("Duplicate output.files names")
    runtime_packages = meta.get("runtime", {}).get("packages", [])
    if not isinstance(runtime_packages, list) or not all(isinstance(x, str) for x in runtime_packages):
        raise HarborExportError("runtime.packages must be a list of package specifications")
    if meta.get("input", {}).get("files") and not (instance / "data").is_dir():
        raise HarborExportError("Declared instance input data/ is missing")

    slug = task_id.replace(".", "-").replace("_", "-") + f"-{level}"
    task_name = f"asi-bench/{slug}"
    exported = destination / "tasks" / slug
    try:
        (exported / "environment" / "workspace").mkdir(parents=True)
        (exported / "tests" / "task_bundle").mkdir(parents=True)
        (exported / "tests" / "instance" / instance.name).mkdir(parents=True)
        (exported / "tests" / "wheels").mkdir(parents=True)
        data = instance / "data"
        if data.exists():
            _copy_public_tree(data, exported / "environment" / "workspace" / "data")
            _copy_public_tree(data, exported / "tests" / "instance" / instance.name / "data")
        _copy_public_tree(reference, exported / "tests" / "instance" / instance.name / "reference")
        for filename in ("instance_meta.json",):
            source = instance / filename
            if source.exists():
                if source.is_symlink() or not source.is_file():
                    raise HarborExportError(f"Unsafe instance file: {source}")
                shutil.copy2(source, exported / "tests" / "instance" / instance.name / filename)
        for source in (meta_path, eval_path, *sorted(task.glob("*_scorer.py")), *sorted(task.glob("*_eval_runtime.py"))):
            if source.is_symlink() or not source.is_file():
                raise HarborExportError(f"Unsafe task bundle file: {source}")
            shutil.copy2(source, exported / "tests" / "task_bundle" / source.name)
        shutil.copy2(wheel_path, exported / "tests" / "wheels" / wheel_path.name)
        (exported / "instruction.md").write_text(
            "Work in /workspace. Input files are under /workspace/data/. "
            "Write the requested outputs under /workspace.\n\n"
            + prompt.read_text(encoding="utf-8"), encoding="utf-8"
        )
        artifact_lines = ", ".join(_toml_string(f"/workspace/{name}") for name in names)
        (exported / "task.toml").write_text(
            f'schema_version = "1.4"\nartifacts = [{artifact_lines}]\n\n'
            f'[task]\nname = {_toml_string(task_name)}\n\n'
            f'[metadata]\nasi_task_id = {_toml_string(task_id)}\n'
            f'instance_id = {_toml_string(instance.name)}\n'
            f'prompt_level = {_toml_string(level)}\nofficial = false\n\n'
            '[agent]\ntimeout_sec = 10800\n\n'
            '[verifier]\nenvironment_mode = "separate"\ntimeout_sec = 1800\n\n'
            '[verifier.environment]\nnetwork_mode = "public"\n'
            'build_timeout_sec = 1800\n\n'
            '[environment]\nnetwork_mode = "public"\n'
            'workdir = "/workspace"\nbuild_timeout_sec = 1800\n',
            encoding="utf-8",
        )
        base = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
        packages = " ".join(shlex.quote(p) for p in runtime_packages)
        (exported / "environment" / "Dockerfile").write_text(
            f"FROM {base}\n"
            + (f"RUN uv pip install --system {packages}\n" if packages else "")
            + "COPY workspace/ /workspace/\nWORKDIR /workspace\n",
            encoding="utf-8",
        )
        (exported / "tests" / "Dockerfile").write_text(
            f"FROM {base}\n"
            f"COPY wheels/{wheel_path.name} /tmp/{wheel_path.name}\n"
            f"RUN uv pip install --system {shlex.quote('/tmp/' + wheel_path.name)} litellm "
            + packages + "\n"
            "COPY task_bundle/ /tests/task_bundle/\n"
            "COPY instance/ /tests/instance/\n"
            "COPY --chmod=755 test.sh /tests/test.sh\n"
            "WORKDIR /workspace\n",
            encoding="utf-8",
        )
        (exported / "tests" / "test.sh").write_text(
            "#!/bin/sh\nset -eu\n"
            f"exec asibench harbor-verify --task-dir /tests/task_bundle "
            f"--instance-dir /tests/instance/{instance.name} "
            f"--outputs-dir /workspace --prompt-level {level} --out /logs/verifier\n",
            encoding="utf-8",
        )
        (exported / "tests" / "test.sh").chmod(0o755)
        (destination / "registry.json").write_text(
            json.dumps([{
                "name": "asi-bench", "version": "seed31415",
                "description": "ASI-Bench seed31415 local non-official tasks",
                "tasks": [{"name": task_name, "path": f"tasks/{slug}"}],
            }], indent=2) + "\n", encoding="utf-8"
        )
        (destination / "export_manifest.json").write_text(
            json.dumps({
                "schema_version": 1, "task_id": task_id, "instance_id": instance.name,
                "level": level, "asibench_wheel": wheel_path.name,
                "asibench_wheel_sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
                "official": False,
            }, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return exported
