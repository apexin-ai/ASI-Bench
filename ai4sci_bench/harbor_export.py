"""Export one materialized seed31415 instance as a Harbor task."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class HarborExportError(ValueError):
    """The public instance or task bundle cannot be exported safely."""


_DOCKER_SCORERS = {
    "cmos_opamp_testbench_judge",
    "max3sat_solver_total_score",
    "tsp_solver_total_score",
}


def _walk_mappings(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _task_config(task_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta_path = task_dir / "task_meta.yaml"
    eval_path = task_dir / "task_eval.yaml"
    if any(path.is_symlink() or not path.is_file() for path in (meta_path, eval_path)):
        raise HarborExportError(f"Task needs real task_meta.yaml and task_eval.yaml: {task_dir}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not isinstance(evaluation, dict):
        raise HarborExportError(f"Task metadata must be mappings: {task_dir}")
    task_id = meta.get("id")
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", task_id):
        raise HarborExportError(f"Invalid task id in {task_dir / 'task_meta.yaml'}")
    if evaluation.get("task_id") != task_id:
        raise HarborExportError(f"task_eval.yaml task_id does not match {task_id}")
    return meta, evaluation


def _dockerfile_has_external_context(dockerfile: Path) -> bool:
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        instruction = line.strip().split(maxsplit=1)
        if instruction and instruction[0].upper() in {"COPY", "ADD"}:
            return True
    return False


def audit_harbor_tasks(
    *,
    tasks_dir: str | Path,
    levels: tuple[str, ...] = ("b1", "b2", "b3", "b4"),
    instances_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Report the static Harbor compatibility of public task-level cells."""
    root = Path(tasks_dir).absolute()
    if root.is_symlink() or not root.is_dir():
        raise HarborExportError("tasks_dir must be a real task root")
    if not levels or any(level not in {"b1", "b2", "b3", "b4"} for level in levels):
        raise HarborExportError("levels must contain one or more of b1, b2, b3, or b4")
    instances_root = Path(instances_dir).absolute() if instances_dir is not None else None
    if instances_root is not None and (instances_root.is_symlink() or not instances_root.is_dir()):
        raise HarborExportError("instances_dir must be a real materialized instance root")

    cells: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for meta_path in sorted(root.glob("*/*/task_meta.yaml")):
        task_dir = meta_path.parent
        meta, task_eval = _task_config(task_dir)
        task_id = meta["id"]
        if task_id in seen_ids:
            raise HarborExportError(f"Duplicate task id in task tree: {task_id}")
        seen_ids.add(task_id)
        runtime = meta.get("runtime", {})
        if not isinstance(runtime, dict):
            raise HarborExportError(f"runtime must be a mapping for {task_id}")
        evaluation = task_eval.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise HarborExportError(f"evaluation must be a mapping for {task_id}")
        scorer_names = {
            item.get(key)
            for item in _walk_mappings(evaluation)
            for key in ("scorer", "mode")
            if isinstance(item.get(key), str)
        }
        blockers: list[str] = []
        notes: list[str] = []
        runtime_mode = evaluation.get("runtime")
        if runtime_mode not in (None, "task"):
            blockers.append(f"unsupported evaluation.runtime={runtime_mode!r}")
        python_spec = runtime.get("python")
        if python_spec:
            from ai4sci_bench.runner.task_env import TaskEnvironmentManager

            try:
                compatible_python = TaskEnvironmentManager(root)._version_satisfies(
                    str(python_spec), "3.12.0"
                )
            except RuntimeError as exc:
                blockers.append(f"unsupported runtime.python={python_spec!r}: {exc}")
            else:
                if not compatible_python:
                    blockers.append(f"Python 3.12 does not satisfy runtime.python={python_spec!r}")
        dockerfile_name = runtime.get("dockerfile")
        if dockerfile_name:
            dockerfile = (task_dir / str(dockerfile_name)).absolute()
            if dockerfile.is_relative_to(root) is False or dockerfile.is_symlink() or not dockerfile.is_file():
                blockers.append(f"runtime.dockerfile is missing or unsafe: {dockerfile_name}")
            elif _dockerfile_has_external_context(dockerfile):
                blockers.append("runtime.dockerfile uses COPY/ADD; custom build context is not exported")
            else:
                notes.append(f"agent image uses task runtime Dockerfile {dockerfile_name}")
        if _DOCKER_SCORERS & scorer_names:
            blockers.append("scorer launches Docker; Harbor verifier has no nested Docker runtime")
        uses_judge = bool(
            {"llm_judge", "vlm_judge", "cmos_opamp_testbench_judge"} & scorer_names
        )
        if uses_judge:
            notes.append("Judge model/API configuration and credentials must be injected into verifier")
        if runtime_mode == "task":
            notes.append("task runtime dependencies are preinstalled in the dedicated verifier image")
        instance_id = f"{task_id}__seed31415"
        instance_path = instances_root / instance_id if instances_root is not None else None
        for level in levels:
            cell_blockers = list(blockers)
            if instance_path is not None:
                if not instance_path.is_dir() or instance_path.is_symlink():
                    cell_blockers.append(f"materialized instance missing: {instance_id}")
                else:
                    for required in (f"prompt_{level}.md", "reference"):
                        path = instance_path / required
                        if path.is_symlink() or not path.exists():
                            cell_blockers.append(f"instance input missing or unsafe: {required}")
                    reference = instance_path / "reference"
                    if reference.is_dir() and not any(reference.iterdir()):
                        cell_blockers.append("public reference directory is empty")
                    if meta.get("input", {}).get("files") and not (instance_path / "data").is_dir():
                        cell_blockers.append("declared task input data/ is missing")
            cells.append({
                "task_id": task_id,
                "instance_id": instance_id,
                "level": level,
                "status": "blocked" if cell_blockers else "ready",
                "blockers": cell_blockers,
                "notes": notes,
                "uses_judge": uses_judge,
            })
    if not cells:
        raise HarborExportError(f"No task_meta.yaml files found under {root}")
    return {
        "schema_version": 1,
        "official": False,
        "python_image": "3.12",
        "task_count": len(seen_ids),
        "cell_count": len(cells),
        "ready_count": sum(cell["status"] == "ready" for cell in cells),
        "blocked_count": sum(cell["status"] == "blocked" for cell in cells),
        "cells": cells,
    }


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
    judge_api_key_env: str = "OPENAI_API_KEY",
    judge_api_base: str | None = None,
    judge_api_protocol: str | None = None,
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

    meta, evaluation = _task_config(task)
    meta_path = task / "task_meta.yaml"
    eval_path = task / "task_eval.yaml"
    task_id = meta["id"]
    if instance.name != f"{task_id}__seed31415":
        raise HarborExportError("Only the matching materialized seed31415 instance is supported")
    evaluation_config = evaluation.get("evaluation", {})
    runtime_mode = evaluation_config.get("runtime") if isinstance(evaluation_config, dict) else None
    if runtime_mode not in (None, "task"):
        raise HarborExportError(f"Unsupported evaluation.runtime: {runtime_mode!r}")
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
    uses_judge = any(
        item.get("scorer") in {"llm_judge", "vlm_judge", "cmos_opamp_testbench_judge"}
        or item.get("mode") in {"llm_judge", "vlm_judge"}
        for item in _walk_mappings(evaluation.get("evaluation", {}))
    )
    if uses_judge:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", judge_api_key_env):
            raise HarborExportError("judge_api_key_env must be an environment-variable name")
        if (judge_api_base is None) != (judge_api_protocol is None):
            raise HarborExportError("judge_api_base and judge_api_protocol must be set together")
        if judge_api_protocol not in (None, "native", "openai"):
            raise HarborExportError("judge_api_protocol must be native or openai")
    runtime_dockerfile = meta.get("runtime", {}).get("dockerfile")
    dockerfile_source: Path | None = None
    if runtime_dockerfile:
        dockerfile_source = (task / str(runtime_dockerfile)).absolute()
        if (
            not dockerfile_source.is_relative_to(task.resolve())
            or dockerfile_source.is_symlink()
            or not dockerfile_source.is_file()
        ):
            raise HarborExportError("runtime.dockerfile must be a safe file inside task_dir")
        if _dockerfile_has_external_context(dockerfile_source):
            raise HarborExportError("runtime.dockerfile COPY/ADD context files are not supported")

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
        task_toml = (
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
            'workdir = "/workspace"\nbuild_timeout_sec = 1800\n'
        )
        if uses_judge:
            task_toml += (
                "\n[verifier.env]\n"
                f"ASIBENCH_JUDGE_API_KEY_ENV = {_toml_string(judge_api_key_env)}\n"
                f"{judge_api_key_env} = {_toml_string('${' + judge_api_key_env + '}')}\n"
            )
            if judge_api_base is not None:
                task_toml += (
                    f"ASIBENCH_JUDGE_API_BASE = {_toml_string(judge_api_base)}\n"
                    f"ASIBENCH_JUDGE_API_PROTOCOL = {_toml_string(judge_api_protocol or '')}\n"
                )
        (exported / "task.toml").write_text(task_toml, encoding="utf-8")
        base = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
        packages = " ".join(shlex.quote(p) for p in runtime_packages)
        task_image_identity = (
            f'LABEL org.asibench.task_id="{task_id}" '
            f'org.asibench.instance_id="{instance.name}"\n'
        )
        if dockerfile_source is None:
            environment_dockerfile = (
                f"FROM {base}\n"
                + task_image_identity
                + (f"RUN uv pip install --system {packages}\n" if packages else "")
                + "COPY workspace/ /workspace/\nWORKDIR /workspace\n"
            )
        else:
            environment_dockerfile = (
                dockerfile_source.read_text(encoding="utf-8").rstrip()
                + "\n"
                + task_image_identity
                + "COPY --chown=1000:1000 workspace/ /workspace/\n"
                + "WORKDIR /workspace\n"
            )
        (exported / "environment" / "Dockerfile").write_text(
            environment_dockerfile, encoding="utf-8"
        )
        (exported / "tests" / "Dockerfile").write_text(
            f"FROM {base}\n"
            + task_image_identity
            + f'LABEL org.asibench.wheel_sha256="{hashlib.sha256(wheel_path.read_bytes()).hexdigest()}"\n'
            f"COPY wheels/{wheel_path.name} /tmp/{wheel_path.name}\n"
            "COPY task_bundle/ /tests/task_bundle/\n"
            f"RUN uv pip install --system {shlex.quote('/tmp/' + wheel_path.name)} "
            "litellm 'numpy>=1.26' "
            + packages + "\n"
            + (
                "RUN python -c 'import platform, yaml; from pathlib import Path; "
                "from ai4sci_bench.runner.task_env import TaskEnvironmentManager; "
                "spec = yaml.safe_load(Path(\"/tests/task_bundle/task_meta.yaml\").read_text())"
                ".get(\"runtime\", {}).get(\"python\") or \"\"; "
                "manager = TaskEnvironmentManager(Path(\"/tests\")); "
                "assert manager._version_satisfies(spec, platform.python_version()), "
                "\"Verifier Python does not satisfy runtime.python\"'\n"
                "RUN uv pip check\n"
                "ENV ASIBENCH_HARBOR_TASK_RUNTIME_PREINSTALLED=1\n"
                if runtime_mode == "task" else ""
            )
            + "COPY instance/ /tests/instance/\n"
            + "COPY --chmod=755 test.sh /tests/test.sh\n"
            + "WORKDIR /workspace\n",
            encoding="utf-8",
        )
        (exported / "tests" / "test.sh").write_text(
            "#!/bin/sh\nset -eu\n"
            f"exec asibench harbor-verify --task-dir /tests/task_bundle "
            f"--instance-dir /tests/instance/{instance.name} "
            f"--outputs-dir /workspace --prompt-level {level} --out /logs/verifier"
            + (" --task-runtime-preinstalled" if runtime_mode == "task" else "")
            + "\n",
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


def export_harbor_dataset(
    *,
    tasks_dir: str | Path,
    instances_dir: str | Path,
    wheel: str | Path,
    output_dir: str | Path,
    levels: tuple[str, ...] = ("b1", "b2", "b3", "b4"),
    skip_unsupported: bool = False,
    judge_api_key_env: str = "OPENAI_API_KEY",
    judge_api_base: str | None = None,
    judge_api_protocol: str | None = None,
) -> dict[str, Any]:
    """Atomically export all compatible materialized task-level cells."""
    tasks_root = Path(tasks_dir).absolute()
    instances_root = Path(instances_dir).absolute()
    wheel_path = Path(wheel).absolute()
    destination = Path(output_dir).absolute()
    for label, path in (("tasks_dir", tasks_root), ("instances_dir", instances_root)):
        if path.is_symlink() or not path.is_dir():
            raise HarborExportError(f"{label} must be a real directory")
    if destination.exists():
        raise HarborExportError(f"output_dir already exists: {destination}")
    if not destination.parent.is_dir():
        raise HarborExportError("output_dir parent directory must already exist")
    for source in (tasks_root, instances_root, wheel_path):
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise HarborExportError("output_dir must be separate from all inputs")
    report = audit_harbor_tasks(
        tasks_dir=tasks_root, levels=levels, instances_dir=instances_root
    )
    blocked = [cell for cell in report["cells"] if cell["status"] == "blocked"]
    if blocked and not skip_unsupported:
        sample = blocked[0]
        raise HarborExportError(
            f"{report['blocked_count']} of {report['cell_count']} cells are blocked; "
            f"first: {sample['task_id']} {sample['level']}: {'; '.join(sample['blockers'])}. "
            "Run `asibench harbor audit` or pass --skip-unsupported to export a partial dataset."
        )
    ready_keys = {
        (cell["task_id"], cell["level"])
        for cell in report["cells"] if cell["status"] == "ready"
    }
    if not ready_keys:
        raise HarborExportError(
            "No compatible task-level cells are available; refusing to create an empty dataset"
        )
    task_dirs: dict[str, Path] = {}
    for meta_path in sorted(tasks_root.glob("*/*/task_meta.yaml")):
        meta, _ = _task_config(meta_path.parent)
        task_dirs[meta["id"]] = meta_path.parent

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    registry_tasks: list[dict[str, str]] = []
    manifests: list[dict[str, Any]] = []
    try:
        for task_id, level in sorted(ready_keys):
            instance_dir = instances_root / f"{task_id}__seed31415"
            task_output = staging / ".single" / f"{task_id.replace('.', '-')}-{level}"
            exported = export_harbor_task(
                task_dir=task_dirs[task_id],
                instance_dir=instance_dir,
                level=level,
                wheel=wheel_path,
                output_dir=task_output,
                judge_api_key_env=judge_api_key_env,
                judge_api_base=judge_api_base,
                judge_api_protocol=judge_api_protocol,
            )
            relative_path = exported.relative_to(task_output)
            final_task_dir = staging / relative_path
            final_task_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported), str(final_task_dir))
            task_name = f"asi-bench/{final_task_dir.name}"
            registry_tasks.append({"name": task_name, "path": f"tasks/{final_task_dir.name}"})
            per_task_manifest = json.loads((task_output / "export_manifest.json").read_text())
            manifests.append(per_task_manifest)
        shutil.rmtree(staging / ".single")
        registry_tasks.sort(key=lambda item: item["name"])
        (staging / "registry.json").write_text(
            json.dumps([{
                "name": "asi-bench", "version": "seed31415",
                "description": "ASI-Bench public seed31415 tasks (non-official scores)",
                "tasks": registry_tasks,
            }], indent=2) + "\n",
            encoding="utf-8",
        )
        report["exported_count"] = len(registry_tasks)
        report["skipped_count"] = report["blocked_count"]
        report["wheel"] = wheel_path.name
        report["wheel_sha256"] = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        (staging / "compatibility_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "export_manifest.json").write_text(
            json.dumps({
                "schema_version": 1,
                "official": False,
                "task_count": len(task_dirs),
                "levels": list(levels),
                "exported_count": len(registry_tasks),
                "skipped_count": len(blocked),
                "wheel": wheel_path.name,
                "wheel_sha256": report["wheel_sha256"],
                "tasks": manifests,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
