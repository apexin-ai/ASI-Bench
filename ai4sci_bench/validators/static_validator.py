"""Static validation for task definitions.

Checks legacy ``task.yaml`` and split ``task_meta.yaml`` / ``task_eval.yaml``
definitions, file existence, path conventions, and information leakage without
executing any code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai4sci_bench.core.task import merge_task_eval, resolve_task_sources


REQUIRED_TASK_YAML_FIELDS = ["id", "name", "version", "domain"]
REQUIRED_EVALUATION_FIELDS = ["evaluation"]
VALID_STATUSES = {"in_development", "test", "sample", "final", "abandoned"}
VALID_DOMAINS = {
    "physics",
    "chemistry",
    "biology",
    "biostat",
    "materials",
    "earth_science",
    "astronomy",
    "math",
    "medicine",
    "electrical_engineering",
    "computer_science",
    "finance",
    "robotics",
}
VALID_GENERATION_MODES = {"infinite", "finite"}

TASK_INFO_LEAKED_FIELDS = {"instance_id", "parameters", "seed", "requires_gpu"}


@dataclass
class ValidationResult:
    """Structured result from static validation."""

    task_dir: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_dir": self.task_dir,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_task_static(
    task_dir: Path,
    tasks_root: Path | None = None,
    scores_dir: Path | None = None,
) -> ValidationResult:
    """Run all static checks on a task directory.

    Args:
        task_dir: Path to the task directory (e.g. tasks/math/beyn_contour_eigenvalue/)
        tasks_root: Optional root of the tasks/ tree, used for path convention checks.
        scores_dir: Optional path to the ``scores/`` directory. When provided,
            tasks with ``status: final`` are checked for a passing difficulty-check
            record (warning only — does not fail CI).

    Returns:
        ValidationResult with errors and warnings.
    """
    result = ValidationResult(task_dir=str(task_dir))

    try:
        meta_path, eval_path = resolve_task_sources(task_dir)
    except FileNotFoundError:
        result.errors.append("task.yaml not found (and task_meta.yaml not found)")
        return result

    try:
        with open(meta_path, encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        result.errors.append(f"{meta_path.name} is not valid YAML: {exc}")
        return result

    if not isinstance(metadata, dict):
        result.errors.append(
            f"{meta_path.name} must be a YAML mapping, got {type(metadata).__name__}"
        )
        return result

    if eval_path is not None:
        try:
            with open(eval_path, encoding="utf-8") as f:
                eval_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            result.errors.append(f"{eval_path.name} is not valid YAML: {exc}")
            return result
        if not isinstance(eval_data, dict):
            result.errors.append(
                f"{eval_path.name} must be a YAML mapping, got {type(eval_data).__name__}"
            )
            return result
        merge_task_eval(metadata, eval_data)

    _check_required_fields(metadata, result)
    _check_status(metadata, result)
    _check_domain(metadata, result)
    _check_version(metadata, result)
    _check_evaluation_block(metadata, result)
    _check_prompt_files(metadata, task_dir, result)
    _check_generation_script(metadata, task_dir, result)
    _check_custom_scorer(metadata, task_dir, result)
    _check_path_convention(metadata, task_dir, tasks_root, result)
    _check_task_info_leakage(task_dir, result)
    _check_status_score_consistency(metadata, scores_dir, result)

    return result


def _check_required_fields(metadata: dict, result: ValidationResult) -> None:
    for f in REQUIRED_TASK_YAML_FIELDS:
        val = metadata.get(f)
        if val is None:
            result.errors.append(f"Missing required field: {f}")
        elif isinstance(val, str) and not val.strip():
            result.errors.append(f"Required field '{f}' is empty")


def _check_status(metadata: dict, result: ValidationResult) -> None:
    status = metadata.get("status")
    if status is None:
        result.errors.append("Missing required field: status")
        return
    if status not in VALID_STATUSES:
        result.errors.append(
            f"Invalid status '{status}', must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )

    if status == "abandoned":
        if not metadata.get("abandoned_reason"):
            result.warnings.append("Task is abandoned but missing 'abandoned_reason' field")
    elif metadata.get("abandoned_reason"):
        result.warnings.append("Task has 'abandoned_reason' but status is not 'abandoned'")


def _check_domain(metadata: dict, result: ValidationResult) -> None:
    domain = metadata.get("domain")
    if domain is None:
        return  # already reported by required fields check
    if domain not in VALID_DOMAINS:
        result.warnings.append(
            f"Domain '{domain}' is not in the standard set: {', '.join(sorted(VALID_DOMAINS))}"
        )


def _check_version(metadata: dict, result: ValidationResult) -> None:
    version = metadata.get("version")
    if version is None:
        return  # already reported by required fields check
    version_str = str(version)
    if not re.match(r"^\d+(\.\d+)*$", version_str):
        result.errors.append(
            f"Invalid version format '{version_str}', expected semver-like (e.g. '1.0', '1.1')"
        )


def _check_evaluation_block(metadata: dict, result: ValidationResult) -> None:
    status = metadata.get("status")
    if status == "abandoned":
        return

    evaluation = metadata.get("evaluation")
    if evaluation is None:
        result.errors.append("Missing required field: evaluation")
        return
    if not isinstance(evaluation, dict):
        result.errors.append("evaluation must be a mapping")
        return

    gates = evaluation.get("gates")
    scoring = evaluation.get("scoring")

    if gates is not None and not isinstance(gates, list):
        result.errors.append("evaluation.gates must be a list")
    if scoring is not None and not isinstance(scoring, list):
        result.errors.append("evaluation.scoring must be a list")

    if not gates and not scoring:
        result.warnings.append("evaluation has neither gates nor scoring defined")

    if isinstance(scoring, list):
        for i, entry in enumerate(scoring):
            if not isinstance(entry, dict):
                result.errors.append(f"evaluation.scoring[{i}] must be a mapping")
                continue
            if "scorer" not in entry:
                result.errors.append(f"evaluation.scoring[{i}] missing 'scorer' field")
            if "weight" not in entry:
                result.errors.append(f"evaluation.scoring[{i}] missing 'weight' field")


def _check_prompt_files(metadata: dict, task_dir: Path, result: ValidationResult) -> None:
    status = metadata.get("status")
    if status == "abandoned":
        return

    prompts = metadata.get("prompts", {})
    if not isinstance(prompts, dict):
        result.errors.append("prompts must be a mapping")
        return

    for level in ["b1", "b2", "b3"]:
        declared = prompts.get(level)
        if declared:
            prompt_path = task_dir / declared
            if not prompt_path.exists():
                result.errors.append(f"Declared prompt file missing: {declared}")
            elif prompt_path.stat().st_size == 0:
                result.errors.append(f"Prompt file is empty: {declared}")
            else:
                _check_prompt_file_bytes(prompt_path, declared, result)
        else:
            fallback = task_dir / f"prompt_{level}.md"
            if not fallback.exists():
                result.errors.append(f"Missing prompt file: prompt_{level}.md")
            elif fallback.stat().st_size == 0:
                result.errors.append(f"Prompt file is empty: prompt_{level}.md")
            else:
                _check_prompt_file_bytes(fallback, fallback.name, result)

    b4_declared = prompts.get("b4")
    if b4_declared:
        prompt_path = task_dir / b4_declared
        if not prompt_path.exists():
            result.errors.append(f"Declared prompt file missing: {b4_declared}")
        elif prompt_path.stat().st_size == 0:
            result.errors.append(f"Prompt file is empty: {b4_declared}")
        else:
            _check_prompt_file_bytes(prompt_path, b4_declared, result)


def _check_prompt_file_bytes(
    prompt_path: Path,
    label: str,
    result: ValidationResult,
) -> None:
    """Reject prompt files that cannot be safely rendered/read as UTF-8 text."""
    try:
        data = prompt_path.read_bytes()
    except OSError as exc:
        result.errors.append(f"Could not read prompt file {label}: {exc}")
        return

    if b"\x00" in data:
        result.errors.append(f"Prompt file contains NUL byte: {label}")
        return

    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        result.errors.append(f"Prompt file is not valid UTF-8: {label}: {exc}")


def _check_generation_script(metadata: dict, task_dir: Path, result: ValidationResult) -> None:
    status = metadata.get("status")
    if status == "abandoned":
        return

    generation = metadata.get("generation", {})
    if not isinstance(generation, dict):
        result.errors.append("generation must be a mapping")
        return

    script = generation.get("script")
    if script:
        script_path = task_dir / script
        if not script_path.exists():
            result.errors.append(f"Generation script not found: {script}")

    mode = generation.get("mode")
    if mode and mode not in VALID_GENERATION_MODES:
        result.errors.append(
            f"Invalid generation mode '{mode}', must be one of: {', '.join(sorted(VALID_GENERATION_MODES))}"
        )


def _check_custom_scorer(metadata: dict, task_dir: Path, result: ValidationResult) -> None:
    status = metadata.get("status")
    if status == "abandoned":
        return

    evaluation = metadata.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return

    for section_key in ("gates", "scoring"):
        entries = evaluation.get(section_key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("scorer") == "custom":
                config = entry.get("config", {})
                if isinstance(config, dict):
                    script = config.get("script")
                    if script:
                        script_path = task_dir / script
                        if not script_path.exists():
                            result.errors.append(f"Custom scorer script not found: {script}")


def _check_path_convention(
    metadata: dict, task_dir: Path, tasks_root: Path | None, result: ValidationResult
) -> None:
    task_id = metadata.get("id")
    if not task_id or not isinstance(task_id, str):
        return

    parts = task_id.split(".")
    if len(parts) < 2:
        result.errors.append(
            f"Task id '{task_id}' must be in format '<domain>.<task_name>'"
        )
        return

    domain_part = parts[0]
    name_part = ".".join(parts[1:])

    declared_domain = metadata.get("domain")
    if declared_domain and declared_domain != domain_part:
        result.errors.append(
            f"Task id domain prefix '{domain_part}' does not match declared domain '{declared_domain}'"
        )

    if tasks_root:
        try:
            relative = task_dir.resolve().relative_to(tasks_root.resolve())
            rel_parts = relative.parts
            if len(rel_parts) >= 2:
                dir_domain = rel_parts[0]
                dir_name = rel_parts[1]
                if dir_domain != domain_part:
                    result.errors.append(
                        f"Directory domain '{dir_domain}' does not match task id domain '{domain_part}'"
                    )
                if dir_name != name_part:
                    result.warnings.append(
                        f"Directory name '{dir_name}' does not match task id name '{name_part}'"
                    )
        except ValueError:
            pass


def _check_status_score_consistency(
    metadata: dict,
    scores_dir: Path | None,
    result: ValidationResult,
) -> None:
    """Warn if a `status: final` task has no passing difficulty-check record.

    Looked up via ``scores/<task_id>.json``; the task PASSES this check when
    at least one historical evaluation has ``verdict: pass``. Failure modes
    surface as warnings (not errors) so CI never blocks on missing scores.
    """
    if scores_dir is None:
        return
    if metadata.get("status") != "final":
        return

    task_id = metadata.get("id")
    if not task_id or not isinstance(task_id, str):
        return

    scores_path = Path(scores_dir) / f"{task_id}.json"
    if not scores_path.exists():
        result.warnings.append(
            f"Task is 'final' but has no difficulty check record at {scores_path}"
        )
        return

    try:
        import json

        data = json.loads(scores_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result.warnings.append(
            f"Task is 'final' but score file could not be parsed: {exc}"
        )
        return

    if not isinstance(data, dict):
        result.warnings.append(
            f"Task is 'final' but score file {scores_path} is not a JSON object"
        )
        return

    evaluations = data.get("evaluations") or []
    if not evaluations:
        result.warnings.append(
            f"Task is 'final' but difficulty check record at {scores_path} is empty"
        )
        return

    if not any(
        isinstance(ev, dict) and ev.get("verdict") == "pass" for ev in evaluations
    ):
        latest = evaluations[-1] if isinstance(evaluations[-1], dict) else {}
        result.warnings.append(
            "Task is 'final' but no difficulty check has ever verdict='pass' "
            f"(latest: {latest.get('verdict', 'unknown')!r} on {latest.get('date', 'unknown')})"
        )


def _check_task_info_leakage(task_dir: Path, result: ValidationResult) -> None:
    """Check for hardcoded task_info.json that might leak sensitive fields."""
    import json

    task_info_path = task_dir / "task_info.json"
    if not task_info_path.exists():
        return

    try:
        with open(task_info_path, encoding="utf-8") as f:
            task_info = json.load(f)
    except (json.JSONDecodeError, OSError):
        result.warnings.append("task_info.json exists but could not be parsed")
        return

    if not isinstance(task_info, dict):
        return

    for leaked_field in TASK_INFO_LEAKED_FIELDS:
        if leaked_field in task_info:
            result.errors.append(
                f"task_info.json contains potentially leaked field: '{leaked_field}' "
                f"(agent-facing task_info must not expose this)"
            )


def detect_changed_tasks(repo_root: Path, base_ref: str = "origin/main") -> list[Path]:
    """Detect which task directories were modified relative to base_ref.

    Uses git diff to find changed files under tasks/, then extracts
    unique task directories (tasks/<domain>/<task_name>/).

    Returns:
        List of absolute paths to changed task directories.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=repo_root,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["git", "diff", "--name-only", base_ref, "HEAD"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=repo_root,
            )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    changed_files = result.stdout.strip().splitlines()
    task_dirs: set[Path] = set()

    for filepath in changed_files:
        parts = Path(filepath).parts
        if len(parts) >= 3 and parts[0] == "tasks" and parts[1] != "_template":
            task_dir = repo_root / parts[0] / parts[1] / parts[2]
            try:
                resolve_task_sources(task_dir)
            except FileNotFoundError:
                continue
            else:
                task_dirs.add(task_dir)

    return sorted(task_dirs)
