"""Task loading and management."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.types import GenerationMode, TaskLifecycle

logger = get_logger(__name__)

LEGACY_TASK_FILE = "task.yaml"
META_TASK_FILE = "task_meta.yaml"
EVAL_TASK_FILE = "task_eval.yaml"


def resolve_task_sources(task_dir: Path) -> tuple[Path, Path | None]:
    """Resolve the metadata source file(s) for a task directory.

    Supports two on-disk layouts:

    * **Legacy** — a single monolithic ``task.yaml`` → ``(task.yaml, None)``.
    * **Split** — ``task_meta.yaml`` (public) plus an optional
      ``task_eval.yaml`` (private) → ``(task_meta.yaml, task_eval.yaml | None)``.
      A public/HF tree may ship only ``task_meta.yaml``.

    When both ``task.yaml`` and ``task_meta.yaml`` exist (e.g. mid-migration),
    the legacy ``task.yaml`` wins so behaviour is unchanged until a task is
    fully migrated (its ``task.yaml`` removed).

    Raises:
        FileNotFoundError: if neither layout is present.
    """
    legacy = task_dir / LEGACY_TASK_FILE
    if legacy.exists():
        meta = task_dir / META_TASK_FILE
        if meta.exists():
            logger.warning(
                "Both %s and %s present in %s; using legacy %s",
                LEGACY_TASK_FILE, META_TASK_FILE, task_dir, LEGACY_TASK_FILE,
            )
        return legacy, None
    meta = task_dir / META_TASK_FILE
    if meta.exists():
        ev = task_dir / EVAL_TASK_FILE
        return meta, (ev if ev.exists() else None)
    raise FileNotFoundError(
        f"No {LEGACY_TASK_FILE} or {META_TASK_FILE} found in {task_dir}"
    )


def status_source(task_dir: Path) -> Path:
    """Return the file that carries the ``status:`` / ``id:`` fields.

    ``task_meta.yaml`` in split layout, or legacy ``task.yaml``.
    Raises FileNotFoundError if the task dir has neither.
    """
    meta, _ = resolve_task_sources(task_dir)
    return meta


def eval_source(task_dir: Path) -> Path:
    """Return the file that carries ``evaluation:`` / ``generation:``.

    ``task_eval.yaml`` in split layout; in legacy layout evaluation lives in
    ``task.yaml`` itself. Raises FileNotFoundError if the task dir has neither
    layout; a split task without ``task_eval.yaml`` also raises (no eval source).
    """
    meta, ev = resolve_task_sources(task_dir)
    if ev is not None:
        return ev
    if meta.name == LEGACY_TASK_FILE:
        return meta  # legacy monolith holds evaluation inline
    raise FileNotFoundError(f"No {EVAL_TASK_FILE} found in {task_dir}")


def task_metadata_files(task_dir: Path) -> list[Path]:
    """Return the existing metadata source files for bundling/replay/diagnosis.

    ``[task.yaml]`` for legacy, ``[task_meta.yaml, task_eval.yaml?]`` for split.
    """
    meta, ev = resolve_task_sources(task_dir)
    return [meta] + ([ev] if ev is not None else [])


def _merge_output_files(meta: dict[str, Any], eval_output: dict[str, Any]) -> None:
    """Overlay eval-only output sub-fields (shape/dtype/...) onto meta files by name."""
    if not isinstance(eval_output, dict):
        return
    meta_output = meta.setdefault("output", {})
    if not isinstance(meta_output, dict):
        return
    meta_files = meta_output.setdefault("files", [])
    if not isinstance(meta_files, list):
        return
    by_name = {
        f.get("name"): f for f in meta_files if isinstance(f, dict) and f.get("name")
    }
    for ef in eval_output.get("files", []) or []:
        if not isinstance(ef, dict):
            continue
        name = ef.get("name")
        target = by_name.get(name)
        if target is not None:
            for k, v in ef.items():
                target[k] = v
        else:
            meta_files.append(dict(ef))


def merge_task_eval(meta: dict[str, Any], eval_data: dict[str, Any]) -> None:
    """Merge a parsed ``task_eval.yaml`` dict into a ``task_meta.yaml`` dict in place.

    The two files carry disjoint fields by design (see repo_split_design.md §9),
    except ``output.files`` which straddles both: ``name``/``type`` come from
    meta, ``shape``/``dtype`` from eval — merged by file name.
    """
    if not isinstance(eval_data, dict):
        return
    for key, val in eval_data.items():
        if key == "task_id":
            continue  # back-reference only; identity lives in meta['id']
        if key == "output":
            _merge_output_files(meta, val)
        else:
            meta[key] = val


class TaskLoader:
    """Load and manage benchmark tasks from a tasks directory."""

    TEMPLATE_TASK_IDS = {"DOMAIN.TASK_NAME"}
    TEMPLATE_TASK_NAMES = {"Human-readable task name"}

    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir

    def _iter_task_source_files(self):
        """Yield one primary metadata file per task dir (legacy or split).

        Finds both ``task.yaml`` (legacy) and ``task_meta.yaml`` (split) trees,
        deduping by directory so a dir mid-migration (both present) is visited
        once. The resolved primary path (legacy-wins) is yielded.
        """
        seen: set[Path] = set()
        candidates = sorted(self.tasks_dir.rglob(LEGACY_TASK_FILE)) + sorted(
            self.tasks_dir.rglob(META_TASK_FILE)
        )
        for cand in candidates:
            task_dir = cand.parent
            if task_dir in seen:
                continue
            seen.add(task_dir)
            try:
                meta_path, _ = resolve_task_sources(task_dir)
            except FileNotFoundError:
                continue
            yield meta_path

    def discover_tasks(
        self,
        include_test: bool = False,
        include_sample: bool = False,
        include_dev: bool = False,
        include_abandoned: bool = False,
    ) -> list[dict[str, Any]]:
        """Discover all tasks (legacy task.yaml or split task_meta.yaml).

        Returns list of parsed task metadata dicts.
        """
        tasks = []
        for task_yaml in self._iter_task_source_files():
            task_dir = task_yaml.parent
            # Skip template directory
            if task_dir.name == "_template":
                continue
            try:
                metadata = self.load_task_metadata(task_yaml)
            except Exception as exc:
                logger.warning("Skipping unparseable %s: %s", task_yaml, exc)
                continue
            if self._is_template_metadata(metadata):
                logger.warning("Skipping template-like task metadata at %s", task_yaml)
                continue
            status = TaskLifecycle(metadata.get("status", "in_development"))
            if status == TaskLifecycle.FINAL:
                tasks.append(metadata)
            elif status == TaskLifecycle.TEST and include_test:
                tasks.append(metadata)
            elif status == TaskLifecycle.SAMPLE and include_sample:
                tasks.append(metadata)
            elif status == TaskLifecycle.IN_DEVELOPMENT and include_dev:
                tasks.append(metadata)
            elif status == TaskLifecycle.ABANDONED and include_abandoned:
                tasks.append(metadata)
        return tasks

    def load_task_metadata(self, task_yaml: Path) -> dict[str, Any]:
        """Load and parse a task's metadata.

        Accepts a path to either ``task.yaml`` (legacy) or ``task_meta.yaml``
        (split). The hint path only identifies the task *directory*; the actual
        source files are resolved via :func:`resolve_task_sources`, so a caller
        may pass a non-existent ``.../task.yaml`` for a split task and still
        load it. A ``task_eval.yaml`` (if present) is merged in, yielding the
        same dict shape as the legacy monolithic file.
        """
        meta_path, eval_path = resolve_task_sources(task_yaml.parent)
        with open(meta_path, encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
        if eval_path is not None:
            with open(eval_path, encoding="utf-8") as f:
                eval_data = yaml.safe_load(f)
            merge_task_eval(metadata, eval_data)
        metadata["_task_dir"] = meta_path.parent
        metadata["_task_yaml"] = meta_path
        metadata["_task_eval_yaml"] = eval_path

        # Parse generation mode (default: infinite)
        gen = metadata.get("generation", {})
        mode_str = gen.get("mode", "infinite")
        metadata["_generation_mode"] = GenerationMode(mode_str)
        metadata["_generation_precomputed"] = gen.get("precomputed", False)
        metadata["_generation_settings"] = gen.get("settings", [])

        runtime = metadata.get("runtime", {})
        packages = runtime.get("packages", [])
        if not isinstance(packages, list):
            raise ValueError(
                f"Task runtime.packages must be a list in {meta_path}"
            )
        metadata["_runtime_python"] = runtime.get("python")
        metadata["_runtime_packages"] = [str(pkg).strip() for pkg in packages if str(pkg).strip()]

        return metadata

    def load_task_by_id(self, task_id: str) -> dict[str, Any]:
        """Load a specific task by its ID (e.g. 'physics.example_task').

        Tries direct path resolution first (task_id → directory path),
        then falls back to scanning all task.yaml files. The fallback
        skips files that fail to parse, logging a warning instead of
        crashing.
        """
        parts = task_id.split(".")
        if len(parts) >= 2:
            candidate_dir = self.tasks_dir / "/".join(parts)
            try:
                meta_path, _ = resolve_task_sources(candidate_dir)
            except FileNotFoundError:
                meta_path = None
            if meta_path is not None:
                try:
                    metadata = self.load_task_metadata(meta_path)
                except yaml.YAMLError:
                    logger.warning("Direct path %s has invalid YAML, falling back to scan", meta_path)
                else:
                    if metadata.get("id") == task_id:
                        return metadata

        for task_yaml in self._iter_task_source_files():
            if task_yaml.parent.name == "_template":
                continue
            try:
                metadata = self.load_task_metadata(task_yaml)
            except Exception as exc:
                logger.warning("Skipping unparseable %s: %s", task_yaml, exc)
                continue
            if self._is_template_metadata(metadata):
                continue
            if metadata.get("id") == task_id:
                return metadata
        raise ValueError(f"Task '{task_id}' not found in {self.tasks_dir}")

    @classmethod
    def _is_template_metadata(cls, metadata: dict[str, Any]) -> bool:
        """Return whether parsed metadata still matches the scaffold template."""
        task_id = str(metadata.get("id", "")).strip()
        task_name = str(metadata.get("name", "")).strip()
        if task_id in cls.TEMPLATE_TASK_IDS or task_name in cls.TEMPLATE_TASK_NAMES:
            return True
        return task_id.startswith("DOMAIN.") or task_id.endswith(".TASK_NAME")

    def load_generate_gt_module(self, task_dir: Path) -> Any:
        """Dynamically load a task's generate_gt.py module."""
        gt_path = task_dir / "generate_gt.py"
        if not gt_path.exists():
            raise FileNotFoundError(f"generate_gt.py not found in {task_dir}")
        spec = importlib.util.spec_from_file_location("generate_gt", gt_path)
        module = importlib.util.module_from_spec(spec)
        # Don't pollute sys.modules with temporary modules
        spec.loader.exec_module(module)
        return module

    def load_custom_scorers(self, task_dir: Path) -> None:
        """Load custom_scorer.py from a task directory (registers scorers via decorator)."""
        scorer_path = task_dir / "custom_scorer.py"
        if not scorer_path.exists():
            return
        spec = importlib.util.spec_from_file_location("custom_scorer", scorer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
