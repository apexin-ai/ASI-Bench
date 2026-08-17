"""Instance generator — generates task instances from generate_gt.py."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.core.types import GenerationMode, PromptLevel, TaskInstance
from ai4sci_bench.generators.param_space import ParamSpace, _sample_params
from ai4sci_bench.runner.linux_ns_sandbox import LinuxNSSandbox
from ai4sci_bench.runner.runtime_root import resolve_runtime_root
from ai4sci_bench.runner.sandbox_support import validate_sandbox_mode
from ai4sci_bench.runner.task_env import TaskEnvironmentManager

logger = get_logger(__name__)
AGENT_TASK_INFO_FILENAME = "task_info.json"
FRAMEWORK_TASK_INFO_FILENAME = "framework_task_info.json"


class InstanceGenerator:
    """Generate task instances by calling generate_gt.py with parameters."""

    def __init__(
        self,
        tasks_dir: Path,
        sandbox: str = "none",
        repo_root: Path | None = None,
        execution_timeout_seconds: int = 10800,
    ):
        self.tasks_dir = tasks_dir
        self.sandbox = sandbox
        self.execution_timeout_seconds = execution_timeout_seconds
        validate_sandbox_mode(
            sandbox,
            supported_modes=("none", "task", "os", "linux_ns"),
            component=self.__class__.__name__,
        )
        self.task_loader = TaskLoader(tasks_dir)
        self.repo_root = Path(repo_root) if repo_root else resolve_runtime_root(tasks_dir)
        self.task_env_manager = (
            TaskEnvironmentManager(self.repo_root)
            if sandbox in ("task", "os", "linux_ns")
            else None
        )

    def generate_instance(
        self,
        task_metadata: dict[str, Any],
        params: dict[str, Any],
        output_dir: Path,
        prompt_level: PromptLevel = PromptLevel.B2,
        *,
        effective_timeout_seconds: int | None = None,
    ) -> TaskInstance:
        """Generate a single task instance.

        Calls the task's generate_gt.py to produce input data, reference outputs,
        and rendered prompts.
        """
        task_dir = task_metadata["_task_dir"]
        task_id = task_metadata["id"]

        # Build instance ID from task_id and key params
        instance_id = self._build_instance_id(task_id, params)

        # Create output directory
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        eff_timeout = (
            effective_timeout_seconds
            if effective_timeout_seconds is not None
            else self.execution_timeout_seconds
        )

        # Call generate_gt.py (skip if reference already exists with files)
        reference_dir = instance_dir / "reference"
        if not reference_dir.exists() or not any(reference_dir.iterdir()):
            self._run_generate_gt(
                task_metadata,
                instance_dir,
                params,
                effective_timeout_seconds=eff_timeout,
            )
        self._copy_missing_prompt_templates(task_metadata, instance_dir)

        # Build TaskInstance — workspace is isolated from instance_dir to prevent
        # agents from traversing up to read reference data or metadata (#7)
        resolved_params = self._load_resolved_params(instance_dir, params)
        workspace_dir = self._create_isolated_workspace(
            output_dir, instance_id, prompt_level
        )

        # Prepare workspace (copy data + prompt)
        self._prepare_workspace(
            instance_dir,
            workspace_dir,
            instance_id,
            prompt_level,
            task_metadata,
            resolved_params,
            effective_timeout_seconds=eff_timeout,
        )

        return TaskInstance(
            task_id=task_id,
            instance_id=instance_id,
            task_dir=task_dir,
            workspace_dir=workspace_dir,
            reference_dir=reference_dir,
            prompt_level=prompt_level,
            parameters=resolved_params,
            metadata=task_metadata,
            retained_workspace_dir=self._maybe_retained_workspace_dir(
                output_dir, instance_id, prompt_level
            ),
            effective_timeout_seconds=eff_timeout,
        )

    def generate_instances_on_the_fly(
        self,
        task_metadata: dict[str, Any],
        seed: int,
        count: int,
        output_dir: Path,
        prompt_level: PromptLevel = PromptLevel.B2,
        *,
        effective_timeout_seconds: int | None = None,
    ) -> list[TaskInstance]:
        """Generate multiple instances using seed+idx sampling or finite settings."""
        mode = task_metadata.get("_generation_mode", GenerationMode.INFINITE)

        if mode == GenerationMode.FINITE:
            return self._generate_finite_instances(
                task_metadata, seed, count, output_dir, prompt_level,
                effective_timeout_seconds=effective_timeout_seconds,
            )

        # Infinite mode: random sampling
        param_space = task_metadata.get("generation", {}).get("parameters", {})
        instances = []
        for idx in range(count):
            params = _sample_params(seed, idx, param_space)
            instance = self.generate_instance(
                task_metadata, params, output_dir, prompt_level,
                effective_timeout_seconds=effective_timeout_seconds,
            )
            instances.append(instance)
        return instances

    def _generate_finite_instances(
        self,
        task_metadata: dict[str, Any],
        seed: int,
        count: int,
        output_dir: Path,
        prompt_level: PromptLevel = PromptLevel.B2,
        *,
        effective_timeout_seconds: int | None = None,
    ) -> list[TaskInstance]:
        """Generate instances from finite settings list."""
        settings = task_metadata.get("_generation_settings", [])
        if not settings:
            raise ValueError(
                f"Task '{task_metadata['id']}' uses finite mode but has no settings defined"
            )

        precomputed = task_metadata.get("_generation_precomputed", False)
        task_dir = task_metadata["_task_dir"]

        # Select settings: cycle through the list using seed for offset
        import random
        rng = random.Random(seed)
        indices = list(range(len(settings)))
        rng.shuffle(indices)

        # Take up to count settings (cycle if count > len(settings))
        selected = []
        for i in range(count):
            selected.append(settings[indices[i % len(indices)]])

        instances = []
        for setting in selected:
            params = {k: v for k, v in setting.items() if k != "id"}

            if precomputed:
                # Load pre-computed GT from reference/<setting_id>/
                setting_id = setting.get("id")
                if not setting_id:
                    raise ValueError(
                        "Precomputed finite mode requires 'id' field in each setting"
                    )
                instance = self._load_precomputed_instance(
                    task_metadata, setting_id, params, output_dir, prompt_level,
                    effective_timeout_seconds=effective_timeout_seconds,
                )
            else:
                # Real-time: call generate_gt.py with the setting params
                instance = self.generate_instance(
                    task_metadata, params, output_dir, prompt_level,
                    effective_timeout_seconds=effective_timeout_seconds,
                )
            instances.append(instance)

        return instances

    def _load_precomputed_instance(
        self,
        task_metadata: dict[str, Any],
        setting_id: str,
        params: dict[str, Any],
        output_dir: Path,
        prompt_level: PromptLevel,
        *,
        effective_timeout_seconds: int | None = None,
    ) -> TaskInstance:
        """Load a precomputed instance from reference/<setting_id>/."""
        import shutil

        task_dir = task_metadata["_task_dir"]
        task_id = task_metadata["id"]
        instance_id = self._build_instance_id(task_id, params)

        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        # Precomputed bundles currently live under reference/<setting_id>/ and may
        # contain the full generated instance layout:
        #   reference/<setting_id>/
        #     data/
        #     reference/
        #     prompt_b*.md
        #     instance_meta.json
        # Older layouts may store reference files directly at the root.
        precomputed_bundle = task_dir / "reference" / setting_id
        if not precomputed_bundle.exists():
            raise FileNotFoundError(
                f"Precomputed reference directory not found: {precomputed_bundle}. "
                f"Run 'asibench generate --task {task_id} --precompute' first."
            )

        nested_reference_dir = precomputed_bundle / "reference"
        if nested_reference_dir.is_dir():
            reference_source = nested_reference_dir
        else:
            reference_source = precomputed_bundle

        ref_dst = instance_dir / "reference"
        if not ref_dst.exists():
            shutil.copytree(reference_source, ref_dst)

        # Data may be stored inside the precomputed bundle.
        data_src = precomputed_bundle / "data"

        data_dst = instance_dir / "data"
        if data_src.exists() and not data_dst.exists():
            shutil.copytree(data_src, data_dst)

        # Prefer precomputed rendered prompts when available. Otherwise, render
        # them from the task templates using the current params.
        copied_any_prompt = False
        for level in ["b1", "b2", "b3", "b4"]:
            prompt_src = precomputed_bundle / f"prompt_{level}.md"
            if prompt_src.exists():
                shutil.copy2(prompt_src, instance_dir / f"prompt_{level}.md")
                copied_any_prompt = True

        try:
            if not copied_any_prompt:
                module = self.task_loader.load_generate_gt_module(task_dir)
                if hasattr(module, "_render_prompt_template"):
                    for level in ["b1", "b2", "b3", "b4"]:
                        template_path = task_dir / f"prompt_{level}.md"
                        if template_path.exists():
                            text = template_path.read_text(encoding="utf-8")
                            rendered = module._render_prompt_template(text, params)
                            (instance_dir / f"prompt_{level}.md").write_text(
                                rendered, encoding="utf-8"
                            )
        except Exception:
            if not copied_any_prompt:
                # Fallback: copy prompt templates as-is
                for level in ["b1", "b2", "b3", "b4"]:
                    src = task_dir / f"prompt_{level}.md"
                    if src.exists():
                        shutil.copy2(src, instance_dir / f"prompt_{level}.md")
        self._copy_missing_prompt_templates(task_metadata, instance_dir)

        # Workspace is isolated from instance_dir to prevent agents from
        # traversing up to read reference data or metadata (#7)
        resolved_params = self._load_resolved_params(instance_dir, params)
        workspace_dir = self._create_isolated_workspace(
            output_dir, instance_id, prompt_level
        )

        eff_timeout = (
            effective_timeout_seconds
            if effective_timeout_seconds is not None
            else self.execution_timeout_seconds
        )

        self._prepare_workspace(
            instance_dir,
            workspace_dir,
            instance_id,
            prompt_level,
            task_metadata,
            resolved_params,
            effective_timeout_seconds=eff_timeout,
        )

        return TaskInstance(
            task_id=task_id,
            instance_id=instance_id,
            task_dir=task_dir,
            workspace_dir=workspace_dir,
            reference_dir=ref_dst,
            prompt_level=prompt_level,
            parameters=resolved_params,
            metadata=task_metadata,
            retained_workspace_dir=self._maybe_retained_workspace_dir(
                output_dir, instance_id, prompt_level
            ),
            effective_timeout_seconds=eff_timeout,
        )

    def get_retained_workspace_dir(
        self,
        output_dir: Path,
        instance_id: str,
        prompt_level: PromptLevel,
        attempt: int | None = None,
    ) -> Path:
        """Return the persistent workspace path under ``_workspaces/``."""
        name = f"workspace_{prompt_level.value}"
        if attempt is not None and attempt > 1:
            name = f"{name}_attempt{attempt}"
        return output_dir / "_workspaces" / instance_id / name

    def _maybe_retained_workspace_dir(
        self,
        output_dir: Path,
        instance_id: str,
        prompt_level: PromptLevel,
    ) -> Path | None:
        """Return persistent workspace path only when sandbox=='os' (else None)."""
        if self.sandbox != "os":
            return None
        return self.get_retained_workspace_dir(
            output_dir, instance_id, prompt_level
        )

    def _build_instance_id(self, task_id: str, params: dict[str, Any]) -> str:
        """Build a deterministic instance ID from task_id and params."""
        # Include key params in the ID for readability
        param_parts = []
        for k, v in sorted(params.items()):
            if k == "seed":
                param_parts.append(f"seed{v}")
            elif isinstance(v, float):
                param_parts.append(f"{k}{v:.2f}")
            else:
                param_parts.append(f"{k}{v}")
        param_str = "_".join(param_parts)
        # Truncate if too long
        if len(param_str) > 80:
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            param_str = param_str[:60] + "_" + param_hash
        return f"{task_id}__{param_str}"

    def _run_generate_gt(
        self,
        task_metadata: dict[str, Any],
        output_dir: Path,
        params: dict[str, Any],
        *,
        effective_timeout_seconds: int | None = None,
    ) -> None:
        """Run generate_gt.py for a task."""
        task_dir = task_metadata["_task_dir"]
        gt_script = task_dir / "generate_gt.py"
        if not gt_script.exists():
            raise FileNotFoundError(f"generate_gt.py not found in {task_dir}")

        task_env = self._get_task_environment(task_metadata)

        if task_env is None:
            # Try to use the module's generate() function directly
            try:
                module = self.task_loader.load_generate_gt_module(task_dir)
                if hasattr(module, "generate"):
                    module.generate(output_dir, params)
                    return
            except Exception:
                pass

        # Fallback to subprocess
        python_cmd = str(task_env.python_executable) if task_env else sys.executable
        cmd = [
            python_cmd, gt_script.name,
            "--output-dir", str(output_dir.resolve()),
            "--params", json.dumps(params),
        ]
        generation_config = task_metadata.get("generation", {})
        timeout = generation_config.get("timeout_seconds")
        if timeout is None:
            timeout = (
                effective_timeout_seconds
                if effective_timeout_seconds is not None
                else 300
            )
        if self.sandbox == "linux_ns":
            sandbox = LinuxNSSandbox()
            success, log, _, stderr = sandbox.run_command(
                command=cmd,
                cwd=task_dir,
                timeout=timeout,
                extra_env=task_env.build_subprocess_env() if task_env else None,
            )
            if not success:
                raise RuntimeError(
                    f"generate_gt.py failed for {task_dir.name}:\n{stderr or log}"
                )
            return
        result = subprocess.run(
            cmd,
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=task_env.build_subprocess_env() if task_env else None,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"generate_gt.py failed for {task_dir.name}:\n{result.stderr}"
            )

    def _get_task_environment(self, task_metadata: dict[str, Any]):
        if self.task_env_manager is None:
            return None
        return self.task_env_manager.ensure_env(task_metadata)

    PERSISTED_BACKUP_SUFFIX = "__previous"

    def _create_isolated_workspace(
        self,
        output_dir: Path,
        instance_id: str,
        prompt_level: PromptLevel,
    ) -> Path:
        """Create an isolated workspace in a temporary directory.

        Each prompt-level workspace is placed under its own temp directory
        so that agents cannot ``cd ..`` to read sibling workspaces of
        other prompt levels (fixes #13).  A stable symlink under
        ``_workspaces/`` is maintained for human debugging and result
        collection.

        When ``--sandbox os`` previously persisted a real directory at the
        same path (issue #17), it is archived to ``<name>__previous`` (a
        single backup slot) before the new symlink is created, so re-running
        with the same ``--output-dir`` does not silently destroy prior
        outputs.
        """
        import tempfile

        # Create a temp directory that is completely isolated
        tmpdir = tempfile.mkdtemp(prefix="ai4sci_ws_")
        workspace_dir = Path(tmpdir) / "workspace"
        workspace_dir.mkdir()

        # Maintain a stable symlink under _workspaces/ for debugging
        ws_root = output_dir / "_workspaces" / instance_id
        ws_root.mkdir(parents=True, exist_ok=True)
        link_path = self.get_retained_workspace_dir(
            output_dir,
            instance_id,
            prompt_level,
        )
        path_file = ws_root / f"workspace_{prompt_level.value}.path"

        self._reset_workspace_slot(link_path)
        if path_file.exists():
            path_file.unlink()

        try:
            link_path.symlink_to(workspace_dir)
        except OSError:
            # Symlink may fail on some Windows configs — fall back to
            # recording the path in a text file.
            path_file.write_text(str(workspace_dir), encoding="utf-8")

        # Reset stale retry workspaces (preserve the most recent run via
        # __previous backup slot for sandbox=os persisted data).
        for stale_ws in ws_root.glob(f"workspace_{prompt_level.value}_attempt*"):
            if stale_ws.name.endswith(self.PERSISTED_BACKUP_SUFFIX):
                continue
            self._reset_workspace_slot(stale_ws)

        return workspace_dir

    def _reset_workspace_slot(self, slot: Path) -> None:
        """Free a workspace slot for a new symlink, preserving real data.

        - Symlinks (live or dangling) are unlinked.
        - Real directories (persisted by ``--sandbox os``) are archived to
          ``<name>__previous``, overwriting any older backup. This bounds
          disk usage to one prior run while preventing silent data loss.
        - Plain files are removed.
        """
        import shutil

        if slot.is_symlink():
            slot.unlink()
            return
        if not slot.exists():
            return
        if slot.is_dir():
            backup = slot.parent / f"{slot.name}{self.PERSISTED_BACKUP_SUFFIX}"
            if backup.exists() or backup.is_symlink():
                if backup.is_symlink() or backup.is_file():
                    backup.unlink()
                else:
                    shutil.rmtree(backup)
            slot.rename(backup)
            logger.info(
                "Archived previously persisted workspace %s -> %s",
                slot,
                backup,
            )
            return
        slot.unlink()

    def _prepare_workspace(
        self,
        instance_dir: Path,
        workspace_dir: Path,
        instance_id: str,
        prompt_level: PromptLevel,
        task_metadata: dict[str, Any],
        params: dict[str, Any],
        *,
        effective_timeout_seconds: int | None = None,
        framework_task_info_dir: Path | None = None,
    ) -> None:
        """Prepare the agent workspace and framework-only metadata.

        instance_dir may be a shared, downloaded --instances-dir and must
        therefore remain read-only. Callers loading pre-generated instances
        provide a run-owned framework_task_info_dir.
        """
        import shutil
        import sys

        # Preflight: check for Windows MAX_PATH (260 chars) risk (fixes #9).
        if sys.platform == "win32":
            # Estimate longest sub-path: workspace/data/<longest_data_file>
            longest_data_name = ""
            data_src = instance_dir / "data"
            if data_src.exists():
                for p in data_src.rglob("*"):
                    rel = str(p.relative_to(data_src))
                    if len(rel) > len(longest_data_name):
                        longest_data_name = rel
            estimated = len(str(workspace_dir.resolve())) + 1 + len("data") + 1 + len(longest_data_name)
            if estimated > 250:
                raise RuntimeError(
                    f"Estimated path length ({estimated} chars) exceeds Windows "
                    f"MAX_PATH limit. Shorten --output-dir or use a shorter "
                    f"base directory. Current workspace: {workspace_dir}"
                )

        # Copy data files
        data_src = instance_dir / "data"
        data_dst = workspace_dir / "data"
        if data_src.exists():
            if data_dst.exists():
                shutil.rmtree(data_dst)
            shutil.copytree(data_src, data_dst)

        # Copy the selected prompt level as prompt.md
        prompt_file = instance_dir / f"prompt_{prompt_level.value}.md"
        if prompt_file.exists():
            shutil.copy2(prompt_file, workspace_dir / "prompt.md")
        else:
            raise FileNotFoundError(
                f"Prompt file not found for {instance_id} level {prompt_level.value}: "
                f"{prompt_file}. Ensure generate_gt.py renders prompts or task.yaml "
                "prompts point to existing task prompt templates."
            )

        task_info = self._build_agent_task_info(
            task_metadata, prompt_level,
            effective_timeout_seconds=effective_timeout_seconds,
        )
        framework_task_info = self._build_framework_task_info(
            task_metadata,
            instance_id=instance_id,
            prompt_level=prompt_level,
            resolved_params=params,
            effective_timeout_seconds=effective_timeout_seconds,
        )
        (workspace_dir / AGENT_TASK_INFO_FILENAME).write_text(
            json.dumps(task_info, indent=2), encoding="utf-8"
        )
        framework_info_dir = framework_task_info_dir or instance_dir
        framework_info_dir.mkdir(parents=True, exist_ok=True)
        (framework_info_dir / FRAMEWORK_TASK_INFO_FILENAME).write_text(
            json.dumps(framework_task_info, indent=2), encoding="utf-8"
        )

        # Strip any stray CLAUDE.md / AGENTS.md that may have been copied in
        # from the task's data directory. Claude Code and other CLI agents
        # will auto-load these and they can leak hints or cross-agent bias.
        # Raw sidecar logs are preserved elsewhere, so nothing user-visible
        # is lost here. See benchmark_issues_and_proposals.md §"workspace
        # CLAUDE.md leakage".
        self._strip_agent_memory_files(workspace_dir)

    def _copy_missing_prompt_templates(
        self,
        task_metadata: dict[str, Any],
        instance_dir: Path,
    ) -> None:
        """Backfill prompt_b*.md from task templates when GT did not render them."""
        task_dir = task_metadata["_task_dir"]
        prompt_map = task_metadata.get("prompts", {})
        for level in ("b1", "b2", "b3", "b4"):
            dst = instance_dir / f"prompt_{level}.md"
            if dst.exists():
                continue
            prompt_name = prompt_map.get(level) or f"prompt_{level}.md"
            src = task_dir / str(prompt_name)
            if src.exists():
                shutil.copy2(src, dst)

    _AGENT_MEMORY_FILENAMES = (
        "CLAUDE.md",
        "CLAUDE.local.md",
        "AGENTS.md",
        ".claude",
    )

    def _strip_agent_memory_files(self, workspace_dir: Path) -> None:
        """Remove agent-memory files that would taint a benchmark run.

        These files (CLAUDE.md, AGENTS.md, .claude/) are auto-loaded by
        the CLI agents we evaluate and would otherwise smuggle context
        from the task authoring repo into the agent's view of the
        workspace. Only the workspace root is scrubbed — we do not walk
        into ``data/`` because task inputs are agent-facing.
        """
        import shutil

        if not workspace_dir.exists():
            return
        for name in self._AGENT_MEMORY_FILENAMES:
            target = workspace_dir / name
            if not target.exists() and not target.is_symlink():
                continue
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to strip agent memory file %s: %s", target, exc
                )

    def _load_resolved_params(
        self,
        instance_dir: Path,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Prefer params_used recorded by generate_gt.py when available."""
        meta_path = instance_dir / "instance_meta.json"
        if not meta_path.exists():
            return dict(params)

        try:
            instance_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Failed to parse instance_meta.json at %s", meta_path)
            return dict(params)

        params_used = instance_meta.get("params_used")
        if isinstance(params_used, dict):
            return params_used
        return dict(params)

    def _build_expected_outputs(self, task_metadata: dict[str, Any]) -> list[dict[str, str]]:
        """Build the agent-facing list of expected output files.

        **Allowlist**: only ``name`` and ``type`` are forwarded to the
        agent-facing ``task_info.json``.  Fields like ``shape``,
        ``dtype``, ``tolerance``, or any evaluation-specific keys must
        **never** be added here — they would leak scoring criteria to
        the agent under test.
        """
        expected_outputs = []
        for output_file in task_metadata.get("output", {}).get("files", []):
            # ALLOWLIST: only name + type — do not add shape/dtype/tolerance
            expected_outputs.append(
                {"name": output_file["name"], "type": output_file.get("type", "data")}
            )
        return expected_outputs

    @staticmethod
    def _anonymize_task_id(task_id: str) -> str:
        """Return a deterministic, de-semanticized alias for a task_id.

        Agents see this instead of the real task_id so that names like
        ``physics.ks_2d_anisotropic`` do not leak equation / method
        family information (fixes #11).
        """
        import hashlib

        return f"task_{hashlib.sha256(task_id.encode()).hexdigest()[:12]}"

    def _build_agent_task_info(
        self,
        task_metadata: dict[str, Any],
        prompt_level: PromptLevel,
        *,
        effective_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Build the agent-facing task contract exposed inside the workspace."""
        # Always anonymize to prevent method/equation leakage (fixes #11).
        agent_task_id = self._anonymize_task_id(task_metadata["id"])
        timeout = (
            effective_timeout_seconds
            if effective_timeout_seconds is not None
            else self.execution_timeout_seconds
        )
        return {
            "schema_version": 3,
            "task_id": agent_task_id,
            "prompt_level": prompt_level.value,
            "expected_outputs": self._build_expected_outputs(task_metadata),
            "timeout_seconds": timeout,
        }

    def _build_framework_task_info(
        self,
        task_metadata: dict[str, Any],
        *,
        instance_id: str,
        prompt_level: PromptLevel,
        resolved_params: dict[str, Any],
        effective_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Build framework-only metadata used for re-eval and provenance."""
        task_info = self._build_agent_task_info(
            task_metadata, prompt_level,
            effective_timeout_seconds=effective_timeout_seconds,
        )
        task_info["instance_id"] = instance_id
        task_info["parameters"] = resolved_params
        return task_info
