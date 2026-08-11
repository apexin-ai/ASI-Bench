"""Tests for parameter space and instance generation."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.generators.param_space import ParamSpace, _sample_params
from ai4sci_bench.generators.instance_generator import InstanceGenerator
from ai4sci_bench.core.types import PromptLevel


class TestSampleParams:
    def test_deterministic(self):
        """Same seed+idx produces same params."""
        space = {
            "size": {"type": "int", "range": [10, 100], "default": 50},
            "scale": {"type": "float", "range": [0.1, 2.0], "default": 1.0},
            "mode": {"type": "choice", "options": ["a", "b", "c"], "default": "a"},
        }
        p1 = _sample_params(42, 0, space)
        p2 = _sample_params(42, 0, space)
        assert p1 == p2

    def test_different_idx_different_params(self):
        space = {
            "size": {"type": "int", "range": [10, 100]},
            "scale": {"type": "float", "range": [0.1, 2.0]},
        }
        p0 = _sample_params(42, 0, space)
        p1 = _sample_params(42, 1, space)
        assert p0 != p1

    def test_seed_injected(self):
        space = {"size": {"type": "int", "range": [10, 100]}}
        params = _sample_params(42, 0, space)
        assert "seed" in params
        assert isinstance(params["seed"], int)

    def test_int_in_range(self):
        space = {"x": {"type": "int", "range": [5, 15]}}
        for idx in range(20):
            p = _sample_params(0, idx, space)
            assert 5 <= p["x"] <= 15

    def test_float_in_range(self):
        space = {"x": {"type": "float", "range": [0.0, 1.0]}}
        for idx in range(20):
            p = _sample_params(0, idx, space)
            assert 0.0 <= p["x"] <= 1.0

    def test_choice_in_options(self):
        space = {"m": {"type": "choice", "options": ["foo", "bar", "baz"]}}
        for idx in range(20):
            p = _sample_params(0, idx, space)
            assert p["m"] in ["foo", "bar", "baz"]


class TestParamSpace:
    def setup_method(self):
        self.param_defs = {
            "grid_size": {"type": "int", "range": [127, 511], "default": 255},
            "cfl": {"type": "float", "range": [0.5, 2.0], "default": 1.0},
            "ic": {"type": "choice", "options": ["leapfrog", "co_rotate"], "default": "leapfrog"},
        }

    def test_defaults(self):
        ps = ParamSpace(self.param_defs)
        defaults = ps.defaults
        assert defaults["grid_size"] == 255
        assert defaults["cfl"] == 1.0
        assert defaults["ic"] == "leapfrog"

    def test_random_sampling(self):
        ps = ParamSpace(self.param_defs)
        samples = ps.sample(5, strategy="random", seed=42)
        assert len(samples) == 5
        for s in samples:
            assert 127 <= s["grid_size"] <= 511
            assert 0.5 <= s["cfl"] <= 2.0
            assert s["ic"] in ["leapfrog", "co_rotate"]

    def test_lhs_sampling(self):
        ps = ParamSpace(self.param_defs)
        samples = ps.sample(10, strategy="latin_hypercube", seed=42)
        assert len(samples) == 10
        # LHS should have better coverage than random
        grid_sizes = [s["grid_size"] for s in samples]
        assert min(grid_sizes) < 200  # should cover lower range
        assert max(grid_sizes) > 400  # should cover upper range

    def test_grid_sampling(self):
        ps = ParamSpace(self.param_defs)
        samples = ps.sample(20, strategy="grid")
        assert len(samples) <= 20
        assert len(samples) > 0

    def test_boundary_sampling(self):
        ps = ParamSpace(self.param_defs)
        samples = ps.sample(10, strategy="boundary")
        assert len(samples) > 0
        # Should include min and max values
        grid_sizes = [s["grid_size"] for s in samples]
        assert 127 in grid_sizes
        assert 511 in grid_sizes

    def test_unknown_strategy(self):
        ps = ParamSpace(self.param_defs)
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            ps.sample(5, strategy="invalid")

    def test_reproducible(self):
        ps = ParamSpace(self.param_defs)
        s1 = ps.sample(5, strategy="random", seed=123)
        s2 = ps.sample(5, strategy="random", seed=123)
        assert s1 == s2


class TestInfiniteModeSeedReproducibility:
    """Verify that generate_instances_on_the_fly with the same seed produces identical results."""

    def test_same_seed_same_instances(self, sample_task_dir, tmp_dir):
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        gen1 = InstanceGenerator(tasks_dir)
        out1 = tmp_dir / "out1"
        instances1 = gen1.generate_instances_on_the_fly(metadata, seed=42, count=3, output_dir=out1)

        gen2 = InstanceGenerator(tasks_dir)
        out2 = tmp_dir / "out2"
        instances2 = gen2.generate_instances_on_the_fly(metadata, seed=42, count=3, output_dir=out2)

        assert len(instances1) == len(instances2) == 3
        for i1, i2 in zip(instances1, instances2):
            assert i1.parameters == i2.parameters
            assert i1.instance_id == i2.instance_id

    def test_different_seed_different_instances(self, sample_task_dir, tmp_dir):
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        gen = InstanceGenerator(tasks_dir)
        out1 = tmp_dir / "out1"
        instances1 = gen.generate_instances_on_the_fly(metadata, seed=42, count=3, output_dir=out1)

        out2 = tmp_dir / "out2"
        instances2 = gen.generate_instances_on_the_fly(metadata, seed=99, count=3, output_dir=out2)

        params1 = [i.parameters for i in instances1]
        params2 = [i.parameters for i in instances2]
        assert params1 != params2


class TestInstanceGenerator:
    def test_generate_instance(self, sample_task_dir, tmp_dir):
        import yaml
        tasks_dir = sample_task_dir.parent.parent  # tasks/physics/test_task → tasks/
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        output_dir = tmp_dir / "instances"
        instance = generator.generate_instance(
            metadata,
            {"size": 50, "seed": 0},
            output_dir,
        )

        assert instance.task_id == "physics.test_task"
        assert instance.workspace_dir.exists()
        assert (instance.workspace_dir / "prompt.md").exists()
        assert (instance.workspace_dir / "task_info.json").exists()
        assert not (instance.workspace_dir / "framework_task_info.json").exists()
        assert instance.reference_dir.exists()

        task_info = json.loads((instance.workspace_dir / "task_info.json").read_text())
        framework_task_info = json.loads(
            (instance.reference_dir.parent / "framework_task_info.json").read_text()
        )
        assert task_info["schema_version"] == 3
        # task_id is anonymized (fixes #11) — should NOT be the raw id
        assert task_info["task_id"] != "physics.test_task"
        assert task_info["task_id"].startswith("task_")
        assert task_info["prompt_level"] == "b2"
        assert "parameters" not in task_info
        assert "instance_id" not in task_info
        assert framework_task_info["instance_id"] == instance.instance_id
        assert framework_task_info["parameters"] == instance.parameters

    def test_generate_instance_backfills_task_prompt_templates(self, sample_task_dir, tmp_dir):
        """If generate_gt.py only writes data/reference, prompt templates still reach workspace."""
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        generate_gt = sample_task_dir / "generate_gt.py"
        original = generate_gt.read_text(encoding="utf-8")
        prompt_generation = (
            '    for level in ["b1", "b2", "b3", "b4"]:\n'
            '        Path(output_dir, f"prompt_{level}.md").write_text('
            'f"# Test\\nSize={p[\'size\']}")\n'
        )
        without_generated_prompts = original.replace(prompt_generation, "")
        assert without_generated_prompts != original, (
            "sample generate_gt fixture no longer contains the expected prompt block"
        )
        generate_gt.write_text(without_generated_prompts, encoding="utf-8")

        instance = generator.generate_instance(
            metadata,
            {"size": 50, "seed": 0},
            tmp_dir / "instances",
            prompt_level=PromptLevel.B2,
        )

        assert (instance.reference_dir.parent / "prompt_b2.md").read_text() == (
            sample_task_dir / "prompt_b2.md"
        ).read_text()
        assert (instance.workspace_dir / "prompt.md").read_text() == (
            sample_task_dir / "prompt_b2.md"
        ).read_text()

    def test_generate_instance_uses_params_used_from_instance_meta(self, sample_task_dir, tmp_dir):
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        output_dir = tmp_dir / "instances"
        instance = generator.generate_instance(
            metadata,
            {"size": 50},
            output_dir,
        )

        assert instance.parameters == {"size": 50, "seed": 0}
        task_info = json.loads((instance.workspace_dir / "task_info.json").read_text())
        framework_task_info = json.loads(
            (instance.reference_dir.parent / "framework_task_info.json").read_text()
        )
        assert "parameters" not in task_info
        assert framework_task_info["parameters"] == {"size": 50, "seed": 0}

    @patch("ai4sci_bench.generators.instance_generator.validate_sandbox_mode")
    @patch("ai4sci_bench.generators.instance_generator.LinuxNSSandbox")
    def test_run_generate_gt_uses_linux_ns_sandbox(
        self,
        mock_linux_ns_cls,
        mock_validate_sandbox,
        sample_task_dir,
        tmp_dir,
    ):
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir, sandbox="linux_ns")
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        mock_linux_ns = MagicMock()
        mock_linux_ns.run_command.return_value = (True, "ok", "", "")
        mock_linux_ns_cls.return_value = mock_linux_ns

        with (
            patch.object(
                generator.task_loader,
                "load_generate_gt_module",
                side_effect=RuntimeError("force subprocess"),
            ),
            patch.object(generator, "_get_task_environment", return_value=None),
        ):
            generator._run_generate_gt(
                metadata,
                tmp_dir / "instance",
                {"size": 50, "seed": 0},
            )

        mock_linux_ns.run_command.assert_called_once()
        call_kwargs = mock_linux_ns.run_command.call_args.kwargs
        assert call_kwargs["cwd"] == sample_task_dir
        assert call_kwargs["command"][0] == sys.executable
        assert call_kwargs["command"][1] == "generate_gt.py"

    def test_workspace_is_isolated_from_instance_dir(self, sample_task_dir, tmp_dir):
        """Issue #7 + #13: workspace must be in an isolated temp directory so
        agents cannot traverse up to read instance metadata, reference data, or
        sibling workspaces of other prompt levels."""
        import yaml
        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        output_dir = tmp_dir / "instances"
        instance = generator.generate_instance(
            metadata,
            {"size": 50, "seed": 0},
            output_dir,
        )

        # workspace_dir must NOT be under instance_dir
        instance_dir = output_dir / instance.instance_id
        assert instance_dir.exists(), "instance dir should exist"
        assert instance.reference_dir.exists(), "reference dir should exist"

        # Workspace is in a temp directory, not under instance_dir
        try:
            instance.workspace_dir.relative_to(instance_dir)
            assert False, "workspace_dir should NOT be inside instance_dir"
        except ValueError:
            pass  # Expected: workspace is not relative to instance_dir

        # From workspace, parent traversal should NOT reach instance_meta.json or reference/
        ws_parent = instance.workspace_dir.parent
        assert not (ws_parent / "reference").exists()
        assert not (ws_parent / "instance_meta.json").exists()

        # A symlink or .path file should exist under _workspaces/ for debugging
        ws_link_root = output_dir / "_workspaces" / instance.instance_id
        assert ws_link_root.exists(), "_workspaces/ debug directory should exist"

    def test_workspace_strips_agent_memory_files(self, sample_task_dir, tmp_dir):
        """Workspace preparation must scrub CLAUDE.md / AGENTS.md / .claude/
        so CLI agents cannot pick up leaked prompt context from the repo."""
        import yaml
        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        output_dir = tmp_dir / "instances"
        instance = generator.generate_instance(
            metadata,
            {"size": 50, "seed": 0},
            output_dir,
        )

        # Simulate a stray CLAUDE.md and AGENTS.md being present in the
        # freshly-prepared workspace (e.g. from a previous run or from a
        # bad data copy). Re-run _prepare_workspace and verify cleanup.
        (instance.workspace_dir / "CLAUDE.md").write_text("# leaked hints")
        (instance.workspace_dir / "AGENTS.md").write_text("# more leaked hints")
        (instance.workspace_dir / ".claude").mkdir(exist_ok=True)
        (instance.workspace_dir / ".claude" / "settings.json").write_text("{}")

        instance_dir = output_dir / instance.instance_id
        resolved = generator._load_resolved_params(instance_dir, {"size": 50, "seed": 0})
        generator._prepare_workspace(
            instance_dir,
            instance.workspace_dir,
            instance.instance_id,
            instance.prompt_level,
            metadata,
            resolved,
        )

        assert not (instance.workspace_dir / "CLAUDE.md").exists()
        assert not (instance.workspace_dir / "AGENTS.md").exists()
        assert not (instance.workspace_dir / ".claude").exists()
        # Normal workspace files are still intact.
        assert (instance.workspace_dir / "prompt.md").exists()
        assert (instance.workspace_dir / "task_info.json").exists()

    def test_strip_agent_memory_handles_missing_workspace(self, sample_task_dir, tmp_dir):
        """The scrub helper must be a no-op when the workspace does not exist."""
        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        # No exception when the directory is missing.
        generator._strip_agent_memory_files(tmp_dir / "nope")

    def test_build_instance_id(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)

        id1 = generator._build_instance_id("physics.test", {"size": 50, "seed": 0})
        id2 = generator._build_instance_id("physics.test", {"size": 100, "seed": 1})
        assert id1 != id2
        assert "physics.test__" in id1

    def test_run_generate_gt_uses_task_sandbox_subprocess(self, sample_task_dir, tmp_dir):
        import yaml
        from ai4sci_bench.runner.task_env import TaskEnvironment

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir, sandbox="task", repo_root=tmp_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"
        metadata["_runtime_python"] = ">=3.11"
        metadata["_runtime_packages"] = ["taichi>=1.7.4"]

        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "bin" / "python",
            bin_dir=tmp_dir / "env" / "bin",
            cache_key="abc123",
            python_requirement=">=3.11",
            packages=["taichi>=1.7.4"],
            cache_hit=True,
        )

        with (
            patch.object(generator.task_env_manager, "ensure_env", return_value=fake_env),
            patch.object(generator.task_loader, "load_generate_gt_module") as mock_loader,
            patch("ai4sci_bench.generators.instance_generator.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            mock_run.return_value.stdout = ""

            generator._run_generate_gt(
                metadata,
                tmp_dir / "out",
                {"size": 50, "seed": 0},
            )

        mock_loader.assert_not_called()
        cmd = mock_run.call_args.kwargs.get("args", mock_run.call_args.args[0])
        assert cmd[0] == str(fake_env.python_executable)
        assert cmd[1] == "generate_gt.py"
        assert cmd[3] == str((tmp_dir / "out").resolve())
        assert mock_run.call_args.kwargs["env"]["VIRTUAL_ENV"] == str(fake_env.env_dir)

    def test_run_generate_gt_uses_task_env_subprocess_for_os_sandbox(self, sample_task_dir, tmp_dir):
        import yaml
        from ai4sci_bench.runner.task_env import TaskEnvironment

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir, sandbox="os", repo_root=tmp_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"
        metadata["_runtime_python"] = ">=3.11"
        metadata["_runtime_packages"] = ["taichi>=1.7.4"]

        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "bin" / "python",
            bin_dir=tmp_dir / "env" / "bin",
            cache_key="abc123",
            python_requirement=">=3.11",
            packages=["taichi>=1.7.4"],
            cache_hit=True,
        )

        with (
            patch.object(generator.task_env_manager, "ensure_env", return_value=fake_env),
            patch.object(generator.task_loader, "load_generate_gt_module") as mock_loader,
            patch("ai4sci_bench.generators.instance_generator.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            mock_run.return_value.stdout = ""

            generator._run_generate_gt(
                metadata,
                tmp_dir / "out",
                {"size": 50, "seed": 0},
            )

        mock_loader.assert_not_called()
        cmd = mock_run.call_args.kwargs.get("args", mock_run.call_args.args[0])
        assert cmd[0] == str(fake_env.python_executable)
        assert cmd[1] == "generate_gt.py"
        assert mock_run.call_args.kwargs["env"]["VIRTUAL_ENV"] == str(fake_env.env_dir)


    def test_run_generate_gt_reads_timeout_from_generation_config(self, sample_task_dir, tmp_dir):
        """GT generation timeout must come from task_metadata['generation']['timeout_seconds'],
        not from params."""
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        with (
            patch.object(
                generator.task_loader,
                "load_generate_gt_module",
                side_effect=RuntimeError("force subprocess"),
            ),
            patch.object(generator, "_get_task_environment", return_value=None),
            patch("ai4sci_bench.generators.instance_generator.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            mock_run.return_value.stdout = ""

            generator._run_generate_gt(
                metadata,
                tmp_dir / "instance",
                {"size": 50, "seed": 0},
            )

        assert mock_run.call_args.kwargs["timeout"] == 60

    def test_run_generate_gt_defaults_timeout_to_300(self, sample_task_dir, tmp_dir):
        """When generation.timeout_seconds is not set, default to 300."""
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"
        del metadata["generation"]["timeout_seconds"]

        with (
            patch.object(
                generator.task_loader,
                "load_generate_gt_module",
                side_effect=RuntimeError("force subprocess"),
            ),
            patch.object(generator, "_get_task_environment", return_value=None),
            patch("ai4sci_bench.generators.instance_generator.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            mock_run.return_value.stdout = ""

            generator._run_generate_gt(
                metadata,
                tmp_dir / "instance",
                {"size": 50, "seed": 0},
            )

        assert mock_run.call_args.kwargs["timeout"] == 300

    def test_run_generate_gt_uses_effective_timeout_when_generation_timeout_missing(
        self, sample_task_dir, tmp_dir
    ):
        """validate --preflight/run pass the resolved task timeout into GT generation."""
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"
        del metadata["generation"]["timeout_seconds"]

        with (
            patch.object(
                generator.task_loader,
                "load_generate_gt_module",
                side_effect=RuntimeError("force subprocess"),
            ),
            patch.object(generator, "_get_task_environment", return_value=None),
            patch("ai4sci_bench.generators.instance_generator.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            mock_run.return_value.stdout = ""

            generator._run_generate_gt(
                metadata,
                tmp_dir / "instance",
                {"size": 50, "seed": 0},
                effective_timeout_seconds=1800,
            )

        assert mock_run.call_args.kwargs["timeout"] == 1800

    def test_generate_instance_passes_effective_timeout_to_generate_gt(
        self, sample_task_dir, tmp_dir
    ):
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"
        del metadata["generation"]["timeout_seconds"]

        def fake_run_generate_gt(_metadata, output_dir, _params, *, effective_timeout_seconds=None):
            assert effective_timeout_seconds == 1800
            (output_dir / "reference").mkdir(parents=True)
            (output_dir / "reference" / "output_ref.npy").write_bytes(b"ref")
            (output_dir / "data").mkdir()
            (output_dir / "prompt_b2.md").write_text("prompt", encoding="utf-8")

        with patch.object(generator, "_run_generate_gt", side_effect=fake_run_generate_gt):
            instance = generator.generate_instance(
                metadata,
                {"size": 50, "seed": 0},
                tmp_dir / "instances",
                effective_timeout_seconds=1800,
            )

        assert instance.effective_timeout_seconds == 1800

    def test_run_generate_gt_ignores_params_timeout(self, sample_task_dir, tmp_dir):
        """params['timeout_seconds'] must NOT override generation config.
        Regression test for bug where params.get('timeout_seconds', 300) was used."""
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        with (
            patch.object(
                generator.task_loader,
                "load_generate_gt_module",
                side_effect=RuntimeError("force subprocess"),
            ),
            patch.object(generator, "_get_task_environment", return_value=None),
            patch("ai4sci_bench.generators.instance_generator.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            mock_run.return_value.stdout = ""

            generator._run_generate_gt(
                metadata,
                tmp_dir / "instance",
                {"size": 50, "seed": 0, "timeout_seconds": 9999},
            )

        assert mock_run.call_args.kwargs["timeout"] == 60

    @patch("ai4sci_bench.generators.instance_generator.validate_sandbox_mode")
    @patch("ai4sci_bench.generators.instance_generator.LinuxNSSandbox")
    def test_run_generate_gt_linux_ns_uses_generation_timeout(
        self,
        mock_linux_ns_cls,
        mock_validate_sandbox,
        sample_task_dir,
        tmp_dir,
    ):
        """Linux namespace sandbox path should also use generation.timeout_seconds."""
        import yaml

        tasks_dir = sample_task_dir.parent.parent
        generator = InstanceGenerator(tasks_dir, sandbox="linux_ns")
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        mock_linux_ns = MagicMock()
        mock_linux_ns.run_command.return_value = (True, "ok", "", "")
        mock_linux_ns_cls.return_value = mock_linux_ns

        with (
            patch.object(
                generator.task_loader,
                "load_generate_gt_module",
                side_effect=RuntimeError("force subprocess"),
            ),
            patch.object(generator, "_get_task_environment", return_value=None),
        ):
            generator._run_generate_gt(
                metadata,
                tmp_dir / "instance",
                {"size": 50, "seed": 0},
            )

        call_kwargs = mock_linux_ns.run_command.call_args.kwargs
        assert call_kwargs["timeout"] == 60


class TestInstanceGeneratorHelpers:
    """Tests for helper methods: _build_expected_outputs, _build_agent_task_info,
    _build_framework_task_info, _build_instance_id edge cases, _load_resolved_params,
    _strip_agent_memory_files edge cases."""

    def test_build_expected_outputs(self, sample_task_dir, tmp_dir):
        import yaml
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        outputs = gen._build_expected_outputs(metadata)
        assert len(outputs) == 1
        assert outputs[0]["name"] == "output.npy"
        assert outputs[0]["type"] == "data"

    def test_build_expected_outputs_empty(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        assert gen._build_expected_outputs({}) == []
        assert gen._build_expected_outputs({"output": {}}) == []

    def test_build_agent_task_info(self, sample_task_dir, tmp_dir):
        import yaml
        from ai4sci_bench.core.types import PromptLevel
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        info = gen._build_agent_task_info(metadata, PromptLevel.B1)
        assert info["schema_version"] == 3
        # task_id is anonymized — must NOT be the raw id
        assert info["task_id"] != "physics.test_task"
        assert info["task_id"].startswith("task_")
        assert info["prompt_level"] == "b1"
        assert "expected_outputs" in info
        # Must NOT leak instance_id, parameters, requires_gpu
        assert "instance_id" not in info
        assert "parameters" not in info
        assert "requires_gpu" not in info

    def test_build_framework_task_info(self, sample_task_dir, tmp_dir):
        import yaml
        from ai4sci_bench.core.types import PromptLevel
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        info = gen._build_framework_task_info(
            metadata,
            instance_id="physics.test_task__size50",
            prompt_level=PromptLevel.B2,
            resolved_params={"size": 50, "seed": 0},
        )
        assert info["instance_id"] == "physics.test_task__size50"
        assert info["parameters"] == {"size": 50, "seed": 0}
        assert info["schema_version"] == 3

    def test_build_instance_id_long_params_truncated(self, sample_task_dir, tmp_dir):
        """Instance IDs with very long param strings should be hash-truncated."""
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        # Create params that produce a very long string
        long_params = {f"param_{i}": i for i in range(20)}
        instance_id = gen._build_instance_id("test.task", long_params)
        # Truncated IDs are capped at ~60 chars + hash
        assert len(instance_id) < 200
        assert instance_id.startswith("test.task__")

    def test_build_instance_id_deterministic(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        id1 = gen._build_instance_id("t", {"a": 1, "b": 2})
        id2 = gen._build_instance_id("t", {"a": 1, "b": 2})
        assert id1 == id2

    def test_load_resolved_params_missing_meta(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        # No instance_meta.json → falls back to input params
        result = gen._load_resolved_params(tmp_dir / "nonexistent", {"size": 99})
        assert result == {"size": 99}

    def test_load_resolved_params_malformed_json(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        inst_dir = tmp_dir / "inst"
        inst_dir.mkdir()
        (inst_dir / "instance_meta.json").write_text("not json")
        result = gen._load_resolved_params(inst_dir, {"size": 42})
        assert result == {"size": 42}

    def test_load_resolved_params_no_params_used_key(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        inst_dir = tmp_dir / "inst"
        inst_dir.mkdir()
        (inst_dir / "instance_meta.json").write_text(json.dumps({"other": "data"}))
        result = gen._load_resolved_params(inst_dir, {"size": 7})
        assert result == {"size": 7}

    def test_strip_agent_memory_files_symlink(self, sample_task_dir, tmp_dir):
        """Symlinked CLAUDE.md should be unlinked (not rmtree'd)."""
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        ws = tmp_dir / "ws"
        ws.mkdir()
        real_file = tmp_dir / "real_claude.md"
        real_file.write_text("real")
        (ws / "CLAUDE.md").symlink_to(real_file)
        gen._strip_agent_memory_files(ws)
        assert not (ws / "CLAUDE.md").exists()
        # Original file should still exist
        assert real_file.exists()

    def test_strip_agent_memory_files_claude_local_md(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        ws = tmp_dir / "ws"
        ws.mkdir()
        (ws / "CLAUDE.local.md").write_text("local secrets")
        gen._strip_agent_memory_files(ws)
        assert not (ws / "CLAUDE.local.md").exists()
