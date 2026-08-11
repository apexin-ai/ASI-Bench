"""Tests for B4 prompt level and finite generation mode.

Covers:
  - PromptLevel.B4 enum value
  - GenerationMode enum
  - TaskLoader parsing of generation mode/settings
  - InstanceGenerator finite mode (real-time and precomputed)
  - CLI validate with B4 prompt
  - CLI generate with --precompute
  - VIC leapfrog B4 prompt validation
  - Template files include B4
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ai4sci_bench.core.types import GenerationMode, PromptLevel, TaskInstance


# ──────────────────────────────────────────────────────────────────────────────
# PromptLevel.B4
# ──────────────────────────────────────────────────────────────────────────────


class TestPromptLevelB4:
    def test_b4_enum_exists(self):
        """B4 is a valid PromptLevel."""
        assert PromptLevel.B4 == PromptLevel("b4")
        assert PromptLevel.B4.value == "b4"

    def test_all_four_levels(self):
        """All four prompt levels exist."""
        levels = [PromptLevel.B1, PromptLevel.B2, PromptLevel.B3, PromptLevel.B4]
        assert len(levels) == 4
        assert [l.value for l in levels] == ["b1", "b2", "b3", "b4"]

    def test_b4_in_iteration(self):
        """B4 appears when iterating PromptLevel."""
        values = [l.value for l in PromptLevel]
        assert "b4" in values


# ──────────────────────────────────────────────────────────────────────────────
# GenerationMode enum
# ──────────────────────────────────────────────────────────────────────────────


class TestGenerationMode:
    def test_infinite_mode(self):
        assert GenerationMode.INFINITE == GenerationMode("infinite")
        assert GenerationMode.INFINITE.value == "infinite"

    def test_finite_mode(self):
        assert GenerationMode.FINITE == GenerationMode("finite")
        assert GenerationMode.FINITE.value == "finite"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            GenerationMode("unknown")


# ──────────────────────────────────────────────────────────────────────────────
# TaskLoader — generation mode parsing
# ──────────────────────────────────────────────────────────────────────────────


class TestTaskLoaderGenerationMode:
    def test_default_infinite_mode(self, tmp_path):
        """Tasks without explicit mode default to infinite."""
        from ai4sci_bench.core.task import TaskLoader

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(yaml.dump({
            "id": "test.task",
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }))
        loader = TaskLoader(tmp_path)
        metadata = loader.load_task_metadata(task_yaml)
        assert metadata["_generation_mode"] == GenerationMode.INFINITE
        assert metadata["_generation_precomputed"] is False
        assert metadata["_generation_settings"] == []

    def test_finite_mode_parsed(self, tmp_path):
        """Finite mode with settings is parsed correctly."""
        from ai4sci_bench.core.task import TaskLoader

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(yaml.dump({
            "id": "test.task",
            "generation": {
                "script": "generate_gt.py",
                "mode": "finite",
                "precomputed": True,
                "settings": [
                    {"id": "s1", "grid_size": 64, "seed": 1},
                    {"id": "s2", "grid_size": 128, "seed": 2},
                ],
            },
        }))
        loader = TaskLoader(tmp_path)
        metadata = loader.load_task_metadata(task_yaml)
        assert metadata["_generation_mode"] == GenerationMode.FINITE
        assert metadata["_generation_precomputed"] is True
        assert len(metadata["_generation_settings"]) == 2
        assert metadata["_generation_settings"][0]["id"] == "s1"

    def test_finite_realtime_mode(self, tmp_path):
        """Finite mode with precomputed=false."""
        from ai4sci_bench.core.task import TaskLoader

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(yaml.dump({
            "id": "test.task",
            "generation": {
                "mode": "finite",
                "precomputed": False,
                "settings": [{"grid_size": 64, "seed": 1}],
            },
        }))
        loader = TaskLoader(tmp_path)
        metadata = loader.load_task_metadata(task_yaml)
        assert metadata["_generation_mode"] == GenerationMode.FINITE
        assert metadata["_generation_precomputed"] is False


# ──────────────────────────────────────────────────────────────────────────────
# InstanceGenerator — finite mode
# ──────────────────────────────────────────────────────────────────────────────


class TestInstanceGeneratorFiniteMode:
    def _make_task_metadata(self, task_dir, mode="finite", precomputed=False, settings=None):
        """Helper to build task metadata dict."""
        return {
            "id": "test.finite_task",
            "_task_dir": task_dir,
            "_generation_mode": GenerationMode(mode),
            "_generation_precomputed": precomputed,
            "_generation_settings": settings or [],
            "generation": {"script": "generate_gt.py"},
            "output": {"files": []},
            "difficulty": {},
        }

    def test_finite_no_settings_raises(self, tmp_path):
        """Finite mode with empty settings list raises ValueError."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        metadata = self._make_task_metadata(task_dir, settings=[])
        gen = InstanceGenerator(tmp_path)

        with pytest.raises(ValueError, match="no settings"):
            gen.generate_instances_on_the_fly(
                metadata, seed=42, count=1, output_dir=tmp_path / "out"
            )

    def test_finite_realtime_calls_generate_instance(self, tmp_path):
        """Finite realtime mode calls generate_instance for each setting."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()

        settings = [
            {"grid_size": 64, "seed": 1},
            {"grid_size": 128, "seed": 2},
        ]
        metadata = self._make_task_metadata(task_dir, settings=settings)

        gen = InstanceGenerator(tmp_path)

        # Mock generate_instance to avoid running actual GT generation
        mock_instance = TaskInstance(
            task_id="test.finite_task",
            instance_id="test_inst",
            task_dir=task_dir,
            workspace_dir=tmp_path / "ws",
            reference_dir=tmp_path / "ref",
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata=metadata,
        )
        gen.generate_instance = MagicMock(return_value=mock_instance)

        instances = gen.generate_instances_on_the_fly(
            metadata, seed=42, count=2, output_dir=tmp_path / "out"
        )

        assert len(instances) == 2
        assert gen.generate_instance.call_count == 2

    def test_finite_setting_selection_deterministic(self, tmp_path):
        """Same seed produces same setting selection order."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()

        settings = [
            {"grid_size": 64, "seed": 1},
            {"grid_size": 128, "seed": 2},
            {"grid_size": 256, "seed": 3},
        ]
        metadata = self._make_task_metadata(task_dir, settings=settings)

        gen = InstanceGenerator(tmp_path)
        called_params = []
        mock_instance = TaskInstance(
            task_id="test.finite_task", instance_id="x",
            task_dir=task_dir, workspace_dir=tmp_path, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={}, metadata=metadata,
        )

        def capture_call(meta, params, out_dir, level, **kwargs):
            called_params.append(params.copy())
            return mock_instance

        gen.generate_instance = capture_call

        gen.generate_instances_on_the_fly(metadata, seed=42, count=3, output_dir=tmp_path / "out1")
        first_run = called_params.copy()
        called_params.clear()

        gen.generate_instances_on_the_fly(metadata, seed=42, count=3, output_dir=tmp_path / "out2")
        second_run = called_params.copy()

        assert first_run == second_run

    def test_finite_cycles_when_count_exceeds_settings(self, tmp_path):
        """When count > len(settings), settings are cycled."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()

        settings = [{"grid_size": 64, "seed": 1}]
        metadata = self._make_task_metadata(task_dir, settings=settings)

        gen = InstanceGenerator(tmp_path)
        mock_instance = TaskInstance(
            task_id="test.finite_task", instance_id="x",
            task_dir=task_dir, workspace_dir=tmp_path, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={}, metadata=metadata,
        )
        gen.generate_instance = MagicMock(return_value=mock_instance)

        instances = gen.generate_instances_on_the_fly(
            metadata, seed=42, count=3, output_dir=tmp_path / "out"
        )
        assert len(instances) == 3

    def test_precomputed_missing_id_raises(self, tmp_path):
        """Precomputed mode requires 'id' in each setting."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "generate_gt.py").write_text("# stub")

        settings = [{"grid_size": 64, "seed": 1}]  # no 'id' field
        metadata = self._make_task_metadata(
            task_dir, precomputed=True, settings=settings
        )

        gen = InstanceGenerator(tmp_path)

        with pytest.raises(ValueError, match="id"):
            gen.generate_instances_on_the_fly(
                metadata, seed=42, count=1, output_dir=tmp_path / "out"
            )

    def test_precomputed_missing_reference_dir_raises(self, tmp_path):
        """Precomputed mode raises if reference/<id>/ doesn't exist."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "generate_gt.py").write_text("# stub")

        settings = [{"id": "s1", "grid_size": 64, "seed": 1}]
        metadata = self._make_task_metadata(
            task_dir, precomputed=True, settings=settings
        )

        gen = InstanceGenerator(tmp_path)

        with pytest.raises(FileNotFoundError, match="Precomputed reference"):
            gen.generate_instances_on_the_fly(
                metadata, seed=42, count=1, output_dir=tmp_path / "out"
            )

    def test_precomputed_loads_reference(self, tmp_path):
        """Precomputed mode loads the generated precompute bundle layout."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[]\nDEFAULT_PARAMS={}\n"
            "def generate(o,p): pass\n"
        )

        # Create the same nested bundle layout produced by `generate --precompute`.
        bundle_dir = task_dir / "reference" / "s1"
        ref_dir = bundle_dir / "reference"
        data_dir = bundle_dir / "data"
        ref_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        (ref_dir / "result_ref.npy").write_text("fake data")
        (data_dir / "input.npy").write_text("fake input")
        for level in ["b1", "b2", "b3", "b4"]:
            (bundle_dir / f"prompt_{level}.md").write_text(f"Rendered prompt {level}")

        settings = [{"id": "s1", "grid_size": 64, "seed": 1}]
        metadata = self._make_task_metadata(
            task_dir, precomputed=True, settings=settings
        )

        gen = InstanceGenerator(tmp_path)
        out_dir = tmp_path / "out"

        instances = gen.generate_instances_on_the_fly(
            metadata, seed=42, count=1, output_dir=out_dir
        )

        assert len(instances) == 1
        inst = instances[0]
        assert inst.reference_dir.exists()
        assert (inst.reference_dir / "result_ref.npy").exists()
        assert not (inst.reference_dir / "reference").exists()
        assert (inst.workspace_dir / "data" / "input.npy").exists()
        assert (inst.workspace_dir / "prompt.md").read_text() == "Rendered prompt b2"
        assert inst.workspace_dir.exists()

    def test_precomputed_loads_legacy_flat_reference_layout(self, tmp_path):
        """Older precomputed layouts with files at the bundle root still load."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[]\nDEFAULT_PARAMS={}\n"
            "def generate(o,p): pass\n"
        )

        bundle_dir = task_dir / "reference" / "s1"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "result_ref.npy").write_text("fake data")
        for level in ["b1", "b2", "b3", "b4"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Prompt {level}")

        settings = [{"id": "s1", "grid_size": 64, "seed": 1}]
        metadata = self._make_task_metadata(
            task_dir, precomputed=True, settings=settings
        )

        gen = InstanceGenerator(tmp_path)
        instances = gen.generate_instances_on_the_fly(
            metadata, seed=42, count=1, output_dir=tmp_path / "out"
        )

        inst = instances[0]
        assert (inst.reference_dir / "result_ref.npy").exists()

    def test_infinite_mode_ignores_settings(self, tmp_path):
        """Infinite mode uses parameter sampling, not settings."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "task"
        task_dir.mkdir()

        metadata = {
            "id": "test.task",
            "_task_dir": task_dir,
            "_generation_mode": GenerationMode.INFINITE,
            "_generation_precomputed": False,
            "_generation_settings": [{"id": "s1"}],  # should be ignored
            "generation": {
                "parameters": {
                    "x": {"type": "int", "range": [1, 10], "default": 5},
                },
            },
            "output": {"files": []},
            "difficulty": {},
        }

        gen = InstanceGenerator(tmp_path)
        mock_instance = TaskInstance(
            task_id="test.task", instance_id="x",
            task_dir=task_dir, workspace_dir=tmp_path, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={}, metadata=metadata,
        )
        gen.generate_instance = MagicMock(return_value=mock_instance)

        instances = gen.generate_instances_on_the_fly(
            metadata, seed=42, count=2, output_dir=tmp_path / "out"
        )
        assert len(instances) == 2
        # Should have been called with sampled params, not settings
        for call in gen.generate_instance.call_args_list:
            params = call[0][1]
            assert "x" in params


# ──────────────────────────────────────────────────────────────────────────────
# CLI — validate with B4
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIValidateB4:
    def test_validate_b4_declared_but_missing(self, tmp_path):
        """Validate reports error when b4 is declared but file missing."""
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli

        task_dir = tmp_path / "tasks" / "test" / "mytask"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "test.mytask",
            "name": "Test",
            "domain": "test",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md",
                        "b3": "prompt_b3.md", "b4": "prompt_b4.md"},
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Content for {level}")

        # generate_gt.py with required interface
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[]\nDEFAULT_PARAMS={}\n"
            "def generate(o,p): pass\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "--task", "test.mytask",
            "--tasks-dir", str(tmp_path / "tasks"),
        ])
        assert result.exit_code != 0
        assert "prompt_b4.md" in result.output

    def test_validate_b4_present_passes(self, tmp_path):
        """Validate passes when b4 is declared and file exists."""
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli

        task_dir = tmp_path / "tasks" / "test" / "mytask"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "test.mytask",
            "name": "Test",
            "domain": "test",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md",
                        "b3": "prompt_b3.md", "b4": "prompt_b4.md"},
        }))
        for level in ["b1", "b2", "b3", "b4"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Content for {level}")
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[]\nDEFAULT_PARAMS={}\n"
            "def generate(o,p): pass\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "--task", "test.mytask",
            "--tasks-dir", str(tmp_path / "tasks"),
        ])
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_validate_no_b4_still_passes(self, tmp_path):
        """Tasks without b4 in prompts still validate fine."""
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli

        task_dir = tmp_path / "tasks" / "test" / "mytask"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "test.mytask",
            "name": "Test",
            "domain": "test",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md",
                        "b3": "prompt_b3.md"},
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Content for {level}")
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[]\nDEFAULT_PARAMS={}\n"
            "def generate(o,p): pass\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "--task", "test.mytask",
            "--tasks-dir", str(tmp_path / "tasks"),
        ])
        assert result.exit_code == 0
        assert "PASSED" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# CLI — generate --precompute
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIPrecompute:
    def test_precompute_requires_finite_mode(self, tmp_path):
        """--precompute errors when task is infinite mode."""
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli

        task_dir = tmp_path / "tasks" / "test" / "mytask"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "test.mytask",
            "name": "Test",
            "domain": "test",
            "status": "test",
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }))
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[]\nDEFAULT_PARAMS={}\n"
            "def generate(o,p): pass\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate", "--task", "test.mytask",
            "--tasks-dir", str(tmp_path / "tasks"),
            "--precompute",
        ])
        assert result.exit_code != 0
        assert "finite" in result.output.lower()

    def test_precompute_no_settings_errors(self, tmp_path):
        """--precompute errors when no settings defined."""
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli

        task_dir = tmp_path / "tasks" / "test" / "mytask"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "test.mytask",
            "name": "Test",
            "domain": "test",
            "status": "test",
            "generation": {"script": "generate_gt.py", "mode": "finite"},
        }))
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[]\nDEFAULT_PARAMS={}\n"
            "def generate(o,p): pass\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate", "--task", "test.mytask",
            "--tasks-dir", str(tmp_path / "tasks"),
            "--precompute",
        ])
        assert result.exit_code != 0
        assert "settings" in result.output.lower()

    def test_precompute_runs_generate_gt(self, tmp_path):
        """--precompute invokes generate_gt.py for each setting."""
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli

        task_dir = tmp_path / "tasks" / "test" / "mytask"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "test.mytask",
            "name": "Test",
            "domain": "test",
            "status": "test",
            "generation": {
                "script": "generate_gt.py",
                "mode": "finite",
                "precomputed": True,
                "settings": [
                    {"id": "s1", "grid_size": 64, "seed": 1},
                    {"id": "s2", "grid_size": 128, "seed": 2},
                ],
            },
        }))

        # generate_gt.py that creates a marker file
        (task_dir / "generate_gt.py").write_text('''
import json, time
from pathlib import Path
INPUT_SPEC=[]
OUTPUT_SPEC=[]
DEFAULT_PARAMS={"seed": 0}

def generate(output_dir, params):
    p = {**DEFAULT_PARAMS, **params}
    ref_dir = Path(output_dir) / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "result.txt").write_text(json.dumps(p))
    meta = {"params_used": p, "input_files": [], "reference_files": ["result.txt"],
            "generation_time_seconds": 0.0}
    (Path(output_dir) / "instance_meta.json").write_text(json.dumps(meta))
    return meta
''')

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate", "--task", "test.mytask",
            "--tasks-dir", str(tmp_path / "tasks"),
            "--precompute",
        ])
        assert result.exit_code == 0
        assert "Precomputed" in result.output
        # Check that generated bundles were created under reference/<id>/.
        assert (task_dir / "reference" / "s1").exists()
        assert (task_dir / "reference" / "s2").exists()
        assert (task_dir / "reference" / "s1" / "reference" / "result.txt").exists()
        assert (task_dir / "reference" / "s2" / "reference" / "result.txt").exists()


class TestCLIGenerateFinite:
    def test_generate_multiple_instances_uses_finite_settings(self, tmp_path):
        """`generate` without --params should use finite settings, not parameters."""
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli

        task_dir = tmp_path / "tasks" / "test" / "finite_task"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "test.finite_task",
            "name": "Finite Task",
            "domain": "test",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md", "b3": "prompt_b3.md"},
            "generation": {
                "script": "generate_gt.py",
                "mode": "finite",
                "parameters": {
                    "wrong_param": {"type": "int", "range": [100, 200], "default": 123},
                },
                "settings": [
                    {"id": "s1", "grid_size": 64, "seed": 1},
                    {"id": "s2", "grid_size": 128, "seed": 2},
                ],
            },
            "output": {"files": []},
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Prompt {level}")
        (task_dir / "generate_gt.py").write_text(
            """
import json
from pathlib import Path

INPUT_SPEC=[]
OUTPUT_SPEC=[]
DEFAULT_PARAMS={}

def generate(output_dir, params):
    output_dir = Path(output_dir)
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "params.json").write_text(json.dumps(params, sort_keys=True))
    for level in ["b1", "b2", "b3"]:
        (output_dir / f"prompt_{level}.md").write_text(f"Prompt {level}")
    return {"params_used": params}
"""
        )

        runner = CliRunner()
        output_dir = tmp_path / "instances"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--task",
                "test.finite_task",
                "--tasks-dir",
                str(tmp_path / "tasks"),
                "--output-dir",
                str(output_dir),
                "--instances-per-task",
                "2",
            ],
        )

        assert result.exit_code == 0
        generated_dirs = sorted(
            p for p in output_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
        assert len(generated_dirs) == 2

        params_seen = []
        for instance_dir in generated_dirs:
            params = json.loads((instance_dir / "reference" / "params.json").read_text())
            params_seen.append(params)
            assert "grid_size" in params
            assert "wrong_param" not in params

        assert sorted(p["grid_size"] for p in params_seen) == [64, 128]


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator — B4 prompt level in run
# ──────────────────────────────────────────────────────────────────────────────


class TestOrchestratorB4:
    def test_b4_prompt_level_accepted(self):
        """RunConfig accepts 'b4' as a valid prompt level."""
        from ai4sci_bench.runner.orchestrator import RunConfig
        from ai4sci_bench.core.agent_interface import AgentAdapter

        mock_agent = MagicMock(spec=AgentAdapter)
        mock_agent.setup = MagicMock()

        config = RunConfig(
            agent=mock_agent,
            prompt_levels=["b3", "b4"],
        )
        assert "b4" in config.prompt_levels
        # Verify PromptLevel("b4") works
        level = PromptLevel(config.prompt_levels[1])
        assert level == PromptLevel.B4


# ──────────────────────────────────────────────────────────────────────────────
# InstanceGenerator — B4 workspace preparation
# ──────────────────────────────────────────────────────────────────────────────


class TestInstanceGeneratorB4:
    def test_prepare_workspace_copies_b4_prompt(self, tmp_path):
        """_prepare_workspace copies prompt_b4.md when B4 level is selected."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        gen = InstanceGenerator(tmp_path)

        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()
        workspace_dir = instance_dir / "workspace"
        workspace_dir.mkdir()

        # Create prompt file
        (instance_dir / "prompt_b4.md").write_text("B4 prompt content here")

        metadata = {"id": "test.task", "output": {"files": []}, "difficulty": {}}

        gen._prepare_workspace(
            instance_dir, workspace_dir, "test.task__seed0", PromptLevel.B4, metadata, {}
        )

        assert (workspace_dir / "prompt.md").exists()
        assert (workspace_dir / "prompt.md").read_text() == "B4 prompt content here"

    def test_prepare_workspace_b4_task_info(self, tmp_path):
        """task_info.json records b4 prompt level."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        gen = InstanceGenerator(tmp_path)

        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()
        workspace_dir = instance_dir / "workspace"
        workspace_dir.mkdir()
        (instance_dir / "prompt_b4.md").write_text("B4 content")

        metadata = {"id": "test.task", "output": {"files": []}, "difficulty": {}}
        gen._prepare_workspace(
            instance_dir, workspace_dir, "test.task__seed0", PromptLevel.B4, metadata, {}
        )

        task_info = json.loads((workspace_dir / "task_info.json").read_text())
        assert task_info["prompt_level"] == "b4"


# ──────────────────────────────────────────────────────────────────────────────
# Template files — B4
# ──────────────────────────────────────────────────────────────────────────────


TEMPLATE_DIR = Path(__file__).parent.parent / "tasks" / "_template"


class TestTemplateB4:
    def test_template_prompts_are_generated_by_cli_not_tracked(self):
        if not TEMPLATE_DIR.exists():
            pytest.skip("Template directory not found")
        assert not list(TEMPLATE_DIR.glob("prompt_b*.md"))

    def test_template_task_meta_declares_b4(self):
        if not TEMPLATE_DIR.exists():
            pytest.skip("Template directory not found")
        metadata = yaml.safe_load((TEMPLATE_DIR / "task_meta.yaml").read_text())
        assert "b4" in metadata.get("prompts", {})

    def test_template_task_eval_has_mode_field(self):
        if not TEMPLATE_DIR.exists():
            pytest.skip("Template directory not found")
        metadata = yaml.safe_load((TEMPLATE_DIR / "task_eval.yaml").read_text())
        gen = metadata.get("generation", {})
        assert "mode" in gen

    def test_template_generate_gt_renders_b4(self):
        """Template generate_gt.py iterates over b4 level."""
        if not TEMPLATE_DIR.exists():
            pytest.skip("Template directory not found")
        content = (TEMPLATE_DIR / "generate_gt.py").read_text()
        assert '"b4"' in content or "'b4'" in content
