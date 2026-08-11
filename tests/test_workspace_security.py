"""Tests for workspace security boundaries and metadata safety.

Covers:
- TaskInstance.agent_safe_metadata property
- _build_expected_outputs() allowlist
- Preflight data/ref_file collision check
- CLIAgentAdapter {instance_id} exposure documentation
- HTTPAgentAdapter instance_id omission
- collect_output_files() / get_task_environment() shared helpers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from ai4sci_bench.core.types import AgentOutput, PromptLevel, RunStatus, TaskInstance


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "prompt.md").write_text("Solve the task.")
    return ws


@pytest.fixture
def full_metadata() -> dict[str, Any]:
    """Task metadata with both safe and unsafe fields."""
    return {
        "id": "physics.test_task",
        "name": "Test Task",
        "domain": "physics",
        "status": "test",
        "output": {
            "files": [
                {"name": "simulation.py", "type": "code"},
                {"name": "output.npy", "type": "data", "shape": "(256, 256)", "dtype": "float64"},
            ]
        },
        "evaluation": {
            "gates": [
                {
                    "scorer": "file_exists",
                    "config": {"file": "simulation.py"},
                    "severity": "hard",
                }
            ],
            "scoring": [
                {
                    "scorer": "numerical",
                    "weight": 100,
                    "config": {
                        "metric": "relative_l2",
                        "pred_file": "output.npy",
                        "ref_file": "gt_output.npy",
                        "threshold": 0.01,
                        "shape": "(256, 256)",
                    },
                }
            ],
        },
        "generation": {
            "script": "generate_gt.py",
            "parameters": {"size": {"type": "int", "range": [1, 10]}},
        },
        "_task_dir": "/some/internal/path",
        "_runtime_python": ">=3.11",
        "_runtime_packages": ["scipy"],
    }


@pytest.fixture
def task_instance(workspace, tmp_path, full_metadata):
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    task_dir = tmp_path / "tasks" / "physics" / "test_task"
    task_dir.mkdir(parents=True)
    return TaskInstance(
        task_id="physics.test_task",
        instance_id="physics.test_task__size50_seed42",
        task_dir=task_dir,
        workspace_dir=workspace,
        reference_dir=ref_dir,
        prompt_level=PromptLevel.B1,
        parameters={"size": 50},
        metadata=full_metadata,
    )


# ── Tests: TaskInstance.agent_safe_metadata ───────────────────────────────────


class TestAgentSafeMetadata:
    def test_only_output_key_exposed(self, task_instance):
        safe = task_instance.agent_safe_metadata
        assert "output" in safe
        assert "evaluation" not in safe
        assert "generation" not in safe
        assert "_task_dir" not in safe
        assert "_runtime_python" not in safe
        assert "_runtime_packages" not in safe

    def test_output_files_preserved(self, task_instance):
        safe = task_instance.agent_safe_metadata
        files = safe["output"]["files"]
        assert len(files) == 2
        assert files[0]["name"] == "simulation.py"
        assert files[1]["name"] == "output.npy"

    def test_evaluation_config_not_leaked(self, task_instance):
        safe = task_instance.agent_safe_metadata
        # Even though full_metadata has evaluation, it's not in safe
        assert "evaluation" not in safe

    def test_empty_metadata(self, workspace, tmp_path):
        inst = TaskInstance(
            task_id="t",
            instance_id="t__s0",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={},
        )
        assert inst.agent_safe_metadata == {}

    def test_metadata_with_only_evaluation(self, workspace, tmp_path):
        inst = TaskInstance(
            task_id="t",
            instance_id="t__s0",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={"evaluation": {"scoring": []}},
        )
        # evaluation is NOT in the safe subset
        assert inst.agent_safe_metadata == {}

    def test_safe_metadata_is_independent_copy(self, task_instance):
        """Modifying agent_safe_metadata should not affect original metadata."""
        safe = task_instance.agent_safe_metadata
        safe["dangerous_key"] = "injected"
        assert "dangerous_key" not in task_instance.metadata

    def test_agent_safe_metadata_keys_frozenset(self):
        """Verify the allowlist is a frozenset (immutable)."""
        assert isinstance(TaskInstance._AGENT_SAFE_METADATA_KEYS, frozenset)
        assert "output" in TaskInstance._AGENT_SAFE_METADATA_KEYS
        assert "evaluation" not in TaskInstance._AGENT_SAFE_METADATA_KEYS


# ── Tests: _build_expected_outputs allowlist ─────────────────────────────────


class TestBuildExpectedOutputsAllowlist:
    def test_only_name_and_type_forwarded(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        metadata = {
            "output": {
                "files": [
                    {
                        "name": "output.npy",
                        "type": "data",
                        "shape": "(256, 256)",
                        "dtype": "float64",
                        "tolerance": 0.01,
                    }
                ]
            }
        }
        result = generator._build_expected_outputs(metadata)
        assert len(result) == 1
        assert result[0] == {"name": "output.npy", "type": "data"}
        # shape, dtype, tolerance must NOT be present
        assert "shape" not in result[0]
        assert "dtype" not in result[0]
        assert "tolerance" not in result[0]

    def test_default_type_is_data(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        metadata = {
            "output": {
                "files": [{"name": "result.csv"}]
            }
        }
        result = generator._build_expected_outputs(metadata)
        assert result[0]["type"] == "data"

    def test_multiple_files(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        metadata = {
            "output": {
                "files": [
                    {"name": "sim.py", "type": "code", "extra": "ignored"},
                    {"name": "out.npy", "type": "data", "shape": "(10,)"},
                ]
            }
        }
        result = generator._build_expected_outputs(metadata)
        assert len(result) == 2
        for entry in result:
            assert set(entry.keys()) == {"name", "type"}

    def test_empty_files_list(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        result = generator._build_expected_outputs({"output": {"files": []}})
        assert result == []

    def test_no_output_key(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        result = generator._build_expected_outputs({})
        assert result == []


# ── Tests: _validate_data_ref_file_no_collision ──────────────────────────────


class TestDataRefFileCollision:
    def test_no_collision(self, tmp_path):
        from ai4sci_bench.cli import _validate_data_ref_file_no_collision

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        data_dir = workspace / "data"
        data_dir.mkdir()
        (data_dir / "input.csv").write_text("data")

        metadata = {
            "evaluation": {
                "scoring": [
                    {
                        "scorer": "numerical",
                        "config": {"ref_file": "gt_output.npy", "pred_file": "output.npy"},
                    }
                ],
                "gates": [],
            }
        }
        errors: list[str] = []
        _validate_data_ref_file_no_collision(metadata, {}, workspace, errors)
        assert errors == []

    def test_collision_detected(self, tmp_path):
        from ai4sci_bench.cli import _validate_data_ref_file_no_collision

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        data_dir = workspace / "data"
        data_dir.mkdir()
        (data_dir / "reference.npy").write_text("data")

        metadata = {
            "evaluation": {
                "scoring": [
                    {
                        "scorer": "numerical",
                        "config": {"ref_file": "reference.npy", "pred_file": "output.npy"},
                    }
                ],
                "gates": [],
            }
        }
        errors: list[str] = []
        _validate_data_ref_file_no_collision(metadata, {}, workspace, errors)
        assert len(errors) == 1
        assert "reference.npy" in errors[0]
        assert "collides" in errors[0]

    def test_collision_in_gates(self, tmp_path):
        from ai4sci_bench.cli import _validate_data_ref_file_no_collision

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        data_dir = workspace / "data"
        data_dir.mkdir()
        (data_dir / "gate_ref.npy").write_text("data")

        metadata = {
            "evaluation": {
                "gates": [
                    {
                        "scorer": "numerical",
                        "config": {"ref_file": "gate_ref.npy"},
                    }
                ],
                "scoring": [],
            }
        }
        errors: list[str] = []
        _validate_data_ref_file_no_collision(metadata, {}, workspace, errors)
        assert len(errors) == 1

    def test_no_data_dir(self, tmp_path):
        from ai4sci_bench.cli import _validate_data_ref_file_no_collision

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # No data/ directory at all

        metadata = {
            "evaluation": {
                "scoring": [
                    {"scorer": "numerical", "config": {"ref_file": "ref.npy"}},
                ],
                "gates": [],
            }
        }
        errors: list[str] = []
        _validate_data_ref_file_no_collision(metadata, {}, workspace, errors)
        assert errors == []

    def test_empty_data_dir(self, tmp_path):
        from ai4sci_bench.cli import _validate_data_ref_file_no_collision

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "data").mkdir()

        metadata = {
            "evaluation": {
                "scoring": [
                    {"scorer": "numerical", "config": {"ref_file": "ref.npy"}},
                ],
                "gates": [],
            }
        }
        errors: list[str] = []
        _validate_data_ref_file_no_collision(metadata, {}, workspace, errors)
        assert errors == []

    def test_no_ref_file_in_config(self, tmp_path):
        from ai4sci_bench.cli import _validate_data_ref_file_no_collision

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        data_dir = workspace / "data"
        data_dir.mkdir()
        (data_dir / "input.csv").write_text("data")

        metadata = {
            "evaluation": {
                "scoring": [
                    {"scorer": "code_analysis", "config": {"checks": []}},
                ],
                "gates": [],
            }
        }
        errors: list[str] = []
        _validate_data_ref_file_no_collision(metadata, {}, workspace, errors)
        assert errors == []

    def test_parameterized_ref_file_exact_match(self, tmp_path):
        """When ref_file is a bare template like '{ref_name}', it resolves via params."""
        from ai4sci_bench.cli import _validate_data_ref_file_no_collision

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        data_dir = workspace / "data"
        data_dir.mkdir()
        (data_dir / "shared.npy").write_text("data")

        metadata = {
            "evaluation": {
                "scoring": [
                    {"scorer": "numerical", "config": {"ref_file": "{ref_name}"}},
                ],
                "gates": [],
            }
        }
        errors: list[str] = []
        # {ref_name} resolves to "shared.npy" which collides with data/shared.npy
        _validate_data_ref_file_no_collision(
            metadata, {"ref_name": "shared.npy"}, workspace, errors
        )
        assert len(errors) == 1
        assert "shared.npy" in errors[0]


# ── Tests: CLIAgentAdapter {instance_id} security warning ───────────────────


class TestCLIAgentInstanceIdWarning:
    def test_docstring_has_security_warning(self):
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        assert "Security warning" in CLIAgentAdapter.__doc__
        assert "instance_id" in CLIAgentAdapter.__doc__
        assert "parameters and random seed" in CLIAgentAdapter.__doc__


# ── Tests: HTTPAgentAdapter instance_id not in payload ───────────────────────


class TestHTTPAgentNoInstanceId:
    def test_payload_omits_instance_id(self, task_instance):
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter

        adapter = HTTPAgentAdapter(base_url="http://fake:8080")
        captured = {}

        def fake_post(endpoint, payload, **kwargs):
            captured.update(payload)
            return {
                "status": "completed",
                "code_files": {},
                "data_files": {},
                "log": "ok",
            }

        adapter._post = fake_post
        adapter.solve(task_instance)

        assert "instance_id" not in captured
        assert "task_id" in captured
        assert captured["task_id"] == "physics.test_task"

    def test_docstring_omits_instance_id(self):
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter
        # The docstring should not mention instance_id as a field
        body_section = HTTPAgentAdapter.__doc__
        assert '"instance_id"' not in body_section


# ── Tests: _build_agent_task_info (workspace task_info.json) ─────────────────


class TestBuildAgentTaskInfo:
    def test_does_not_contain_evaluation(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        generator.execution_timeout_seconds = 3600
        metadata = {
            "id": "physics.test",
            "output": {"files": [{"name": "out.npy"}]},
            "evaluation": {"scoring": [{"scorer": "numerical"}]},
        }
        info = generator._build_agent_task_info(metadata, PromptLevel.B1)
        assert "evaluation" not in info
        assert "instance_id" not in info
        assert "parameters" not in info
        # task_id is anonymized — must NOT expose raw task name
        assert info["task_id"] != "physics.test"
        assert info["task_id"].startswith("task_")
        assert info["prompt_level"] == "b1"
        assert info["timeout_seconds"] == 3600

    def test_expected_outputs_allowlisted(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        generator.execution_timeout_seconds = 3600
        metadata = {
            "id": "physics.test",
            "output": {
                "files": [
                    {"name": "out.npy", "type": "data", "shape": "(10,)", "dtype": "float64"}
                ]
            },
        }
        info = generator._build_agent_task_info(metadata, PromptLevel.B1)
        outputs = info["expected_outputs"]
        assert len(outputs) == 1
        assert outputs[0] == {"name": "out.npy", "type": "data"}


# ── Tests: _build_framework_task_info (includes sensitive fields) ────────────


class TestBuildFrameworkTaskInfo:
    def test_includes_instance_id_and_parameters(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        generator = InstanceGenerator.__new__(InstanceGenerator)
        generator.execution_timeout_seconds = 3600
        metadata = {"id": "t", "output": {"files": []}}
        info = generator._build_framework_task_info(
            metadata,
            instance_id="t__size50_seed42",
            prompt_level=PromptLevel.B1,
            resolved_params={"size": 50},
        )
        assert info["instance_id"] == "t__size50_seed42"
        assert info["parameters"] == {"size": 50}


# ── Tests: Preflight integration — data/ref collision via CLI ────────────────


class TestPreflightDataRefCollisionIntegration:
    """Integration test using the actual CLI validate --preflight."""

    def test_preflight_catches_data_ref_collision(self, tmp_path):
        from ai4sci_bench.cli import cli

        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "physics" / "collision_task"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.collision_task",
            "name": "Collision Task",
            "domain": "physics",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md", "b3": "prompt_b3.md"},
            "input": {"files": [{"name": "shared.npy", "type": "data"}]},
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {
                "gates": [],
                "scoring": [
                    {
                        "scorer": "numerical",
                        "weight": 100,
                        "config": {
                            "metric": "relative_l2",
                            "pred_file": "output.npy",
                            "ref_file": "shared.npy",  # same name as input file!
                            "threshold": 0.1,
                        },
                    }
                ],
            },
            "generation": {
                "script": "generate_gt.py",
                "parameters": {"size": {"type": "int", "range": [1, 10], "default": 3}},
            },
        }))

        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Prompt for {level}")

        # generate_gt.py that creates the collision: data/shared.npy
        (task_dir / "generate_gt.py").write_text(
            "import json, numpy as np\n"
            "from pathlib import Path\n"
            "INPUT_SPEC = [{'name': 'shared.npy'}]\n"
            "OUTPUT_SPEC = [{'name': 'output.npy'}]\n"
            "DEFAULT_PARAMS = {'size': 3}\n"
            "def generate(output_dir, params):\n"
            "    out = Path(output_dir)\n"
            "    (out / 'data').mkdir(parents=True, exist_ok=True)\n"
            "    np.save(out / 'data' / 'shared.npy', np.zeros(3))\n"
            "    ref = out / 'reference'\n"
            "    ref.mkdir(parents=True, exist_ok=True)\n"
            "    np.save(ref / 'shared.npy', np.ones(3))\n"
            "    for level in ['b1', 'b2', 'b3']:\n"
            "        (out / f'prompt_{level}.md').write_text(\n"
            "            (Path(__file__).parent / f'prompt_{level}.md').read_text())\n"
            "    (out / 'instance_meta.json').write_text(json.dumps({\n"
            "        'params_used': params,\n"
            "        'input_files': ['shared.npy'],\n"
            "        'reference_files': ['shared.npy'],\n"
            "    }))\n"
            "    return {'params_used': params, 'input_files': ['shared.npy'], 'reference_files': ['shared.npy']}\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.collision_task",
            "--tasks-dir", str(tasks_dir),
            "--preflight",
        ])
        assert result.exit_code != 0
        assert "collides" in result.output or "ref_file" in result.output
