"""Tests for per-instance timeout resolution, propagation, and thread safety.

Covers:
- §7.1: CLI-only timeout resolution
- §7.2: End-to-end timeout propagation
- §7.3: Multi-path consistency
- §7.4: Thread safety under --parallel > 1
- §7.8: Clone workspace preserves timeout
- §7.9: Pre-generated metadata cannot override CLI
- §7.11: TaskInstance backward compatibility
- §7.12: task.yaml timeout fields are ignored
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ai4sci_bench.core.types import PromptLevel, TaskInstance


# ---------------------------------------------------------------------------
# §7.1 — resolve_instance_timeout priority logic
# ---------------------------------------------------------------------------

class TestResolveInstanceTimeout:
    """The CLI value is the only effective timeout source."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from ai4sci_bench.runner.orchestrator import resolve_instance_timeout
        self.resolve = resolve_instance_timeout

    @pytest.mark.parametrize(
        "cli_timeout, yaml_block, expected",
        [
            (120, {"execution": {"agent_timeout_seconds": 7200}}, 120),
            (None, {"execution": {"agent_timeout_seconds": 7200}}, 10800),
            (None, {"timeout": 60}, 10800),
            (None, {}, 10800),
            (None, {"execution": {}}, 10800),
            (0, {"execution": {"agent_timeout_seconds": 7200}}, 0),
            (5400, {}, 5400),
            (None, {"execution": "fast"}, 10800),
            (None, {"execution": True}, 10800),
        ],
        ids=[
            "cli_value_wins",
            "execution_timeout_ignored",
            "legacy_timeout_ignored",
            "default_10800",
            "execution_block_without_timeout_field",
            "cli_zero_is_explicit",
            "cli_explicit_no_yaml",
            "non_dict_execution_string",
            "non_dict_execution_bool",
        ],
    )
    def test_priority(self, cli_timeout, yaml_block, expected):
        result = self.resolve(yaml_block, cli_timeout)
        assert result == expected

    def test_execution_not_in_agent_safe_metadata(self):
        """execution block must NOT leak to agent_safe_metadata."""
        inst = TaskInstance(
            task_id="test.task",
            instance_id="test__x",
            task_dir=Path("/tmp"),
            workspace_dir=Path("/tmp"),
            reference_dir=Path("/tmp"),
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata={"output": {"files": []}, "execution": {"agent_timeout_seconds": 7200}},
        )
        safe = inst.agent_safe_metadata
        assert "execution" not in safe
        assert "output" in safe


# ---------------------------------------------------------------------------
# §7.2 — End-to-end timeout propagation
# ---------------------------------------------------------------------------

class TestE2ETimeoutPropagation:
    """If these fail, timeout doesn't flow from CLI through to workspace files."""

    def test_generate_instance_propagates_to_task_info(self, tmp_path):
        """task_info.json inside workspace must reflect effective_timeout_seconds."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "tasks" / "physics" / "test_task"
        task_dir.mkdir(parents=True)

        # Minimal generate_gt.py
        (task_dir / "generate_gt.py").write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "def generate(output_dir, params, **kwargs):\n"
            "    ref = output_dir / 'reference'\n"
            "    ref.mkdir(exist_ok=True)\n"
            "    (ref / 'out.npy').write_text('data')\n"
            "    return params\n",
            encoding="utf-8",
        )
        (task_dir / "prompt_b2.md").write_text("Test prompt", encoding="utf-8")
        (task_dir / "task.yaml").write_text(
            "id: physics.test_task\nname: Test\nstatus: test\n"
            "output:\n  files:\n    - name: out.npy\n      type: data\n",
            encoding="utf-8",
        )

        gen = InstanceGenerator(tmp_path / "tasks")
        metadata = gen.task_loader.load_task_by_id("physics.test_task")
        instance = gen.generate_instance(
            metadata, {}, tmp_path / "out", PromptLevel.B2,
            effective_timeout_seconds=120,
        )

        # Check TaskInstance field
        assert instance.effective_timeout_seconds == 120

        # Check task_info.json
        task_info = json.loads(
            (instance.workspace_dir / "task_info.json").read_text(encoding="utf-8")
        )
        assert task_info["timeout_seconds"] == 120

        # Check framework_task_info.json
        instance_dir = tmp_path / "out" / instance.instance_id
        fti = json.loads(
            (instance_dir / "framework_task_info.json").read_text(encoding="utf-8")
        )
        assert fti["timeout_seconds"] == 120

    def test_generate_instance_defaults_to_generator_timeout(self, tmp_path):
        """Without explicit effective_timeout_seconds, uses generator default."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        task_dir = tmp_path / "tasks" / "physics" / "test_task"
        task_dir.mkdir(parents=True)
        (task_dir / "generate_gt.py").write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "def generate(output_dir, params, **kwargs):\n"
            "    ref = output_dir / 'reference'\n"
            "    ref.mkdir(exist_ok=True)\n"
            "    (ref / 'out.npy').write_text('data')\n"
            "    return params\n",
            encoding="utf-8",
        )
        (task_dir / "prompt_b2.md").write_text("Test", encoding="utf-8")
        (task_dir / "task.yaml").write_text(
            "id: physics.test_task\nname: Test\nstatus: test\n"
            "output:\n  files:\n    - name: out.npy\n      type: data\n",
            encoding="utf-8",
        )

        gen = InstanceGenerator(tmp_path / "tasks", execution_timeout_seconds=9999)
        metadata = gen.task_loader.load_task_by_id("physics.test_task")
        instance = gen.generate_instance(metadata, {}, tmp_path / "out", PromptLevel.B2)
        assert instance.effective_timeout_seconds == 9999

        task_info = json.loads(
            (instance.workspace_dir / "task_info.json").read_text(encoding="utf-8")
        )
        assert task_info["timeout_seconds"] == 9999


# ---------------------------------------------------------------------------
# §7.3 — Multi-path consistency
# ---------------------------------------------------------------------------

class TestMultiPathConsistency:
    """Timeout must propagate identically through all generation paths."""

    def test_on_the_fly_propagates_effective_timeout(self, tmp_path):
        """generate_instances_on_the_fly passes timeout to generate_instance."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        gen = InstanceGenerator(tmp_path)
        mock_instance = TaskInstance(
            task_id="test.task", instance_id="x",
            task_dir=tmp_path, workspace_dir=tmp_path, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
            effective_timeout_seconds=120,
        )

        captured = []

        def fake_generate(meta, params, out_dir, level, **kwargs):
            captured.append(kwargs.get("effective_timeout_seconds"))
            return mock_instance

        gen.generate_instance = fake_generate

        metadata = {
            "id": "test.task",
            "_task_dir": tmp_path,
            "_generation_mode": "infinite",
            "generation": {"parameters": {}},
        }
        from ai4sci_bench.core.types import GenerationMode
        metadata["_generation_mode"] = GenerationMode.INFINITE

        gen.generate_instances_on_the_fly(
            metadata, seed=42, count=2, output_dir=tmp_path / "out",
            effective_timeout_seconds=5400,
        )
        assert all(t == 5400 for t in captured)

    def test_finite_mode_propagates_effective_timeout(self, tmp_path):
        """_generate_finite_instances passes timeout to generate_instance."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        from ai4sci_bench.core.types import GenerationMode

        gen = InstanceGenerator(tmp_path)
        mock_instance = TaskInstance(
            task_id="test.task", instance_id="x",
            task_dir=tmp_path, workspace_dir=tmp_path, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
        )

        captured = []

        def fake_generate(meta, params, out_dir, level, **kwargs):
            captured.append(kwargs.get("effective_timeout_seconds"))
            return mock_instance

        gen.generate_instance = fake_generate

        metadata = {
            "id": "test.task",
            "_task_dir": tmp_path,
            "_generation_mode": GenerationMode.FINITE,
            "_generation_settings": [{"grid": 64}, {"grid": 128}],
            "_generation_precomputed": False,
        }

        gen._generate_finite_instances(
            metadata, seed=42, count=2, output_dir=tmp_path / "out",
            effective_timeout_seconds=7200,
        )
        assert all(t == 7200 for t in captured)

    def test_task_metadata_cannot_change_timeout(self):
        """Different task metadata receives the same CLI/default timeout."""
        from ai4sci_bench.runner.orchestrator import resolve_instance_timeout

        task_a = {"execution": {"agent_timeout_seconds": 7200}}
        task_b = {"timeout": 60}

        assert resolve_instance_timeout(task_a, None) == 10800
        assert resolve_instance_timeout(task_b, None) == 10800
        assert resolve_instance_timeout(task_a, 300) == 300
        assert resolve_instance_timeout(task_b, 300) == 300


# ---------------------------------------------------------------------------
# §7.8 — Clone workspace preserves timeout
# ---------------------------------------------------------------------------

class TestCloneWorkspaceTimeout:
    """If this fails, retry attempts lose the per-instance timeout."""

    def test_clone_workspace_preserves_effective_timeout(self, tmp_path):
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
        from ai4sci_bench.core.agent_interface import AgentAdapter
        from ai4sci_bench.core.types import AgentOutput, RunStatus

        class DummyAgent(AgentAdapter):
            def solve(self, task_instance):
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=task_instance.workspace_dir,
                    code_files=[], data_files=[], log="ok",
                    execution_time_seconds=1.0, status=RunStatus.COMPLETED,
                )

        config = RunConfig(
            agent=DummyAgent(),
            output_dir=str(tmp_path / "results"),
            tasks_dir=str(tmp_path / "tasks"),
        )
        (tmp_path / "tasks").mkdir(exist_ok=True)
        orch = BenchmarkOrchestrator(config)

        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "prompt.md").write_text("test")
        (ws / "task_info.json").write_text("{}")

        instance = TaskInstance(
            task_id="test.task",
            instance_id="test__x",
            task_dir=tmp_path,
            workspace_dir=ws,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata={},
            effective_timeout_seconds=5400,
        )

        cloned = orch._clone_workspace(instance, attempt=2)
        assert cloned.effective_timeout_seconds == 5400


# ---------------------------------------------------------------------------
# §7.9 — Pre-generated metadata cannot override CLI
# ---------------------------------------------------------------------------

class TestResumeSnapshot:
    """Saved instance metadata must not become a timeout source."""

    def test_pregenerated_ignores_framework_task_info_timeout(self, tmp_path):
        """The current CLI-selected timeout wins over a saved snapshot."""
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
        from ai4sci_bench.core.agent_interface import AgentAdapter
        from ai4sci_bench.core.types import AgentOutput, RunStatus

        class DummyAgent(AgentAdapter):
            def solve(self, task_instance):
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=task_instance.workspace_dir,
                    code_files=[], data_files=[], log="",
                    execution_time_seconds=0, status=RunStatus.COMPLETED,
                )

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        instances_dir = tmp_path / "instances"
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Create a pre-generated instance dir with framework_task_info
        inst_dir = instances_dir / "physics.test__seed42"
        inst_dir.mkdir(parents=True)
        (inst_dir / "reference").mkdir()
        (inst_dir / "reference" / "out.npy").write_text("data")
        (inst_dir / "prompt_b2.md").write_text("prompt")
        (inst_dir / "instance_meta.json").write_text(json.dumps({"params_used": {}}))
        (inst_dir / "framework_task_info.json").write_text(json.dumps({
            "timeout_seconds": 9999,
            "task_id": "test_hash",
            "prompt_level": "b2",
        }))

        task_metadata = {
            "id": "physics.test",
            "_task_dir": tasks_dir,
            "output": {"files": [{"name": "out.npy", "type": "data"}]},
        }

        config = RunConfig(
            agent=DummyAgent(),
            output_dir=str(results_dir),
            tasks_dir=str(tasks_dir),
            instances_dir=str(instances_dir),
        )
        orch = BenchmarkOrchestrator(config)

        instances = orch._load_pregenerated_instances(
            task_metadata, PromptLevel.B2, effective_timeout_seconds=3600,
        )

        assert len(instances) == 1
        assert instances[0].effective_timeout_seconds == 3600

    def test_missing_framework_task_info_falls_back(self, tmp_path):
        """If framework_task_info.json missing, uses the passed-in timeout."""
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
        from ai4sci_bench.core.agent_interface import AgentAdapter
        from ai4sci_bench.core.types import AgentOutput, RunStatus

        class DummyAgent(AgentAdapter):
            def solve(self, task_instance):
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=task_instance.workspace_dir,
                    code_files=[], data_files=[], log="",
                    execution_time_seconds=0, status=RunStatus.COMPLETED,
                )

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        instances_dir = tmp_path / "instances"
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        inst_dir = instances_dir / "physics.test__seed42"
        inst_dir.mkdir(parents=True)
        (inst_dir / "reference").mkdir()
        (inst_dir / "reference" / "out.npy").write_text("data")
        (inst_dir / "prompt_b2.md").write_text("prompt")
        # No framework_task_info.json

        task_metadata = {
            "id": "physics.test",
            "_task_dir": tasks_dir,
            "output": {"files": [{"name": "out.npy", "type": "data"}]},
        }

        config = RunConfig(
            agent=DummyAgent(),
            output_dir=str(results_dir),
            tasks_dir=str(tasks_dir),
            instances_dir=str(instances_dir),
        )
        orch = BenchmarkOrchestrator(config)

        instances = orch._load_pregenerated_instances(
            task_metadata, PromptLevel.B2, effective_timeout_seconds=5400,
        )

        assert len(instances) == 1
        assert instances[0].effective_timeout_seconds == 5400


# ---------------------------------------------------------------------------
# §7.11 — TaskInstance backward compatibility
# ---------------------------------------------------------------------------

class TestTaskInstanceCompat:
    """New field must not break existing construction sites."""

    def test_default_effective_timeout(self):
        inst = TaskInstance(
            task_id="test.t", instance_id="x",
            task_dir=Path("/tmp"), workspace_dir=Path("/tmp"),
            reference_dir=Path("/tmp"), prompt_level=PromptLevel.B2,
            parameters={}, metadata={},
        )
        assert inst.effective_timeout_seconds == 10800

    def test_custom_effective_timeout(self):
        inst = TaskInstance(
            task_id="test.t", instance_id="x",
            task_dir=Path("/tmp"), workspace_dir=Path("/tmp"),
            reference_dir=Path("/tmp"), prompt_level=PromptLevel.B2,
            parameters={}, metadata={},
            effective_timeout_seconds=120,
        )
        assert inst.effective_timeout_seconds == 120


# ---------------------------------------------------------------------------
# §7.12 — task.yaml timeout fields are inert
# ---------------------------------------------------------------------------

class TestTaskYamlTimeoutIgnored:
    """TaskLoader accepts old fields, but they cannot affect execution."""

    @pytest.fixture
    def task_dir(self, tmp_path):
        d = tmp_path / "tasks" / "physics" / "test_task"
        d.mkdir(parents=True)
        return d

    def _write_task_yaml(self, task_dir, extra=""):
        (task_dir / "task.yaml").write_text(
            f"id: physics.test_task\nname: Test\nstatus: test\n{extra}",
            encoding="utf-8",
        )

    def _load(self, task_dir):
        from ai4sci_bench.core.task import TaskLoader
        loader = TaskLoader(task_dir.parent.parent)
        return loader.load_task_metadata(task_dir / "task.yaml")

    @pytest.mark.parametrize("value", [7200, 0, -1, 3.14, "fast"])
    def test_execution_timeout_is_accepted_but_ignored(self, task_dir, value):
        self._write_task_yaml(
            task_dir, f"execution:\n  agent_timeout_seconds: {value}\n"
        )
        meta = self._load(task_dir)

        from ai4sci_bench.runner.orchestrator import resolve_instance_timeout

        assert resolve_instance_timeout(meta, None) == 10800
        assert resolve_instance_timeout(meta, 240) == 240

    def test_legacy_top_level_timeout_is_ignored(self, task_dir):
        self._write_task_yaml(task_dir, "timeout: 30\n")
        meta = self._load(task_dir)

        from ai4sci_bench.runner.orchestrator import resolve_instance_timeout

        assert resolve_instance_timeout(meta, None) == 10800

    def test_empty_execution_block_accepted(self, task_dir):
        self._write_task_yaml(task_dir, "execution: {}\n")
        meta = self._load(task_dir)
        assert meta.get("execution") == {}

    def test_no_execution_block_accepted(self, task_dir):
        self._write_task_yaml(task_dir)
        meta = self._load(task_dir)
        assert "execution" not in meta

    def test_execution_not_leaked_to_agent_safe_metadata(self, task_dir):
        self._write_task_yaml(task_dir, "execution:\n  agent_timeout_seconds: 7200\n"
                              "output:\n  files:\n    - name: out.npy\n      type: data\n")
        meta = self._load(task_dir)
        inst = TaskInstance(
            task_id="physics.test_task", instance_id="x",
            task_dir=task_dir, workspace_dir=task_dir, reference_dir=task_dir,
            prompt_level=PromptLevel.B2, parameters={}, metadata=meta,
        )
        assert "execution" not in inst.agent_safe_metadata


# ---------------------------------------------------------------------------
# §7.10 — CLI sentinel tests
# ---------------------------------------------------------------------------

class TestCLISentinel:
    """CLI and programmatic runs default to 10800 seconds."""

    def test_run_config_timeout_default_is_10800(self):
        from ai4sci_bench.runner.orchestrator import RunConfig
        from ai4sci_bench.core.agent_interface import AgentAdapter
        from ai4sci_bench.core.types import AgentOutput, RunStatus

        class Dummy(AgentAdapter):
            def solve(self, ti):
                return AgentOutput(
                    instance_id=ti.instance_id, output_dir=ti.workspace_dir,
                    code_files=[], data_files=[], log="", execution_time_seconds=0,
                    status=RunStatus.COMPLETED,
                )

        config = RunConfig(agent=Dummy())
        assert config.timeout == 10800

    @pytest.mark.parametrize("command_name", ["run", "difficulty-check"])
    def test_cli_timeout_default_is_10800(self, command_name):
        from ai4sci_bench.cli import cli

        command = cli.commands[command_name]
        timeout_param = next(p for p in command.params if p.name == "timeout")
        assert timeout_param.default == 10800

    def test_run_timeout_explicit_is_int(self):
        from ai4sci_bench.runner.orchestrator import RunConfig
        from ai4sci_bench.core.agent_interface import AgentAdapter
        from ai4sci_bench.core.types import AgentOutput, RunStatus

        class Dummy(AgentAdapter):
            def solve(self, ti):
                return AgentOutput(
                    instance_id=ti.instance_id, output_dir=ti.workspace_dir,
                    code_files=[], data_files=[], log="", execution_time_seconds=0,
                    status=RunStatus.COMPLETED,
                )

        config = RunConfig(agent=Dummy(), timeout=120)
        assert config.timeout == 120


# ---------------------------------------------------------------------------
# §7.4 — Thread safety (--parallel > 1)
# ---------------------------------------------------------------------------

class TestParallelTimeoutIsolation:
    """If this test fails, adapter solve() mutates self instead of using locals."""

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_concurrent_solve_uses_per_instance_timeout(self, mock_run):
        from ai4sci_bench.adapters.subprocess_base import SubprocessAgentAdapter

        class EchoAdapter(SubprocessAgentAdapter):
            def _build_command(self, task_instance, task_env):
                return ["echo", "ok"]

        barrier = threading.Barrier(2, timeout=5)
        captured_timeouts = []

        def side_effect(*args, **kwargs):
            barrier.wait()
            captured_timeouts.append(kwargs.get("timeout"))
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        adapter = EchoAdapter(timeout_seconds=3600)

        inst_short = TaskInstance(
            task_id="t", instance_id="short", task_dir=Path("/tmp"),
            workspace_dir=Path("/tmp"), reference_dir=Path("/tmp"),
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
            effective_timeout_seconds=60,
        )
        inst_long = TaskInstance(
            task_id="t", instance_id="long", task_dir=Path("/tmp"),
            workspace_dir=Path("/tmp"), reference_dir=Path("/tmp"),
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
            effective_timeout_seconds=9999,
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(adapter.solve, inst_short)
            f2 = pool.submit(adapter.solve, inst_long)
            f1.result()
            f2.result()

        assert set(captured_timeouts) == {60, 9999}


# ---------------------------------------------------------------------------
# §7.5 — Claude adapter command line verification
# ---------------------------------------------------------------------------

class TestClaudeCommandVerification:
    """Verify --max-turns is gone and tool isolation preserved."""

    def test_claude_cmd_no_max_turns_flag(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        import tempfile
        adapter = ClaudeCodeCLIAdapter()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "prompt.md").write_text("test")
            cmd = adapter._build_command(
                TaskInstance(
                    task_id="t", instance_id="x", task_dir=ws,
                    workspace_dir=ws, reference_dir=ws,
                    prompt_level=PromptLevel.B2, parameters={}, metadata={},
                ),
                None,
            )
        assert "--max-turns" not in cmd

    def test_claude_os_cmd_no_max_turns_flag(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        import tempfile
        adapter = ClaudeCodeCLIAdapter()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "prompt.md").write_text("test")
            cmd = adapter._build_os_agent_cmd(ws)
        assert "--max-turns" not in cmd

    def test_claude_init_rejects_max_turns_param(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        with pytest.raises(TypeError):
            ClaudeCodeCLIAdapter(max_turns=20)

    def test_claude_tool_isolation_preserved_after_max_turns_removal(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        import tempfile
        adapter = ClaudeCodeCLIAdapter()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "prompt.md").write_text("test")
            cmd = adapter._build_os_agent_cmd(ws)
        assert "--tools" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd


# ---------------------------------------------------------------------------
# §7.6 — Adapter multi-path timeout propagation
# ---------------------------------------------------------------------------

class TestAdapterMultiPathTimeout:
    """Each adapter path must read effective_timeout_seconds from task_instance."""

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_subprocess_base_uses_effective_timeout(self, mock_run):
        from ai4sci_bench.adapters.subprocess_base import SubprocessAgentAdapter

        class EchoAdapter(SubprocessAgentAdapter):
            def _build_command(self, task_instance, task_env):
                return ["echo", "ok"]

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = EchoAdapter(timeout_seconds=9999)
        inst = TaskInstance(
            task_id="t", instance_id="x", task_dir=Path("/tmp"),
            workspace_dir=Path("/tmp"), reference_dir=Path("/tmp"),
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
            effective_timeout_seconds=120,
        )
        adapter.solve(inst)
        assert mock_run.call_args.kwargs["timeout"] == 120

    @patch("ai4sci_bench.adapters.docker_agent.subprocess.run")
    def test_docker_subprocess_timeout_equals_effective_plus_30(self, mock_run):
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        adapter = DockerAgentAdapter(image="test:latest", timeout=9999)
        inst = TaskInstance(
            task_id="t", instance_id="x", task_dir=Path("/tmp"),
            workspace_dir=Path("/tmp"), reference_dir=Path("/tmp"),
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
            effective_timeout_seconds=120,
        )
        adapter.solve(inst)
        assert mock_run.call_args.kwargs["timeout"] == 150  # 120 + 30


# ---------------------------------------------------------------------------
# §7.7 — direct_llm setup fallback
# ---------------------------------------------------------------------------

class TestDirectLLMSetupFallback:
    """Ensure direct_llm doesn't silently fall back to 600s."""

    def test_direct_llm_default_timeout_is_10800(self):
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        adapter = DirectLLMAdapter()
        assert adapter.timeout == 10800

    def test_direct_llm_setup_without_timeout_keeps_10800(self):
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        adapter = DirectLLMAdapter()
        adapter.setup({"sandbox": "none"})
        assert adapter.timeout == 10800

    def test_direct_llm_setup_with_explicit_timeout(self):
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        adapter = DirectLLMAdapter()
        adapter.setup({"timeout": 120, "sandbox": "none"})
        assert adapter.timeout == 120

    @patch("ai4sci_bench.adapters.direct_llm.subprocess.run")
    def test_direct_llm_execute_uses_instance_timeout(self, mock_run):
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        adapter = DirectLLMAdapter()
        adapter.timeout = 600  # stale value
        inst = TaskInstance(
            task_id="t", instance_id="x", task_dir=Path("/tmp"),
            workspace_dir=Path("/tmp"), reference_dir=Path("/tmp"),
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
            effective_timeout_seconds=7200,
        )
        (Path("/tmp") / "simulation.py").write_text("print('ok')")  # noqa
        adapter._execute(inst, "simulation.py")
        assert mock_run.call_args.kwargs["timeout"] == 7200


# ---------------------------------------------------------------------------
# §7.13 — Static guard against self.timeout references in solve paths
# ---------------------------------------------------------------------------

class TestNoStaleTimeoutReferences:
    """Ensures no adapter solve() path uses self.timeout for subprocess timeout."""

    def test_subprocess_base_solve_no_self_timeout_reference(self):
        import inspect
        from ai4sci_bench.adapters.subprocess_base import SubprocessAgentAdapter
        src = inspect.getsource(SubprocessAgentAdapter.solve)
        lines = [l for l in src.splitlines() if "timeout=" in l and "self.timeout" in l]
        assert lines == [], f"solve() still references self.timeout in timeout= assignment: {lines}"

    def test_cli_agent_no_self_timeout_attribute(self):
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        adapter = CLIAgentAdapter(cmd_template="echo test")
        assert not hasattr(adapter, "timeout"), \
            "CLIAgentAdapter still has self.timeout — should be removed"

    def test_docker_solve_no_self_timeout_reference(self):
        import inspect
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter
        src = inspect.getsource(DockerAgentAdapter.solve)
        lines = [l for l in src.splitlines() if "timeout=" in l and "self.timeout" in l]
        assert lines == [], f"solve() still references self.timeout: {lines}"

    def test_direct_llm_execute_no_self_timeout_reference(self):
        import inspect
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        src = inspect.getsource(DirectLLMAdapter._execute)
        lines = [l for l in src.splitlines() if "timeout=" in l and "self.timeout" in l]
        assert lines == [], f"_execute() still references self.timeout: {lines}"

    def test_repository_tasks_do_not_declare_dead_timeout_fields(self):
        tasks_dir = Path(__file__).resolve().parents[1] / "tasks"
        metadata_paths = [
            *tasks_dir.rglob("task.yaml"),
            *tasks_dir.rglob("task_meta.yaml"),
        ]
        offenders = []
        for path in metadata_paths:
            metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict) and (
                "timeout" in metadata or "execution" in metadata
            ):
                offenders.append(str(path.relative_to(tasks_dir)))
        assert offenders == []


# ---------------------------------------------------------------------------
# §thread-safety — _pending_stdin elimination
# ---------------------------------------------------------------------------

class TestNoPendingStdinAttribute:
    """_pending_stdin must not exist in production code (thread-unsafe pattern)."""

    def test_claude_no_pending_stdin_after_build_command(self, tmp_path):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "prompt.md").write_text("hello")
        adapter = ClaudeCodeCLIAdapter()
        inst = TaskInstance(
            task_id="t", instance_id="x", task_dir=ws,
            workspace_dir=ws, reference_dir=ws,
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
        )
        adapter._build_command(inst, None)
        assert not hasattr(adapter, "_pending_stdin"), \
            "ClaudeCodeCLIAdapter still sets self._pending_stdin in _build_command"

    def test_codex_no_pending_stdin_after_build_command(self, tmp_path):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "prompt.md").write_text("hello")
        adapter = CodexCLIAdapter()
        inst = TaskInstance(
            task_id="t", instance_id="x", task_dir=ws,
            workspace_dir=ws, reference_dir=ws,
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        adapter._build_command(inst, None)
        assert not hasattr(adapter, "_pending_stdin"), \
            "CodexCLIAdapter still sets self._pending_stdin in _build_command"

    def test_no_pending_stdin_in_source(self):
        """Static guard: _pending_stdin must not appear in adapter source."""
        import inspect
        from ai4sci_bench.adapters import claude_code_cli, codex_cli
        for mod in (claude_code_cli, codex_cli):
            src = inspect.getsource(mod)
            assert "_pending_stdin" not in src, \
                f"{mod.__name__} still references _pending_stdin"


class TestStdinThreadSafety:
    """Concurrent solve() calls must not cross-contaminate prompts."""

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_concurrent_claude_solve_gets_correct_prompts(self, mock_run, tmp_path):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        barrier = threading.Barrier(2, timeout=5)
        captured_inputs = []

        def side_effect(*args, **kwargs):
            barrier.wait()
            captured_inputs.append(kwargs.get("input"))
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        adapter = ClaudeCodeCLIAdapter()

        ws_a = tmp_path / "ws_a"
        ws_a.mkdir()
        (ws_a / "prompt.md").write_text("prompt_for_task_A")
        inst_a = TaskInstance(
            task_id="t", instance_id="a", task_dir=ws_a,
            workspace_dir=ws_a, reference_dir=ws_a,
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
        )

        ws_b = tmp_path / "ws_b"
        ws_b.mkdir()
        (ws_b / "prompt.md").write_text("prompt_for_task_B")
        inst_b = TaskInstance(
            task_id="t", instance_id="b", task_dir=ws_b,
            workspace_dir=ws_b, reference_dir=ws_b,
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(adapter.solve, inst_a)
            f2 = pool.submit(adapter.solve, inst_b)
            f1.result()
            f2.result()

        assert set(captured_inputs) == {"prompt_for_task_A", "prompt_for_task_B"}

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_concurrent_codex_solve_gets_correct_prompts(self, mock_run, tmp_path):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter

        barrier = threading.Barrier(2, timeout=5)
        captured_inputs = []

        def side_effect(*args, **kwargs):
            barrier.wait()
            captured_inputs.append(kwargs.get("input"))
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        adapter = CodexCLIAdapter()

        ws_a = tmp_path / "ws_a"
        ws_a.mkdir()
        (ws_a / "prompt.md").write_text("codex_prompt_A")
        inst_a = TaskInstance(
            task_id="t", instance_id="a", task_dir=ws_a,
            workspace_dir=ws_a, reference_dir=ws_a,
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )

        ws_b = tmp_path / "ws_b"
        ws_b.mkdir()
        (ws_b / "prompt.md").write_text("codex_prompt_B")
        inst_b = TaskInstance(
            task_id="t", instance_id="b", task_dir=ws_b,
            workspace_dir=ws_b, reference_dir=ws_b,
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(adapter.solve, inst_a)
            f2 = pool.submit(adapter.solve, inst_b)
            f1.result()
            f2.result()

        assert set(captured_inputs) == {"codex_prompt_A", "codex_prompt_B"}
