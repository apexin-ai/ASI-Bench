"""Tests for the orchestrator and runner components."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.result_schema import (
    CURRENT_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION_FIELD,
)
from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    PromptLevel,
    RunStatus,
    ScoreDetail,
    TaskInstance,
)
from ai4sci_bench.runner.orchestrator import (
    BenchmarkOrchestrator,
    RunConfig,
    _eval_param_expr,
    _evaluate_gates_and_scores,
    _resolve_config_templates,
)

class TestRunConfigDefaults:
    """Verify RunConfig default values match expectations."""

    def test_instances_per_task_default(self):
        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(agent=agent)
        assert config.instances_per_task == 1

    def test_seed_default(self):
        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(agent=agent)
        assert config.seed == 42

    def test_prompt_levels_default(self):
        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(agent=agent)
        assert config.prompt_levels == ["b1", "b2", "b3", "b4"]


class DummyAgent(AgentAdapter):
    """Agent that creates output files matching reference."""
    def __init__(self, output_data=None):
        self.output_data = output_data

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        workspace = task_instance.workspace_dir
        # Copy reference files as "predictions" if available
        if self.output_data:
            for name, data in self.output_data.items():
                np.save(workspace / name, data)

        code_files = [f for f in self.output_data or {} if f.endswith(".py")]
        data_files = [f for f in self.output_data or {} if f.endswith(".npy")]
        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=code_files,
            data_files=data_files,
            log="dummy agent completed",
            execution_time_seconds=0.1,
            status=RunStatus.COMPLETED,
        )


class FailingAgent(AgentAdapter):
    """Agent that always fails."""
    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=task_instance.workspace_dir,
            code_files=[],
            data_files=[],
            log="error occurred",
            execution_time_seconds=0.1,
            status=RunStatus.FAILED,
            error_message="Test failure",
        )


class TestBenchmarkOrchestrator:
    def test_prepare_instances_with_fixed_params(self, sample_task_dir, tmp_dir):
        """Fixed params should generate exactly one instance per prompt level."""
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"
        metadata["_generation_mode"] = "infinite"
        metadata["_generation_precomputed"] = False
        metadata["_generation_settings"] = []

        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            prompt_levels=["b1", "b2"],
            fixed_params={"size": 12, "seed": 7},
            instances_per_task=1,
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)
        orch._load_tasks = MagicMock(return_value=[metadata])

        instances = orch._prepare_instances([metadata])

        assert len(instances) == 2
        assert all(instance.parameters == {"size": 12, "seed": 7} for instance in instances)
        assert {instance.prompt_level for instance in instances} == {PromptLevel.B1, PromptLevel.B2}

    def test_prepare_instances_with_fixed_params_requires_single_instance(self, sample_task_dir, tmp_dir):
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            fixed_params={"size": 12, "seed": 7},
            instances_per_task=2,
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        with pytest.raises(ValueError, match="instances_per_task=1"):
            orch._prepare_instances([metadata])

    def test_evaluate_with_gates_and_scoring(self, sample_task_instance, tmp_dir):
        """Test the evaluation pipeline with gates and scoring."""
        # Create matching pred/ref data
        ref_data = np.random.randn(50).astype(np.float32)
        np.save(sample_task_instance.reference_dir / "output_ref.npy", ref_data)
        np.save(sample_task_instance.workspace_dir / "output.npy", ref_data.copy())

        agent = DummyAgent(output_data={"output.npy": ref_data})
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        agent_output = AgentOutput(
            instance_id=sample_task_instance.instance_id,
            output_dir=sample_task_instance.workspace_dir,
            code_files=[],
            data_files=["output.npy"],
            log="ok",
            execution_time_seconds=0.1,
            status=RunStatus.COMPLETED,
        )

        result = orch._evaluate(sample_task_instance, agent_output)
        assert result.gates_passed is True
        assert result.final_score == 100.0

    def test_evaluate_gates_fail(self, sample_task_instance, tmp_dir):
        """If gates fail, scoring is skipped and score is 0."""
        # Add a gate that will fail
        sample_task_instance.metadata["evaluation"]["gates"] = [
            {
                "scorer": "file_match",
                "config": {
                    "checks": [{"file": "nonexistent_file.npy"}],
                },
            }
        ]

        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        agent_output = AgentOutput(
            instance_id=sample_task_instance.instance_id,
            output_dir=sample_task_instance.workspace_dir,
            code_files=[],
            data_files=[],
            log="",
            execution_time_seconds=0.1,
            status=RunStatus.COMPLETED,
        )

        result = orch._evaluate(sample_task_instance, agent_output)
        assert result.gates_passed is False
        assert result.final_score == 0.0
        assert len(result.score_results) == 0  # scoring skipped

    def test_evaluate_soft_gate_does_not_block_scoring(self, sample_task_instance, tmp_dir):
        ref_data = np.random.randn(50).astype(np.float32)
        np.save(sample_task_instance.reference_dir / "output_ref.npy", ref_data)
        np.save(sample_task_instance.workspace_dir / "output.npy", ref_data.copy())
        (sample_task_instance.workspace_dir / "analysis.py").write_text("import numpy as np\n")

        sample_task_instance.metadata["evaluation"]["gates"] = [
            {
                "scorer": "code_analysis",
                "severity": "soft",
                "config": {
                    "target_file": "analysis.py",
                    "checks": [{"pattern": "import scipy", "required": True}],
                },
            }
        ]
        sample_task_instance.metadata["evaluation"]["scoring"] = [
            {
                "scorer": "numerical",
                "weight": 100,
                "config": {
                    "metric": "relative_l2",
                    "pred_file": "output.npy",
                    "ref_file": "output_ref.npy",
                    "threshold": 0.1,
                },
            }
        ]

        agent = DummyAgent(output_data={"output.npy": ref_data})
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        agent_output = AgentOutput(
            instance_id=sample_task_instance.instance_id,
            output_dir=sample_task_instance.workspace_dir,
            code_files=["analysis.py"],
            data_files=["output.npy"],
            log="ok",
            execution_time_seconds=0.1,
            status=RunStatus.COMPLETED,
        )

        result = orch._evaluate(sample_task_instance, agent_output)
        assert result.gates_passed is True
        assert result.hard_gates_passed is True
        assert result.soft_gate_failures == 1
        assert result.final_score == 100.0
        assert len(result.score_results) == 1

    def test_evaluate_resolves_parametric_file_shapes(self, sample_task_instance, tmp_dir):
        """Evaluation should resolve shape expressions against instance parameters."""
        sample_task_instance.parameters = {"grid_size": 4, "n_frames": 2}
        sample_task_instance.metadata["evaluation"]["gates"] = [
            {
                "scorer": "file_match",
                "config": {
                    "checks": [
                        {"file": "vorticity_frames.npy", "shape": "{5,grid_size,grid_size}", "dtype": "float32"},
                        {"file": "kinetic_energy.npy", "shape": "{n_frames+1}", "dtype": "float32"},
                    ],
                },
            }
        ]
        sample_task_instance.metadata["evaluation"]["scoring"] = []

        np.save(
            sample_task_instance.workspace_dir / "vorticity_frames.npy",
            np.zeros((5, 4, 4), dtype=np.float32),
        )
        np.save(
            sample_task_instance.workspace_dir / "kinetic_energy.npy",
            np.zeros((3,), dtype=np.float32),
        )

        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        agent_output = AgentOutput(
            instance_id=sample_task_instance.instance_id,
            output_dir=sample_task_instance.workspace_dir,
            code_files=[],
            data_files=["vorticity_frames.npy", "kinetic_energy.npy"],
            log="ok",
            execution_time_seconds=0.1,
            status=RunStatus.COMPLETED,
        )

        result = orch._evaluate(sample_task_instance, agent_output)
        assert result.gates_passed is True
        assert result.final_score == 0.0

    def test_save_and_load_result(self, sample_task_instance, tmp_dir):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        eval_result = EvalResult(
            instance_id="test__id",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="DummyAgent",
            parameters={"size": 50},
            gate_results=[
                ScoreDetail(scorer_name="file_match", score=1.0, max_score=1.0, passed=True, details={}),
            ],
            gates_passed=True,
            score_results=[
                ScoreDetail(scorer_name="numerical:relative_l2", score=80.0, max_score=100.0, passed=True, details={}),
            ],
            final_score=80.0,
        )

        orch._save_result(eval_result)

        # Verify saved file
        result_file = tmp_dir / "results" / "physics.test_task" / "test__id__b2.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data[RESULT_SCHEMA_VERSION_FIELD] == CURRENT_RESULT_SCHEMA_VERSION
        assert data["final_score"] == 80.0
        assert data["gates_passed"] is True
        assert data["hard_gates_passed"] is True
        assert data["soft_gate_failures"] == 0

    def test_save_result_persists_agent_output_debug_info(self, sample_task_instance, tmp_dir):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        eval_result = EvalResult(
            instance_id="test__id",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="DummyAgent",
            parameters={"size": 50},
            gate_results=[],
            gates_passed=False,
            score_results=[],
            final_score=0.0,
            agent_output=AgentOutput(
                instance_id="test__id",
                output_dir=tmp_dir,
                code_files=["simulation.py"],
                data_files=["output.npy"],
                log="adapter stderr details",
                execution_time_seconds=0.2,
                status=RunStatus.FAILED,
                error_message="command failed",
            ),
        )

        orch._save_result(eval_result)

        data = json.loads((tmp_dir / "results" / "physics.test_task" / "test__id__b2.json").read_text())
        assert data["agent_output"]["code_files"] == ["simulation.py"]
        assert data["agent_output"]["data_files"] == ["output.npy"]
        assert data["agent_output"]["log"] == "adapter stderr details"
        assert data["agent_output"]["error_message"] == "command failed"
        assert data["agent_output"]["status"] == "failed"

    def test_save_result_persists_raw_agent_log_artifacts(self, sample_task_instance, tmp_dir):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        eval_result = EvalResult(
            instance_id="test__raw",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="DummyAgent",
            parameters={"size": 50},
            gate_results=[],
            gates_passed=False,
            score_results=[],
            final_score=0.0,
            agent_output=AgentOutput(
                instance_id="test__raw",
                output_dir=tmp_dir,
                code_files=["simulation.py"],
                data_files=["output.npy"],
                log="summary log",
                execution_time_seconds=0.2,
                status=RunStatus.FAILED,
                error_message="command failed",
                raw_stdout='{"type":"assistant"}\n',
                raw_stderr="stderr details",
                raw_stdout_format="jsonl",
                raw_model_output="```python\nprint('hello')\n```",
                raw_model_output_format="markdown",
            ),
        )

        orch._save_result(eval_result)

        result_dir = tmp_dir / "results" / "physics.test_task"
        data = json.loads((result_dir / "test__raw__b2.json").read_text())
        assert data["agent_output"]["raw_model_output_file"] == "test__raw__b2.agent_model_output.md"
        assert data["agent_output"]["raw_model_output_format"] == "markdown"
        assert data["agent_output"]["raw_stdout_file"] == "test__raw__b2.agent_stdout.jsonl"
        assert data["agent_output"]["raw_stdout_format"] == "jsonl"
        assert data["agent_output"]["raw_stderr_file"] == "test__raw__b2.agent_stderr.log"
        assert (result_dir / "test__raw__b2.agent_model_output.md").read_text() == "```python\nprint('hello')\n```"
        assert (result_dir / "test__raw__b2.agent_stdout.jsonl").read_text() == '{"type":"assistant"}\n'
        assert (result_dir / "test__raw__b2.agent_stderr.log").read_text() == "stderr details"

    def test_save_result_sanitizes_host_paths_in_saved_logs_and_artifacts(self, sample_task_instance, tmp_dir):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
            agent_metadata={
                "agent_name": "custom_cli",
                "cmd_template": f"python {tmp_dir}/agent.py --workspace {sample_task_instance.workspace_dir}",
                "config": {"cwd": str(tmp_dir / "cwd")},
            },
        )
        orch = BenchmarkOrchestrator(config)
        workspace = sample_task_instance.workspace_dir

        raw_jsonl = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "command": f"/bin/zsh -lc 'pwd && ls {workspace}'",
                            "aggregated_output": str(workspace),
                            "changes": [{"path": str(workspace / 'analysis.py')}],
                        },
                    }
                ),
                "",
            ]
        )

        eval_result = EvalResult(
            instance_id="test__sanitize",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="DummyAgent",
            parameters={"size": 50},
            gate_results=[],
            gates_passed=False,
            score_results=[],
            final_score=0.0,
            agent_output=AgentOutput(
                instance_id="test__sanitize",
                output_dir=workspace,
                code_files=["analysis.py"],
                data_files=[],
                log=f"see workspace {workspace}",
                execution_time_seconds=0.2,
                status=RunStatus.FAILED,
                error_message=f"failed in {workspace / 'analysis.py'}",
                raw_stdout=raw_jsonl,
                raw_stderr=f"traceback at {workspace / 'analysis.py'}",
                raw_stdout_format="jsonl",
            ),
        )

        orch._save_result(eval_result)

        result_dir = tmp_dir / "results" / "physics.test_task"
        data = json.loads((result_dir / "test__sanitize__b2.json").read_text())
        agent_meta = data["provenance"]["agent"]
        assert "<workspace>" in data["agent_output"]["log"]
        assert str(workspace) not in data["agent_output"]["log"]
        assert "<workspace>/analysis.py" in data["agent_output"]["error_message"]
        assert str(workspace / "analysis.py") not in data["agent_output"]["error_message"]
        assert "<abs_path>" in agent_meta["cmd_template"]
        assert str(tmp_dir / "agent.py") not in agent_meta["cmd_template"]
        assert str(workspace) not in agent_meta["cmd_template"]
        assert "<abs_path>" in agent_meta["config"]["cwd"]
        assert str(tmp_dir / "cwd") not in agent_meta["config"]["cwd"]

        stdout_text = (result_dir / "test__sanitize__b2.agent_stdout.jsonl").read_text()
        stderr_text = (result_dir / "test__sanitize__b2.agent_stderr.log").read_text()
        assert "<workspace>" in stdout_text
        assert str(workspace) not in stdout_text
        assert "<workspace>/analysis.py" in stdout_text
        assert str(workspace / "analysis.py") not in stdout_text
        assert "<workspace>/analysis.py" in stderr_text
        assert str(workspace / "analysis.py") not in stderr_text

    def test_sanitize_keeps_repo_root_placeholder_path_suffix(self, sample_task_instance, tmp_dir):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)
        workspace = sample_task_instance.workspace_dir

        value = (
            f"{orch.repo_root}/.venv/bin/python --workspace {workspace}"
        )

        sanitized = orch._sanitize_persisted_text(value, workspace=workspace)

        assert "<repo_root>/.venv/bin/python" in sanitized
        assert "<workspace>" in sanitized
        assert "<repo_root><abs_path>" not in sanitized
        assert str(orch.repo_root) not in sanitized
        assert str(workspace) not in sanitized

    def test_cleanup_workspace_transients_removes_cache_files_only(self, sample_task_instance, tmp_dir):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)
        workspace = sample_task_instance.workspace_dir

        keep_files = [
            workspace / "analysis.py",
            workspace / "potential.npy",
            workspace / "prompt.md",
            workspace / "task_info.json",
        ]
        for path in keep_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npy":
                np.save(path, np.zeros((2,), dtype=np.float32))
            else:
                path.write_text("keep", encoding="utf-8")

        transient_dir = workspace / ".mplconfig"
        transient_dir.mkdir()
        (transient_dir / "fontlist-v390.json").write_text("cache", encoding="utf-8")
        pycache_dir = workspace / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "analysis.cpython-312.pyc").write_bytes(b"cache")
        (workspace / ".DS_Store").write_text("noise", encoding="utf-8")

        orch._cleanup_workspace_transients(workspace)

        assert not transient_dir.exists()
        assert not pycache_dir.exists()
        assert not (workspace / ".DS_Store").exists()
        assert (workspace / "analysis.py").exists()
        assert (workspace / "potential.npy").exists()
        assert (workspace / "prompt.md").exists()
        assert (workspace / "task_info.json").exists()

    def test_run_single_instance_cleans_transient_workspace_cache(self, sample_task_instance, tmp_dir):
        class CacheWritingAgent(AgentAdapter):
            def solve(self, task_instance: TaskInstance) -> AgentOutput:
                workspace = task_instance.workspace_dir
                mpl_dir = workspace / ".mplconfig"
                mpl_dir.mkdir(exist_ok=True)
                (mpl_dir / "fontlist-v390.json").write_text("cache", encoding="utf-8")
                np.save(workspace / "output.npy", np.array([1.0], dtype=np.float32))
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=workspace,
                    code_files=[],
                    data_files=["output.npy"],
                    log="ok",
                    execution_time_seconds=0.1,
                    status=RunStatus.COMPLETED,
                )

        agent = CacheWritingAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        result = orch._run_single_instance(sample_task_instance)

        assert result.status == RunStatus.COMPLETED
        assert not (sample_task_instance.workspace_dir / ".mplconfig").exists()
        assert (sample_task_instance.workspace_dir / "output.npy").exists()

    def test_run_single_instance_retains_workspace_under_output_dir(self, sample_task_instance, tmp_dir):
        class OutputWritingAgent(AgentAdapter):
            def solve(self, task_instance: TaskInstance) -> AgentOutput:
                workspace = task_instance.workspace_dir
                np.save(workspace / "output.npy", np.array([1.0], dtype=np.float32))
                (workspace / "analysis.py").write_text("print('ok')", encoding="utf-8")
                cache_dir = workspace / ".mplconfig"
                cache_dir.mkdir(exist_ok=True)
                (cache_dir / "fontlist.json").write_text("cache", encoding="utf-8")
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=workspace,
                    code_files=["analysis.py"],
                    data_files=["output.npy"],
                    log="ok",
                    execution_time_seconds=0.1,
                    status=RunStatus.COMPLETED,
                )

        # The fixture seeds workspace under tmp_dir; relocate it to a true
        # temp root so the post-run rmtree-of-temp simulation does not also
        # wipe the retained directory under tmp_dir/results/.
        import shutil
        import tempfile

        temp_root = Path(tempfile.mkdtemp(prefix="ai4sci_ws_test_"))
        relocated_workspace = temp_root / "workspace"
        shutil.copytree(sample_task_instance.workspace_dir, relocated_workspace)
        sample_task_instance.workspace_dir = relocated_workspace

        # The fixture only seeds prompt.md; populate task_info.json so the
        # assertion below covers the full agent-facing contract.
        (relocated_workspace / "task_info.json").write_text(
            '{"task_id": "task_test"}', encoding="utf-8"
        )

        sample_task_instance.retained_workspace_dir = (
            tmp_dir
            / "results"
            / "instances"
            / "_workspaces"
            / sample_task_instance.instance_id
            / "workspace_b2"
        )

        agent = OutputWritingAgent()
        config = RunConfig(
            agent=agent,
            sandbox="os",
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        result = orch._run_single_instance(sample_task_instance)

        retained = sample_task_instance.retained_workspace_dir
        assert result.status == RunStatus.COMPLETED
        assert retained is not None
        assert retained.exists()
        assert not retained.is_symlink()
        assert (retained / "prompt.md").exists()
        assert (retained / "task_info.json").exists()
        assert (retained / "analysis.py").exists()
        assert (retained / "output.npy").exists()
        assert not (retained / ".mplconfig").exists()

        # Simulate /tmp being cleared (WSL restart): retained must survive.
        shutil.rmtree(temp_root)

        assert retained.exists()
        assert (retained / "analysis.py").exists()
        assert (retained / "output.npy").exists()

    def test_run_single_instance_preserves_result_when_retention_fails(
        self, sample_task_instance, tmp_dir, monkeypatch
    ):
        ref_data = np.load(sample_task_instance.reference_dir / "output_ref.npy")
        agent = DummyAgent(output_data={"output.npy": ref_data.copy()})
        config = RunConfig(
            agent=agent,
            sandbox="os",
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        def fail_retain(_instance):
            raise OSError("[WinError 3] The system cannot find the path specified.")

        monkeypatch.setattr(orch, "_retain_workspace", fail_retain)

        result = orch._run_single_instance(sample_task_instance)

        assert result.status == RunStatus.COMPLETED
        assert result.final_score > 0

        result_file = (
            tmp_dir
            / "results"
            / sample_task_instance.task_id
            / f"{sample_task_instance.run_key}.json"
        )
        saved = json.loads(result_file.read_text(encoding="utf-8"))
        assert saved["status"] == RunStatus.COMPLETED.value
        assert saved["final_score"] == result.final_score

    def test_retain_workspace_skips_windows_max_path_risk(
        self, sample_task_instance, tmp_dir, monkeypatch
    ):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            sandbox="os",
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        retained = tmp_dir / "retained"
        for i in range(14):
            retained = retained / f"segment_{i:02d}_long"
        retained = retained / sample_task_instance.instance_id / "workspace_b2"
        sample_task_instance.retained_workspace_dir = retained
        monkeypatch.setattr("ai4sci_bench.runner.orchestrator.sys.platform", "win32")

        assert orch._retain_workspace(sample_task_instance) is None
        assert not retained.exists()

    def test_create_isolated_workspace_archives_persisted_dir_on_rerun(self, sample_task_instance, tmp_dir):
        """Re-running with the same --output-dir must not silently destroy
        a workspace previously persisted by --sandbox os (issue #17 follow-up)."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        gen = InstanceGenerator(
            tasks_dir=sample_task_instance.task_dir.parent.parent,
            sandbox="os",
        )
        output_dir = tmp_dir / "rerun_results" / "instances"
        instance_id = sample_task_instance.instance_id
        prompt_level = sample_task_instance.prompt_level

        # Simulate a prior run's persisted workspace at link_path.
        link_path = gen.get_retained_workspace_dir(output_dir, instance_id, prompt_level)
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.mkdir()
        (link_path / "prior_output.npy").write_bytes(b"prior data")

        # And a prior attempt's persisted workspace.
        attempt_path = gen.get_retained_workspace_dir(
            output_dir, instance_id, prompt_level, attempt=2
        )
        attempt_path.mkdir()
        (attempt_path / "prior_attempt.npy").write_bytes(b"prior attempt")

        gen._create_isolated_workspace(output_dir, instance_id, prompt_level)

        backup = link_path.parent / f"{link_path.name}__previous"
        attempt_backup = attempt_path.parent / f"{attempt_path.name}__previous"
        assert backup.is_dir()
        assert (backup / "prior_output.npy").read_bytes() == b"prior data"
        assert attempt_backup.is_dir()
        assert (attempt_backup / "prior_attempt.npy").read_bytes() == b"prior attempt"

        # A second re-run should overwrite the single backup slot, not pile up.
        link_path2 = gen.get_retained_workspace_dir(output_dir, instance_id, prompt_level)
        if link_path2.is_symlink():
            link_path2.unlink()
        link_path2.mkdir()
        (link_path2 / "second_run.npy").write_bytes(b"second run")
        gen._create_isolated_workspace(output_dir, instance_id, prompt_level)
        assert backup.is_dir()
        assert (backup / "second_run.npy").read_bytes() == b"second run"
        assert not (backup / "prior_output.npy").exists()

    def test_retain_workspace_ignores_task_venv(self, sample_task_instance, tmp_dir):
        """Persisting OS workspaces must not copy task venv symlinks."""
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            sandbox="os",
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        workspace = sample_task_instance.workspace_dir
        venv_bin = workspace / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (workspace / "analysis.py").write_text("# generated")
        try:
            (venv_bin / "python3").symlink_to(workspace / "missing-python")
        except OSError:
            (venv_bin / "python3").write_text("broken venv placeholder")

        retained = tmp_dir / "retained" / "workspace_b2"
        sample_task_instance.retained_workspace_dir = retained

        orch._retain_workspace(sample_task_instance)

        assert retained.exists()
        assert (retained / "analysis.py").exists()
        assert not (retained / ".venv").exists()

    def test_save_result_persists_gate_severity_fields(self, sample_task_instance, tmp_dir):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        eval_result = EvalResult(
            instance_id="test__severity",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="DummyAgent",
            parameters={"size": 50},
            gate_results=[
                ScoreDetail(
                    scorer_name="code_analysis",
                    score=0.0,
                    max_score=1.0,
                    passed=False,
                    details={},
                    severity="soft",
                ),
            ],
            gates_passed=True,
            hard_gates_passed=True,
            soft_gate_failures=1,
            score_results=[],
            final_score=15.0,
        )

        orch._save_result(eval_result)
        data = json.loads((tmp_dir / "results" / "physics.test_task" / "test__severity__b2.json").read_text())
        assert data["hard_gates_passed"] is True
        assert data["soft_gate_failures"] == 1
        assert data["gate_results"][0]["severity"] == "soft"

    def test_run_agent_exception_handling(self, sample_task_instance, tmp_dir):
        """Agent exceptions are caught and returned as FAILED status."""
        class CrashAgent(AgentAdapter):
            def solve(self, task_instance):
                raise RuntimeError("Agent crashed!")

        config = RunConfig(
            agent=CrashAgent(),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)
        result = orch._run_agent(sample_task_instance)
        assert result.status == RunStatus.FAILED
        assert "Agent crashed!" in result.log

    def test_resume_skips_completed(self, sample_task_instance, tmp_dir):
        """Resume mode should skip already completed instances."""
        results_dir = tmp_dir / "results" / "physics.test_task"
        results_dir.mkdir(parents=True)

        # Write a completed result
        (results_dir / "test_instance__b2.json").write_text(json.dumps({
            RESULT_SCHEMA_VERSION_FIELD: CURRENT_RESULT_SCHEMA_VERSION,
            "instance_id": "test_instance",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "gates_passed": True,
            "gate_results": [],
            "score_results": [],
            "final_score": 90.0,
            "status": "completed",
        }))

        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
            resume=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)
        completed = orch._load_completed_ids()
        assert "test_instance__b2" in completed

    def test_resume_ignores_non_result_json_and_unsupported_schema(self, sample_task_instance, tmp_dir):
        results_dir = tmp_dir / "results" / "physics.test_task"
        results_dir.mkdir(parents=True)

        (results_dir / "task_info.json").write_text(json.dumps({
            "instance_id": "test_instance",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "expected_outputs": [],
        }))
        (results_dir / "unsupported.json").write_text(json.dumps({
            RESULT_SCHEMA_VERSION_FIELD: 99,
            "instance_id": "test_instance",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "gates_passed": True,
            "gate_results": [],
            "score_results": [],
            "final_score": 90.0,
            "status": "completed",
        }))

        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
            resume=str(tmp_dir / "results"),
        )
        orch = BenchmarkOrchestrator(config)

        assert orch._load_completed_ids() == set()


class TestEvalParamExpr:
    """Tests for _eval_param_expr — the safe arithmetic evaluator."""

    def test_constant_int(self):
        assert _eval_param_expr("42", {}) == 42

    def test_constant_float(self):
        assert _eval_param_expr("3.14", {}) == 3.14

    def test_variable_lookup(self):
        assert _eval_param_expr("grid_size", {"grid_size": 255}) == 255

    def test_addition(self):
        assert _eval_param_expr("n_frames+1", {"n_frames": 200}) == 201

    def test_subtraction(self):
        assert _eval_param_expr("n-2", {"n": 10}) == 8

    def test_multiplication(self):
        assert _eval_param_expr("n*3", {"n": 5}) == 15

    def test_floor_division(self):
        assert _eval_param_expr("n//2", {"n": 7}) == 3

    def test_modulo(self):
        assert _eval_param_expr("n%3", {"n": 10}) == 1

    def test_true_division(self):
        assert _eval_param_expr("n/4", {"n": 10}) == 2.5

    def test_unary_negative(self):
        assert _eval_param_expr("-n", {"n": 5}) == -5

    def test_unary_positive(self):
        assert _eval_param_expr("+n", {"n": 5}) == 5

    def test_complex_expression(self):
        assert _eval_param_expr("a+b*c", {"a": 1, "b": 2, "c": 3}) == 7

    def test_unknown_parameter_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown parameter"):
            _eval_param_expr("missing_var", {"grid_size": 10})

    def test_unsupported_expression_raises_value_error(self):
        # Power operator ** is not supported
        with pytest.raises(ValueError, match="Unsupported parameter expression"):
            _eval_param_expr("n**2", {"n": 3})

    def test_function_call_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported parameter expression"):
            _eval_param_expr("abs(n)", {"n": -3})

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            _eval_param_expr("n/0", {"n": 1})

    def test_string_value_in_params(self):
        # Arithmetic on non-numeric values should raise TypeError at runtime
        with pytest.raises(TypeError):
            _eval_param_expr("n+1", {"n": "abc"})


class TestResolveConfigTemplates:
    """Tests for _resolve_config_templates — recursive template resolution."""

    def test_single_variable(self):
        result = _resolve_config_templates("{grid_size}", {"grid_size": 255})
        assert result == 255

    def test_arithmetic_expression(self):
        result = _resolve_config_templates("{n_frames+1}", {"n_frames": 200})
        assert result == 201

    def test_comma_separated_produces_list(self):
        result = _resolve_config_templates(
            "{5,grid_size,grid_size}", {"grid_size": 255}
        )
        assert result == [5, 255, 255]

    def test_non_template_string_passthrough(self):
        result = _resolve_config_templates("hello world", {"grid_size": 10})
        assert result == "hello world"

    def test_dict_recursion(self):
        config = {
            "shape": "{5,n,n}",
            "dtype": "float32",
            "nested": {"size": "{n+1}"},
        }
        result = _resolve_config_templates(config, {"n": 10})
        assert result["shape"] == [5, 10, 10]
        assert result["dtype"] == "float32"
        assert result["nested"]["size"] == 11

    def test_list_recursion(self):
        config = ["{n}", "static", "{n+1}"]
        result = _resolve_config_templates(config, {"n": 3})
        assert result == [3, "static", 4]

    def test_non_string_passthrough(self):
        assert _resolve_config_templates(42, {}) == 42
        assert _resolve_config_templates(3.14, {}) == 3.14
        assert _resolve_config_templates(True, {}) is True
        assert _resolve_config_templates(None, {}) is None

    def test_string_without_braces_passthrough(self):
        assert _resolve_config_templates("float32", {"n": 10}) == "float32"

    def test_braces_not_at_edges_passthrough(self):
        # Only strings where the whole content is {expr} get resolved
        result = _resolve_config_templates("prefix {n} suffix", {"n": 10})
        assert result == "prefix {n} suffix"

    def test_whitespace_around_braces(self):
        result = _resolve_config_templates(" {n} ", {"n": 42})
        assert result == 42

    def test_empty_dict(self):
        assert _resolve_config_templates({}, {"n": 1}) == {}

    def test_empty_list(self):
        assert _resolve_config_templates([], {"n": 1}) == []

    def test_empty_parameters(self):
        # Non-template values should work even with empty params
        assert _resolve_config_templates("hello", {}) == "hello"
        assert _resolve_config_templates(42, {}) == 42

    def test_deeply_nested_config(self):
        config = {"level1": {"level2": {"checks": [{"shape": "{n,n}"}]}}}
        result = _resolve_config_templates(config, {"n": 8})
        assert result["level1"]["level2"]["checks"][0]["shape"] == [8, 8]


class TestEvaluateGatesAndScoresPromptLevel:
    def test_prompt_level_reaches_gate_and_scoring_configs(self, monkeypatch, tmp_path):
        captured_configs: list[tuple[str, dict]] = []

        class RecordingScorer:
            def __init__(self, name: str):
                self.name = name

            def score(self, pred_dir, ref_dir, config):
                captured_configs.append((self.name, dict(config)))
                weight = float(config.get("weight", 1.0))
                return ScoreDetail(
                    scorer_name=self.name,
                    score=weight,
                    max_score=weight,
                    passed=True,
                    details={},
                )

        monkeypatch.setattr(
            "ai4sci_bench.runner.orchestrator.get_scorer",
            lambda name: RecordingScorer(name),
        )

        evaluation = {
            "gates": [
                {
                    "scorer": "level_gate",
                    "config": {"prompt_level": "auto", "threshold": "{n+1}"},
                },
            ],
            "scoring": [
                {
                    "scorer": "level_score",
                    "weight": 7.0,
                    "config": {"metric": "strict"},
                },
            ],
        }

        _evaluate_gates_and_scores(
            evaluation,
            tmp_path,
            tmp_path,
            {"n": 2},
            prompt_level=PromptLevel.B3,
        )

        assert captured_configs == [
            ("level_gate", {"prompt_level": "b3", "threshold": 3, "weight": 1.0}),
            ("level_score", {"metric": "strict", "prompt_level": "b3", "weight": 7.0}),
        ]

    def test_omitted_prompt_level_keeps_legacy_configs(self, monkeypatch, tmp_path):
        captured_configs: list[dict] = []

        class RecordingScorer:
            def score(self, pred_dir, ref_dir, config):
                captured_configs.append(dict(config))
                return ScoreDetail(
                    scorer_name="score",
                    score=1.0,
                    max_score=1.0,
                    passed=True,
                    details={},
                )

        monkeypatch.setattr(
            "ai4sci_bench.runner.orchestrator.get_scorer",
            lambda name: RecordingScorer(),
        )

        _evaluate_gates_and_scores(
            {"scoring": [{"scorer": "score", "config": {"metric": "plain"}}]},
            tmp_path,
            tmp_path,
            {},
        )

        assert captured_configs == [{"metric": "plain", "weight": 1.0}]


class TestInstancesDirWorkspaceIsolation:
    """Bug #1: two runs sharing --instances-dir must not stomp on each
    other's workspaces.

    Prior to the fix, ``_load_pregenerated_instances`` passed the shared
    ``instances_dir`` to ``_create_isolated_workspace``, which does a
    ``shutil.rmtree`` on any existing same-named workspace. The second
    run would therefore delete the workspace a concurrent first run was
    writing into, manifesting as "workspace cwd deleted" / empty output
    / timeout. After the fix, each run must place its ``_workspaces``
    subtree under ``self.output_dir / "instances"``.
    """

    def _build_instances_dir(self, sample_task_dir, tmp_dir):
        """Pre-generate a shared instances dir the way ``run`` would."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        tasks_dir = sample_task_dir.parent.parent
        gen = InstanceGenerator(tasks_dir)
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        metadata["_task_yaml"] = sample_task_dir / "task.yaml"

        shared_dir = tmp_dir / "shared_instances"
        gen.generate_instance(metadata, {"size": 50, "seed": 0}, shared_dir)
        return shared_dir, metadata, tasks_dir

    def test_workspace_created_under_output_dir_not_instances_dir(
        self, sample_task_dir, tmp_dir
    ):
        shared_dir, metadata, tasks_dir = self._build_instances_dir(
            sample_task_dir, tmp_dir
        )
        # A user-pregenerated instances dir will not have a "_workspaces"
        # subtree yet (only the instance directory itself). Wipe any
        # residual workspace left behind by generate_instance above so
        # we test the pregenerated code path cleanly.
        ws_root = shared_dir / "_workspaces"
        if ws_root.exists():
            import shutil
            shutil.rmtree(ws_root)

        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            prompt_levels=["b2"],
            instances_dir=str(shared_dir),
            tasks_dir=str(tasks_dir),
            output_dir=str(tmp_dir / "run_a"),
        )
        orch = BenchmarkOrchestrator(config)
        instances = orch._prepare_instances([metadata])

        assert len(instances) == 1
        inst = instances[0]
        # Workspace must NOT live under the shared instances_dir.
        # After the temp-dir isolation fix (#13), workspace is in a system
        # temp directory, with a debug symlink under _workspaces/.
        try:
            inst.workspace_dir.relative_to(shared_dir)
            assert False, "workspace should NOT be under shared instances dir"
        except ValueError:
            pass  # Expected
        # And importantly: no new _workspaces tree has been created
        # inside the shared instances dir.
        assert not (shared_dir / "_workspaces").exists()

    @pytest.mark.parametrize("existing_framework_info", [False, True])
    def test_prepare_instances_keeps_shared_source_tree_byte_identical(
        self, sample_task_dir, tmp_dir, existing_framework_info
    ):
        shared_dir, metadata, tasks_dir = self._build_instances_dir(
            sample_task_dir, tmp_dir
        )
        if (shared_dir / "_workspaces").exists():
            import shutil
            shutil.rmtree(shared_dir / "_workspaces")

        instance_dir = next(
            path for path in shared_dir.iterdir() if path.is_dir()
        )
        framework_info = instance_dir / "framework_task_info.json"
        if not existing_framework_info:
            framework_info.unlink()

        before = {
            path.relative_to(shared_dir): path.read_bytes()
            for path in shared_dir.rglob("*")
            if path.is_file()
        }

        output_dir = tmp_dir / "run_read_only"
        config = RunConfig(
            agent=DummyAgent(),
            tasks=["physics.test_task"],
            prompt_levels=["b1"],
            instances_dir=str(shared_dir),
            tasks_dir=str(tasks_dir),
            output_dir=str(output_dir),
        )
        orch = BenchmarkOrchestrator(config)
        instances = orch._prepare_instances([metadata])

        after = {
            path.relative_to(shared_dir): path.read_bytes()
            for path in shared_dir.rglob("*")
            if path.is_file()
        }
        assert after == before

        run_framework_info = (
            output_dir
            / "instances"
            / instances[0].instance_id
            / "framework_task_info.json"
        )
        assert run_framework_info.is_file()
        run_info = json.loads(run_framework_info.read_text(encoding="utf-8"))
        assert run_info["instance_id"] == instances[0].instance_id
        assert run_info["prompt_level"] == "b1"

    def test_second_run_does_not_delete_first_runs_workspace(
        self, sample_task_dir, tmp_dir
    ):
        """Regression: sequential runs sharing --instances-dir must keep
        separate workspace trees. Prior to the fix, the second run's
        ``shutil.rmtree`` on ``<shared>/_workspaces/<id>/workspace_b2``
        would blow away the first run's workspace."""
        shared_dir, metadata, tasks_dir = self._build_instances_dir(
            sample_task_dir, tmp_dir
        )
        import shutil
        if (shared_dir / "_workspaces").exists():
            shutil.rmtree(shared_dir / "_workspaces")

        agent = DummyAgent()
        cfg_a = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            prompt_levels=["b2"],
            instances_dir=str(shared_dir),
            tasks_dir=str(tasks_dir),
            output_dir=str(tmp_dir / "run_a"),
        )
        orch_a = BenchmarkOrchestrator(cfg_a)
        inst_a = orch_a._prepare_instances([metadata])[0]

        # Simulate run A having written agent output into its workspace.
        canary = inst_a.workspace_dir / "agent_wrote_this.txt"
        canary.write_text("run_a output")

        # Run B starts using the same instances_dir.
        cfg_b = RunConfig(
            agent=DummyAgent(),
            tasks=["physics.test_task"],
            prompt_levels=["b2"],
            instances_dir=str(shared_dir),
            tasks_dir=str(tasks_dir),
            output_dir=str(tmp_dir / "run_b"),
        )
        orch_b = BenchmarkOrchestrator(cfg_b)
        inst_b = orch_b._prepare_instances([metadata])[0]

        # The workspaces must be different on disk...
        assert inst_a.workspace_dir != inst_b.workspace_dir
        # ...and run A's canary file must still exist after run B
        # finishes preparing instances. This is the direct regression
        # check for the bug described in
        # docs/benchmark_issues_and_proposals.md §"`--instances-dir`
        # 共享导致 workspace 被破坏".
        assert canary.exists(), (
            "Run B preparing instances clobbered run A's workspace — "
            "Bug #1 regression"
        )
        assert canary.read_text() == "run_a output"


class TestResolveGateSeverity:
    """Tests for _resolve_gate_severity edge cases."""

    def test_default_is_hard(self):
        from ai4sci_bench.runner.orchestrator import _resolve_gate_severity
        assert _resolve_gate_severity({}) == "hard"

    def test_explicit_hard(self):
        from ai4sci_bench.runner.orchestrator import _resolve_gate_severity
        assert _resolve_gate_severity({"severity": "hard"}) == "hard"

    def test_explicit_soft(self):
        from ai4sci_bench.runner.orchestrator import _resolve_gate_severity
        assert _resolve_gate_severity({"severity": "soft"}) == "soft"

    def test_case_insensitive(self):
        from ai4sci_bench.runner.orchestrator import _resolve_gate_severity
        assert _resolve_gate_severity({"severity": "SOFT"}) == "soft"
        assert _resolve_gate_severity({"severity": " Hard "}) == "hard"

    def test_invalid_raises(self):
        from ai4sci_bench.runner.orchestrator import _resolve_gate_severity
        import pytest
        with pytest.raises(ValueError, match="Unsupported gate severity"):
            _resolve_gate_severity({"severity": "medium"})


class TestEvalParamExprEdgeCases:
    """Tests for _eval_param_expr unary ops and edge cases."""

    def test_unary_plus(self):
        assert _eval_param_expr("+n", {"n": 5}) == 5

    def test_unary_minus(self):
        assert _eval_param_expr("-n", {"n": 3}) == -3

    def test_unary_minus_constant(self):
        assert _eval_param_expr("-7", {}) == -7

    def test_floor_division(self):
        assert _eval_param_expr("n // 2", {"n": 7}) == 3

    def test_modulo(self):
        assert _eval_param_expr("n % 3", {"n": 10}) == 1

    def test_unknown_parameter_raises(self):
        import pytest
        with pytest.raises(KeyError, match="Unknown parameter"):
            _eval_param_expr("x + 1", {})

    def test_unsupported_expression_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported"):
            _eval_param_expr("[1, 2, 3]", {})

    def test_compound_expression(self):
        assert _eval_param_expr("n * 2 + 1", {"n": 4}) == 9


class TestCloneWorkspace:
    """Tests for BenchmarkOrchestrator._clone_workspace edge cases."""

    def _make_orchestrator(self, tmp_path):
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            output_dir=str(tmp_path / "results"),
            tasks_dir=str(tmp_path / "tasks"),
        )
        (tmp_path / "tasks").mkdir(exist_ok=True)
        return BenchmarkOrchestrator(config)

    def test_clone_copies_prompt_and_task_info(self, tmp_dir):
        orch = self._make_orchestrator(tmp_dir)
        ws = tmp_dir / "_workspaces" / "inst" / "workspace_b2"
        ws.mkdir(parents=True)
        (ws / "prompt.md").write_text("# Prompt")
        (ws / "task_info.json").write_text('{"task_id": "test"}')
        (ws / "data").mkdir()
        (ws / "data" / "input.npy").write_bytes(b"data")

        instance = TaskInstance(
            task_id="test", instance_id="inst",
            task_dir=tmp_dir, workspace_dir=ws,
            reference_dir=tmp_dir, prompt_level=PromptLevel.B2,
            parameters={}, metadata={},
        )
        clone = orch._clone_workspace(instance, 2)
        assert clone.workspace_dir != ws
        assert (clone.workspace_dir / "prompt.md").read_text() == "# Prompt"
        assert (clone.workspace_dir / "task_info.json").exists()
        assert (clone.workspace_dir / "data" / "input.npy").exists()

    def test_clone_missing_optional_files(self, tmp_dir):
        """Clone should work even if prompt.md / task_info.json / data/ are missing."""
        orch = self._make_orchestrator(tmp_dir)
        ws = tmp_dir / "_workspaces" / "inst" / "workspace_b2"
        ws.mkdir(parents=True)
        # Empty workspace — no prompt.md, no task_info.json, no data/

        instance = TaskInstance(
            task_id="test", instance_id="inst",
            task_dir=tmp_dir, workspace_dir=ws,
            reference_dir=tmp_dir, prompt_level=PromptLevel.B2,
            parameters={}, metadata={},
        )
        clone = orch._clone_workspace(instance, 3)
        assert clone.workspace_dir.exists()
        assert not (clone.workspace_dir / "prompt.md").exists()
        assert not (clone.workspace_dir / "data").exists()

    def test_clone_does_not_copy_agent_output(self, tmp_dir):
        """Agent output from a previous attempt must not leak into the clone."""
        orch = self._make_orchestrator(tmp_dir)
        ws = tmp_dir / "_workspaces" / "inst" / "workspace_b2"
        ws.mkdir(parents=True)
        (ws / "prompt.md").write_text("# Prompt")
        # Simulate agent output from attempt 1
        (ws / "simulation.py").write_text("print('agent code')")
        (ws / "output.npy").write_bytes(b"agent data")

        instance = TaskInstance(
            task_id="test", instance_id="inst",
            task_dir=tmp_dir, workspace_dir=ws,
            reference_dir=tmp_dir, prompt_level=PromptLevel.B2,
            parameters={}, metadata={},
        )
        clone = orch._clone_workspace(instance, 2)
        assert (clone.workspace_dir / "prompt.md").exists()
        assert not (clone.workspace_dir / "simulation.py").exists()
        assert not (clone.workspace_dir / "output.npy").exists()


class TestBuildResultProvenance:
    """Tests for _build_result_provenance graceful degradation."""

    def _make_orchestrator(self, tmp_path, tasks_dir=None):
        agent = DummyAgent()
        td = tasks_dir or str(tmp_path / "tasks")
        config = RunConfig(
            agent=agent,
            output_dir=str(tmp_path / "results"),
            tasks_dir=td,
        )
        (Path(td)).mkdir(parents=True, exist_ok=True)
        return BenchmarkOrchestrator(config)

    def test_provenance_with_missing_task(self, tmp_dir):
        """Missing task metadata should omit runtime, not crash."""
        orch = self._make_orchestrator(tmp_dir)
        eval_result = EvalResult(
            instance_id="inst", task_id="nonexistent.task",
            prompt_level=PromptLevel.B2, agent_name="Test",
            parameters={}, gate_results=[], gates_passed=True,
            score_results=[], final_score=0.0,
            execution_time_seconds=1.0, status=RunStatus.COMPLETED,
        )
        prov = orch._build_result_provenance(eval_result)
        assert "agent" in prov
        assert "sandbox" in prov
        assert "runtime" not in prov

    def test_provenance_with_existing_task(self, tmp_dir, sample_task_dir):
        """When task exists, provenance should include runtime."""
        tasks_dir = sample_task_dir.parent.parent
        orch = self._make_orchestrator(tmp_dir, tasks_dir=str(tasks_dir))
        eval_result = EvalResult(
            instance_id="inst", task_id="physics.test_task",
            prompt_level=PromptLevel.B2, agent_name="Test",
            parameters={}, gate_results=[], gates_passed=True,
            score_results=[], final_score=0.0,
            execution_time_seconds=1.0, status=RunStatus.COMPLETED,
        )
        prov = orch._build_result_provenance(eval_result)
        assert "agent" in prov
        assert "sandbox" in prov
        assert "runtime" in prov

    def test_build_agent_provenance(self, tmp_dir):
        orch = self._make_orchestrator(tmp_dir)
        prov = orch._build_agent_provenance()
        assert prov["adapter_class"] == "DummyAgent"


# ── TODO-5: Trajectory summary statistics tests ─────────────────────────


class TestTrajectorySummaryInOrchestrator:
    """Tests for trajectory summary computation in _save_result."""

    def _make_orchestrator(self, tmp_dir):
        task_dir = tmp_dir / "tasks" / "physics" / "test_task"
        task_dir.mkdir(parents=True)
        task_config = {
            "id": "physics.test_task",
            "name": "Test Task",
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {"gates": [], "scoring": []},
        }
        (task_dir / "task.yaml").write_text(yaml.dump(task_config))
        agent = DummyAgent()
        config = RunConfig(
            agent=agent,
            tasks_dir=str(tmp_dir / "tasks"),
            output_dir=str(tmp_dir / "output"),
        )
        return BenchmarkOrchestrator(config)

    def test_save_result_includes_trajectory_summary(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        jsonl_events = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "name": "Write", "input": {"file_path": "sim.py", "content": "code"}},
            ]}}),
            json.dumps({"type": "user", "message": {"content": [
                {"type": "tool_result", "content": "ok"},
            ]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "input": {"command": "python sim.py"}},
            ]}}),
        ]
        raw_stdout = "\n".join(jsonl_events)
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        agent_output = AgentOutput(
            instance_id="test_inst",
            output_dir=workspace,
            code_files=["sim.py"],
            data_files=[],
            log="ok",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_stdout=raw_stdout,
            raw_stdout_format="jsonl",
        )
        eval_result = EvalResult(
            instance_id="test_inst",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="Test",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=80.0,
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            agent_output=agent_output,
        )
        orch._save_result(eval_result)

        result_file = tmp_path / "output" / "physics.test_task" / "test_inst__b2.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert "trajectory_summary" in data.get("agent_output", {})
        ts = data["agent_output"]["trajectory_summary"]
        assert ts["total_turns"] >= 1
        assert ts["total_tool_calls"] >= 1

    def test_save_result_parses_current_codex_item_events(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        jsonl_events = [
            json.dumps({"type": "thread.started", "thread_id": "thread_1"}),
            json.dumps({
                "type": "item.completed",
                "item": {"id": "item_1", "type": "agent_message", "text": "planning"},
            }),
            json.dumps({
                "type": "item.started",
                "item": {
                    "id": "item_2",
                    "type": "command_execution",
                    "command": "python -V",
                },
            }),
            json.dumps({
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "command_execution",
                    "command": "python -V",
                    "aggregated_output": "Python 3.13",
                    "exit_code": 0,
                    "status": "completed",
                },
            }),
            json.dumps({
                "type": "item.completed",
                "item": {
                    "id": "item_3",
                    "type": "file_change",
                    "changes": [{"path": "analysis.py", "kind": "modify"}],
                    "status": "completed",
                },
            }),
        ]
        workspace = tmp_path / "codex_ws"
        workspace.mkdir()
        agent_output = AgentOutput(
            instance_id="test_codex_schema",
            output_dir=workspace,
            code_files=["analysis.py"],
            data_files=[],
            log="ok",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_stdout="\n".join(jsonl_events),
            raw_stdout_format="jsonl",
        )
        eval_result = EvalResult(
            instance_id="test_codex_schema",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="Test",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=80.0,
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            agent_output=agent_output,
        )
        orch._save_result(eval_result)

        result_file = tmp_path / "output" / "physics.test_task" / "test_codex_schema__b2.json"
        data = json.loads(result_file.read_text())
        ao = data["agent_output"]
        ts = ao["trajectory_summary"]
        assert ts["total_turns"] == 1
        assert ts["total_tool_calls"] == 2
        assert ts["tool_call_distribution"]["command_execution"] == 1
        assert ts["tool_call_distribution"]["file_change"] == 1
        assert ts["total_code_executions"] == 1
        assert ts["unique_files_modified"] == ["analysis.py"]
        assert "analysis.py" in ao["file_versions"]

    def test_trajectory_summary_tool_call_distribution(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        jsonl_events = []
        for _ in range(3):
            jsonl_events.append(json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "a.py", "content": "x"}},
            ]}}))
        for _ in range(2):
            jsonl_events.append(json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ]}}))
        jsonl_events.append(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
        ]}}))

        workspace = tmp_path / "ws2"
        workspace.mkdir()
        agent_output = AgentOutput(
            instance_id="test_dist",
            output_dir=workspace,
            code_files=[],
            data_files=[],
            log="",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_stdout="\n".join(jsonl_events),
            raw_stdout_format="jsonl",
        )

        data = orch._compute_trajectory_data(agent_output)
        ts = data.get("trajectory_summary", {})
        dist = ts.get("tool_call_distribution", {})
        assert dist.get("Write") == 3
        assert dist.get("Bash") == 2
        assert dist.get("Read") == 1

    def test_trajectory_summary_thinking_total_chars(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        jsonl_events = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "x" * 500},
            ]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "y" * 800},
            ]}}),
        ]
        workspace = tmp_path / "ws3"
        workspace.mkdir()
        agent_output = AgentOutput(
            instance_id="test_think",
            output_dir=workspace,
            code_files=[],
            data_files=[],
            log="",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_stdout="\n".join(jsonl_events),
            raw_stdout_format="jsonl",
        )
        data = orch._compute_trajectory_data(agent_output)
        assert data["trajectory_summary"]["thinking_total_chars"] == 1300

    def test_trajectory_summary_absent_when_no_jsonl(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        workspace = tmp_path / "ws4"
        workspace.mkdir()
        agent_output = AgentOutput(
            instance_id="test_nojsonl",
            output_dir=workspace,
            code_files=[],
            data_files=[],
            log="plain text log",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_stdout="plain text",
            raw_stdout_format="log",
        )
        data = orch._compute_trajectory_data(agent_output)
        assert "trajectory_summary" not in data

    def test_direct_llm_json_generates_trajectory_summary(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        workspace = tmp_path / "ws_direct"
        workspace.mkdir()
        structured_log = [
            {
                "step": "prompt",
                "user_prompt": "solve task",
                "system_prompt": "system",
                "model": "openai/test",
                "timestamp_ms": 0,
            },
            {
                "step": "llm_response",
                "content": "```python\nprint('hi')\n```",
                "content_length": 24,
                "timestamp_ms": 10,
            },
            {
                "step": "code_extraction",
                "num_candidates": 1,
                "candidate_scores": [5],
                "selected_index": 0,
                "selected_code": "print('hi')\n",
                "selected_code_length": 12,
                "timestamp_ms": 12,
            },
            {
                "step": "code_execution",
                "code_file": "simulation.py",
                "exit_code": 0,
                "duration_ms": 4,
                "timestamp_ms": 16,
            },
        ]
        agent_output = AgentOutput(
            instance_id="direct_inst",
            output_dir=workspace,
            code_files=["simulation.py"],
            data_files=[],
            log="ok",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_model_output=json.dumps(structured_log),
            raw_model_output_format="json",
        )

        data = orch._compute_trajectory_data(agent_output)

        assert data["trajectory_summary"]["total_turns"] == 1
        assert data["trajectory_summary"]["total_code_executions"] == 1
        assert len(data["trajectory"]) == 4

    def test_save_result_persists_direct_llm_audit_artifacts(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        workspace = tmp_path / "ws_direct_artifacts"
        workspace.mkdir()
        code = "from pathlib import Path\nprint(Path('./data/input.txt'))\n"
        (workspace / "simulation.py").write_text(code, encoding="utf-8")
        (workspace / "output.npy").write_bytes(b"artifact")
        structured_log = [
            {
                "step": "llm_response",
                "content": f"```python\n{code}```",
                "timestamp_ms": 10,
            },
            {
                "step": "code_extraction",
                "selected_code": code,
                "timestamp_ms": 12,
            },
            {
                "step": "code_execution",
                "code_file": "simulation.py",
                "exit_code": 0,
                "timestamp_ms": 16,
            },
        ]
        agent_output = AgentOutput(
            instance_id="direct_artifact",
            output_dir=workspace,
            code_files=["simulation.py"],
            data_files=["output.npy"],
            log="ok",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_model_output=json.dumps(structured_log),
            raw_model_output_format="json",
        )
        eval_result = EvalResult(
            instance_id="direct_artifact",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="Test",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=80.0,
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            agent_output=agent_output,
        )

        orch._save_result(eval_result)

        result_file = tmp_path / "output" / "physics.test_task" / "direct_artifact__b2.json"
        data = json.loads(result_file.read_text())
        ao = data["agent_output"]
        assert ao["llm_response_file"] == "direct_artifact__b2.llm_response.md"
        assert ao["selected_code_file"] == "direct_artifact__b2.selected_code.py"
        assert ao["trajectory_file"] == "direct_artifact__b2.trajectory.json"
        assert "direct_artifact__b2.artifact.simulation.py" in ao["execution_artifacts"]
        assert "direct_artifact__b2.artifact.output.npy" in ao["execution_artifacts"]
        result_dir = tmp_path / "output" / "physics.test_task"
        assert (result_dir / ao["llm_response_file"]).read_text(encoding="utf-8")
        assert (result_dir / ao["selected_code_file"]).read_text(encoding="utf-8") == code
        assert (result_dir / "direct_artifact__b2.artifact.simulation.py").read_text(
            encoding="utf-8"
        ) == code
        assert json.loads((result_dir / ao["trajectory_file"]).read_text(encoding="utf-8"))

    def test_file_history_saved_in_result_json(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        jsonl_events = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "sim.py", "content": "code v1"}},
            ]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "sim.py", "old_string": "v1", "new_string": "v2"}},
            ]}}),
        ]

        workspace = tmp_path / "ws5"
        workspace.mkdir()
        agent_output = AgentOutput(
            instance_id="test_fv",
            output_dir=workspace,
            code_files=["sim.py"],
            data_files=[],
            log="",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            raw_stdout="\n".join(jsonl_events),
            raw_stdout_format="jsonl",
        )
        eval_result = EvalResult(
            instance_id="test_fv",
            task_id="physics.test_task",
            prompt_level=PromptLevel.B2,
            agent_name="Test",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=80.0,
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            agent_output=agent_output,
        )
        orch._save_result(eval_result)

        result_file = tmp_path / "output" / "physics.test_task" / "test_fv__b2.json"
        data = json.loads(result_file.read_text())
        fv = data["agent_output"].get("file_versions", {})
        assert "sim.py" in fv
        assert len(fv["sim.py"]) == 2
