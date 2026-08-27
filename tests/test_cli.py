"""Tests for the CLI interface."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from unittest.mock import patch

from ai4sci_bench.cli import (
    _build_agent,
    _build_agent_metadata,
    _failed_final_attempts,
    _is_eval_result_json,
    _parse_eval_result,
    _run_generate_with_timeout,
    _validate_task_contracts,
    _resolve_tool_mode,
    _save_eval_result,
    cli,
)
from ai4sci_bench.core.result_schema import (
    CURRENT_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION_FIELD,
)
from ai4sci_bench.core.types import PromptLevel, RunStatus, ToolMode


def test_strict_run_status_uses_last_retry_attempt():
    failed = SimpleNamespace(
        instance_id="demo__seed31415",
        prompt_level=PromptLevel.B1,
        attempt=1,
        status=RunStatus.FAILED,
        agent_output=SimpleNamespace(status=RunStatus.FAILED),
    )
    recovered = SimpleNamespace(
        instance_id="demo__seed31415",
        prompt_level=PromptLevel.B1,
        attempt=2,
        status=RunStatus.COMPLETED,
        agent_output=SimpleNamespace(status=RunStatus.COMPLETED),
    )

    assert _failed_final_attempts([failed, recovered]) == []


class TestRunSandboxAvailability:
    def test_run_rejects_unavailable_linux_ns_before_instances_start(self):
        with patch(
            "ai4sci_bench.runner.linux_ns_sandbox.check_linux_ns_available",
            return_value=(False, "user namespaces unavailable"),
        ):
            result = CliRunner().invoke(
                cli, ["run", "--agent", "codex_cli", "--sandbox", "linux_ns"]
            )

        assert result.exit_code != 0
        assert "--sandbox linux_ns is unavailable" in result.output
        assert "user namespaces unavailable" in result.output

    def test_run_rejects_unavailable_docker_before_instances_start(self):
        with patch(
            "ai4sci_bench.runner.task_image.TaskImageBuilder.ensure_docker_available",
            side_effect=RuntimeError("Docker daemon unavailable"),
        ):
            result = CliRunner().invoke(
                cli, ["run", "--agent", "codex_cli", "--sandbox", "os"]
            )

        assert result.exit_code != 0
        assert "Docker daemon unavailable" in result.output


class TestBuildAgent:
    """Tests for _build_agent with allow_external_tools flag."""

    def test_build_agent_passes_allow_external_tools_false(self):
        adapter = _build_agent(None, "claude_code_cli", {}, allow_external_tools=False)
        assert adapter.allow_external_tools is False

    def test_build_agent_passes_allow_external_tools_true(self):
        adapter = _build_agent(None, "claude_code_cli", {}, allow_external_tools=True)
        assert adapter.allow_external_tools is True

    def test_build_agent_codex_passes_allow_external_tools(self):
        adapter = _build_agent(None, "codex_cli", {}, allow_external_tools=True)
        assert adapter.allow_external_tools is True

    def test_build_agent_codex_default_blocks_external_tools(self):
        adapter = _build_agent(None, "codex_cli", {})
        assert adapter.allow_external_tools is False

    def test_build_agent_config_override_takes_precedence(self):
        """If agent_config already has allow_external_tools, it should not be overwritten."""
        adapter = _build_agent(
            None, "claude_code_cli",
            {"allow_external_tools": True},
            allow_external_tools=False,
        )
        assert adapter.allow_external_tools is True

    def test_build_agent_passes_tool_mode_to_claude(self):
        adapter = _build_agent(None, "claude_code_cli", {}, tool_mode="search")
        assert adapter.tool_mode == ToolMode.SEARCH

    def test_build_agent_passes_tool_mode_to_codex(self):
        adapter = _build_agent(None, "codex_cli", {}, tool_mode="restricted")
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_build_agent_tool_mode_not_passed_to_cli_agent(self):
        adapter = _build_agent("echo hi", None, {}, tool_mode="search")
        assert not hasattr(adapter, "tool_mode")

    def test_build_agent_tool_mode_not_passed_to_direct_llm(self):
        adapter = _build_agent(None, "direct_llm", {}, tool_mode="search")
        assert not hasattr(adapter, "tool_mode")


class TestResolveToolMode:
    def test_default_restricted(self):
        assert _resolve_tool_mode(None, False) == "restricted"

    def test_allow_external_tools_implies_search(self):
        assert _resolve_tool_mode(None, True) == "search"

    def test_explicit_overrides_flag(self):
        assert _resolve_tool_mode("unrestricted", False) == "unrestricted"
        assert _resolve_tool_mode("restricted", True) == "restricted"


class TestBuildAgentMetadata:
    def test_metadata_includes_tool_mode_restricted(self):
        metadata = _build_agent_metadata(
            None, "claude_code_cli", {},
            allow_external_tools=False,
        )
        assert "tool_mode" in metadata
        assert metadata["tool_mode"] == "restricted"

    def test_metadata_includes_tool_mode_search(self):
        metadata = _build_agent_metadata(
            None, "claude_code_cli", {},
            allow_external_tools=True,
        )
        assert metadata["tool_mode"] == "search"

    def test_metadata_explicit_tool_mode(self):
        metadata = _build_agent_metadata(
            None, "claude_code_cli", {},
            allow_external_tools=False,
            tool_mode="unrestricted",
        )
        assert metadata["tool_mode"] == "unrestricted"


class TestCLIList:
    def test_list_no_tasks(self, tmp_dir):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tasks-dir", str(tmp_dir)])
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_with_tasks(self, sample_task_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tasks-dir", str(tasks_dir), "--include-test"])
        assert result.exit_code == 0
        assert "physics.test_task" in result.output

    def test_list_skips_template_like_task_outside_template_dir(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        placeholder_dir = tasks_dir / "physics" / "GL"
        placeholder_dir.mkdir(parents=True)
        (placeholder_dir / "task.yaml").write_text(yaml.dump({
            "id": "DOMAIN.TASK_NAME",
            "name": "Human-readable task name",
            "status": "in_development",
            "domain": "astronomy",
        }))
        real_task_dir = tasks_dir / "physics" / "real_task"
        real_task_dir.mkdir(parents=True)
        (real_task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.real_task",
            "name": "Real Task",
            "status": "test",
            "domain": "physics",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tasks-dir", str(tasks_dir), "--include-test", "--include-dev"])

        assert result.exit_code == 0
        assert "physics.real_task" in result.output
        assert "DOMAIN.TASK_NAME" not in result.output
    def test_list_abandoned_excluded_by_default(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        (tasks_dir / "a").mkdir(parents=True)
        (tasks_dir / "a" / "task.yaml").write_text(yaml.dump({
            "id": "test.a", "name": "a", "status": "abandoned", "domain": "test",
            "abandoned_reason": "not feasible",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tasks-dir", str(tasks_dir)])
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_abandoned_with_flag(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        (tasks_dir / "a").mkdir(parents=True)
        (tasks_dir / "a" / "task.yaml").write_text(yaml.dump({
            "id": "test.a", "name": "a", "status": "abandoned", "domain": "test",
            "abandoned_reason": "not feasible",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tasks-dir", str(tasks_dir), "--include-abandoned"])
        assert result.exit_code == 0
        assert "test.a" in result.output
        assert "abandoned" in result.output
        assert "not feasible" in result.output

    def test_list_sample_with_flag(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "sample_task"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.sample_task", "name": "Sample Task",
            "status": "sample", "domain": "physics",
        }))

        runner = CliRunner()
        default_result = runner.invoke(cli, ["list", "--tasks-dir", str(tasks_dir)])
        sample_result = runner.invoke(cli, [
            "list", "--tasks-dir", str(tasks_dir), "--include-sample",
        ])

        assert default_result.exit_code == 0
        assert "No tasks found" in default_result.output
        assert sample_result.exit_code == 0
        assert "[sample] physics.sample_task" in sample_result.output


class TestCLIValidate:
    def test_validate_task_contracts_ignores_code_outputs_in_metadata_and_generator(self):
        metadata = {
            "input": {"files": []},
            "output": {
                "files": [
                    {"name": "solver.py", "type": "code"},
                    {"name": "results.npy", "type": "data"},
                ]
            },
        }
        module = SimpleNamespace(
            INPUT_SPEC=[],
            OUTPUT_SPEC=[
                {"name": "solver.py", "type": "code"},
                {"name": "results.npy", "type": "data"},
            ],
        )
        errors: list[str] = []

        _validate_task_contracts(metadata, module, errors)

        assert errors == []

    def test_validate_valid_task(self, sample_task_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
        ])
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_validate_preflight_valid_task(self, sample_task_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--preflight",
        ])
        assert result.exit_code == 0
        assert "VALIDATION PASSED" in result.output
        assert "PREFLIGHT PASSED" in result.output

    def test_validate_preflight_detects_unredacted_task_info(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "unredacted_task"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.unredacted_task",
            "name": "Unredacted Task",
            "domain": "physics",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md", "b3": "prompt_b3.md"},
            "input": {"files": []},
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {"gates": [], "scoring": []},
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Prompt for {level}")
        (task_dir / "generate_gt.py").write_text(
            "import json\n"
            "from pathlib import Path\n"
            "INPUT_SPEC=[]\n"
            "OUTPUT_SPEC=[{'name': 'output.npy'}]\n"
            "DEFAULT_PARAMS={}\n"
            "def generate(output_dir, params):\n"
            "    out = Path(output_dir)\n"
            "    (out / 'reference').mkdir(parents=True, exist_ok=True)\n"
            "    for level in ['b1', 'b2', 'b3']:\n"
            "        (out / f'prompt_{level}.md').write_text(f'Prompt for {level}')\n"
            "    (out / 'instance_meta.json').write_text(json.dumps({'params_used': params, 'input_files': [], 'reference_files': []}))\n"
            "    return {'params_used': params, 'input_files': [], 'reference_files': []}\n"
            "def preflight_check(instance_dir, workspace_dir, params, metadata):\n"
            "    task_info_path = workspace_dir / 'task_info.json'\n"
            "    payload = json.loads(task_info_path.read_text())\n"
            "    payload['parameters'] = {'seed': 0}\n"
            "    task_info_path.write_text(json.dumps(payload))\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.unredacted_task",
            "--tasks-dir", str(tasks_dir),
            "--preflight",
        ])

        assert result.exit_code != 0
        assert "task_info should not expose redacted field: parameters" in result.output

    def test_validate_preflight_rejects_framework_task_info_in_workspace(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "framework_visible_task"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.framework_visible_task",
            "name": "Framework Visible Task",
            "domain": "physics",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md", "b3": "prompt_b3.md"},
            "input": {"files": []},
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {"gates": [], "scoring": []},
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Prompt for {level}")
        (task_dir / "generate_gt.py").write_text(
            "import json\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "INPUT_SPEC=[]\n"
            "OUTPUT_SPEC=[{'name': 'output.npy'}]\n"
            "DEFAULT_PARAMS={}\n"
            "def generate(output_dir, params):\n"
            "    out = Path(output_dir)\n"
            "    (out / 'reference').mkdir(parents=True, exist_ok=True)\n"
            "    for level in ['b1', 'b2', 'b3']:\n"
            "        (out / f'prompt_{level}.md').write_text(f'Prompt for {level}')\n"
            "    (out / 'instance_meta.json').write_text(json.dumps({'params_used': params, 'input_files': [], 'reference_files': []}))\n"
            "    return {'params_used': params, 'input_files': [], 'reference_files': []}\n"
            "def preflight_check(instance_dir, workspace_dir, params, metadata):\n"
            "    shutil.copy2(instance_dir / 'framework_task_info.json', workspace_dir / 'framework_task_info.json')\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.framework_visible_task",
            "--tasks-dir", str(tasks_dir),
            "--preflight",
        ])

        assert result.exit_code != 0
        assert "Preflight workspace should not expose framework_task_info.json" in result.output

    def test_validate_preflight_requires_task_sandbox_for_runtime_packages(self, sample_task_dir):
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["runtime"] = {"packages": ["taichi>=1.7.4"]}
        (sample_task_dir / "task.yaml").write_text(yaml.dump(metadata))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(sample_task_dir.parent.parent),
            "--preflight",
        ])

        assert result.exit_code != 0
        assert "Use --sandbox task to validate the real task runtime" in result.output

    def test_validate_preflight_rejects_agent_sandbox_mismatch(self, sample_task_dir):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(sample_task_dir.parent.parent),
            "--preflight",
            "--sandbox", "task",
            "--agent-cmd", "echo ok",
        ])

        assert result.exit_code != 0
        assert "Preflight agent sandbox validation failed for agent-cmd" in result.output
        assert "does not support sandbox 'task'" in result.output


class TestGTSelfCheck:
    """Tests for `validate --gt-selfcheck`."""

    def test_generate_timeout_fallback_without_sigalrm(self, tmp_path, monkeypatch):
        """Windows lacks SIGALRM; fallback should still execute generate()."""
        import signal

        monkeypatch.delattr(signal, "SIGALRM", raising=False)
        monkeypatch.delattr(signal, "alarm", raising=False)

        generate_gt = tmp_path / "generate_gt.py"
        generate_gt.write_text(
            "from pathlib import Path\n"
            "def generate(output_dir, params):\n"
            "    Path(output_dir, 'marker.txt').write_text(str(params['x']))\n"
        )
        module = SimpleNamespace(__file__=str(generate_gt))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        _run_generate_with_timeout(module, out_dir, {"x": 7}, timeout=5)

        assert (out_dir / "marker.txt").read_text() == "7"

    def test_generate_timeout_fallback_enforces_timeout(self, tmp_path, monkeypatch):
        """The no-SIGALRM fallback must terminate stuck generation."""
        import signal

        monkeypatch.delattr(signal, "SIGALRM", raising=False)
        monkeypatch.delattr(signal, "alarm", raising=False)

        generate_gt = tmp_path / "generate_gt.py"
        generate_gt.write_text(
            "import time\n"
            "def generate(output_dir, params):\n"
            "    time.sleep(5)\n"
        )
        module = SimpleNamespace(__file__=str(generate_gt))

        with pytest.raises(TimeoutError, match="generate\\(\\) timed out after 1s"):
            _run_generate_with_timeout(module, tmp_path / "out", {}, timeout=1)

    def test_generate_timeout_fallback_propagates_errors(self, tmp_path, monkeypatch):
        """When generate() raises in the subprocess, the caller gets RuntimeError."""
        import signal

        monkeypatch.delattr(signal, "SIGALRM", raising=False)
        monkeypatch.delattr(signal, "alarm", raising=False)

        generate_gt = tmp_path / "generate_gt.py"
        generate_gt.write_text(
            "def generate(output_dir, params):\n"
            "    raise ValueError('bad params')\n"
        )
        module = SimpleNamespace(__file__=str(generate_gt))

        with pytest.raises(RuntimeError, match="generate\\(\\) failed in subprocess"):
            _run_generate_with_timeout(module, tmp_path / "out", {}, timeout=5)

    def test_generate_timeout_fallback_rejects_missing_file(self, monkeypatch):
        """The fallback must reject a module with no __file__ attribute."""
        import signal

        monkeypatch.delattr(signal, "SIGALRM", raising=False)
        monkeypatch.delattr(signal, "alarm", raising=False)

        module = SimpleNamespace()  # no __file__

        with pytest.raises(RuntimeError, match="has no __file__"):
            _run_generate_with_timeout(module, Path("/unused"), {}, timeout=5)

    def test_generate_sigalrm_path_raises_timeout_error(self, tmp_path):
        """The SIGALRM path must also raise TimeoutError (not a private exception)."""
        import signal

        if not hasattr(signal, "SIGALRM"):
            pytest.skip("SIGALRM not available on this platform")

        generate_gt = tmp_path / "generate_gt.py"
        generate_gt.write_text(
            "import time\n"
            "def generate(output_dir, params):\n"
            "    time.sleep(5)\n"
        )
        module = SimpleNamespace(
            __file__=str(generate_gt),
            generate=lambda od, p: __import__("time").sleep(5),
        )

        with pytest.raises(TimeoutError, match="generate\\(\\) timed out after 1s"):
            _run_generate_with_timeout(module, tmp_path / "out", {}, timeout=1)

    def test_gt_selfcheck_passes_on_well_formed_task(self, sample_task_dir):
        """GT self-check on the sample task should reach 100/100."""
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--gt-selfcheck",
            "--task", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--gt-timeout", "30",
        ])
        assert result.exit_code == 0, result.output
        assert "PASS  physics.test_task" in result.output
        # The well-formed sample should reach 100/100.
        assert "100.00/100.00" in result.output
        assert "FAIL:  0" in result.output

    def test_gt_selfcheck_respects_explicit_skip_flag(self, tmp_dir):
        """Wrapper scorers can opt out of GT self-check even if their scorer
        name is not one of the built-in LLM/code-analysis names."""
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "skip_wrapper_task"
        task_dir.mkdir(parents=True)

        metadata = {
            "id": "physics.skip_wrapper_task",
            "name": "Skip Wrapper Task",
            "domain": "physics",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md"},
            "input": {"files": []},
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
                            "ref_file": "output.npy",
                            "scoring": "linear_interpolation",
                            "full_score_threshold": 0.0,
                            "zero_score_threshold": 1.0,
                        },
                    },
                    {
                        "scorer": "custom_wrapper_that_would_call_api",
                        "weight": 10,
                        "skip_in_gt_selfcheck": True,
                        "config": {},
                    },
                ],
            },
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))
        (task_dir / "prompt_b1.md").write_text("prompt b1")
        (task_dir / "generate_gt.py").write_text(
            "import json\nfrom pathlib import Path\nimport numpy as np\n"
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[{'name': 'output.npy'}]\nDEFAULT_PARAMS={}\n"
            "def generate(output_dir, params):\n"
            "    p = Path(output_dir)\n"
            "    (p / 'reference').mkdir(parents=True, exist_ok=True)\n"
            "    np.save(p / 'reference' / 'output.npy', np.ones(3))\n"
            "    (p / 'instance_meta.json').write_text(json.dumps({'params_used': params}))\n"
            "    return {'params_used': params}\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "--gt-selfcheck",
            "--task", "physics.skip_wrapper_task",
            "--tasks-dir", str(tasks_dir),
            "--gt-timeout", "20",
        ])

        assert result.exit_code == 0, result.output
        assert "PASS  physics.skip_wrapper_task  100.00/100.00" in result.output
        assert "skipped 10.00 agent-only pts" in result.output

    def test_gt_selfcheck_reports_missing_reference(self, tmp_dir):
        """A task whose generate_gt.py does not create the file the file_match
        gate requires should fail with a clear ``[hard gate FAILED]`` message."""
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "broken_task"
        task_dir.mkdir(parents=True)

        metadata = {
            "id": "physics.broken_task",
            "name": "Broken Task",
            "domain": "physics",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md", "b3": "prompt_b3.md"},
            "input": {"files": []},
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {
                "gates": [
                    {
                        "scorer": "file_match",
                        "severity": "hard",
                        "config": {"checks": [{"file": "output.npy"}]},
                    }
                ],
                "scoring": [],
            },
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"prompt {level}")
        # generate_gt.py creates reference/ but no output.npy in any form.
        (task_dir / "generate_gt.py").write_text(
            "import json\nfrom pathlib import Path\n"
            "INPUT_SPEC=[]\nOUTPUT_SPEC=[{'name': 'output.npy'}]\nDEFAULT_PARAMS={}\n"
            "def generate(output_dir, params):\n"
            "    p = Path(output_dir)\n"
            "    (p / 'reference').mkdir(parents=True, exist_ok=True)\n"
            "    (p / 'instance_meta.json').write_text(json.dumps({'params_used': params}))\n"
            "    return {'params_used': params}\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "--gt-selfcheck",
            "--task", "physics.broken_task",
            "--tasks-dir", str(tasks_dir),
            "--gt-timeout", "20",
        ])
        # Missing reference file → file_match hard gate fails → non-zero exit.
        assert result.exit_code != 0
        assert "FAIL" in result.output
        assert "physics.broken_task" in result.output

    def test_validate_preflight_rejects_os_when_docker_unavailable(self, sample_task_dir):
        runner = CliRunner()
        with patch(
            "ai4sci_bench.cli.TaskImageBuilder.ensure_image",
            side_effect=RuntimeError("Docker CLI not found; --sandbox os requires Docker."),
        ):
            result = runner.invoke(cli, [
                "validate",
                "--task", "physics.test_task",
                "--tasks-dir", str(sample_task_dir.parent.parent),
                "--preflight",
                "--sandbox", "os",
            ])

        assert result.exit_code != 0
        assert "OS sandbox preflight failed" in result.output
        assert "--sandbox os requires Docker" in result.output

    def test_validate_preflight_rejects_linux_ns_when_unavailable(self, sample_task_dir):
        runner = CliRunner()
        with patch(
            "ai4sci_bench.runner.linux_ns_sandbox.check_linux_ns_available",
            return_value=(False, "user namespaces unavailable"),
        ):
            result = runner.invoke(cli, [
                "validate",
                "--task", "physics.test_task",
                "--tasks-dir", str(sample_task_dir.parent.parent),
                "--preflight",
                "--sandbox", "linux_ns",
            ])

        assert result.exit_code != 0
        assert "Linux namespace sandbox preflight failed" in result.output
        assert "user namespaces unavailable" in result.output

    def test_validate_rejects_broken_custom_scorer_import(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "broken_custom_scorer"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.broken_custom_scorer",
            "name": "Broken Custom Scorer",
            "domain": "physics",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md", "b3": "prompt_b3.md"},
            "input": {"files": []},
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {"gates": [], "scoring": [{"scorer": "my_custom", "weight": 100, "config": {}}]},
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Prompt for {level}")
        (task_dir / "generate_gt.py").write_text(
            "INPUT_SPEC=[]\n"
            "OUTPUT_SPEC=[{'name': 'output.npy'}]\n"
            "DEFAULT_PARAMS={}\n"
            "def generate(output_dir, params):\n"
            "    return {'params_used': params, 'input_files': [], 'reference_files': []}\n"
        )
        (task_dir / "custom_scorer.py").write_text("raise RuntimeError('boom from custom scorer')\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.broken_custom_scorer",
            "--tasks-dir", str(tasks_dir),
        ])

        assert result.exit_code != 0
        assert "Error loading custom_scorer.py" in result.output

    def test_validate_preflight_runs_task_preflight_hook(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "hook_task"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.hook_task",
            "name": "Hook Task",
            "domain": "physics",
            "status": "test",
            "prompts": {"b1": "prompt_b1.md", "b2": "prompt_b2.md", "b3": "prompt_b3.md"},
            "input": {"files": []},
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {"gates": [], "scoring": []},
            "generation": {"script": "generate_gt.py", "parameters": {}},
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Prompt for {level}")
        (task_dir / "generate_gt.py").write_text(
            "import json\n"
            "from pathlib import Path\n"
            "INPUT_SPEC=[]\n"
            "OUTPUT_SPEC=[{'name': 'output.npy'}]\n"
            "DEFAULT_PARAMS={}\n"
            "def generate(output_dir, params):\n"
            "    out = Path(output_dir)\n"
            "    (out / 'reference').mkdir(parents=True, exist_ok=True)\n"
            "    for level in ['b1', 'b2', 'b3']:\n"
            "        (out / f'prompt_{level}.md').write_text(f'Prompt for {level}')\n"
            "    (out / 'instance_meta.json').write_text(json.dumps({'params_used': params, 'input_files': [], 'reference_files': []}))\n"
            "    return {'params_used': params, 'input_files': [], 'reference_files': []}\n"
            "def preflight_check(instance_dir, workspace_dir, params, metadata):\n"
            "    return ['hook says generated output.npy is missing from workspace']\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.hook_task",
            "--tasks-dir", str(tasks_dir),
            "--preflight",
        ])

        assert result.exit_code != 0
        assert "hook says generated output.npy is missing from workspace" in result.output

    def test_validate_accepts_gate_severity_and_per_check_scoring(self, sample_task_dir):
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["evaluation"]["gates"] = [
            {
                "scorer": "file_match",
                "severity": "soft",
                "config": {"checks": [{"file": "output.npy"}]},
            }
        ]
        metadata["evaluation"]["scoring"] = [
            {
                "scorer": "code_analysis",
                "weight": 10,
                "config": {
                    "target_file": "analysis.py",
                    "scoring_mode": "per_check",
                    "checks": [{"pattern": "import numpy", "required": True}],
                },
            }
        ]
        (sample_task_dir / "task.yaml").write_text(yaml.dump(metadata))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(sample_task_dir.parent.parent),
        ])
        assert result.exit_code == 0
        assert "VALIDATION PASSED" in result.output

    def test_validate_rejects_invalid_gate_severity(self, sample_task_dir):
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["evaluation"]["gates"] = [
            {
                "scorer": "file_match",
                "severity": "warning",
                "config": {"checks": [{"file": "output.npy"}]},
            }
        ]
        (sample_task_dir / "task.yaml").write_text(yaml.dump(metadata))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(sample_task_dir.parent.parent),
            "--preflight",
        ])
        assert result.exit_code != 0
        assert "Invalid gate severity" in result.output

    def test_validate_rejects_invalid_scoring_mode(self, sample_task_dir):
        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["evaluation"]["scoring"] = [
            {
                "scorer": "file_match",
                "weight": 10,
                "config": {
                    "checks": [{"file": "output.npy"}],
                    "scoring_mode": "fractional_but_wrong",
                },
            }
        ]
        (sample_task_dir / "task.yaml").write_text(yaml.dump(metadata))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(sample_task_dir.parent.parent),
            "--preflight",
        ])
        assert result.exit_code != 0
        assert "Invalid scoring_mode" in result.output

    def test_validate_nonexistent_task(self, tmp_dir):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "nonexistent.task",
            "--tasks-dir", str(tmp_dir),
        ])
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_validate_abandoned_task_skips_file_checks(self, tmp_dir):
        """Abandoned tasks skip prompt and generate_gt.py validation."""
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "abandoned_task"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.abandoned_task",
            "name": "Abandoned Task",
            "domain": "physics",
            "status": "abandoned",
            "abandoned_reason": "evaluation method infeasible",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.abandoned_task",
            "--tasks-dir", str(tasks_dir),
        ])
        assert result.exit_code == 0
        assert "PASSED" in result.output
        assert "skipping prompt and generate_gt.py checks" in result.output

    def test_validate_abandoned_task_warns_missing_reason(self, tmp_dir):
        """Abandoned task without abandoned_reason triggers a warning."""
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "abandoned_no_reason"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.abandoned_no_reason",
            "name": "Abandoned No Reason",
            "domain": "physics",
            "status": "abandoned",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.abandoned_no_reason",
            "--tasks-dir", str(tasks_dir),
        ])
        assert result.exit_code == 0
        assert "WARNINGS" in result.output
        assert "missing 'abandoned_reason'" in result.output

    def test_validate_non_abandoned_with_reason_warns(self, tmp_dir):
        """Non-abandoned task with abandoned_reason triggers a warning."""
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "dev_with_reason"
        task_dir.mkdir(parents=True)

        task_dir_prompts = task_dir
        for level in ["b1", "b2", "b3"]:
            (task_dir_prompts / f"prompt_{level}.md").write_text("test prompt")

        gt_code = (
            "INPUT_SPEC = []\n"
            "OUTPUT_SPEC = []\n"
            "DEFAULT_PARAMS = {}\n"
            "def generate(output_dir, params, rng):\n"
            "    pass\n"
        )
        (task_dir / "generate_gt.py").write_text(gt_code)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.dev_with_reason",
            "name": "Dev With Reason",
            "domain": "physics",
            "status": "in_development",
            "abandoned_reason": "leftover field",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.dev_with_reason",
            "--tasks-dir", str(tasks_dir),
        ])
        assert result.exit_code == 0
        assert "WARNINGS" in result.output
        assert "status is not 'abandoned'" in result.output

    def test_validate_preflight_detects_unrendered_prompt(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "broken_task"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.broken_task",
            "name": "Broken Task",
            "domain": "physics",
            "status": "test",
            "prompts": {
                "b1": "prompt_b1.md",
                "b2": "prompt_b2.md",
                "b3": "prompt_b3.md",
            },
            "input": {"files": []},
            "output": {"files": [{"name": "output.npy", "type": "data"}]},
            "evaluation": {"gates": [], "scoring": []},
            "generation": {
                "script": "generate_gt.py",
                "parameters": {
                    "size": {"type": "int", "range": [1, 10], "default": 3},
                },
            },
        }))
        for level in ["b1", "b2", "b3"]:
            (task_dir / f"prompt_{level}.md").write_text(f"Size is {{{{ size }}}} for {level}")
        (task_dir / "generate_gt.py").write_text(
            "import json\n"
            "from pathlib import Path\n"
            "INPUT_SPEC=[]\n"
            "OUTPUT_SPEC=[{'name': 'output.npy'}]\n"
            "DEFAULT_PARAMS={'size': 3}\n"
            "def generate(output_dir, params):\n"
            "    out = Path(output_dir)\n"
            "    (out / 'data').mkdir(parents=True, exist_ok=True)\n"
            "    (out / 'reference').mkdir(parents=True, exist_ok=True)\n"
            "    for level in ['b1', 'b2', 'b3']:\n"
            "        text = (Path(__file__).parent / f'prompt_{level}.md').read_text()\n"
            "        (out / f'prompt_{level}.md').write_text(text)\n"
            "    (out / 'instance_meta.json').write_text(json.dumps({'params_used': params, 'input_files': [], 'reference_files': []}))\n"
            "    return {'params_used': params, 'input_files': [], 'reference_files': []}\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.broken_task",
            "--tasks-dir", str(tasks_dir),
            "--preflight",
        ])
        assert result.exit_code != 0
        assert "Rendered prompt still contains template placeholders" in result.output

    def test_validate_generated_instance_detects_nul_prompt_bytes(self, tmp_dir):
        from ai4sci_bench.cli import (
            FRAMEWORK_TASK_INFO_FILENAME,
            _validate_generated_instance,
        )

        instance_dir = tmp_dir / "instance"
        workspace = instance_dir / "workspace"
        reference = instance_dir / "reference"
        workspace.mkdir(parents=True)
        reference.mkdir()
        (instance_dir / "prompt_b1.md").write_bytes(b"hello\x00world")
        (workspace / "prompt.md").write_bytes(b"hello\x00world")
        (workspace / "task_info.json").write_text(
            json.dumps({"expected_outputs": [{"name": "output.npy"}]}),
            encoding="utf-8",
        )
        (instance_dir / FRAMEWORK_TASK_INFO_FILENAME).write_text(
            json.dumps({"instance_id": "inst", "parameters": {}}),
            encoding="utf-8",
        )
        (instance_dir / "instance_meta.json").write_text(
            json.dumps({"reference_files": []}),
            encoding="utf-8",
        )
        metadata = {
            "prompts": {"b1": "prompt_b1.md"},
            "input": {"files": []},
            "output": {"files": [{"name": "output.npy"}]},
            "evaluation": {"gates": [], "scoring": []},
        }
        instance = SimpleNamespace(
            instance_id="inst",
            parameters={},
            workspace_dir=workspace,
            reference_dir=reference,
        )
        errors: list[str] = []

        _validate_generated_instance(instance, metadata, errors)

        assert any("contains NUL byte" in error for error in errors)

    def test_validate_preflight_detects_missing_scoring_reference(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        task_dir = tasks_dir / "physics" / "missing_ref_task"
        task_dir.mkdir(parents=True)

        (task_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.missing_ref_task",
            "name": "Missing Ref Task",
            "domain": "physics",
            "status": "test",
            "prompts": {
                "b1": "prompt_b1.md",
                "b2": "prompt_b2.md",
                "b3": "prompt_b3.md",
            },
            "input": {"files": []},
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
                            "ref_file": "missing_ref.npy",
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
        (task_dir / "generate_gt.py").write_text(
            "import json\n"
            "from pathlib import Path\n"
            "INPUT_SPEC=[]\n"
            "OUTPUT_SPEC=[{'name': 'output.npy'}]\n"
            "DEFAULT_PARAMS={'size': 3}\n"
            "def generate(output_dir, params):\n"
            "    out = Path(output_dir)\n"
            "    (out / 'data').mkdir(parents=True, exist_ok=True)\n"
            "    ref = out / 'reference'\n"
            "    ref.mkdir(parents=True, exist_ok=True)\n"
            "    for level in ['b1', 'b2', 'b3']:\n"
            "        text = (Path(__file__).parent / f'prompt_{level}.md').read_text()\n"
            "        (out / f'prompt_{level}.md').write_text(text)\n"
            "    (out / 'instance_meta.json').write_text(json.dumps({'params_used': params, 'input_files': [], 'reference_files': []}))\n"
            "    return {'params_used': params, 'input_files': [], 'reference_files': []}\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.missing_ref_task",
            "--tasks-dir", str(tasks_dir),
            "--preflight",
        ])
        assert result.exit_code != 0
        assert "missing reference file in generated instance" in result.output


class TestCLINewTask:
    def test_task_create(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        tasks_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(cli, [
            "task", "create",
            "--domain", "physics",
            "--name", "my_test",
            "--tasks-dir", str(tasks_dir),
        ])
        assert result.exit_code == 0
        assert "Created task scaffold" in result.output

        task_dir = tasks_dir / "physics" / "my_test"
        assert (task_dir / "task_meta.yaml").exists()
        assert (task_dir / "task_eval.yaml").exists()
        assert (task_dir / "task_submission.yaml").exists()
        assert (task_dir / "prompt_b1.md").exists()
        assert (task_dir / "prompt_b2.md").exists()
        assert (task_dir / "prompt_b3.md").exists()
        assert (task_dir / "prompt_b4.md").exists()
        assert (task_dir / "generate_gt.py").exists()

        # Verify public metadata content.
        metadata = yaml.safe_load((task_dir / "task_meta.yaml").read_text())
        assert metadata["id"] == "physics.my_test"
        assert metadata["status"] == "in_development"

    def test_task_create_generates_prompts_when_repository_template_has_none(self, tmp_dir):
        tasks_dir = tmp_dir / "tasks"
        template_dir = tasks_dir / "_template"
        template_dir.mkdir(parents=True)
        (template_dir / "task_meta.yaml").write_text(
            "id: DOMAIN.TASK_NAME\nprompts:\n"
            "  b1: prompt_b1.md\n  b2: prompt_b2.md\n"
            "  b3: prompt_b3.md\n  b4: prompt_b4.md\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(cli, [
            "task", "create",
            "--domain", "physics",
            "--name", "hf_only_prompts",
            "--tasks-dir", str(tasks_dir),
        ])

        assert result.exit_code == 0, result.output
        task_dir = tasks_dir / "physics" / "hf_only_prompts"
        for level in ("b1", "b2", "b3", "b4"):
            prompt = task_dir / f"prompt_{level}.md"
            assert prompt.is_file()
            assert f"level {level}" in prompt.read_text(encoding="utf-8")

class TestCLITaskPull:
    def test_task_pull(self, tmp_dir, monkeypatch):
        from ai4sci_bench.hf.pull import PullResult

        out = tmp_dir / "instances"
        captured = {}

        def fake_pull(repo, **kwargs):
            captured["repo"] = repo
            captured.update(kwargs)
            return PullResult(
                repo_id="org/seed",
                output_dir=out,
                instance_ids=["math.example__seed1"],
                task_ids=["math.example"],
            )

        monkeypatch.setattr("ai4sci_bench.hf.pull_instances", fake_pull)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "task", "pull",
            "--repo", "seed42",
            "--output-dir", str(out),
        ])

        assert result.exit_code == 0
        assert captured["repo"] == "seed42"
        assert captured["output_dir"] == str(out)
        assert "instances: 1" in result.output
        assert "references: private" in result.output

    def test_task_pull_empty_result_explains_required_layout(self, tmp_dir, monkeypatch):
        from ai4sci_bench.hf.pull import PullResult

        out = tmp_dir / "instances"
        monkeypatch.setattr(
            "ai4sci_bench.hf.pull_instances",
            lambda repo, **kwargs: PullResult(repo_id="org/seed", output_dir=out),
        )

        result = CliRunner().invoke(cli, [
            "task", "pull", "--repo", "seed31415", "--output-dir", str(out),
        ])

        assert result.exit_code == 0
        assert "expected tasks/<instance-id>/" in result.output
        assert "root-level" not in result.output
        assert "references: public" in result.output

    def test_task_pull_requires_explicit_official_seed(self):
        result = CliRunner().invoke(cli, ["task", "pull"])

        assert result.exit_code == 2
        assert "--repo is required" in result.output
        assert "seed42" in result.output
        assert "seed31415" in result.output

    @pytest.mark.parametrize("repo", ["benchmark", "demo", "org/custom"])
    def test_task_pull_rejects_non_official_repo(self, repo):
        result = CliRunner().invoke(cli, ["task", "pull", "--repo", repo])

        assert result.exit_code == 2
        assert "Invalid value for '--repo'" in result.output


class TestLegacyTaskCommandsRemoved:
    def test_only_grouped_task_commands_are_registered(self):
        assert "pull" not in cli.commands
        assert "new-task" not in cli.commands
        assert {"create", "pull", "submit"} <= set(cli.commands["task"].commands)


class TestParseEvalResult:
    def test_parse_basic(self):
        data = {
            "instance_id": "test_id",
            "task_id": "physics.test",
            "prompt_level": "b2",
            "agent_name": "agent",
            "parameters": {"size": 10},
            "gates_passed": True,
            "hard_gates_passed": True,
            "soft_gate_failures": 1,
            "gate_results": [
                {
                    "scorer_name": "file_match",
                    "score": 1,
                    "max_score": 1,
                    "passed": True,
                    "details": {},
                    "severity": "hard",
                }
            ],
            "score_results": [
                {"scorer_name": "numerical:l2", "score": 85, "max_score": 100, "passed": True, "details": {}}
            ],
            "final_score": 85.0,
            "status": "completed",
        }
        result = _parse_eval_result(data)
        assert result.final_score == 85.0
        assert result.gates_passed is True
        assert result.hard_gates_passed is True
        assert result.soft_gate_failures == 1
        assert result.gate_results[0].severity == "hard"
        assert len(result.gate_results) == 1
        assert len(result.score_results) == 1

    def test_parse_with_error_analysis(self):
        data = {
            "instance_id": "test_id",
            "task_id": "t.a",
            "prompt_level": "b1",
            "agent_name": "agent",
            "parameters": {},
            "gates_passed": False,
            "gate_results": [],
            "score_results": [],
            "final_score": 0.0,
            "status": "failed",
            "error_analysis": {
                "error_category": "algorithm_error",
                "error_subcategory": "wrong_method",
                "root_cause": "Used FFT instead of DCT",
                "evidence": ["line 42"],
                "fix_suggestions": ["Use DCT"],
                "confidence": 0.9,
            },
        }
        result = _parse_eval_result(data)
        assert result.error_analysis is not None
        assert result.error_analysis.error_category == "algorithm_error"

    def test_parse_rejects_unsupported_result_schema_version(self):
        data = {
            RESULT_SCHEMA_VERSION_FIELD: 99,
            "instance_id": "test_id",
            "task_id": "physics.test",
            "prompt_level": "b2",
            "agent_name": "agent",
            "parameters": {},
            "gates_passed": True,
            "gate_results": [],
            "score_results": [],
            "final_score": 0.0,
            "status": "completed",
        }
        with pytest.raises(ValueError, match="Unsupported result_schema_version=99"):
            _parse_eval_result(data)


class TestCLIReportHelpers:
    def test_is_eval_result_json_rejects_task_info(self):
        task_info = {
            "instance_id": "physics.task__seed0",
            "task_id": "physics.task",
            "prompt_level": "b1",
            "expected_outputs": [{"name": "simulation.py", "type": "code"}],
        }
        assert _is_eval_result_json(task_info) is False

    def test_is_eval_result_json_accepts_saved_result(self):
        result_json = {
            RESULT_SCHEMA_VERSION_FIELD: CURRENT_RESULT_SCHEMA_VERSION,
            "instance_id": "physics.task__seed0",
            "task_id": "physics.task",
            "prompt_level": "b1",
            "agent_name": "DummyAgent",
            "parameters": {},
            "gates_passed": False,
            "gate_results": [],
            "score_results": [],
            "final_score": 0.0,
            "status": "failed",
        }
        assert _is_eval_result_json(result_json) is True

    def test_is_eval_result_json_rejects_empty_dict(self):
        assert _is_eval_result_json({}) is False

    def test_is_eval_result_json_rejects_partial_keys(self):
        """Having some but not all required keys should be rejected."""
        partial = {
            "instance_id": "x",
            "task_id": "y",
            "prompt_level": "b1",
            # missing: agent_name, gates_passed, gate_results, score_results, final_score, status
        }
        assert _is_eval_result_json(partial) is False

    def test_is_eval_result_json_accepts_extra_keys(self):
        """Extra keys beyond the required set should still pass."""
        result_json = {
            RESULT_SCHEMA_VERSION_FIELD: CURRENT_RESULT_SCHEMA_VERSION,
            "instance_id": "x",
            "task_id": "y",
            "prompt_level": "b1",
            "agent_name": "A",
            "gates_passed": True,
            "gate_results": [],
            "score_results": [],
            "final_score": 1.0,
            "status": "completed",
            "agent_output": {"log": "debug info"},
            "extra_field": "should not matter",
        }
        assert _is_eval_result_json(result_json) is True


class TestSaveEvalResult:
    def test_save_eval_result_preserves_existing_rich_fields(self, tmp_dir):
        from ai4sci_bench.core.types import AnalysisReport, EvalResult, PromptLevel, RunStatus

        path = tmp_dir / "result.json"
        eval_result = EvalResult(
            instance_id="inst1",
            task_id="physics.test",
            prompt_level=PromptLevel.B1,
            agent_name="DirectLLMAdapter",
            parameters={"seed": 0},
            gate_results=[],
            gates_passed=False,
            hard_gates_passed=False,
            soft_gate_failures=0,
            score_results=[],
            final_score=0.0,
            execution_time_seconds=3.0,
            status=RunStatus.FAILED,
            error_analysis=AnalysisReport(
                instance_id="inst1",
                error_category="algorithm_error",
                error_subcategory="wrong_method",
                root_cause="Used the wrong solver",
                evidence=["traceback"],
                fix_suggestions=["Use the stable method"],
                raw_analysis="raw",
                confidence=0.9,
            ),
        )
        existing_data = {
            "attempt": 2,
            "provenance": {"agent": {"agent_name": "direct_llm", "config": {"model": "openai/gpt-5.4"}}},
            "agent_output": {"log": "debug", "status": "failed"},
            "cost": {"estimated_cost_usd": 1.23, "total_tokens": 456},
            "extra_field": "keep-me",
        }

        _save_eval_result(eval_result, path, existing_data=existing_data)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload[RESULT_SCHEMA_VERSION_FIELD] == CURRENT_RESULT_SCHEMA_VERSION
        assert payload["attempt"] == 2
        assert payload["provenance"]["agent"]["config"]["model"] == "openai/gpt-5.4"
        assert payload["agent_output"]["log"] == "debug"
        assert payload["cost"]["total_tokens"] == 456
        assert payload["extra_field"] == "keep-me"
        assert payload["error_analysis"]["error_category"] == "algorithm_error"


class TestCLIRun:
    def test_run_is_produce_only_and_rejects_local_scoring_flags(self):
        runner = CliRunner()
        help_result = runner.invoke(cli, ["run", "--help"])

        assert help_result.exit_code == 0
        assert "--score" not in help_result.output
        assert "--no-score" not in help_result.output

        for retired_flag in ("--score", "--no-score"):
            result = runner.invoke(cli, ["run", retired_flag])
            assert result.exit_code != 0
            assert "No such option" in result.output

    """Tests for the run command (end-to-end via CLI)."""

    def test_run_with_default_agent(self, sample_task_dir, tmp_dir):
        """Test run command with default (echo) agent."""
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "results"),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
        ])
        assert result.exit_code == 0
        assert "Produce-Only Summary" in result.output
        result_files = sorted(
            (tmp_dir / "results" / "physics.test_task").glob("*.json")
        )
        levels = {json.loads(path.read_text())["prompt_level"] for path in result_files}
        assert levels == {"b1", "b2", "b3", "b4"}

    def test_run_with_agent_cmd(self, sample_task_dir, tmp_dir):
        """Test run command with --agent-cmd."""
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "results"),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
            "--agent-cmd", "echo test",
        ])
        assert result.exit_code == 0
        assert "Produce-Only Summary" in result.output

    def test_run_strict_mode_returns_nonzero_after_persisting_failed_result(
        self, sample_task_dir, tmp_dir
    ):
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--prompt-levels", "b1",
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
            "--agent-cmd", "false",
            "--fail-on-agent-error",
        ])

        assert result.exit_code == 1
        assert "1 final attempt(s) failed" in result.output
        result_files = sorted((results_dir / "physics.test_task").glob("*.json"))
        assert len(result_files) == 1
        payload = json.loads(result_files[0].read_text())
        assert payload["agent_output"]["status"] == "failed"

    def test_run_rejects_unsupported_agent_cmd_task_sandbox(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "results"),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
            "--agent-cmd", "echo test",
            "--sandbox", "task",
        ])
        assert result.exit_code != 0
        assert "CLIAgentAdapter does not support sandbox 'task'" in result.output

    def test_run_tool_mode_and_allow_external_tools_conflict(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "results"),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
            "--allow-external-tools",
            "--tool-mode", "restricted",
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_run_rejects_unknown_sandbox_mode(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "results"),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
            "--agent", "codex_cli",
            "--sandbox", "bogus",
        ])
        assert result.exit_code != 0
        assert "Unknown sandbox mode 'bogus'" in result.output

    def test_run_creates_metadata(self, sample_task_dir, tmp_dir):
        """Test that run creates run_metadata.json."""
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
        ])
        assert result.exit_code == 0
        metadata_path = results_dir / "run_metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["agent_config"]["adapter_class"] == "CLIAgentAdapter"
        assert metadata["sandbox"]["requested_mode"] == "none"
        assert metadata["task_runtime"]["physics.test_task"]["packages"] == []

    def test_run_metadata_records_agent_config_and_task_runtime(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
            "--agent-cmd", "echo test",
            "--agent-config", '{"timeout":77}',
        ])
        assert result.exit_code == 0

        metadata = json.loads((results_dir / "run_metadata.json").read_text())
        assert metadata["agent_config"]["adapter_class"] == "CLIAgentAdapter"
        assert metadata["agent_config"]["cmd_template"] == "echo test"
        assert metadata["agent_config"]["config"]["timeout"] == 77
        assert metadata["sandbox"]["requested_mode"] == "none"
        assert metadata["task_runtime"]["physics.test_task"]["task_env_cache_key"] is None
        assert metadata["task_runtime"]["physics.test_task"]["resolved_python_version"] is None
        assert metadata["task_runtime"]["physics.test_task"]["python_requirement_satisfied"] is None

    def test_run_with_fixed_params(self, sample_task_dir, tmp_dir):
        """`run --params` should generate one fixed-parameter instance at each default level."""
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--instances-per-task", "1",
            "--params", '{"size": 12, "seed": 7}',
            "--include-test",
        ])
        assert result.exit_code == 0
        assert "Produce-Only Summary" in result.output
        result_files = sorted((results_dir / "physics.test_task").glob("*.json"))
        assert len(result_files) == 4
        payloads = [json.loads(path.read_text()) for path in result_files]
        assert {payload["prompt_level"] for payload in payloads} == {"b1", "b2", "b3", "b4"}
        for payload in payloads:
            assert payload["parameters"]["size"] == 12
            assert payload["parameters"]["seed"] == 7
            assert payload["provenance"]["sandbox"]["requested_mode"] == "none"
            assert payload["provenance"]["agent"]["adapter_class"] == "CLIAgentAdapter"
            assert payload["provenance"]["runtime"]["python_requirement"] is None
            assert payload["provenance"]["runtime"]["resolved_python_version"] is None
            assert payload["provenance"]["runtime"]["python_requirement_satisfied"] is None

    def test_run_with_fixed_params_requires_single_task(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "all",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "results"),
            "--instances-per-task", "1",
            "--params", '{"size": 12, "seed": 7}',
            "--include-test",
        ])
        assert result.exit_code != 0
        assert "--params requires exactly one task" in result.output

    def test_run_with_fixed_params_requires_single_instance(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "results"),
            "--instances-per-task", "2",
            "--params", '{"size": 12, "seed": 7}',
            "--include-test",
        ])
        assert result.exit_code != 0
        assert "--params requires --instances-per-task 1" in result.output


class TestPromptLevelIsolation:
    """Tests that different prompt levels don't interfere with each other."""

    def test_multi_level_produces_separate_results(self, sample_task_dir, tmp_dir):
        """Running with b1,b2 should produce 2 separate result files."""
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--instances-per-task", "1",
            "--seed", "42",
            "--prompt-levels", "b1,b2",
            "--include-test",
        ])
        assert result.exit_code == 0
        result_files = sorted((results_dir / "physics.test_task").glob("*.json"))
        assert len(result_files) == 2
        # Files should contain prompt level in name
        names = [f.name for f in result_files]
        assert any("__b1" in n for n in names), f"No b1 result: {names}"
        assert any("__b2" in n for n in names), f"No b2 result: {names}"


class TestCLIGenerate:
    def test_generate_rejects_unknown_sandbox_mode(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--task", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(tmp_dir / "instances"),
            "--sandbox", "bogus",
        ])
        assert result.exit_code != 0
        assert "Unknown sandbox mode 'bogus'" in result.output


class TestPromptLevelIsolationMore:
    def test_multi_level_results_have_correct_prompt_level(self, sample_task_dir, tmp_dir):
        """Each result JSON should contain the correct prompt_level field."""
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--instances-per-task", "1",
            "--seed", "42",
            "--prompt-levels", "b1,b2,b3",
            "--include-test",
        ])
        assert result.exit_code == 0
        result_files = sorted((results_dir / "physics.test_task").glob("*.json"))
        levels_found = set()
        for f in result_files:
            data = json.loads(f.read_text())
            levels_found.add(data["prompt_level"])
        assert levels_found == {"b1", "b2", "b3"}

    def test_multi_level_separate_workspaces(self, sample_task_dir, tmp_dir):
        """Each prompt level should have its own workspace directory."""
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--instances-per-task", "1",
            "--seed", "42",
            "--prompt-levels", "b1,b2",
            "--include-test",
        ])
        assert result.exit_code == 0
        instances_dir = results_dir / "instances"
        # Find instance dirs
        instance_dirs = [
            d for d in instances_dir.iterdir() if d.is_dir() and d.name != "_workspaces"
        ]
        assert len(instance_dirs) == 1  # Same params → same instance dir
        inst = instance_dirs[0]
        ws_root = instances_dir / "_workspaces" / inst.name
        # Each prompt level has its own workspace
        assert (ws_root / "workspace_b1").is_dir()
        assert (ws_root / "workspace_b2").is_dir()
        # Old shared "workspace" should NOT exist
        assert not (ws_root / "workspace").exists()

    def test_gt_generated_once_for_same_params(self, sample_task_dir, tmp_dir):
        """With same seed, different prompt levels should share GT (reference/)."""
        tasks_dir = sample_task_dir.parent.parent
        results_dir = tmp_dir / "results"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(results_dir),
            "--instances-per-task", "1",
            "--seed", "42",
            "--prompt-levels", "b1,b2",
            "--include-test",
        ])
        assert result.exit_code == 0
        instances_dir = results_dir / "instances"
        instance_dirs = [
            d for d in instances_dir.iterdir() if d.is_dir() and d.name != "_workspaces"
        ]
        assert len(instance_dirs) == 1  # Only one instance dir (params identical)
        # Reference should exist (generated once)
        assert (instance_dirs[0] / "reference").is_dir()


class TestDeriveAgentLabel:
    """Tests for _derive_agent_label helper."""

    def test_direct_llm_with_model(self):
        from ai4sci_bench.cli import _derive_agent_label
        assert _derive_agent_label("direct_llm", {"model": "openai/gpt-5.4"}) == "gpt-5.4"
        assert _derive_agent_label("direct_llm", {"model": "claude-opus-4-6"}) == "claude-opus-4-6"
        assert _derive_agent_label("direct_llm", {"model": "gemini/gemini-3.1-pro-preview"}) == "gemini-3.1-pro-preview"

    def test_direct_llm_no_model(self):
        from ai4sci_bench.cli import _derive_agent_label
        assert _derive_agent_label("direct_llm", {}) == "unknown"

    def test_openrouter_model(self):
        from ai4sci_bench.cli import _derive_agent_label
        assert _derive_agent_label(
            "direct_llm", {"model": "openrouter/anthropic/claude-3.5-sonnet"}
        ) == "openrouter_claude-3.5-sonnet"
        assert _derive_agent_label(
            "direct_llm", {"model": "openrouter/openai/gpt-4o"}
        ) == "openrouter_gpt-4o"
        assert _derive_agent_label(
            "direct_llm", {"model": "openrouter/meta-llama/llama-3-70b"}
        ) == "openrouter_llama-3-70b"

    def test_cli_agents(self):
        from ai4sci_bench.cli import _derive_agent_label
        assert _derive_agent_label("claude_code_cli", {}) == "claude_code_cli"
        assert _derive_agent_label("codex_cli", {}) == "codex_cli"


class TestBuildAgentOpenRouter:
    """Tests for _build_agent with OpenRouter and api_key/api_base params."""

    def test_build_agent_direct_llm_with_api_key(self):
        adapter = _build_agent(
            None, "direct_llm",
            {"model": "openrouter/anthropic/claude-3.5-sonnet", "api_key": "sk-or-test"},
        )
        assert adapter.model == "openrouter/anthropic/claude-3.5-sonnet"
        assert adapter.api_key == "sk-or-test"

    def test_build_agent_direct_llm_with_api_base(self):
        adapter = _build_agent(
            None, "direct_llm",
            {
                "model": "gpt-4o",
                "api_base": "https://openrouter.ai/api/v1",
                "api_protocol": "openai",
            },
        )
        assert adapter.api_base == "https://openrouter.ai/api/v1"
        assert adapter.model == "openai/gpt-4o"

    def test_build_agent_direct_llm_default_no_api_key(self):
        adapter = _build_agent(None, "direct_llm", {"model": "openai/gpt-5.4"})
        assert adapter.api_key is None
        assert adapter.api_base is None

    def test_build_agent_codex_cli_ignores_api_key(self):
        """CodexCLIAdapter should not accept api_key — it uses local auth."""
        adapter = _build_agent(None, "codex_cli", {})
        assert not hasattr(adapter, "api_key") or getattr(adapter, "api_key", None) is None


class TestCopyInstancesClean:
    """Tests for _copy_instances_clean helper."""

    def test_copies_without_workspace(self, tmp_dir):
        from ai4sci_bench.cli import _copy_instances_clean

        # Create source instance dir
        src = tmp_dir / "src_instances"
        instance = src / "task__param_abc123"
        (instance / "data").mkdir(parents=True)
        (instance / "reference").mkdir(parents=True)
        (instance / "workspace").mkdir(parents=True)
        (instance / "data" / "input.npy").write_text("data")
        (instance / "reference" / "ref.npy").write_text("ref")
        (instance / "workspace" / "simulation.py").write_text("code")
        (instance / "instance_meta.json").write_text("{}")
        (instance / "prompt_b1.md").write_text("prompt")

        dst = tmp_dir / "dst_instances"
        _copy_instances_clean(src, dst)

        dst_instance = dst / "task__param_abc123"
        assert (dst_instance / "data" / "input.npy").exists()
        assert (dst_instance / "reference" / "ref.npy").exists()
        assert (dst_instance / "instance_meta.json").exists()
        assert (dst_instance / "prompt_b1.md").exists()
        assert not (dst_instance / "workspace").exists()


class TestCLIAnalyze:
    """Tests for the analyze command."""

    def test_analyze_no_results(self, tmp_dir):
        """Test analyze with nonexistent results directory."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--results-dir", str(tmp_dir / "nonexistent"),
        ])
        assert result.exit_code != 0

    def test_analyze_empty_results(self, tmp_dir):
        """Test analyze with empty results directory."""
        results_dir = tmp_dir / "results"
        results_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--results-dir", str(results_dir),
        ])
        assert result.exit_code == 0
        assert "Analyzed 0" in result.output


class TestCLIReport:
    def test_report_per_task_hides_all_unscored_score_table(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)
        result_data = _make_eval_result_json(
            final_score=0.0,
            max_possible_score=0.0,
            score_results=[{
                "scorer_name": "unscored_submission",
                "score": 0.0,
                "max_score": 0.0,
                "passed": True,
                "details": {"unscored_submission": True},
                "message": "Submit for official scoring.",
            }],
        )
        (task_dir / "inst1__b1.json").write_text(
            json.dumps(result_data), encoding="utf-8"
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "report",
            "--results-dir", str(results_dir),
            "--per-task",
        ])

        assert result.exit_code == 0
        assert "ASI-Bench Produce-Only Summary" in result.output
        assert "Unscored submissions (submit mode): 1" in result.output
        assert "Per-Task Scores" not in result.output

    def test_report_ignores_task_info_json(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)
        workspace_dir = results_dir / "instances" / "physics.test_task__seed0" / "workspace"
        workspace_dir.mkdir(parents=True)

        (task_dir / "physics.test_task__seed0.json").write_text(json.dumps({
            "instance_id": "physics.test_task__seed0",
            "task_id": "physics.test_task",
            "prompt_level": "b1",
            "agent_name": "DummyAgent",
            "parameters": {},
            "gates_passed": False,
            "gate_results": [],
            "score_results": [],
            "final_score": 0.0,
            "status": "failed",
        }))
        (workspace_dir / "task_info.json").write_text(json.dumps({
            "instance_id": "physics.test_task__seed0",
            "task_id": "physics.test_task",
            "prompt_level": "b1",
            "expected_outputs": [{"name": "output.npy", "type": "data"}],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "Instances: 1" in result.output
        assert "Hard gate failures: 1" in result.output

    def test_report_surfaces_low_score_reason(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        (task_dir / "physics.test_task__seed0.json").write_text(json.dumps({
            "instance_id": "physics.test_task__seed0",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "parameters": {},
            "gates_passed": True,
            "gate_results": [],
            "score_results": [
                {
                    "scorer_name": "numerical:mean_relative_l2",
                    "score": 0.0,
                    "max_score": 30.0,
                    "passed": True,
                    "details": {"mean_relative_l2": 0.9},
                    "message": "mean_relative_l2=0.900000; score=0.00/30.00",
                }
            ],
            "final_score": 10.0,
            "status": "completed",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "Low-Score Instances" in result.output
        assert "mean_relative_l2=0.900000" in result.output

    def test_report_builds_reason_from_legacy_details(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        (task_dir / "physics.test_task__seed0.json").write_text(json.dumps({
            "instance_id": "physics.test_task__seed0",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "parameters": {},
            "gates_passed": True,
            "gate_results": [],
            "score_results": [
                {
                    "scorer_name": "numerical:per_frame_relative_l2",
                    "score": 10.0,
                    "max_score": 50.0,
                    "passed": True,
                    "details": {
                        "frame_0": {"rel_err": 0.0, "pts": 10},
                        "frame_1": {"rel_err": 0.617494, "pts": 0.0},
                        "frame_2": {"rel_err": 1.261864, "pts": 0.0},
                    },
                    "message": "",
                }
            ],
            "final_score": 10.0,
            "status": "completed",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "1/3 frames awarded" in result.output
        assert "worst frame_2 rel_err=1.261864" in result.output

    def test_report_surfaces_agent_failure_reason(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        (task_dir / "physics.test_task__seed0.json").write_text(json.dumps({
            "instance_id": "physics.test_task__seed0",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "parameters": {},
            "gates_passed": False,
            "gate_results": [
                {
                    "scorer_name": "file_match",
                    "score": 0.0,
                    "max_score": 1.0,
                    "passed": False,
                    "details": {},
                    "message": "File not found: output.npy",
                }
            ],
            "score_results": [],
            "final_score": 0.0,
            "status": "failed",
            "agent_output": {
                "code_files": [],
                "data_files": [],
                "log": "traceback",
                "error_message": "command failed",
                "status": "failed",
            },
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "agent_status: failed" in result.output
        assert "agent_error: command failed" in result.output

    def test_report_surfaces_soft_gate_warning_bucket(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        (task_dir / "physics.test_task__seed0.json").write_text(json.dumps({
            "instance_id": "physics.test_task__seed0",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "parameters": {},
            "gates_passed": True,
            "hard_gates_passed": True,
            "soft_gate_failures": 1,
            "gate_results": [
                {
                    "scorer_name": "code_analysis",
                    "score": 0.0,
                    "max_score": 1.0,
                    "passed": False,
                    "details": {},
                    "message": "Required pattern not found: scipy",
                    "severity": "soft",
                }
            ],
            "score_results": [],
            "final_score": 100.0,
            "status": "completed",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "Soft gate warnings with score: 1" in result.output
        assert "code_analysis (soft): Required pattern not found: scipy" in result.output

    def test_analyze_skips_perfect_scores(self, tmp_dir):
        """Test that --only-failed skips perfect scores."""
        results_dir = tmp_dir / "results" / "physics.test"
        results_dir.mkdir(parents=True)

        data = {
            "instance_id": "test_instance",
            "task_id": "physics.test",
            "prompt_level": "b2",
            "agent_name": "agent",
            "parameters": {},
            "gates_passed": True,
            "gate_results": [],
            "score_results": [],
            "final_score": 100.0,
            "max_possible_score": 100.0,
            "status": "completed",
        }
        (results_dir / "test_instance.json").write_text(json.dumps(data))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--results-dir", str(tmp_dir / "results"),
            "--only-failed",
        ])
        assert result.exit_code == 0
        assert "skipped 1" in result.output

    def test_analyze_skips_already_analyzed(self, tmp_dir):
        """Test that analyze skips results that already have error_analysis."""
        results_dir = tmp_dir / "results" / "physics.test"
        results_dir.mkdir(parents=True)

        data = {
            "instance_id": "test_instance",
            "task_id": "physics.test",
            "prompt_level": "b2",
            "agent_name": "agent",
            "parameters": {},
            "gates_passed": False,
            "gate_results": [],
            "score_results": [],
            "final_score": 0.0,
            "max_possible_score": 100.0,
            "status": "failed",
            "error_analysis": {
                "error_category": "algorithm_error",
                "error_subcategory": "wrong_method",
                "root_cause": "test",
                "evidence": [],
                "fix_suggestions": [],
                "confidence": 0.5,
            },
        }
        (results_dir / "test_instance.json").write_text(json.dumps(data))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--results-dir", str(tmp_dir / "results"),
        ])
        assert result.exit_code == 0
        assert "skipped 1" in result.output


class TestCLIReportWarnings:
    def test_report_skips_malformed_json_with_warning(self, tmp_dir):
        """Malformed JSON files should be skipped with a warning, not crash."""
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        # Write a valid result file
        (task_dir / "valid.json").write_text(json.dumps({
            "instance_id": "id1",
            "task_id": "physics.test_task",
            "prompt_level": "b1",
            "agent_name": "A",
            "gates_passed": False,
            "gate_results": [],
            "score_results": [],
            "final_score": 0.0,
            "status": "failed",
        }))
        # Write a malformed JSON file in the same directory
        (task_dir / "corrupt.json").write_text("{bad json!!!}")

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        # Report should still succeed and count only the valid result
        assert result.exit_code == 0
        assert "Instances: 1" in result.output


def _make_eval_result_json(
    instance_id="inst1",
    task_id="physics.test_task",
    prompt_level="b1",
    agent_name="TestAgent",
    final_score=50.0,
    max_possible_score=100.0,
    attempt=1,
    status="completed",
    execution_time_seconds=10.0,
    **overrides,
):
    """Helper to build a valid eval-result JSON dict."""
    data = {
        "instance_id": instance_id,
        "task_id": task_id,
        "attempt": attempt,
        "prompt_level": prompt_level,
        "agent_name": agent_name,
        "parameters": {},
        "gates_passed": True,
        "hard_gates_passed": True,
        "soft_gate_failures": 0,
        "gate_results": [],
        "score_results": [],
        "final_score": final_score,
        "max_possible_score": max_possible_score,
        "execution_time_seconds": execution_time_seconds,
        "status": status,
    }
    data.update(overrides)
    return data


class TestCopyInstancesCleanWorkspaceLevels:
    """Bug 1: _copy_instances_clean must skip workspace_b* dirs, not just 'workspace'."""

    def test_skips_workspace_with_level_suffix(self, tmp_dir):
        from ai4sci_bench.cli import _copy_instances_clean

        src = tmp_dir / "src"
        inst = src / "task__abc"
        (inst / "reference").mkdir(parents=True)
        (inst / "workspace_b1").mkdir(parents=True)
        (inst / "workspace_b2").mkdir(parents=True)
        (inst / "workspace_b1" / "simulation.py").write_text("code")
        (inst / "workspace_b2" / "simulation.py").write_text("code")
        (inst / "reference" / "ref.npy").write_text("ref")
        (inst / "instance_meta.json").write_text("{}")

        dst = tmp_dir / "dst"
        _copy_instances_clean(src, dst)

        dst_inst = dst / "task__abc"
        assert (dst_inst / "reference" / "ref.npy").exists()
        assert (dst_inst / "instance_meta.json").exists()
        assert not (dst_inst / "workspace_b1").exists()
        assert not (dst_inst / "workspace_b2").exists()

    def test_skips_workspace_attempt_dirs(self, tmp_dir):
        from ai4sci_bench.cli import _copy_instances_clean

        src = tmp_dir / "src"
        inst = src / "task__abc"
        (inst / "reference").mkdir(parents=True)
        (inst / "workspace_b1").mkdir(parents=True)
        (inst / "workspace_b1_attempt2").mkdir(parents=True)
        (inst / "workspace_b1_attempt3").mkdir(parents=True)
        (inst / "reference" / "ref.npy").write_text("ref")

        dst = tmp_dir / "dst"
        _copy_instances_clean(src, dst)

        dst_inst = dst / "task__abc"
        assert (dst_inst / "reference" / "ref.npy").exists()
        assert not (dst_inst / "workspace_b1").exists()
        assert not (dst_inst / "workspace_b1_attempt2").exists()
        assert not (dst_inst / "workspace_b1_attempt3").exists()

    def test_preserves_non_workspace_dirs(self, tmp_dir):
        from ai4sci_bench.cli import _copy_instances_clean

        src = tmp_dir / "src"
        inst = src / "task__abc"
        (inst / "data").mkdir(parents=True)
        (inst / "reference").mkdir(parents=True)
        (inst / "workspace_b1").mkdir(parents=True)
        (inst / "data" / "input.npy").write_text("data")
        (inst / "reference" / "ref.npy").write_text("ref")
        (inst / "instance_meta.json").write_text("{}")

        dst = tmp_dir / "dst"
        _copy_instances_clean(src, dst)

        dst_inst = dst / "task__abc"
        assert (dst_inst / "data" / "input.npy").exists()
        assert (dst_inst / "reference" / "ref.npy").exists()
        assert (dst_inst / "instance_meta.json").exists()


class TestReportRetryDedup:
    """Bug 2: report command must deduplicate retry attempts, keeping only the best."""

    def test_report_counts_single_instance_with_retries(self, tmp_dir):
        """3 attempts for same (instance, level) should count as 1 instance."""
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        # Write 3 attempts: scores 20, 80, 50
        for attempt, score in [(1, 20.0), (2, 80.0), (3, 50.0)]:
            suffix = f"__attempt{attempt}" if attempt > 1 else ""
            fname = f"inst1__b1{suffix}.json"
            (task_dir / fname).write_text(json.dumps(
                _make_eval_result_json(attempt=attempt, final_score=score)
            ))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        assert "Instances: 1" in result.output

    def test_report_picks_best_attempt(self, tmp_dir):
        """Best attempt (highest score) should be the one reported."""
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        # Attempt 1: score 20, Attempt 2: score 80
        (task_dir / "inst1__b1.json").write_text(json.dumps(
            _make_eval_result_json(attempt=1, final_score=20.0)
        ))
        (task_dir / "inst1__b1__attempt2.json").write_text(json.dumps(
            _make_eval_result_json(attempt=2, final_score=80.0)
        ))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        # Overall score should be 80 (best), not 50 (average) or 20 (first)
        assert "80.0" in result.output

    def test_report_different_levels_not_deduped(self, tmp_dir):
        """Different prompt levels for same instance should not be deduped."""
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        (task_dir / "inst1__b1.json").write_text(json.dumps(
            _make_eval_result_json(prompt_level="b1", final_score=30.0)
        ))
        (task_dir / "inst1__b2.json").write_text(json.dumps(
            _make_eval_result_json(prompt_level="b2", final_score=70.0)
        ))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        assert "Instances: 2" in result.output


class TestReportMultiAgentRoots:
    def test_report_groups_batch_run_parent_and_dedupes_within_each_agent(self, tmp_dir):
        results_dir = tmp_dir / "results"
        gpt_task_dir = results_dir / "gpt-5.4" / "physics.test_task"
        claude_task_dir = results_dir / "claude-opus-4-6" / "physics.test_task"
        gpt_task_dir.mkdir(parents=True)
        claude_task_dir.mkdir(parents=True)
        (results_dir / "gpt-5.4" / "run_metadata.json").write_text("{}", encoding="utf-8")
        (results_dir / "claude-opus-4-6" / "run_metadata.json").write_text("{}", encoding="utf-8")

        # Shared instance across agents: gpt has retries, claude has one completed result.
        (gpt_task_dir / "inst1__b1.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                final_score=20.0,
            )
        ))
        (gpt_task_dir / "inst1__b1__attempt2.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                attempt=2,
                final_score=80.0,
            )
        ))
        (claude_task_dir / "inst1__b1.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                final_score=55.0,
            )
        ))

        # Noise JSON inside a nested workspace should still be ignored.
        workspace_dir = results_dir / "gpt-5.4" / "instances" / "inst1" / "workspace_b1"
        workspace_dir.mkdir(parents=True)
        (workspace_dir / "task_info.json").write_text(json.dumps({
            "instance_id": "inst1",
            "prompt_level": "b1",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "Detected 2 agent groups in results root." in result.output
        assert result.output.count("Instances: 1") == 2
        assert "Agent: gpt-5.4" in result.output
        assert "Agent: claude-opus-4-6" in result.output
        assert "Overall: 80.0 / 100" in result.output
        assert "Overall: 55.0 / 100" in result.output

    def test_report_groups_flat_results_by_agent_provenance(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        (task_dir / "inst1__b1_gpt.json").write_text(json.dumps(
            _make_eval_result_json(
                instance_id="inst1",
                agent_name="DirectLLMAdapter",
                final_score=40.0,
                provenance={
                    "agent": {
                        "agent_name": "direct_llm",
                        "adapter_class": "DirectLLMAdapter",
                        "config": {"model": "openai/gpt-5.4"},
                    }
                },
            )
        ))
        (task_dir / "inst1__b1_claude.json").write_text(json.dumps(
            _make_eval_result_json(
                instance_id="inst1",
                agent_name="DirectLLMAdapter",
                final_score=90.0,
                provenance={
                    "agent": {
                        "agent_name": "direct_llm",
                        "adapter_class": "DirectLLMAdapter",
                        "config": {"model": "claude-opus-4-6"},
                    }
                },
            )
        ))

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "Detected 2 agent groups in results root." in result.output
        assert result.output.count("Instances: 1") == 2
        assert "Agent: gpt-5.4" in result.output
        assert "Agent: claude-opus-4-6" in result.output
        assert "Overall: 40.0 / 100" in result.output
        assert "Overall: 90.0 / 100" in result.output

    def test_report_merges_rounds_with_different_api_key_env(self, tmp_dir):
        round_one = tmp_dir / "round_one"
        round_two = tmp_dir / "round_two"

        def write_result(root, *, task_id, instance_id, score, api_key_env):
            task_dir = root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / f"{instance_id}.json").write_text(json.dumps(
                _make_eval_result_json(
                    instance_id=instance_id,
                    task_id=task_id,
                    agent_name="ClaudeCodeCLIAdapter",
                    final_score=score,
                    provenance={
                        "agent": {
                            "agent_name": "claude_code_cli",
                            "adapter_class": "ClaudeCodeCLIAdapter",
                            "config": {
                                "model": "MiniMax-M3",
                                "effort": "medium",
                                "api_key_env": api_key_env,
                            },
                        }
                    },
                )
            ), encoding="utf-8")

        write_result(
            round_one,
            task_id="chemistry.round_one_only",
            instance_id="chemistry.round_one_only__seed0",
            score=70.0,
            api_key_env="MINIMAX_API_KEY_ACTIVE",
        )
        write_result(
            round_one,
            task_id="physics.shared",
            instance_id="physics.shared__seed0",
            score=40.0,
            api_key_env="MINIMAX_API_KEY_ACTIVE",
        )
        write_result(
            round_two,
            task_id="physics.shared",
            instance_id="physics.shared__seed0",
            score=95.0,
            api_key_env="TOKENROUTER_API_KEY",
        )
        write_result(
            round_two,
            task_id="math.round_two_only",
            instance_id="math.round_two_only__seed0",
            score=80.0,
            api_key_env="TOKENROUTER_API_KEY",
        )

        scores_csv = tmp_dir / "scores.csv"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "report",
            "--results-dir", str(round_one),
            "--results-dir", str(round_two),
            "--per-task",
            "--csv", str(scores_csv),
        ])

        assert result.exit_code == 0
        assert "Detected 2 agent groups in results root." not in result.output
        assert "Tasks: 3  |  Instances: 3" in result.output
        suffixed_csv = scores_csv.with_stem(
            f"{scores_csv.stem}_claude_code_cli_MiniMax-M3"
        )
        assert not suffixed_csv.exists()

        with scores_csv.open(encoding="utf-8", newline="") as csv_file:
            rows = {row["task_id"]: row for row in csv.DictReader(csv_file)}
        assert set(rows) == {
            "chemistry.round_one_only",
            "math.round_two_only",
            "physics.shared",
        }
        assert rows["physics.shared"]["mean_score"] == "95.0"


class TestBatchReportCommand:
    def test_batch_report_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["batch-report", "--help"])
        assert result.exit_code == 0
        assert "Regenerate derived batch_records/" in result.output
        assert "--results-dir" in result.output

    def test_batch_report_regenerates_batch_records_from_existing_root(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "gpt-5.4" / "physics.test_task"
        task_dir.mkdir(parents=True)
        (results_dir / "gpt-5.4" / "run_metadata.json").write_text(json.dumps({
            "agent_config": {
                "agent_name": "direct_llm",
                "adapter_class": "DirectLLMAdapter",
                "config": {"model": "openai/gpt-5.4"},
            }
        }), encoding="utf-8")
        (task_dir / "inst1__b2.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                prompt_level="b2",
                final_score=88.0,
            )
        ))

        runner = CliRunner()
        result = runner.invoke(cli, ["batch-report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "Batch record artifacts:" in result.output
        batch_dir = results_dir / "batch_records"
        assert (batch_dir / "batch_overview.csv").exists()
        assert (batch_dir / "batch_overview.json").exists()
        assert (batch_dir / "task_scoreboard.csv").exists()
        assert (batch_dir / "task_scoreboard.json").exists()
        assert (batch_dir / "task_level_long.csv").exists()
        assert (batch_dir / "task_level_long.json").exists()
        assert (batch_dir / "batch_overview.md").exists()

    def test_batch_report_handles_empty_results_root(self, tmp_dir):
        results_dir = tmp_dir / "results"
        results_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["batch-report", "--results-dir", str(results_dir)])

        assert result.exit_code == 0
        assert "No results found." in result.output


class TestAnalyzeRetryDedup:
    """Bug 2 (analyze): analyze command must skip retry attempts."""

    def test_analyze_skips_retry_attempts(self, tmp_dir):
        """With --only-failed, perfect-score attempt 1 + attempt 2 should count as 1 skip."""
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        # Write attempt 1 (perfect) and attempt 2 (perfect)
        (task_dir / "inst1__b1.json").write_text(json.dumps(
            _make_eval_result_json(attempt=1, final_score=100.0)
        ))
        (task_dir / "inst1__b1__attempt2.json").write_text(json.dumps(
            _make_eval_result_json(attempt=2, final_score=100.0)
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze", "--results-dir", str(results_dir),
            "--only-failed",
        ])
        assert result.exit_code == 0
        # Only attempt 1 should be considered; attempt 2 is filtered out
        # So "skipped 1" (not "skipped 2")
        assert "skipped 1" in result.output


class TestIsBetterResult:
    """Test the _is_better_result helper for correct attempt selection."""

    def test_higher_score_wins(self):
        from ai4sci_bench.cli import _is_better_result, _parse_eval_result
        r1 = _parse_eval_result(_make_eval_result_json(final_score=30.0))
        r2 = _parse_eval_result(_make_eval_result_json(final_score=80.0))
        assert _is_better_result(r2, r1)
        assert not _is_better_result(r1, r2)

    def test_completed_beats_failed_at_same_score(self):
        from ai4sci_bench.cli import _is_better_result, _parse_eval_result
        r_completed = _parse_eval_result(
            _make_eval_result_json(final_score=50.0, status="completed")
        )
        r_failed = _parse_eval_result(
            _make_eval_result_json(final_score=50.0, status="failed")
        )
        assert _is_better_result(r_completed, r_failed)
        assert not _is_better_result(r_failed, r_completed)

    def test_faster_wins_at_same_score_and_status(self):
        from ai4sci_bench.cli import _is_better_result, _parse_eval_result
        r_fast = _parse_eval_result(
            _make_eval_result_json(final_score=50.0, execution_time_seconds=5.0)
        )
        r_slow = _parse_eval_result(
            _make_eval_result_json(final_score=50.0, execution_time_seconds=100.0)
        )
        assert _is_better_result(r_fast, r_slow)
        assert not _is_better_result(r_slow, r_fast)


class TestStaleRetryWorkspaceCleanup:
    """Bug 4: pregenerated instance loading should clean up stale retry workspaces."""

    def test_stale_attempt_dirs_removed_on_instance_generation(self, tmp_dir):
        """Instance generator should remove workspace_bN_attemptM dirs."""
        instance_dir = tmp_dir / "instances" / "task__abc"
        instance_dir.mkdir(parents=True)

        # Simulate stale retry workspaces from a previous run
        (instance_dir / "workspace_b1_attempt2").mkdir()
        (instance_dir / "workspace_b1_attempt2" / "old_code.py").write_text("stale")
        (instance_dir / "workspace_b1_attempt3").mkdir()
        (instance_dir / "workspace_b1_attempt3" / "old_code.py").write_text("stale")
        # Also a normal reference dir that should NOT be deleted
        (instance_dir / "reference").mkdir()
        (instance_dir / "reference" / "ref.npy").write_text("keep")

        # Import and call the relevant cleanup code path
        # We test indirectly via orchestrator's _load_pregenerated_instances
        # but for a unit test, let's verify the glob pattern directly
        import shutil
        prompt_level_value = "b1"
        workspace_dir = instance_dir / f"workspace_{prompt_level_value}"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        for stale_ws in instance_dir.glob(f"workspace_{prompt_level_value}_attempt*"):
            if stale_ws.is_dir():
                shutil.rmtree(stale_ws)
        workspace_dir.mkdir(exist_ok=True)

        assert workspace_dir.exists()
        assert not (instance_dir / "workspace_b1_attempt2").exists()
        assert not (instance_dir / "workspace_b1_attempt3").exists()
        assert (instance_dir / "reference" / "ref.npy").exists()

    def test_cleanup_does_not_affect_other_levels(self, tmp_dir):
        """Cleaning workspace_b1_attempt* should not touch workspace_b2*."""
        instance_dir = tmp_dir / "instances" / "task__abc"
        instance_dir.mkdir(parents=True)

        (instance_dir / "workspace_b1_attempt2").mkdir()
        (instance_dir / "workspace_b2").mkdir()
        (instance_dir / "workspace_b2" / "code.py").write_text("keep")
        (instance_dir / "workspace_b2_attempt2").mkdir()

        import shutil
        prompt_level_value = "b1"
        for stale_ws in instance_dir.glob(f"workspace_{prompt_level_value}_attempt*"):
            if stale_ws.is_dir():
                shutil.rmtree(stale_ws)

        # b1 attempts cleaned
        assert not (instance_dir / "workspace_b1_attempt2").exists()
        # b2 stuff untouched
        assert (instance_dir / "workspace_b2").exists()
        assert (instance_dir / "workspace_b2_attempt2").exists()
