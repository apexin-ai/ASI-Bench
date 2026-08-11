"""Tests for the trajectory system: schema, extractors, file history, validator."""

from __future__ import annotations

import json
import textwrap

import pytest

from ai4sci_bench.core.trajectory import (
    VALID_STEP_TYPES,
    Trajectory,
    TrajectoryStep,
    TrajectorySummary,
)
from ai4sci_bench.trajectory.claude_extractor import extract_from_jsonl as claude_extract
from ai4sci_bench.trajectory.codex_extractor import extract_from_jsonl as codex_extract
from ai4sci_bench.trajectory.direct_llm_extractor import extract_from_structured_log
from ai4sci_bench.trajectory.generic_extractor import extract_from_raw
from ai4sci_bench.trajectory.file_history import extract_file_versions, FileVersion
from ai4sci_bench.trajectory.validator import validate_trajectory


# ── TODO-3: Unified Trajectory Schema tests ──────────────────────────────


class TestTrajectoryStep:
    def test_trajectory_step_dataclass(self):
        step = TrajectoryStep(
            step_index=0,
            timestamp_ms=1000,
            step_type="llm_response",
            content="Hello world",
            metadata={"role": "assistant"},
        )
        d = step.to_dict()
        assert d["step_index"] == 0
        assert d["timestamp_ms"] == 1000
        assert d["step_type"] == "llm_response"
        assert d["content"] == "Hello world"
        assert d["metadata"]["role"] == "assistant"

    def test_trajectory_step_roundtrip(self):
        step = TrajectoryStep(
            step_index=5,
            timestamp_ms=None,
            step_type="tool_call",
            content="Write",
            metadata={"tool_name": "Write", "key_args": {"file_path": "sim.py"}},
        )
        d = step.to_dict()
        restored = TrajectoryStep.from_dict(d)
        assert restored.step_index == step.step_index
        assert restored.step_type == step.step_type
        assert restored.content == step.content
        assert restored.metadata == step.metadata


class TestTrajectorySummary:
    def test_trajectory_summary_computation(self):
        steps = [
            TrajectoryStep(0, None, "llm_response", "text", {"role": "assistant"}),
            TrajectoryStep(1, None, "tool_call", "Write", {"tool_name": "Write", "key_args": {"file_path": "a.py"}}),
            TrajectoryStep(2, None, "tool_result", "ok", {"is_error": False}),
            TrajectoryStep(3, None, "thinking", "x" * 500, {}),
            TrajectoryStep(4, None, "llm_response", "more text", {"role": "assistant"}),
            TrajectoryStep(5, None, "tool_call", "Bash", {"tool_name": "Bash", "key_args": {"command": "python"}}),
            TrajectoryStep(6, None, "thinking", "y" * 800, {}),
            TrajectoryStep(7, None, "tool_call", "Write", {"tool_name": "Write", "key_args": {"file_path": "b.py"}}),
        ]
        summary = TrajectorySummary.from_steps(steps)
        assert summary.total_turns == 2
        assert summary.total_tool_calls == 3
        assert summary.tool_call_distribution == {"Write": 2, "Bash": 1}
        assert summary.thinking_total_chars == 1300
        assert sorted(summary.unique_files_modified) == ["a.py", "b.py"]


class TestTrajectoryRoundtrip:
    def test_trajectory_json_roundtrip(self):
        steps = [
            TrajectoryStep(0, 1000, "llm_response", "hello", {"role": "assistant"}),
            TrajectoryStep(1, 2000, "tool_call", "Read", {"tool_name": "Read"}),
        ]
        summary = TrajectorySummary.from_steps(steps)
        traj = Trajectory(
            instance_id="test_id",
            adapter_type="claude_code_cli",
            total_steps=2,
            total_duration_ms=1000,
            steps=steps,
            summary=summary,
        )
        json_str = traj.to_json()
        restored = Trajectory.from_json(json_str)
        assert restored.instance_id == traj.instance_id
        assert restored.adapter_type == traj.adapter_type
        assert restored.total_steps == traj.total_steps
        assert restored.total_duration_ms == traj.total_duration_ms
        assert len(restored.steps) == len(traj.steps)
        assert restored.steps[0].content == "hello"
        assert restored.summary.total_turns == summary.total_turns


# ── TODO-3: Claude Extractor tests ──────────────────────────────────────


class TestClaudeExtractor:
    def test_claude_extractor_basic(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "I'll help."},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}, "id": "tc1"},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc1", "content": "file content"},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "Let me analyze..."},
                {"type": "text", "text": "Done."},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl, "inst1")
        assert traj.instance_id == "inst1"
        assert traj.adapter_type == "claude_code_cli"
        assert traj.total_steps == 5  # text + tool_use + tool_result + thinking + text

        step_types = [s.step_type for s in traj.steps]
        assert "llm_response" in step_types
        assert "tool_call" in step_types
        assert "tool_result" in step_types
        assert "thinking" in step_types

    def test_claude_extractor_preserves_full_content(self):
        long_thinking = "x" * 10000
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": long_thinking},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl)
        thinking_step = [s for s in traj.steps if s.step_type == "thinking"][0]
        assert len(thinking_step.content) == 10000

    def test_claude_extractor_handles_malformed_lines(self):
        jsonl = "not json\n" + json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}})
        traj = claude_extract(jsonl)
        assert traj.total_steps == 2
        error_steps = [s for s in traj.steps if s.step_type == "error"]
        assert len(error_steps) == 1

    def test_claude_extractor_maps_event_types(self):
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "X", "input": {}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
            {"type": "system", "message": "init"},
            {"type": "error", "message": "bad"},
            {"type": "rate_limit", "message": "slow"},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl)
        types = [s.step_type for s in traj.steps]
        assert "llm_response" in types  # assistant text
        assert "tool_call" in types     # tool_use
        assert "tool_result" in types   # user tool_result
        assert "system" in types
        assert "error" in types
        assert "rate_limit" in types

    def test_claude_extractor_counts_tool_only_assistant_turns(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tc1", "name": "Bash", "input": {"command": "pwd"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc1", "content": "/tmp"},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tc2", "name": "Bash", "input": {"command": "ls"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc2", "content": "analysis.py"},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "done"},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl)
        assert traj.summary.total_turns == 3
        assert traj.summary.total_tool_calls == 2


# ── TODO-3: Codex Extractor tests ───────────────────────────────────────


class TestCodexExtractor:
    def test_codex_extractor_basic(self):
        events = [
            {"type": "message", "role": "assistant", "content": "I'll help."},
            {"type": "function_call", "name": "shell", "arguments": json.dumps({"command": "ls"})},
            {"type": "function_call_output", "output": "file1.py\nfile2.py"},
            {"type": "reasoning", "text": "Let me think about this..."},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = codex_extract(jsonl, "inst2")
        assert traj.adapter_type == "codex_cli"
        assert traj.total_steps == 4

        step_types = [s.step_type for s in traj.steps]
        assert "llm_response" in step_types
        assert "tool_call" in step_types
        assert "tool_result" in step_types
        assert "thinking" in step_types

    def test_codex_extractor_handles_bytes_content(self):
        event = {"type": "message", "role": "assistant", "content": "hello bytes"}
        jsonl = json.dumps(event)
        traj = codex_extract(jsonl)
        assert traj.steps[0].content == "hello bytes"

    def test_codex_extractor_handles_item_events(self):
        events = [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_0",
                    "type": "agent_message",
                    "text": "I will inspect the data.",
                },
            },
            {
                "type": "item.started",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "python analysis.py",
                    "status": "in_progress",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "python analysis.py",
                    "aggregated_output": "ok",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "file_change",
                    "changes": [{"path": "analysis.py", "kind": "add"}],
                    "status": "completed",
                },
            },
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)

        traj = codex_extract(jsonl)

        assert [s.step_type for s in traj.steps] == [
            "llm_response",
            "tool_call",
            "tool_result",
            "tool_call",
        ]
        assert traj.summary.total_turns == 1
        assert traj.summary.total_tool_calls == 2
        assert traj.summary.total_code_executions == 1
        assert traj.summary.unique_files_modified == ["analysis.py"]


# ── TODO-3: Direct LLM Extractor tests ─────────────────────────────────


class TestDirectLLMExtractor:
    def test_direct_llm_extractor(self):
        structured_log = [
            {"step": "prompt", "system_prompt": "You are an expert.", "user_prompt": "Write code.", "model": "gpt-4"},
            {"step": "llm_response", "content_length": 500, "usage": {"input_tokens": 100, "output_tokens": 200}},
            {"step": "code_extraction", "num_candidates": 2, "candidate_scores": [3, 7], "selected_index": 1, "selected_code_length": 200},
            {"step": "code_execution", "code_file": "sim.py", "exit_code": 0, "duration_ms": 5000},
        ]
        log_json = json.dumps(structured_log)
        traj = extract_from_structured_log(log_json, "inst3")
        assert traj.adapter_type == "direct_llm"
        assert traj.total_steps == 4

        step_types = [s.step_type for s in traj.steps]
        assert step_types == ["llm_request", "llm_response", "system", "code_execution"]


# ── TODO-3: Generic Extractor tests ────────────────────────────────────


class TestGenericExtractor:
    def test_generic_extractor_wraps_raw_output(self):
        raw = "Some raw agent output text\nLine 2"
        traj = extract_from_raw(raw, "inst4", "cli_agent")
        assert traj.total_steps == 1
        assert traj.steps[0].step_type == "llm_response"
        assert traj.steps[0].content == raw
        assert traj.adapter_type == "cli_agent"


# ── TODO-3: Trajectory sidecar tests ───────────────────────────────────


class TestTrajectorySidecar:
    def test_trajectory_sidecar_path_sanitized(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Working in /home/user/workspace/code.py"},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl, "test")
        d = traj.to_dict()
        json_str = json.dumps(d)
        assert "/home/user/workspace/code.py" in json_str


# ── TODO-6: File History tests ──────────────────────────────────────────


class TestFileHistory:
    def test_file_history_write_then_edit(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "sim.py", "content": "print('v1')"}},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "sim.py", "old_string": "v1", "new_string": "v2"}},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        versions = extract_file_versions(jsonl)
        assert "sim.py" in versions
        assert len(versions["sim.py"]) == 2
        assert versions["sim.py"][0]["version"] == 1
        assert versions["sim.py"][0]["action"] == "write"
        assert versions["sim.py"][1]["version"] == 2
        assert versions["sim.py"][1]["action"] == "edit"

    def test_file_history_multiple_files(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "a.py", "content": "code_a"}},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "b.py", "content": "code_b"}},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py", "old_string": "old", "new_string": "new"}},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        versions = extract_file_versions(jsonl)
        assert len(versions["a.py"]) == 2
        assert len(versions["b.py"]) == 1

    def test_file_history_content_hash_differs(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "x.py", "content": "content_A"}},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "x.py", "content": "content_B"}},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        versions = extract_file_versions(jsonl)
        assert versions["x.py"][0]["content_hash"] != versions["x.py"][1]["content_hash"]

    def test_file_history_content_hash_stable(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write", "input": {"file_path": "x.py", "content": "same_content"}},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        v1 = extract_file_versions(jsonl)
        v2 = extract_file_versions(jsonl)
        assert v1["x.py"][0]["content_hash"] == v2["x.py"][0]["content_hash"]

    def test_file_history_edit_diff_summary(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "sim.py", "old_string": "dt=0.01", "new_string": "dt=0.001"}},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        versions = extract_file_versions(jsonl)
        assert versions["sim.py"][0]["diff_summary"] is not None
        assert "dt=0.01" in versions["sim.py"][0]["diff_summary"]

    def test_file_history_ignores_non_write_tools(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                {"type": "tool_use", "name": "Glob", "input": {"pattern": "*.py"}},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        versions = extract_file_versions(jsonl)
        assert versions == {}

    def test_file_history_empty_jsonl(self):
        versions = extract_file_versions("")
        assert versions == {}

    def test_file_history_malformed_tool_use(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write"},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        versions = extract_file_versions(jsonl)
        assert versions == {}

    def test_file_history_codex_item_completed_file_change(self):
        events = [
            {
                "type": "item.completed",
                "timestamp": 1700000000.5,
                "item": {
                    "id": "item_1",
                    "type": "file_change",
                    "changes": [
                        {"path": "simulation.py", "kind": "add"},
                        {"path": "analysis.py", "kind": "modify", "diff": "@@ -1 +1 @@"},
                    ],
                    "status": "completed",
                },
            },
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)

        versions = extract_file_versions(jsonl)

        assert set(versions) == {"simulation.py", "analysis.py"}
        assert versions["simulation.py"][0]["action"] == "write"
        assert versions["simulation.py"][0]["timestamp_ms"] == 1700000000500
        assert versions["simulation.py"][0]["diff_summary"] == "add: simulation.py"
        assert versions["analysis.py"][0]["action"] == "edit"
        assert versions["analysis.py"][0]["content_length"] == len("@@ -1 +1 @@")

    def test_codex_item_schema_reproduction_has_summary_and_file_versions(self):
        sample = "\n".join([
            json.dumps({
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "agent_message",
                    "text": "planning",
                },
            }),
            json.dumps({
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "command_execution",
                    "command": "python -V",
                    "aggregated_output": "Python 3.12",
                    "exit_code": 0,
                    "status": "completed",
                },
            }),
            json.dumps({
                "type": "item.completed",
                "item": {
                    "id": "item_3",
                    "type": "file_change",
                    "changes": [{"path": "simulation.py", "kind": "add"}],
                    "status": "completed",
                },
            }),
        ])

        traj = codex_extract(sample, "demo")
        versions = extract_file_versions(sample)

        assert traj.total_steps == 3
        assert traj.summary.total_turns == 1
        assert traj.summary.total_code_executions == 1
        assert traj.summary.unique_files_modified == ["simulation.py"]
        assert "simulation.py" in versions


# ── TODO-9: Validator tests ─────────────────────────────────────────────


class TestTrajectoryValidator:
    def test_validate_trajectory_valid(self):
        data = [
            {"step_type": "llm_response", "content": "hi", "timestamp_ms": 100},
            {"step_type": "tool_call", "content": "Read", "timestamp_ms": 200},
            {"step_type": "tool_result", "content": "ok", "timestamp_ms": 300},
        ]
        issues = validate_trajectory(data)
        assert len(issues) == 0

    def test_validate_trajectory_missing_step_type(self):
        data = [{"content": "hi"}]
        issues = validate_trajectory(data)
        assert any("missing step_type" in i for i in issues)

    def test_validate_trajectory_invalid_step_type(self):
        data = [{"step_type": "bogus_type"}]
        issues = validate_trajectory(data)
        assert any("invalid step_type" in i for i in issues)

    def test_validate_trajectory_non_monotonic_timestamp(self):
        data = [
            {"step_type": "llm_response", "timestamp_ms": 300},
            {"step_type": "tool_call", "timestamp_ms": 100},
        ]
        issues = validate_trajectory(data)
        assert any("non-monotonic" in i for i in issues)


# ── Regression tests for discovered bugs ────────────────────────────────


class TestBugfixCodeExecutionCounting:
    """Bug: TrajectorySummary.from_steps() had duplicate elif, broken code exec counting."""

    def test_bash_error_counted_as_failure_and_execution(self):
        steps = [
            TrajectoryStep(0, None, "tool_call", "Bash", {"tool_name": "Bash"}),
            TrajectoryStep(1, None, "tool_result", "error output", {
                "is_error": True,
                "parent_tool_name": "Bash",
            }),
        ]
        summary = TrajectorySummary.from_steps(steps)
        assert summary.total_code_executions == 1
        assert summary.code_execution_failures == 1

    def test_bash_success_counted_as_execution_not_failure(self):
        steps = [
            TrajectoryStep(0, None, "tool_call", "Bash", {"tool_name": "Bash"}),
            TrajectoryStep(1, None, "tool_result", "ok", {
                "is_error": False,
                "parent_tool_name": "Bash",
            }),
        ]
        summary = TrajectorySummary.from_steps(steps)
        assert summary.total_code_executions == 1
        assert summary.code_execution_failures == 0

    def test_non_bash_tool_result_not_counted_as_code_execution(self):
        steps = [
            TrajectoryStep(0, None, "tool_call", "Read", {"tool_name": "Read"}),
            TrajectoryStep(1, None, "tool_result", "file content", {
                "is_error": False,
                "parent_tool_name": "Read",
            }),
        ]
        summary = TrajectorySummary.from_steps(steps)
        assert summary.total_code_executions == 0
        assert summary.code_execution_failures == 0

    def test_mixed_bash_success_and_failure(self):
        steps = [
            TrajectoryStep(0, None, "tool_call", "Bash", {"tool_name": "Bash"}),
            TrajectoryStep(1, None, "tool_result", "ok", {"is_error": False, "parent_tool_name": "Bash"}),
            TrajectoryStep(2, None, "tool_call", "Bash", {"tool_name": "Bash"}),
            TrajectoryStep(3, None, "tool_result", "err", {"is_error": True, "parent_tool_name": "Bash"}),
            TrajectoryStep(4, None, "tool_call", "Bash", {"tool_name": "Bash"}),
            TrajectoryStep(5, None, "tool_result", "ok", {"is_error": False, "parent_tool_name": "Bash"}),
        ]
        summary = TrajectorySummary.from_steps(steps)
        assert summary.total_code_executions == 3
        assert summary.code_execution_failures == 1


class TestBugfixParentToolNameInExtractors:
    """Bug: Extractors didn't set parent_tool_name, so code exec counting was broken."""

    def test_claude_extractor_sets_parent_tool_name(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "id": "tc1", "input": {"command": "python sim.py"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc1", "content": "ImportError", "is_error": True},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl, "test")
        tool_results = [s for s in traj.steps if s.step_type == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0].metadata["parent_tool_name"] == "Bash"

    def test_claude_extractor_code_execution_failures_counted(self):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "id": "tc1", "input": {"command": "python sim.py"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc1", "content": "Error", "is_error": True},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "id": "tc2", "input": {"command": "python sim.py"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc2", "content": "Success"},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl, "test")
        assert traj.summary.total_code_executions == 2
        assert traj.summary.code_execution_failures == 1

    def test_codex_extractor_sets_parent_tool_name_flat(self):
        events = [
            {"type": "function_call", "name": "Bash", "arguments": json.dumps({"command": "ls"})},
            {"type": "function_call_output", "output": "file.py", "is_error": False},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = codex_extract(jsonl, "test")
        tool_results = [s for s in traj.steps if s.step_type == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0].metadata["parent_tool_name"] == "Bash"

    def test_codex_extractor_code_execution_failures_counted(self):
        events = [
            {"type": "function_call", "name": "Bash"},
            {"type": "function_call_output", "output": "Error", "is_error": True},
            {"type": "function_call", "name": "Bash"},
            {"type": "function_call_output", "output": "OK"},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = codex_extract(jsonl, "test")
        assert traj.summary.total_code_executions == 2
        assert traj.summary.code_execution_failures == 1


class TestBugfixTimestampErrorHandling:
    """Bug: Non-numeric timestamp crashed with ValueError."""

    def test_claude_extractor_non_numeric_timestamp(self):
        events = [
            {"type": "assistant", "timestamp": "not_a_number", "message": {"content": [
                {"type": "text", "text": "hello"},
            ]}},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = claude_extract(jsonl, "test")
        assert traj.total_steps == 1
        assert traj.steps[0].timestamp_ms is None

    def test_codex_extractor_non_numeric_timestamp(self):
        events = [
            {"type": "message", "role": "assistant", "content": "hello", "timestamp": "bad"},
        ]
        jsonl = "\n".join(json.dumps(e) for e in events)
        traj = codex_extract(jsonl, "test")
        assert traj.total_steps == 1
        assert traj.steps[0].timestamp_ms is None

    def test_claude_adapter_non_numeric_timestamp_no_crash(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({
            "type": "assistant",
            "timestamp": "not_a_number",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        })
        result = adapter._parse_claude_log(event)
        assert "=== Turn 1 ===" in result
        assert "[+" not in result

    def test_codex_adapter_non_numeric_timestamp_no_crash(self):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        adapter = CodexCLIAdapter()
        event = json.dumps({
            "type": "message",
            "role": "assistant",
            "content": "hello",
            "timestamp": "bad_ts",
        })
        result = adapter._parse_codex_log(event)
        assert "=== Turn 1 ===" in result
        assert "[+" not in result


class TestBugfixClaudeUsageEarlyReturn:
    """Bug: _extract_usage_from_jsonl returned None on first result without usage."""

    def test_claude_usage_continues_past_result_without_usage(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        lines = [
            json.dumps({"type": "result", "subtype": "success"}),
            json.dumps({"type": "result", "usage": {"input_tokens": 5000, "output_tokens": 1000}}),
        ]
        cost = adapter._extract_usage_from_jsonl("\n".join(lines))
        assert cost is not None
        assert cost.input_tokens == 5000

    def test_claude_usage_returns_none_when_no_result_has_usage(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        lines = [
            json.dumps({"type": "result", "subtype": "success"}),
            json.dumps({"type": "assistant", "message": {"content": []}}),
        ]
        cost = adapter._extract_usage_from_jsonl("\n".join(lines))
        assert cost is None
