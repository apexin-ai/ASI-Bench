"""Tests for pi / opencode trajectory extractors and JSONL schema detection."""

from __future__ import annotations

import json

from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator
from ai4sci_bench.trajectory.opencode_extractor import extract_from_jsonl as oc_extract
from ai4sci_bench.trajectory.pi_extractor import extract_from_jsonl as pi_extract


def _detect(raw: str) -> str:
    return BenchmarkOrchestrator._detect_jsonl_trajectory_schema(raw)


# ── Fixtures ──────────────────────────────────────────────────

PI_FIXTURE = "\n".join([
    json.dumps({"type": "session", "version": 3, "id": "ses-1",
                "timestamp": "2026-08-27T15:28:40.139Z", "cwd": "/w"}),
    json.dumps({"type": "agent_start"}),
    json.dumps({"type": "turn_start"}),
    json.dumps({"type": "tool_execution_start", "toolCallId": "tc-1",
                "toolName": "bash", "args": {"command": "python solve.py"}}),
    json.dumps({"type": "tool_execution_end", "toolCallId": "tc-1",
                "toolName": "bash", "result": {"output": "ok"}, "isError": False}),
    json.dumps({"type": "tool_execution_start", "toolCallId": "tc-2",
                "toolName": "write",
                "args": {"path": "/workspace/out/result.json", "content": "{}"}}),
    json.dumps({"type": "tool_execution_end", "toolCallId": "tc-2",
                "toolName": "write", "result": {"output": "written"},
                "isError": False}),
    json.dumps({"type": "message_end", "message": {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "deep thought"},
            {"type": "text", "text": "Solved."},
            {"type": "toolCall", "id": "tc-1", "name": "bash", "arguments": {}},
        ],
        "usage": {"input": 100, "output": 40, "cacheRead": 0, "cacheWrite": 0,
                  "totalTokens": 140},
        "stopReason": "stop",
    }}),
    json.dumps({"type": "agent_end", "messages": []}),
])

OPENCODE_FIXTURE = "\n".join([
    json.dumps({"type": "step_start", "timestamp": 1787844581500, "sessionID": "ses-1",
                "part": {"id": "p1", "type": "step-start"}}),
    json.dumps({"type": "tool_use", "timestamp": 1787844581600, "sessionID": "ses-1",
                "part": {"id": "p2", "type": "tool", "callID": "c1", "tool": "write",
                         "state": {"status": "completed",
                                   "input": {"filePath": "/workspace/out/ans.txt"},
                                   "output": "ok"}}}),
    json.dumps({"type": "tool_use", "timestamp": 1787844581650, "sessionID": "ses-1",
                "part": {"id": "p3", "type": "tool", "callID": "c2", "tool": "bash",
                         "state": {"status": "error",
                                   "input": {"command": "make"},
                                   "error": "No Makefile"}}}),
    json.dumps({"type": "reasoning", "timestamp": 1787844581700, "sessionID": "ses-1",
                "part": {"id": "p4", "type": "reasoning", "text": "hmm"}}),
    json.dumps({"type": "text", "timestamp": 1787844581800, "sessionID": "ses-1",
                "part": {"id": "p5", "type": "text", "text": "Final answer"}}),
    json.dumps({"type": "step_finish", "timestamp": 1787844581900, "sessionID": "ses-1",
                "part": {"id": "p6", "type": "step-finish",
                         "tokens": {"input": 90, "output": 30, "reasoning": 10,
                                    "cache": {"read": 0, "write": 0}},
                         "cost": 0.002}}),
])


# ── Schema detection ──────────────────────────────────────────

def test_schema_detection_pi():
    assert _detect(PI_FIXTURE) == "pi"


def test_schema_detection_opencode():
    assert _detect(OPENCODE_FIXTURE) == "opencode"
    assert _detect('{"type":"tool_use","part":{}}') == "opencode"
    assert _detect('{"type":"text","part":{}}') == "opencode"


def test_schema_detection_existing_schemas_unchanged():
    assert _detect('{"type":"assistant","message":{}}') == "claude"
    assert _detect('{"type":"thread.started"}') == "codex"
    assert _detect('{"type":"message","role":"assistant"}') == "codex"
    assert _detect('{"type":"tool_use"}') == "codex"  # no `part` → codex
    assert _detect("garbage") == "claude"


# ── pi extractor ──────────────────────────────────────────────

def test_pi_extractor_steps_and_summary():
    trajectory = pi_extract(PI_FIXTURE, instance_id="inst-1")
    assert trajectory.adapter_type == "pi"
    assert trajectory.instance_id == "inst-1"
    types = [s.step_type for s in trajectory.steps]
    assert types[0] == "system"          # session header
    assert any(s.content == "agent_start" and s.step_type == "system"
               for s in trajectory.steps)
    assert types.count("tool_call") == 2
    assert types.count("tool_result") == 2
    assert types.count("thinking") == 1
    assert types.count("llm_response") == 1

    summary = trajectory.summary
    assert summary.total_turns == 1
    assert summary.total_tool_calls == 2
    assert summary.tool_call_distribution == {"bash": 1, "write": 1}
    assert summary.thinking_total_chars == len("deep thought")
    assert summary.total_code_executions == 1  # bash tool result
    assert summary.code_execution_failures == 0
    assert "/workspace/out/result.json" in summary.unique_files_modified


def test_pi_extractor_error_message_becomes_error_step():
    fixture = json.dumps({"type": "message_end", "message": {
        "role": "assistant", "content": [], "stopReason": "error",
        "errorMessage": "boom"}})
    trajectory = pi_extract(fixture)
    error_steps = [s for s in trajectory.steps if s.step_type == "error"]
    assert len(error_steps) == 1
    assert error_steps[0].content == "boom"


def test_pi_extractor_malformed_json_tolerated():
    trajectory = pi_extract("not json\n" + PI_FIXTURE)
    parse_errors = [s for s in trajectory.steps if s.metadata.get("parse_error")]
    assert len(parse_errors) == 1


# ── opencode extractor ────────────────────────────────────────

def test_opencode_extractor_steps_and_summary():
    trajectory = oc_extract(OPENCODE_FIXTURE, instance_id="inst-2")
    assert trajectory.adapter_type == "opencode"
    assert trajectory.instance_id == "inst-2"
    types = [s.step_type for s in trajectory.steps]
    assert types.count("tool_call") == 2
    assert types.count("thinking") == 1
    assert types.count("llm_response") == 1

    summary = trajectory.summary
    assert summary.total_turns == 1
    assert summary.total_tool_calls == 2
    assert summary.tool_call_distribution == {"write": 1, "bash": 1}
    assert "/workspace/out/ans.txt" in summary.unique_files_modified
    # step_finish carries no trajectory steps
    assert "step_finish" not in types


def test_opencode_extractor_error_tool_marks_is_error():
    trajectory = oc_extract(OPENCODE_FIXTURE)
    bash_steps = [s for s in trajectory.steps
                  if s.metadata.get("tool_name") == "bash"]
    assert len(bash_steps) == 1
    assert bash_steps[0].metadata.get("is_error") is True
    assert "No Makefile" in bash_steps[0].content


def test_opencode_extractor_error_event_becomes_error_step():
    fixture = json.dumps({
        "type": "error", "timestamp": 1787844567027, "sessionID": "ses-2",
        "error": {"name": "APIError",
                  "data": {"message": "API key is invalid.", "statusCode": 401}},
    })
    trajectory = oc_extract(fixture)
    error_steps = [s for s in trajectory.steps if s.step_type == "error"]
    assert len(error_steps) == 1
    assert "APIError" in error_steps[0].content
    assert "API key is invalid." in error_steps[0].content


def test_extractors_satisfy_multi_step_acceptance():
    """Acceptance: real multi-step trajectories, not generic single-step."""
    assert len(pi_extract(PI_FIXTURE).steps) > 1
    assert len(oc_extract(OPENCODE_FIXTURE).steps) > 1
