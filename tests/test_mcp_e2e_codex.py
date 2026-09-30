"""Codex CLI evidence for the MCP E2E verifier and trajectory (offline; no MCP server, no Codex).

Event shapes are those emitted by ``codex exec --json`` (codex-cli 0.159.2) in
real runs against the pyscf MCP server: ``mcp_tool_call`` items with server,
tool, arguments and a result whose content blocks are MCP blocks (images carry
``data`` and ``mimeType``).
"""
import importlib.util
import json
import sys
from pathlib import Path

from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator
from ai4sci_bench.trajectory.codex_extractor import extract_from_jsonl

ROOT = Path(__file__).resolve().parents[1]
E2E_TASKS = ROOT / "examples/mcp-e2e-tasks"
RHF_TASK, BOND_TASK = "mcp_e2e.pyscf_rhf_energy", "mcp_e2e.pyscf_bond_stretch"

REF_ENERGY = -75.98321949952147
ATOM = "O 0.016793 -0.002165 0.119622; H 0.016159 0.740617 -0.451404; H 0.012259 -0.760455 -0.484141"
RHF_REFERENCE = {"energy_hartree": REF_ENERGY, "atom": ATOM, "basis": "6-31g"}

LENGTHS = [0.85, 0.95, 1.0499999999999998, 1.15]
ENERGIES = [-74.93598322678581, -74.96327081558326, -74.96087193724398, -74.94160963237066]
SCAN_ARGS = {"smiles_string": "O", "atom1_idx": 0, "atom2_idx": 1, "start_dist": 0.85, "end_dist": 1.15,
             "num_points": 4}
BOND_REFERENCE = {"smiles": "O", "atom1_idx": 0, "atom2_idx": 1, "start_dist": 0.85, "end_dist": 1.15,
                  "num_points": 4, "bond_lengths": LENGTHS, "energies_hartree": ENERGIES,
                  "min_bond_length": LENGTHS[1], "min_energy_hartree": ENERGIES[1]}
BOND_ANSWER = {"molecule": "water", "bond_lengths": LENGTHS, "energies_hartree": ENERGIES,
               "min_bond_length": LENGTHS[1], "min_energy_hartree": ENERGIES[1]}
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAA+gAAAJYCAYAAADxHswl+/9j/4AAQSkZJRg"  # contains path-like runs on purpose


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify = _load(ROOT / "scripts/mcp/e2e/verify_run.py", "mcp_e2e_codex_verify_run")


# --- Codex JSONL fixtures -------------------------------------------------------

def _mcp(item_id, tool, arguments, content=None, error=None):
    base = {"id": item_id, "type": "mcp_tool_call", "server": "pyscf", "tool": tool, "arguments": arguments}
    started = {"type": "item.started", "item": {**base, "result": None, "error": None, "status": "in_progress"}}
    if error is not None:
        done = {**base, "result": None, "error": {"message": error}, "status": "failed"}
    else:
        done = {**base, "result": {"content": content, "structured_content": None}, "error": None,
                "status": "completed"}
    return [started, {"type": "item.completed", "item": done}]


def _text(value):
    return [{"type": "text", "text": value}]


def _shell(item_id, command, output="", exit_code=0):
    base = {"id": item_id, "type": "command_execution", "command": command}
    return [{"type": "item.started", "item": {**base, "aggregated_output": "", "exit_code": None,
                                              "status": "in_progress"}},
            {"type": "item.completed", "item": {**base, "aggregated_output": output, "exit_code": exit_code,
                                                "status": "completed"}}]


def _stream(*groups, final="done"):
    events = [{"type": "thread.started", "thread_id": "01a0f1fc-c68d-73e3-8c18-ba20543905b9"},
              {"type": "turn.started"},
              {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "**Planning**"}},
              # Codex lists MCP resources on its own; reported under the server name.
              *_mcp("item_1", "list_mcp_resources", {"server": "pyscf"},
                    _text('{"server":"pyscf","resources":[]}'))]
    for group in groups:
        events += group
    events += [{"type": "item.completed", "item": {"id": "item_99", "type": "agent_message", "text": final}},
               {"type": "turn.completed", "usage": {"input_tokens": 55966, "cached_input_tokens": 31104,
                                                    "output_tokens": 331, "reasoning_output_tokens": 116}}]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _rhf_call(value=REF_ENERGY, atom=ATOM):
    return _mcp("item_3", "pyscf_rhf_energy", {"atom": atom, "basis": "6-31g"}, _text(repr(value)))


def _scan_call(args=SCAN_ARGS):
    return _mcp("item_5", "run_bond_stretch_calculation_mcp", args,
                _text(json.dumps({"bond_lengths": LENGTHS, "energies": ENERGIES}, indent=2)))


def _plot_call(arguments=None, content=None):
    return _mcp("item_7", "plot_energy_scan_image_mcp", arguments or {"bond_lengths": LENGTHS, "energies": ENERGIES},
                content or [{"type": "image", "data": PNG_B64, "mimeType": "image/png"}])


# --- run artefacts as `asibench run` writes them ----------------------------------

def _persist_like_run(stream, tmp_path):
    orchestrator = object.__new__(BenchmarkOrchestrator)
    orchestrator.output_dir, orchestrator.repo_root = tmp_path / "out", ROOT
    persisted = orchestrator._sanitize_raw_artifact_text(stream, raw_format="jsonl", workspace=tmp_path / "ws")
    return persisted, [step.to_dict() for step in extract_from_jsonl(stream, "inst").steps]


def _run(tmp_path, task_id, reference, answer, stream, *, raw=True):
    instance_id = f"{task_id}__seed31415"
    results, instances = tmp_path / "out", tmp_path / "instances"
    task_out = results / task_id
    outputs = task_out / f"{instance_id}__b1.outputs"
    outputs.mkdir(parents=True)
    ref = instances / instance_id / "reference"
    ref.mkdir(parents=True)
    ref.joinpath("reference.json").write_text(json.dumps(reference))
    if answer is not None:
        outputs.joinpath("result.json").write_text(json.dumps(answer))
    persisted, steps = _persist_like_run(stream, tmp_path)
    traj = f"{instance_id}__b1.trajectory.json"
    task_out.joinpath(traj).write_text(json.dumps(steps))
    agent_output = {"trajectory_file": traj, "persisted_outputs": {"dir": outputs.name}}
    if raw:
        agent_output["raw_stdout_file"] = f"{instance_id}__b1.agent_stdout.jsonl"
        task_out.joinpath(agent_output["raw_stdout_file"]).write_text(persisted)
    result = {"task_id": task_id, "instance_id": instance_id, "prompt_level": "b1", "status": "completed",
              "agent_output": agent_output}
    path = task_out / f"{instance_id}__b1.json"
    path.write_text(json.dumps(result))
    return verify.verify_one(path, result, instances, E2E_TASKS)


def _rhf(tmp_path, stream, answer=REF_ENERGY, **kwargs):
    return _run(tmp_path, RHF_TASK, RHF_REFERENCE, None if answer is None else {"energy_hartree": answer},
                stream, **kwargs)


def _bond(tmp_path, stream, answer=BOND_ANSWER, **kwargs):
    return _run(tmp_path, BOND_TASK, BOND_REFERENCE, answer, stream, **kwargs)


def _status(row):
    return {name: check["status"] for name, check in row["checks"].items()}


# --- trajectory extractor -------------------------------------------------------

def test_extractor_records_mcp_calls_and_results():
    traj = extract_from_jsonl(_stream(_scan_call(), _plot_call()), "inst")
    calls = [s for s in traj.steps if s.step_type == "tool_call"]
    results = [s for s in traj.steps if s.step_type == "tool_result"]
    assert [c.metadata["tool_name"] for c in calls] == [
        "mcp__pyscf__list_mcp_resources",
        "mcp__pyscf__run_bond_stretch_calculation_mcp",
        "mcp__pyscf__plot_energy_scan_image_mcp",
    ]
    assert [c.metadata["tool_call_id"] for c in calls] == [r.metadata["tool_call_id"] for r in results]
    assert calls[1].metadata["mcp_server"] == "pyscf"
    assert calls[1].metadata["mcp_tool"] == "run_bond_stretch_calculation_mcp"
    assert json.loads(results[1].content)["energies"] == ENERGIES
    assert results[1].metadata["content_types"] == ["text"] and results[1].metadata["is_error"] is False
    assert traj.summary.tool_call_distribution["mcp__pyscf__plot_energy_scan_image_mcp"] == 1


def test_extractor_keeps_image_results_observable_without_copying_them():
    traj = extract_from_jsonl(_stream(_scan_call(), _plot_call()), "inst")
    plot = [s for s in traj.steps if s.step_type == "tool_result"][-1]
    assert plot.content == "" and PNG_B64 not in json.dumps(traj.to_dict())
    assert plot.metadata["content_types"] == ["image"]
    assert plot.metadata["image_media_types"] == ["image/png"]


def test_extractor_marks_failed_mcp_calls():
    failed = _mcp("item_3", "pyscf_rhf_energy", {"atom": ATOM}, error="tool call timed out")
    traj = extract_from_jsonl(_stream(failed), "inst")
    result = [s for s in traj.steps if s.step_type == "tool_result"][-1]
    assert result.metadata["is_error"] is True and "timed out" in result.content
    # A call that never completed still shows up as a call without a result.
    pending = extract_from_jsonl(_stream(failed[:1]), "inst")
    assert [s.step_type for s in pending.steps].count("tool_call") == 2
    assert [s.step_type for s in pending.steps].count("tool_result") == 1


# --- verifier ---------------------------------------------------------------------

def test_codex_tool_results_survive_persistence(tmp_path):
    # Unlike Claude stream-json (tool results sit in redacted user events), the
    # persisted Codex JSONL keeps arguments and results of MCP calls.
    persisted, _steps = _persist_like_run(_stream(_rhf_call(), _plot_call()), tmp_path)
    assert repr(REF_ENERGY) in persisted and ATOM in persisted
    assert '"mimeType":"image/png"' in persisted


def test_genuine_codex_rhf_run_passes(tmp_path):
    row = _rhf(tmp_path, _stream(_shell("item_2", "cat data/molecule.json"), _rhf_call(),
                                 _shell("item_4", "printf '%s' '{...}' > result.json")))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert row["evidence_source"].startswith("codex exec JSONL")
    assert row["tool_call_counts"]["mcp__pyscf__pyscf_rhf_energy"] == 1
    assert row["bash_commands"] == ["cat data/molecule.json", "printf '%s' '{...}' > result.json"]
    assert row["agent_final"]["subtype"] == "turn.completed" and row["last_assistant_text"] == "done"
    assert "no server list" in row["checks"]["mcp_connected"]["detail"]


def test_genuine_codex_scan_and_plot_run_passes(tmp_path):
    row = _bond(tmp_path, _stream(_scan_call(), _plot_call()))
    assert row["verdict"] == "PASS", row["checks"]
    assert _status(row) == {name: "PASS" for name in verify.CHECK_ORDER}
    assert row["checks"]["tool_correct"]["per_call"]["plot"]["calls"][0]["media_types"] == ["image/png"]


def test_resource_listing_is_not_a_tool_call(tmp_path):
    row = _rhf(tmp_path, _stream(), answer=None)
    assert row["tool_sequence"] == ["mcp__pyscf__list_mcp_resources"]
    assert row["failure"] == "tool_called"
    assert _status(row)["mcp_connected"] == "WARN"  # nothing proves the server's tools were offered


def test_direct_backend_use_in_a_shell_command_is_flagged(tmp_path):
    row = _rhf(tmp_path, _stream(_rhf_call(), _shell("item_4", "python3 -c 'import pyscf; print(1)'")))
    assert row["verdict"] == "FAIL" and row["failure"] == "no_bypass"


def test_failed_mcp_call_fails_tool_correct(tmp_path):
    failed = _mcp("item_3", "pyscf_rhf_energy", {"atom": ATOM, "basis": "6-31g"}, error="tool call timed out")
    row = _rhf(tmp_path, _stream(failed))
    assert _status(row)["tool_called"] == "PASS" and row["failure"] == "tool_correct"
    assert _status(row)["mcp_connected"] == "WARN"
    assert "timed out" in row["checks"]["tool_correct"]["per_call"]["pyscf_rhf_energy"]["calls"][0]["result_text"]


def test_answer_must_copy_the_tool_value(tmp_path):
    row = _rhf(tmp_path, _stream(_rhf_call()), answer=REF_ENERGY + 5e-7)
    assert row["failure"] == "answer_from_tool"


def test_reformatted_inputs_only_warn(tmp_path):
    row = _rhf(tmp_path, _stream(_rhf_call(atom=ATOM.replace("0.016793", "0.0168"))))
    assert _status(row)["tool_correct"] == "WARN" and row["verdict"] == "PASS"


def test_plot_with_retyped_data_breaks_the_chain(tmp_path):
    rounded = {"bond_lengths": LENGTHS, "energies": [round(e, 4) for e in ENERGIES]}
    row = _bond(tmp_path, _stream(_scan_call(), _plot_call(arguments=rounded)))
    assert row["failure"] == "tool_chain"


def test_plot_without_image_fails(tmp_path):
    row = _bond(tmp_path, _stream(_scan_call(), _plot_call(content=_text("saved plot.png"))))
    assert row["failure"] == "tool_correct"


def test_arguments_serialised_as_json_string_are_parsed(tmp_path):
    call = _mcp("item_3", "pyscf_rhf_energy", json.dumps({"atom": ATOM, "basis": "6-31g"}), _text(repr(REF_ENERGY)))
    assert _status(_rhf(tmp_path, _stream(call)))["tool_correct"] == "PASS"


def test_trajectory_alone_is_enough_for_calls_and_results(tmp_path):
    # Without the raw JSONL the trajectory still proves the calls, their results
    # and the image; only tool inputs are missing, so the chain cannot be checked.
    row = _bond(tmp_path, _stream(_scan_call(), _plot_call(), _shell("item_8", "ls")), raw=False)
    assert row["evidence_source"].startswith("trajectory")
    assert row["verdict"] == "PASS", row["checks"]
    assert _status(row) == {"mcp_connected": "PASS", "tool_called": "PASS", "tool_correct": "PASS",
                            "tool_chain": "WARN", "answer_from_tool": "PASS", "no_bypass": "PASS"}
    assert row["bash_commands"] == ["ls"]


def test_non_agent_logs_are_not_mistaken_for_codex(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"type": "message", "role": "assistant", "content": "hi"}\nplain text\n')
    assert verify.parse_codex_stream(path) is None
