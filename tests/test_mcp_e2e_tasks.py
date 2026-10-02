"""Offline checks for the MCP E2E fake tasks and the run verifier (no MCP server, no PySCF)."""
import importlib.util
import json
import sys
from pathlib import Path

from ai4sci_bench.core.task import TaskLoader

ROOT = Path(__file__).resolve().parents[1]
E2E_TASKS = ROOT / "examples/mcp-e2e-tasks"
PYSCF_TASK = E2E_TASKS / "mcp_e2e/pyscf_rhf_energy"
TASK_ID = "mcp_e2e.pyscf_rhf_energy"
INSTANCE_ID = f"{TASK_ID}__seed31415"
REF_ENERGY = -75.98321949952147
ATOM = "O 0.016793 -0.002165 0.119622; H 0.016159 0.740617 -0.451404; H 0.012259 -0.760455 -0.484141"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(module)
    return module


generate_gt = _load(PYSCF_TASK / "generate_gt.py", "mcp_e2e_pyscf_generate_gt")
scorer = _load(PYSCF_TASK / "custom_scorer.py", "mcp_e2e_pyscf_custom_scorer")
verify = _load(ROOT / "scripts/mcp/e2e/verify_run.py", "mcp_e2e_verify_run")


def test_task_is_test_status_and_only_discovered_from_its_own_tasks_dir():
    loader = TaskLoader(E2E_TASKS)
    assert sorted(t["id"] for t in loader.discover_tasks(include_test=True)) == [
        "mcp_e2e.arxiv_search_snippets", "mcp_e2e.jsbsim_engine_run", "mcp_e2e.psi4_opt_freq",
        "mcp_e2e.pyscf_bond_stretch", TASK_ID, "mcp_e2e.s4_grating_spectrum"]
    assert loader.discover_tasks() == []
    meta = loader.load_task_by_id(TASK_ID)
    assert meta["_runtime_packages"] == ["pyscf==2.9.0"]
    assert TASK_ID not in {t["id"] for t in TaskLoader(ROOT / "tasks").discover_tasks(
        include_test=True, include_sample=True, include_dev=True)}


def test_instances_are_deterministic_and_perturbed():
    first, again, other = generate_gt.build_case(31415), generate_gt.build_case(31415), generate_gt.build_case(7)
    assert first == again and first != other
    assert first["basis"] in generate_gt.BASES and first["molecule"] in generate_gt.MOLECULES
    template = generate_gt.MOLECULES[first["molecule"]]
    for (symbol, *xyz), rendered in zip(template, first["atom"].split("; ")):
        parts = rendered.split()
        assert parts[0] == symbol
        assert all(abs(float(v) - c) <= generate_gt.PERTURBATION_ANGSTROM + 1e-6 for v, c in zip(parts[1:], xyz))


def test_prompts_name_the_tool_only_at_b1_and_forbid_direct_pyscf():
    for level in ("b1", "b2", "b3", "b4"):
        text = (PYSCF_TASK / f"prompt_{level}.md").read_text()
        assert "result.json" in text and "Do not install, import or run PySCF" in text
        assert ("`pyscf_rhf_energy`" in text) == (level == "b1")
        # Harness-specific tool names (e.g. Claude's mcp__<server>__<tool>) primed
        # agents to look for a tool-loading step and give up; keep prompts agent-neutral.
        assert "mcp__" not in text and "Claude" not in text


def _dirs(tmp_path, prediction, reference=REF_ENERGY):
    pred, ref = tmp_path / "pred", tmp_path / "ref"
    pred.mkdir(parents=True)
    ref.mkdir(parents=True)
    if prediction is not None:
        (pred / "result.json").write_text(json.dumps(prediction))
    if reference is not None:
        (ref / "reference.json").write_text(json.dumps({"energy_hartree": reference, "basis": "6-31g"}))
    return pred, ref


CONFIG = {"weight": 100, "full_score_tol": 1e-6, "zero_score_tol": 1e-3}


def test_energy_scorer_credit_curve(tmp_path):
    pred, ref = _dirs(tmp_path, {"energy_hartree": REF_ENERGY + 1e-4, "basis": "6-31G"})
    detail = scorer.PySCFEnergy().score(pred, ref, dict(CONFIG))
    assert abs(detail.score - 100 / 3) < 1e-9 and not detail.passed
    assert detail.details["basis_match"] is True
    assert scorer.credit(5e-7, 1e-6, 1e-3) == 1.0 and scorer.credit(2e-3, 1e-6, 1e-3) == 0.0


def test_submission_failures_are_valid_zero_scores(tmp_path):
    pred, ref = _dirs(tmp_path, {"energy_hartree": "not a number"})
    gate = scorer.PySCFResultSchema().score(pred, ref, {"weight": 1.0})
    assert not gate.passed and "scorer_internal_error" not in gate.details
    missing = scorer.PySCFEnergy().score(*_dirs(tmp_path / "m", None), dict(CONFIG))
    assert missing.score == 0 and "scorer_internal_error" not in missing.details


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    pred, ref = _dirs(tmp_path, {"energy_hartree": REF_ENERGY}, reference=None)
    detail = scorer.PySCFEnergy().score(pred, ref, dict(CONFIG))
    assert detail.details["scorer_internal_error"] is True
    assert detail.details["failure_kind"] == "missing_evaluator_input"


def _stream(tool_value=REF_ENERGY, atom=ATOM, server_status="connected", call_tool=True, bash=()):
    tool = "mcp__pyscf__pyscf_rhf_energy"
    events = [{"type": "system", "subtype": "init", "mcp_servers": [{"name": "pyscf", "status": server_status}],
               "tools": ["Bash", "Read", "Write"] + ([tool] if server_status == "connected" else [])}]
    for i, command in enumerate(bash):
        events.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"b{i}", "name": "Bash", "input": {"command": command}}]}})
    if call_tool:
        events += [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": tool, "input": {"atom": atom, "basis": "6-31g"}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": False,
                 "content": [{"type": "text", "text": repr(tool_value)}]}]}},
        ]
    events.append({"type": "result", "subtype": "success", "is_error": False, "num_turns": 3})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _persist_like_run(stream: str) -> tuple[str, str]:
    """Return (persisted stream, trajectory JSON) exactly as `asibench run` writes them."""
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator
    from ai4sci_bench.trajectory.claude_extractor import extract_from_jsonl

    orchestrator = object.__new__(BenchmarkOrchestrator)
    persisted = "".join(
        json.dumps(orchestrator._redact_raw_prompt_fields(json.loads(line))) + "\n"
        for line in stream.splitlines() if line.strip()
    )
    steps = [step.to_dict() for step in extract_from_jsonl(stream, INSTANCE_ID).steps]
    return persisted, json.dumps(steps)


def _run(tmp_path, stream, answer=REF_ENERGY, persist=True):
    results, instances = tmp_path / "out", tmp_path / "instances"
    task_out = results / TASK_ID
    outputs = task_out / f"{INSTANCE_ID}__b1.outputs"
    outputs.mkdir(parents=True)
    ref = instances / INSTANCE_ID / "reference"
    ref.mkdir(parents=True)
    ref.joinpath("reference.json").write_text(json.dumps(
        {"energy_hartree": REF_ENERGY, "atom": ATOM, "basis": "6-31g"}))
    if answer is not None:
        outputs.joinpath("result.json").write_text(json.dumps({"energy_hartree": answer}))
    stdout = f"{INSTANCE_ID}__b1.agent_stdout.jsonl"
    agent_output = {"raw_stdout_file": stdout, "persisted_outputs": {"dir": outputs.name}}
    if persist:
        stream, trajectory = _persist_like_run(stream)
        traj = f"{INSTANCE_ID}__b1.trajectory.json"
        task_out.joinpath(traj).write_text(trajectory)
        agent_output["trajectory_file"] = traj
    task_out.joinpath(stdout).write_text(stream)
    result = {"task_id": TASK_ID, "instance_id": INSTANCE_ID, "prompt_level": "b1", "status": "completed",
              "agent_output": agent_output}
    path = task_out / f"{INSTANCE_ID}__b1.json"
    path.write_text(json.dumps(result))
    return verify.verify_one(path, result, instances, E2E_TASKS)


def test_run_redaction_removes_tool_results_from_the_stream():
    persisted, _trajectory = _persist_like_run(_stream())
    assert repr(REF_ENERGY) not in persisted  # why the verifier must use the trajectory


def test_verifier_reads_unredacted_stream_too(tmp_path):
    assert _run(tmp_path, _stream(), persist=False)["verdict"] == "PASS"


def test_verifier_passes_a_genuine_mcp_run(tmp_path):
    row = _run(tmp_path, _stream(bash=["cat data/molecule.json"]))
    assert row["verdict"] == "PASS", row["checks"]
    assert row["tool_call_counts"]["mcp__pyscf__pyscf_rhf_energy"] == 1


def test_verifier_flags_direct_pyscf_even_with_correct_answer(tmp_path):
    row = _run(tmp_path, _stream(bash=["uv run --with pyscf python -c 'import pyscf'"]))
    assert row["verdict"] == "FAIL" and row["failure"] == "no_bypass"


def test_verifier_distinguishes_unavailable_server_from_uncalled_tool(tmp_path):
    down = _run(tmp_path / "down", _stream(server_status="failed", call_tool=False))
    assert down["failure"] == "mcp_connected"
    skipped = _run(tmp_path / "skip", _stream(call_tool=False))
    assert skipped["failure"] == "tool_called"


def test_verifier_requires_answer_to_come_from_tool(tmp_path):
    row = _run(tmp_path, _stream(), answer=REF_ENERGY + 5e-7)
    assert row["checks"]["tool_correct"]["status"] == "PASS"
    assert row["failure"] == "answer_from_tool"


def test_verifier_tolerates_string_message_payloads(tmp_path):
    odd = "\n".join(json.dumps(e) for e in (
        {"type": "user", "message": "plain string payload"},
        {"type": "user", "message": {"role": "user", "content": "string content"}},
        {"type": "assistant", "message": "x"},
    ))
    row = _run(tmp_path, odd + "\n" + _stream(), persist=False)
    assert row["verdict"] == "PASS", row["checks"]


def test_verifier_reports_missing_outputs_as_answer_failure(tmp_path):
    row = _run(tmp_path, _stream(), answer=None)
    assert row["failure"] == "answer_from_tool"


def test_verifier_warns_when_tool_inputs_were_reformatted(tmp_path):
    row = _run(tmp_path, _stream(atom=ATOM.replace("0.016793", "0.0168")))
    assert row["checks"]["tool_correct"]["status"] == "WARN" and row["verdict"] == "PASS"
