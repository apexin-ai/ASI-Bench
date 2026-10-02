"""pyscf_rhf_energy (one tool, one scalar; schema-1 e2e_check): generator, scorer, verifier scenarios
(Claude stream-json). Codex runs of this task are in test_verify_evidence.py."""
import json

from ai4sci_bench.core.task import TaskLoader

from . import support
from .support import E2E_TASKS, claude, jsonl

TASK = support.Task("mcp_e2e.pyscf_rhf_energy")
PYSCF_TASK, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id
REF_ENERGY = -75.98321949952147
ATOM = "O 0.016793 -0.002165 0.119622; H 0.016159 0.740617 -0.451404; H 0.012259 -0.760455 -0.484141"
REFERENCE = {"energy_hartree": REF_ENERGY, "atom": ATOM, "basis": "6-31g"}
TOOL = "mcp__pyscf__pyscf_rhf_energy"
CONFIG = {"weight": 100, "full_score_tol": 1e-6, "zero_score_tol": 1e-3}

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify


def _dirs(tmp_path, prediction, reference=REF_ENERGY):
    return support.score_dirs(tmp_path, prediction,
                              None if reference is None else {"energy_hartree": reference, "basis": "6-31g"})


def _stream(tool_value=REF_ENERGY, atom=ATOM, server_status="connected", call_tool=True, bash=()):
    tools = ["Bash", "Read", "Write"] + ([TOOL] if server_status == "connected" else [])
    events = [claude.init("pyscf", tools, server_status)]
    events += [claude.tool_use(f"b{i}", "Bash", {"command": command}) for i, command in enumerate(bash)]
    if call_tool:
        events += claude.call("t1", TOOL, {"atom": atom, "basis": "6-31g"}, repr(tool_value))
    events.append(claude.result(3))
    return jsonl(events)


def _persist_like_run(stream):
    persisted, steps = support.persist_like_run(stream, instance_id=INSTANCE_ID)
    return persisted, json.dumps(steps)


def _run(tmp_path, stream, answer=REF_ENERGY, persist=True):
    return TASK.verify(tmp_path, stream, reference=REFERENCE, persist=persist,
                       answer=None if answer is None else {"energy_hartree": answer})


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


def test_runtime_pins_pyscf():
    assert TaskLoader(E2E_TASKS).load_task_by_id(TASK_ID)["_runtime_packages"] == ["pyscf==2.9.0"]


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
