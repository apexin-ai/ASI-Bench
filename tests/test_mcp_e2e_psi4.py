"""Offline checks for the psi4 optimize → frequency MCP E2E task and the verify_run features it
needs (dotted result keys, a geometry-valued tool chain). No MCP server; PySCF only for the
optional regeneration test."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E_TASKS = ROOT / "examples/mcp-e2e-tasks"
TASK_DIR = E2E_TASKS / "mcp_e2e/psi4_opt_freq"
TASK_ID = "mcp_e2e.psi4_opt_freq"
INSTANCE_ID = f"{TASK_ID}__seed31415"
SERVER = "psi4"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_gt = _load(TASK_DIR / "generate_gt.py", "mcp_e2e_psi4_generate_gt")
scorer = _load(TASK_DIR / "custom_scorer.py", "mcp_e2e_psi4_custom_scorer")
verify = _load(ROOT / "scripts/mcp/e2e/verify_run.py", "mcp_e2e_psi4_verify_run")

# Seed 31415 (methane, HF/STO-3G) as generated with PySCF 2.14.0 / geomeTRIC 1.1.1 on 2026-10-02.
REFERENCE = {
    "molecule": "methane",
    "geometry_xyz": ("C 0.0336 -0.0043 0.0037\nH 0.6614 0.5994 0.6686\nH -0.6046 -0.6391 0.6031\n"
                     "H -0.6297 0.5930 -0.6158\nH 0.6059 -0.6151 -0.5950"),
    "charge": 0, "multiplicity": 1, "method": "HF", "basis": "sto-3g",
    "final_energy_hartree": -39.72701071605472,
    "frequencies_cm_inv": [1675.7637665211773, 1675.763927211305, 1675.7642910716747, 1903.701286216498,
                           1903.701345345241, 3525.942298906059, 3786.7035380966663, 3786.7047458292377,
                           3786.7052811254007],
    "zpe_hartree": 0.05403984581951306,
}
OPT_GEOMETRY = ("0 1\n C    0.000000183968    0.000000183867   -0.000000496868\n"
                " H    0.624646961412    0.617402384774    0.633673566314\n"
                " H   -0.625822419613   -0.633108429762    0.616774691633\n"
                " H   -0.624721384455    0.633046007280   -0.617946714492\n"
                " H    0.625894652181   -0.617342151569   -0.632495627327\n")
# What the pinned server returned for the B1 calls of this instance (VM, 2026-10-02).
OPT_PAYLOAD = {"ok": True, "result": {"final_energy": {"value": -39.72701072, "unit": "Hartree"},
                                      "optimized_geometry_xyz": OPT_GEOMETRY, "n_iterations": 0, "converged": True},
               "warnings": [], "meta": {"psi4_version": "1.11", "wall_time_s": 1.28,
                                        "output_path": "/tmp/chemaster_psi4_x/optimize_output.log"}}
FREQS = [1675.7456, 1675.7483, 1675.7575, 1903.6959, 1903.6981, 3525.9929, 3786.7117, 3786.7549, 3786.8098]
FREQ_PAYLOAD = {"ok": True, "result": {"frequencies_cm_inv": FREQS, "ir_intensities_km_per_mol": [0.0] * 9,
                                       "n_imaginary": 0, "zpe": {"value": 0.05404022, "unit": "Hartree"},
                                       "thermal_corrections": {"h_corr": {"value": 0.05782578, "unit": "Hartree"}},
                                       "temperature_K": 298.15, "pressure_atm": 1.0},
                "warnings": [], "meta": {"psi4_version": "1.11", "wall_time_s": 0.27,
                                         "output_path": "/tmp/chemaster_psi4_y/frequency_output.log"}}
COMMON = {"method": "HF", "basis": "sto-3g", "charge": 0, "multiplicity": 1}


def _good_answer():
    return {"molecule": "methane", "final_energy_hartree": -39.72701072, "frequencies_cm_inv": list(FREQS),
            "zpe_hartree": 0.05404022}


# --- task ---------------------------------------------------------------------

def test_cases_are_deterministic_varied_and_distorted():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415)
    cases = [generate_gt.build_case(s) for s in range(60)]
    assert {c["molecule"] for c in cases} == set(generate_gt.MOLECULES)
    assert {c["basis"] for c in cases} == {"sto-3g", "cc-pvdz"}       # never 6-31G (Cartesian JK fit in psi4)
    for case in cases:
        template = generate_gt.MOLECULES[case["molecule"]].splitlines()
        lines = case["geometry_xyz"].splitlines()
        assert [t.split()[0] for t in template] == [line.split()[0] for line in lines]
        shifts = [abs(float(a) - float(b)) for t, line in zip(template, lines)
                  for a, b in zip(line.split()[1:], t.split()[1:])]
        assert max(shifts) <= generate_gt.DISTORTION + 1e-4 and max(shifts) > 0
        assert case["method"] == "HF" and case["charge"] == 0 and case["multiplicity"] == 1
    assert generate_gt.build_case(31415)["geometry_xyz"] == REFERENCE["geometry_xyz"]
    assert set(generate_gt.JK_FIT) == {"sto-3g", "cc-pvdz"}


def test_regenerated_reference_matches_the_recorded_one(tmp_path):
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    generate_gt.generate(tmp_path, {"seed": 31415})
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    assert ref["final_energy_hartree"] == pytest.approx(REFERENCE["final_energy_hartree"], abs=1e-8)
    assert max(abs(a - b) for a, b in zip(ref["frequencies_cm_inv"], REFERENCE["frequencies_cm_inv"])) < 0.05
    assert json.loads((tmp_path / "data/molecule.json").read_text())["geometry_xyz"] == REFERENCE["geometry_xyz"]
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{" not in text and "methane" in text or level == "b4"


def test_reference_carries_every_key_the_checks_use():
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    assert keys <= set(REFERENCE)
    manifest = json.loads((ROOT / "scripts/mcp/e2e/manifest.json").read_text())
    entry = next(e for e in manifest["servers"] if e["id"] == SERVER)
    assert spec["server_tools"] == entry["expected_tools"]


def test_prompts_name_the_server_only_at_b1_b2_and_keep_the_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert ("`psi4` MCP server" in text) == (level in ("b1", "b2"))
        assert ("`optimize`" in text and "`frequency`" in text) == (level == "b1")
        assert "Do not modify `data/molecule.json`" in text and "no `import psi4`" in text
        assert "optimised geometry, not at the starting geometry" in text and "full precision" in text
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text
    b1 = (TASK_DIR / "prompt_b1.md").read_text()
    assert "**verbatim**" in b1 and "result.optimized_geometry_xyz" in b1 and "result.zpe.value" in b1


def test_task_meta_is_test_status_with_pinned_reference_packages():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == ["pyscf==2.14.0", "geometric==1.1.1"]


# --- scorer -------------------------------------------------------------------

def _dirs(tmp_path, prediction, reference=REFERENCE):
    pred, ref = tmp_path / "pred", tmp_path / "ref"
    pred.mkdir(parents=True)
    ref.mkdir(parents=True)
    if prediction is not None:
        (pred / "result.json").write_text(json.dumps(prediction) if not isinstance(prediction, str) else prediction)
    if reference is not None:
        (ref / "reference.json").write_text(json.dumps(reference))
    return pred, ref


def _eval():
    import yaml
    return yaml.safe_load((TASK_DIR / "task_eval.yaml").read_text())["evaluation"]


def _total(pred, ref):
    from ai4sci_bench.core.scorer import get_scorer
    return sum(get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]}).score
               for item in _eval()["scoring"])


def test_tool_values_score_full(tmp_path):
    assert _total(*_dirs(tmp_path, _good_answer())) == pytest.approx(100.0)
    shuffled = {**_good_answer(), "frequencies_cm_inv": list(reversed(FREQS))}   # order does not matter
    assert _total(*_dirs(tmp_path / "b", shuffled)) == pytest.approx(100.0)


def test_wrong_procedures_lose_credit(tmp_path):
    # frequencies at the (non-stationary) starting geometry: rotations leak in, 12 modes instead of 9
    start = {**_good_answer(), "frequencies_cm_inv": [595.0, 480.0, 251.0, *FREQS]}
    assert _total(*_dirs(tmp_path / "a", start)) == pytest.approx(50.0)
    # a loose optimisation (measured on this instance: 2.3e-6 Eh, 8.3 cm^-1, 1.3e-5 Eh)
    loose = {"final_energy_hartree": REFERENCE["final_energy_hartree"] + 2.3e-6,
             "frequencies_cm_inv": [f + 8.26 for f in REFERENCE["frequencies_cm_inv"]],
             "zpe_hartree": REFERENCE["zpe_hartree"] + 1.3e-5}
    assert 20.0 < _total(*_dirs(tmp_path / "b", loose)) < 70.0
    # the server's default method/basis (B3LYP-D3(BJ)/def2-TZVP) is far off everywhere
    dft = {"final_energy_hartree": -40.53, "frequencies_cm_inv": [f * 0.9 for f in FREQS], "zpe_hartree": 0.0447}
    assert _total(*_dirs(tmp_path / "c", dft)) == 0.0


def test_frequency_error_requires_the_same_mode_count():
    assert scorer.frequency_error([1.0, 2.0], [2.0, 1.5]) == pytest.approx(0.5)
    assert math.isinf(scorer.frequency_error([1.0], [1.0, 2.0]))
    assert scorer.credit(0.5, 1.0, 20.0) == 1.0 and scorer.credit(20.0, 1.0, 20.0) == 0.0
    assert 0.0 < scorer.credit(4.0, 1.0, 20.0) < 1.0


@pytest.mark.parametrize("prediction", [None, "not json", [], {"final_energy_hartree": -1.0},
                                        {**_good_answer(), "frequencies_cm_inv": []},
                                        {**_good_answer(), "frequencies_cm_inv": "1675.7"},
                                        {**_good_answer(), "zpe_hartree": True},
                                        {**_good_answer(), "final_energy_hartree": "low"},
                                        {**_good_answer(), "frequencies_cm_inv": [1.0, float("nan")]}])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, prediction)
    gate = get_scorer("psi4_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in _eval()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error")


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, _good_answer(), reference=None)
    for item in _eval()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.details["scorer_internal_error"] and detail.details["failure_kind"] == "missing_evaluator_input"


# --- verify_run: Claude stream-json and Codex JSONL -----------------------------

def _b1_calls(opt_args=None, freq_args=None, opt_payload=OPT_PAYLOAD, freq_payload=FREQ_PAYLOAD):
    opt = {"geometry_xyz": REFERENCE["geometry_xyz"], **COMMON, **(opt_args or {})}
    freq = {"geometry_xyz": OPT_GEOMETRY, **COMMON, **(freq_args or {})}
    return [("optimize", opt, opt_payload), ("frequency", freq, freq_payload)]


def _stream(calls, extra=(), structured=True):
    """Claude Code stream-json. The psi4 tools return dicts, so FastMCP declares an object
    outputSchema and its structuredContent is the payload itself (no {"result": ...} wrapper);
    Claude shows that object, otherwise the indented text block."""
    tools = ["Bash", "Read", "Write", "WebFetch", "WebSearch",
             *(f"mcp__{SERVER}__{t}" for t in ("frequency", "optimize", "optimize_excited_state",
                                                 "single_point", "tddft"))]
    events = [{"type": "system", "subtype": "init", "mcp_servers": [{"name": SERVER, "status": "connected"}],
               "tools": tools}]

    def event(cid, name, args, text):
        return [{"type": "assistant", "message": {"content": [{"type": "tool_use", "id": cid, "name": name,
                                                               "input": args}]}},
                {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": cid,
                                                          "is_error": False,
                                                          "content": [{"type": "text", "text": text}]}]}}]
    for k, (name, args) in enumerate(extra):
        events += event(f"x{k}", name, args, "ok")
    for k, (tool, args, payload) in enumerate(calls):
        events += event(f"c{k}", f"mcp__{SERVER}__{tool}", args,
                        json.dumps(payload) if structured else json.dumps(payload, indent=2))
    events.append({"type": "result", "subtype": "success", "is_error": False, "num_turns": len(calls)})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _codex(calls):
    events = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"}]
    for k, (tool, args, payload) in enumerate(calls):
        item = {"id": f"m{k}", "type": "mcp_tool_call", "server": SERVER, "tool": tool, "arguments": args,
                "status": "completed", "error": None,
                "result": {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                           "structured_content": payload}}
        events += [{"type": "item.started", "item": {**item, "status": "in_progress", "result": None}},
                   {"type": "item.completed", "item": item}]
    events.append({"type": "turn.completed", "usage": {}})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _persist_like_run(stream):
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator
    from ai4sci_bench.trajectory.claude_extractor import extract_from_jsonl

    orchestrator = object.__new__(BenchmarkOrchestrator)
    persisted = "".join(json.dumps(orchestrator._redact_raw_prompt_fields(json.loads(line))) + "\n"
                        for line in stream.splitlines() if line.strip())
    return persisted, [step.to_dict() for step in extract_from_jsonl(stream, INSTANCE_ID).steps]


def _run(tmp_path, stream, answer, codex=False, files=None):
    results, instances = tmp_path / "out", tmp_path / "instances"
    task_out = results / TASK_ID
    outputs = task_out / f"{INSTANCE_ID}__b1.outputs"
    outputs.mkdir(parents=True)
    ref = instances / INSTANCE_ID / "reference"
    ref.mkdir(parents=True)
    ref.joinpath("reference.json").write_text(json.dumps(REFERENCE))
    outputs.joinpath("result.json").write_text(json.dumps(answer))
    for name, text in (files or {}).items():
        outputs.joinpath(name).write_text(text)
    stdout = f"{INSTANCE_ID}__b1.agent_stdout.jsonl"
    agent = {"raw_stdout_file": stdout, "persisted_outputs": {"dir": outputs.name}}
    if codex:
        task_out.joinpath(stdout).write_text(stream)
    else:
        persisted, steps = _persist_like_run(stream)
        traj = f"{INSTANCE_ID}__b1.trajectory.json"
        task_out.joinpath(stdout).write_text(persisted)
        task_out.joinpath(traj).write_text(json.dumps(steps))
        agent["trajectory_file"] = traj
    result = {"task_id": TASK_ID, "instance_id": INSTANCE_ID, "prompt_level": "b1", "status": "completed",
              "agent_output": agent}
    path = task_out / f"{INSTANCE_ID}__b1.json"
    path.write_text(json.dumps(result))
    return verify.verify_one(path, result, instances, E2E_TASKS)


def _status(row):
    return {name: check["status"] for name, check in row["checks"].items()}


@pytest.mark.parametrize("structured", [True, False])
def test_genuine_b1_run_passes_every_check(tmp_path, structured):
    row = _run(tmp_path, _stream(_b1_calls(), structured=structured), _good_answer())
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


def test_codex_run_passes(tmp_path):
    row = _run(tmp_path, _codex(_b1_calls()), _good_answer(), codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


def test_reformatted_geometry_still_chains(tmp_path):
    # header dropped, coordinates rounded to 6 decimals: same structure within 1e-4 Angstrom
    lines = [line.split() for line in OPT_GEOMETRY.splitlines()[1:]]
    rounded = "\n".join(f"{s} {float(x):.6f} {float(y):.6f} {float(z):.6f}" for s, x, y, z in lines)
    row = _run(tmp_path, _stream(_b1_calls(freq_args={"geometry_xyz": rounded})), _good_answer())
    assert _status(row)["tool_chain"] == "PASS"


def test_frequencies_at_the_starting_geometry_break_the_chain(tmp_path):
    start_freqs = {**FREQ_PAYLOAD, "result": {**FREQ_PAYLOAD["result"],
                                              "frequencies_cm_inv": [595.0, 480.0, 251.0, *FREQS]}}
    calls = _b1_calls(freq_args={"geometry_xyz": REFERENCE["geometry_xyz"]}, freq_payload=start_freqs)
    answer = {**_good_answer(), "frequencies_cm_inv": [595.0, 480.0, 251.0, *FREQS]}
    row = _run(tmp_path, _stream(calls), answer)
    status = _status(row)
    assert status["tool_chain"] == "FAIL" and status["tool_correct"] == "FAIL"
    assert status["answer_from_tool"] == "FAIL" and row["verdict"] == "FAIL"


def test_in_band_error_is_not_a_correct_result(tmp_path):
    failed = {"ok": False, "error_code": "PSI4_INTERNAL_ERROR", "details": "No module named 'psi4'"}
    row = _run(tmp_path, _stream(_b1_calls(opt_payload=failed)), _good_answer())
    assert _status(row)["tool_correct"] == "FAIL" and _status(row)["answer_from_tool"] == "FAIL"


def test_answers_must_be_copied_from_tool_results(tmp_path):
    # the PySCF reference values are right but were never returned by a tool
    answer = {**_good_answer(), "frequencies_cm_inv": REFERENCE["frequencies_cm_inv"]}
    row = _run(tmp_path, _stream(_b1_calls()), answer)
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "frequencies_cm_inv" in row["checks"]["answer_from_tool"]["detail"]


def test_default_method_warns_on_inputs_and_fails_on_values(tmp_path):
    dft = {**OPT_PAYLOAD, "result": {**OPT_PAYLOAD["result"], "final_energy": {"value": -40.53, "unit": "Hartree"}}}
    calls = _b1_calls(opt_args={"method": "B3LYP-D3(BJ)", "basis": "def2-TZVP"}, opt_payload=dft)
    row = _run(tmp_path, _stream(calls), {**_good_answer(), "final_energy_hartree": -40.53})
    assert _status(row)["tool_correct"] == "FAIL" and _status(row)["answer_from_tool"] == "FAIL"


@pytest.mark.parametrize("command,status", [
    ("python3 -c 'import psi4'", "FAIL"),
    ("python3 -c 'from pyscf import gto'", "FAIL"),
    ("pip install pyscf", "FAIL"),
    ("micromamba install -c conda-forge psi4", "FAIL"),
    ("uv run --with pyscf python opt.py", "FAIL"),
    ("psi4 input.dat", "FAIL"),
    ("~/mcp/psi4/.venv/bin/python -c 'print(1)'", "FAIL"),
    ("python3 -m chemaster.mcp.calc_psi4.server", "FAIL"),
    ("ls ~/mcp/psi4", "WARN"),
    ("python3 -c \"import json; json.dump({'zpe_hartree': 1}, open('result.json','w'))\"", "PASS"),
])
def test_shell_access_to_psi4(tmp_path, command, status):
    row = _run(tmp_path, _stream(_b1_calls(), extra=[("Bash", {"command": command})]), _good_answer())
    assert _status(row)["no_bypass"] == status


def test_excited_state_tool_is_only_suspicious(tmp_path):
    calls = [("optimize_excited_state", {"geometry_xyz": OPT_GEOMETRY, **COMMON}, OPT_PAYLOAD), *_b1_calls()]
    row = _run(tmp_path, _stream(calls), _good_answer())
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"
    assert "optimize_excited_state" in row["checks"]["no_bypass"]["detail"]


# --- verify_run: dotted keys and geometry comparison ------------------------------

def test_field_walks_nested_objects_and_prefers_literal_keys():
    data = {"result": {"zpe": {"value": 0.05}}, "a.b": 1, "a": {"b": 2}}
    assert verify.values.field(data, "result.zpe.value") == 0.05
    assert verify.values.field(data, "a.b") == 1
    assert verify.values.field(data, "result.missing.value") is None
    assert verify.values.field(data, "result.zpe.value.more") is None
    assert verify.values.field([1], "x") is None and verify.values.field(data, None) is None
    call = verify.evidence.ToolCall(result_text=json.dumps(OPT_PAYLOAD), is_error=False)
    assert verify.values.read(call, verify.spec.Selector(key="result.final_energy.value")) == -39.72701072
    raw = verify.spec.Selector(key="result.optimized_geometry_xyz", raw=True)
    assert verify.values.read(call, raw) == OPT_PAYLOAD["result"]["optimized_geometry_xyz"]


def test_same_geometry():
    plain = "O 0 0 0.1\nH 0 0.75 -0.47\nH 0 -0.75 -0.47"
    assert verify.values.same_geometry("0 1\n" + plain + "\nsymmetry c1\n", "3\nwater\n" + plain, 1e-4)
    assert verify.values.same_geometry(plain.replace("0.75", "0.75004"), plain, 1e-4)
    assert not verify.values.same_geometry(plain.replace("0.75", "0.7502"), plain, 1e-4)
    assert not verify.values.same_geometry(plain.replace("O", "S", 1), plain, 1e-4)
    assert not verify.values.same_geometry("\n".join(plain.splitlines()[:2]), plain, 1e-4)
    assert not verify.values.same_geometry("no atoms here", plain, 1e-4) and not verify.values.same_geometry(None, plain, 1e-4)
    link = verify.spec.Binding(("g",), verify.spec.Selector(key="g", raw=True))
    assert verify.values.link_comparator(link)("abc", "abc") and not verify.values.link_comparator(link)("0.1", "0.2")
    geo = verify.spec.Binding(("g",), verify.spec.Selector(key="g", raw=True), "geometry", 1e-4)
    assert verify.values.link_comparator(geo)(plain.replace("0.75", "0.75004"), plain)
