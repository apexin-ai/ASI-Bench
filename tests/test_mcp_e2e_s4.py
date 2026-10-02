"""Offline checks for the s4 grating-spectrum MCP E2E task and the verify_run answer `select`
it needs (an element of a returned spectrum). No MCP server, no S4."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E_TASKS = ROOT / "examples/mcp-e2e-tasks"
TASK_DIR = E2E_TASKS / "mcp_e2e/s4_grating_spectrum"
TASK_ID = "mcp_e2e.s4_grating_spectrum"
INSTANCE_ID = f"{TASK_ID}__seed31415"
SERVER = "s4"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_gt = _load(TASK_DIR / "generate_gt.py", "mcp_e2e_s4_generate_gt")
scorer = _load(TASK_DIR / "custom_scorer.py", "mcp_e2e_s4_custom_scorer")
verify = _load(ROOT / "scripts/mcp/e2e/verify_run.py", "mcp_e2e_s4_verify_run")
sys.path.insert(0, str(ROOT / "scripts/mcp/e2e"))
smoke = _load(ROOT / "scripts/mcp/e2e/smoke_s4.py", "mcp_e2e_s4_smoke")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    generate_gt.generate(out, {"seed": 31415})
    return out


@pytest.fixture(scope="module")
def reference(generated):
    return json.loads((generated / "reference/reference.json").read_text())


def _tool_spectrum(reference, orders=None, noise=3e-13):
    """What the server returns for the B1 call. The real server (S4 built from source, Linux
    aarch64, 2026-10-02) matched the reference to <= 1.1e-12 over 61 seeds; `noise` mimics that.
    ``orders`` simulates another truncation (e.g. 4 = the server default of 51 harmonics)."""
    if orders is None:
        r, t = reference["R_spectrum"], reference["T_spectrum"]
    else:
        spec = generate_gt.spectrum(reference["case"], orders, "laurent")
        r, t = [v[0] for v in spec], [v[1] for v in spec]
    r = [x + noise for x in r]
    t = [x - noise for x in t]
    return {"wavelength": reference["wavelengths_um"], "R": r, "T": t, "A": [1 - a - b for a, b in zip(r, t)]}


def _answer(spectrum, reference):
    k = reference["report_point_index"]
    j = spectrum["R"].index(max(spectrum["R"]))
    return {"R_at_report": spectrum["R"][k], "T_at_report": spectrum["T"][k], "R_max": spectrum["R"][j],
            "wavelength_at_R_max_um": spectrum["wavelength"][j]}


# --- task ---------------------------------------------------------------------

def test_cases_are_deterministic_varied_and_meet_every_selection_rule():
    assert generate_gt.build_case(31415)["case"] == generate_gt.build_case(31415)["case"]
    built = [generate_gt.build_case(s) for s in range(25)]
    assert len({json.dumps(b["case"], sort_keys=True) for b in built}) == 25
    assert {b["case"]["n_harmonics"] for b in built} == set(generate_gt.HARMONICS_ORDERS)
    for b in built:
        case, ref = b["case"], b["ref"]
        grid = generate_gt.wavelength_grid(case)
        anomalies = generate_gt.rayleigh_wavelengths(case["period_um"], case["illumination"]["theta_deg"], (1.0, 1.5))
        assert min(abs(lam - a) for lam in grid for a in anomalies) >= generate_gt.ANOMALY_CLEARANCE
        assert case["illumination"] == {**case["illumination"], "polarization": "TM"}
        assert case["illumination"]["theta_deg"] > 0        # an omitted theta_deg would not be the default
        assert 0 < ref["argmax"] < len(grid) - 1 and case["report_point_index"] != ref["argmax"]
        assert sorted(ref["R"])[-1] - sorted(ref["R"])[-2] >= generate_gt.MIN_MAX_GAP
        assert b["bypass_margin"] >= generate_gt.MIN_BYPASS_MARGIN
        assert case["report_wavelength_um"] == grid[case["report_point_index"]]
        step = grid[1] - grid[0]
        assert round(case["sweep"]["wavelength_start_um"] / 0.005, 6) % 1 == 0 and step >= 0.01 - 1e-12


def test_harmonics_are_complete_shells_and_never_the_server_default():
    for harmonics, orders in generate_gt.HARMONICS_ORDERS.items():
        assert smoke.square_shell_orders(harmonics) == orders
    assert 4 not in generate_gt.HARMONICS_ORDERS.values()      # default 51 harmonics keeps orders +-4


def test_reference_rcwa_is_the_smoke_rcwa():
    layer = [(0.3, 1.0, 2.5 ** 2, 0.2, 0.0)]
    for pol in ("TE", "TM"):
        for rule in ("li", "laurent"):
            assert generate_gt.rcwa_1d(pol, 1.07, 12.0, 0.8, 1.0, 1.5, layer, 5, rule) == \
                smoke.rcwa_1d(pol, 1.07, 12.0, 0.8, 1.0, 1.5, layer, 5, rule)
    assert generate_gt.rayleigh_wavelengths(0.8, 20.0, (1.0, 1.5)) == smoke.rayleigh_wavelengths(0.8, 20.0, (1.0, 1.5))


def test_tool_arguments_follow_the_b1_recipe(reference):
    args, case = reference["tool_arguments"], reference["case"]
    assert [layer["name"] for layer in args["layers"]] == ["substrate", "grating", "superstrate"]
    pattern = args["layers"][1]["pattern"]
    assert pattern["halfwidths"] == [case["grating"]["ridge_width_um"] / 2, case["period_um"] / 2]
    assert args["incidence_layer"] == "superstrate" and args["include_plot"] is False
    assert args["n_harmonics"] == case["n_harmonics"] and args["polarization"] == "TM"
    # the smoke's own reference path (same arguments, as the server takes them) gives the same spectrum
    again = smoke.grating_reference({**args, "layers": args["layers"]}, reference["orders"], "laurent")
    assert max(abs(a - r) for (a, _), r in zip(again, reference["R_spectrum"])) < 1e-13


def test_generated_reference_carries_every_key_the_checks_use(reference):
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    keys |= {src["select"]["equals_reference_key"] for a in spec["answers"] for src in a["from_calls"]
             if "equals_reference_key" in src.get("select", {})}
    assert keys <= set(reference)
    assert all(k in reference for k in scorer.KEYS)
    assert reference["sanity_R"] == pytest.approx(0.305336596450, abs=1e-12)


def test_rendered_prompts_and_data(generated, reference):
    case = json.loads((generated / "data/grating.json").read_text())
    assert case == reference["case"]
    for level in ("b1", "b2", "b3", "b4"):
        text = (generated / f"prompt_{level}.md").read_text()
        assert "{{" not in text and "result.json" in text


def test_prompts_name_the_tool_only_at_b1_b2_and_keep_the_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert ("`simulate_stack_spectrum`" in text) == (level in ("b1", "b2"))
        assert ("`s4` MCP server" in text) == (level in ("b1", "b2"))
        assert "no RCWA/FMM/TMM code of your own" in text and "Do not modify `data/grating.json`" in text
        assert "`report_point_index`" in text and "full precision" in text
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text
    b1 = (TASK_DIR / "prompt_b1.md").read_text()
    assert "**bottom to top**" in b1 and "`include_plot` = `false`" in b1 and "period_um / 2" in b1


def test_task_meta_is_test_status_with_numpy_only():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == ["numpy>=2.0"]


# --- scorer -------------------------------------------------------------------

def _dirs(tmp_path, prediction, reference):
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


def test_tool_values_score_full(tmp_path, reference):
    assert _total(*_dirs(tmp_path, _answer(_tool_spectrum(reference), reference), reference)) == pytest.approx(100.0)


def test_home_made_rcwa_and_wrong_procedures_lose_credit(tmp_path, reference):
    tool = _answer(_tool_spectrum(reference), reference)
    # a converged (Li's rule) RCWA at the reported point: >= 1e-3 off by construction
    converged = {**tool, "R_at_report": reference["converged_R_at_report"],
                 "T_at_report": 1 - reference["converged_R_at_report"]}
    assert _total(*_dirs(tmp_path / "a", converged, reference)) <= 40.0 + 1e-9
    # the server default of 51 harmonics (orders +-4) instead of the given number
    default = _answer(_tool_spectrum(reference, orders=4), reference)
    assert _total(*_dirs(tmp_path / "b", default, reference)) < 60.0
    # the neighbouring sweep point as the maximum, or rounded values
    neighbour = {**tool, "wavelength_at_R_max_um": tool["wavelength_at_R_max_um"] + 0.01}
    assert _total(*_dirs(tmp_path / "c", neighbour, reference)) == pytest.approx(85.0)
    rounded = {k: round(v, 3) for k, v in tool.items()}
    assert _total(*_dirs(tmp_path / "d", rounded, reference)) < 85.0


@pytest.mark.parametrize("prediction", [None, "not json", [], {"R_at_report": 0.1},
                                        {"R_at_report": True, "T_at_report": 0.9, "R_max": 0.2,
                                         "wavelength_at_R_max_um": 1.0},
                                        {"R_at_report": "high", "T_at_report": 0.9, "R_max": 0.2,
                                         "wavelength_at_R_max_um": 1.0}])
def test_submission_failures_are_valid_zero_scores(tmp_path, reference, prediction):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, prediction, reference)
    gate = get_scorer("s4_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in _eval()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error")


def test_missing_reference_and_bad_key_are_evaluator_failures(tmp_path, reference):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, _answer(_tool_spectrum(reference), reference), None)
    item = _eval()["scoring"][0]
    detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": 30})
    assert detail.details["scorer_internal_error"] and detail.details["failure_kind"] == "missing_evaluator_input"
    detail = get_scorer("s4_e2e_value").score(pred, ref, {"key": "A", "full_score_tol": 1, "zero_score_tol": 2})
    assert detail.details["scorer_internal_error"]


# --- verify_run: Claude stream-json and Codex JSONL -----------------------------

def _b1_calls(reference, sanity=False, orders=None, **overrides):
    args = {**reference["tool_arguments"], **overrides}
    calls = [("check_engine_sanity", {}, {"R": reference["sanity_R"] + 1e-17, "T": 1 - reference["sanity_R"],
                                          "A": 0.0, "expected_R": 0.3055, "ok": True})] if sanity else []
    calls.append(("simulate_stack_spectrum", args, _tool_spectrum(reference, orders=orders)))
    return calls


def _stream(calls, extra=()):
    tools = ["Bash", "Read", "Write", "WebFetch", "WebSearch",
             f"mcp__{SERVER}__check_engine_sanity", f"mcp__{SERVER}__simulate_stack_spectrum"]
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
        events += event(f"c{k}", f"mcp__{SERVER}__{tool}", args, json.dumps(payload))
    events.append({"type": "result", "subtype": "success", "is_error": False, "num_turns": len(calls)})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _codex(calls):
    events = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"}]
    for k, (tool, args, payload) in enumerate(calls):
        item = {"id": f"m{k}", "type": "mcp_tool_call", "server": SERVER, "tool": tool, "arguments": args,
                "status": "completed", "error": None,
                "result": {"content": [{"type": "text", "text": json.dumps(payload)}], "structured_content": None}}
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


def _run(tmp_path, stream, reference, answer, codex=False, files=None):
    results, instances = tmp_path / "out", tmp_path / "instances"
    task_out = results / TASK_ID
    outputs = task_out / f"{INSTANCE_ID}__b1.outputs"
    outputs.mkdir(parents=True)
    ref = instances / INSTANCE_ID / "reference"
    ref.mkdir(parents=True)
    ref.joinpath("reference.json").write_text(json.dumps(reference))
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


def _good(reference):
    return _answer(_tool_spectrum(reference), reference)


def test_genuine_b1_run_passes_every_check(tmp_path, reference):
    row = _run(tmp_path, _stream(_b1_calls(reference)), reference, _good(reference))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert row["checks"]["tool_correct"]["per_call"]["sanity"]["status"] == "SKIP"


def test_sanity_call_is_judged_when_made(tmp_path, reference):
    row = _run(tmp_path, _stream(_b1_calls(reference, sanity=True)), reference, _good(reference))
    assert row["verdict"] == "PASS" and row["checks"]["tool_correct"]["per_call"]["sanity"]["status"] == "PASS"


def test_codex_run_passes(tmp_path, reference):
    row = _run(tmp_path, _codex(_b1_calls(reference, sanity=True)), reference, _good(reference), codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


def test_default_harmonics_give_a_wrong_spectrum(tmp_path, reference):
    calls = _b1_calls(reference, orders=4, n_harmonics=51)
    answer = _answer(_tool_spectrum(reference, orders=4), reference)
    row = _run(tmp_path, _stream(calls), reference, answer)
    assert _status(row)["tool_correct"] == "FAIL" and _status(row)["answer_from_tool"] == "FAIL"


def test_answers_must_be_copied_from_the_returned_spectrum(tmp_path, reference):
    answer = {**_good(reference), "R_at_report": reference["R_at_report"] + 4e-7}   # right, but not returned
    row = _run(tmp_path, _stream(_b1_calls(reference)), reference, answer)
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "R_at_report" in row["checks"]["answer_from_tool"]["detail"]


def test_other_inputs_warn_when_the_result_is_still_right(tmp_path, reference):
    # numbers given as strings still match; an omitted theta_deg (default 0) is reported
    row = _run(tmp_path, _stream(_b1_calls(reference, theta_deg=str(reference["theta_deg"]))), reference,
               _good(reference))
    assert _status(row)["tool_correct"] == "PASS"
    calls = _b1_calls(reference)
    calls[0][1].pop("theta_deg")
    row = _run(tmp_path / "b", _stream(calls), reference, _good(reference))
    assert _status(row)["tool_correct"] == "WARN"


@pytest.mark.parametrize("command,status", [
    ("python3 -c 'import S4'", "FAIL"),
    ("pip install grcwa", "FAIL"),
    ("uv run --with rcwa python solve.py", "FAIL"),
    ("python3 -c \"import ctypes; ctypes.CDLL('/home/e2e/mcp/s4/src/mcp_s4_rcwa/s4lib/libS4.so')\"", "FAIL"),
    ("ls ~/mcp/s4/src/mcp_s4_rcwa/s4lib", "WARN"),
    ("~/mcp/s4/.venv/bin/python -c 'print(1)'", "FAIL"),
    ("python3 -c 'import numpy as np; np.linalg.eig(np.eye(2))'", "WARN"),
    ("echo 'values from S4 via MCP'", "WARN"),
    ("python3 -c \"import json; json.dump({'R_max': 1}, open('result.json','w'))\"", "PASS"),
])
def test_shell_access_to_s4(tmp_path, reference, command, status):
    row = _run(tmp_path, _stream(_b1_calls(reference), extra=[("Bash", {"command": command})]), reference,
               _good(reference))
    assert _status(row)["no_bypass"] == status


def test_home_made_rcwa_file_is_flagged(tmp_path, reference):
    source = "import numpy as np\n# R from S4 is what we want\nE = toeplitz(c)\n"
    row = _run(tmp_path, _stream(_b1_calls(reference)), reference, _good(reference), files={"solve.py": source})
    assert _status(row)["no_bypass"] == "WARN"
    row = _run(tmp_path / "b", _stream(_b1_calls(reference)), reference, _good(reference),
               files={"solve.py": "from grcwa import obj\n"})
    assert _status(row)["no_bypass"] == "FAIL"


# --- verify_run: answer `select` ----------------------------------------------

def _call(payload):
    return {"result_text": json.dumps(payload), "is_error": False}


def test_source_value_select_modes():
    call = _call({"wavelength": [1.0, 1.1, 1.2], "R": [0.1, 0.3, 0.2], "T": [0.9, 0.7, 0.8]})
    ref = {"lam": 1.1, "far": 1.15}
    assert verify._source_value(call, {"result_key": "R", "select": {"reduce": "max"}}, ref) == 0.3
    assert verify._source_value(call, {"result_key": "T", "select": {"reduce": "min"}}, ref) == 0.7
    assert verify._source_value(call, {"result_key": "wavelength", "select": {"argmax_of": "R"}}, ref) == 1.1
    assert verify._source_value(call, {"result_key": "T", "select": {"argmin_of": "R"}}, ref) == 0.9
    where = {"where_key": "wavelength", "equals_reference_key": "lam"}
    assert verify._source_value(call, {"result_key": "T", "select": where}, ref) == 0.7
    assert verify._source_value(call, {"result_key": "T", "select": {**where, "equals_reference_key": "far"}},
                                ref) is None
    assert verify._source_value(call, {"result_key": "A", "select": {"reduce": "max"}}, ref) is None
    assert verify._source_value(call, {"result_key": "R", "select": {"argmax_of": "missing"}}, ref) is None
    assert verify._source_value(_call({"R": 0.5}), {"result_key": "R", "select": {"reduce": "max"}}, ref) is None
    assert verify._source_value(_call({"R": 0.5}), {"result_key": "R"}, ref) == 0.5
    assert verify._source_value({"result_text": "not json"}, {"result_key": "R", "select": {"reduce": "max"}},
                                ref) is None
