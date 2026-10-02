"""Offline checks for the jsbsim engine-run MCP E2E task and the verify_run extensions it needs
(optional/grouped calls, multi-source answers, string-valued chaining). No MCP server, no JSBSim."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E_TASKS = ROOT / "examples/mcp-e2e-tasks"
TASK_DIR = E2E_TASKS / "mcp_e2e/jsbsim_engine_run"
TASK_ID = "mcp_e2e.jsbsim_engine_run"
INSTANCE_ID = f"{TASK_ID}__seed31415"
SERVER = "jsbsim"
SID = "b03a88ab5beb"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_gt = _load(TASK_DIR / "generate_gt.py", "mcp_e2e_jsbsim_generate_gt")
scorer = _load(TASK_DIR / "custom_scorer.py", "mcp_e2e_jsbsim_custom_scorer")
verify = _load(ROOT / "scripts/mcp/e2e/verify_run.py", "mcp_e2e_jsbsim_verify_run")

CASE = generate_gt.build_case(31415)
# Reference (jsbsim 1.3.1 in generate_gt.py) and what the real server returned for
# the same scenario, 2026-10-02 (the server converts ft/s -> kt with 0.592484).
FLOWN = {"altitude_ft": 5493.534629531205, "airspeed_kt": 104.99751646892263,
         "thrust_lbs": 393.150160146237, "sim_time_s": 10.000000000000076, "jsbsim_version": "1.3.1"}
TOOL = {"altitude_ft": 5493.534898169339, "airspeed_kt": 104.9975171280067,
        "thrust_lbs": 393.1501527026332, "sim_time_s": 10.000000000000076}
PATHS = generate_gt.QUANTITIES


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """reference.json exactly as generate() writes it, with fly() stubbed (no JSBSim)."""
    out = tmp_path_factory.mktemp("gen")
    original = generate_gt.fly
    generate_gt.fly = lambda case, root=None: dict(FLOWN)
    try:
        generate_gt.generate(out, {"seed": 31415})
    finally:
        generate_gt.fly = original
    return json.loads((out / "reference/reference.json").read_text())


# --- task ---------------------------------------------------------------------

def test_cases_are_deterministic_and_varied():
    assert generate_gt.build_case(31415) == CASE
    cases = [generate_gt.build_case(s) for s in range(50)]
    assert len({json.dumps(c["initial_conditions"], sort_keys=True) for c in cases}) == 50
    for case in cases:
        ic = case["initial_conditions"]
        assert list(ic) == ["altitude_ft", "latitude_deg", "longitude_deg", "airspeed_fps", "heading_deg",
                            "pitch_deg", "roll_deg"]
        assert 3000 <= ic["altitude_ft"] <= 6000 and 150 <= ic["airspeed_fps"] <= 185 and ic["roll_deg"] == 0
        assert [s["path"] for s in case["settings"]] == ["propulsion/set-running", "fcs/mixture-cmd-norm",
                                                         "fcs/throttle-cmd-norm"]
        assert case["settings"][0]["value"] == -1.0 and case["duration_s"] in (6.0, 8.0, 10.0)
        assert case["aircraft"] == "c172x" and case["timestep_s"] == 1 / 60


def test_reference_uses_jsbsim_initial_condition_properties():
    props = generate_gt.ic_properties(CASE["initial_conditions"])
    assert props["ic/h-sl-ft"] == CASE["initial_conditions"]["altitude_ft"]
    assert abs(props["ic/vc-kts"] * 6076.115485564304 / 3600 - CASE["initial_conditions"]["airspeed_fps"]) < 1e-12
    assert set(props) == {"ic/h-sl-ft", "ic/lat-geod-deg", "ic/long-gc-deg", "ic/vc-kts", "ic/psi-true-deg",
                          "ic/theta-deg", "ic/phi-deg"}


def test_generated_reference_carries_every_key_the_checks_use(reference):
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    assert keys <= set(reference)
    assert reference["setting0_path"] == "propulsion/set-running" and reference["frames"] == 600
    assert all(reference[k] == FLOWN[k] for k in scorer.KEYS)


def test_prompts_name_tools_only_at_b1_and_keep_procedure_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert "result.json" in text and "Do not install, import or run JSBSim" in text
        assert "**all seven** initial conditions" in text and "Do not run a trim/balancing routine" in text
        for path in PATHS.values():
            assert f"`{path}`" in text
        named = [f"`{t}`" in text for t in ("create_session", "set_initial_conditions", "set_property", "step",
                                            "get_property")]
        assert named == [level == "b1"] * 5
        assert ("`jsbsim` MCP server" in text) == (level in ("b1", "b2"))
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text


def test_task_meta_is_test_status_with_pinned_jsbsim():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == ["jsbsim==1.3.1"]


# --- scorer -------------------------------------------------------------------

def _dirs(tmp_path, prediction, reference=FLOWN):
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
    total = 0.0
    for item in _eval()["scoring"]:
        total += get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]}).score
    return total


def test_tool_values_and_rounded_telemetry_score_full(tmp_path):
    assert _total(*_dirs(tmp_path / "a", TOOL)) == pytest.approx(100.0)
    rounded = {k: round(v, 2) for k, v in TOOL.items()}
    assert _total(*_dirs(tmp_path / "b", rounded)) == pytest.approx(100.0)


def test_procedure_errors_lose_credit(tmp_path):
    # settings applied before the initial conditions: >= 0.33 ft / 0.27 kt off (60 seeds)
    wrong = {**TOOL, "altitude_ft": TOOL["altitude_ft"] + 0.33, "airspeed_kt": TOOL["airspeed_kt"] + 0.27}
    assert _total(*_dirs(tmp_path / "a", wrong)) < 40.0
    one_more_second = {**TOOL, "sim_time_s": 11.000000000000076}
    assert _total(*_dirs(tmp_path / "b", one_more_second)) == pytest.approx(90.0)


@pytest.mark.parametrize("prediction", [None, "not json", [], {**TOOL, "thrust_lbs": None},
                                        {**TOOL, "altitude_ft": True}, {**TOOL, "airspeed_kt": "fast"},
                                        {k: v for k, v in TOOL.items() if k != "sim_time_s"}])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, prediction)
    gate = get_scorer("jsbsim_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in _eval()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error")


def test_missing_reference_and_bad_key_are_evaluator_failures(tmp_path):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, TOOL, reference=None)
    item = _eval()["scoring"][0]
    detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": 35})
    assert detail.details["scorer_internal_error"] and detail.details["failure_kind"] == "missing_evaluator_input"
    detail = get_scorer("jsbsim_e2e_value").score(pred, ref, {"key": "rpm", "full_score_tol": 1, "zero_score_tol": 2})
    assert detail.details["scorer_internal_error"]


# --- verify_run: Claude stream-json -------------------------------------------

def _payload(tool, args):
    """What the real server returns (texts as recorded 2026-10-02)."""
    if tool == "create_session":
        return {"session_id": SID, "aircraft": "c172x", "dt": 1 / 60, "sim_time": 0.0, "root": "/x/jsbsim_data"}
    if tool == "set_initial_conditions":
        return {"ok": True}
    if tool == "set_property":
        return {"ok": True, "path": args["path"], "value": float(args["value"])}
    if tool == "step":
        return {"session_id": args["session_id"], "frames": round(args["seconds"] * 60), "dt": 1 / 60,
                "sim_time": args.get("_t", TOOL["sim_time_s"])}
    if tool == "get_property":
        key = {v: k for k, v in PATHS.items()}.get(args["path"])
        return {"path": args["path"], "value": TOOL[key] if key else 0.0, "present": True}
    if tool == "get_telemetry":
        return {"aircraft": "c172x", "session_id": SID, "t": TOOL["sim_time_s"], "alt_ft": round(TOOL["altitude_ft"], 2),
                "airspeed_kt": round(TOOL["airspeed_kt"], 2), "thrust_lbs": round(TOOL["thrust_lbs"], 2),
                "pitch_deg": 0.0}
    if tool == "trim":
        return {"ok": True, "mode": "longitudinal", "throttle": 0.7}
    raise KeyError(tool)


def _b1_calls(read="property", ic_in_create=False, step_chunks=(10.0,), sid=SID):
    calls = [("create_session", {"aircraft": "c172x", **({"initial_conditions": CASE["initial_conditions"]}
                                                         if ic_in_create else {})})]
    if not ic_in_create:
        calls.append(("set_initial_conditions", {"session_id": sid, **CASE["initial_conditions"]}))
    calls += [("set_property", {"session_id": sid, "path": s["path"], "value": s["value"]}) for s in CASE["settings"]]
    t = 0.0
    for chunk in step_chunks:
        t += chunk
        calls.append(("step", {"session_id": sid, "seconds": chunk,
                               "_t": TOOL["sim_time_s"] if t == 10.0 else t}))
    if read in ("property", "both"):
        calls += [("get_property", {"session_id": sid, "path": p}) for p in PATHS.values()]
    if read in ("telemetry", "both"):
        calls.append(("get_telemetry", {"session_id": sid}))
    return calls


def _stream(calls, extra=()):
    tools = ["Bash", "Read", "Write", "WebFetch", "WebSearch"] + [f"mcp__{SERVER}__{t}" for t in (
        "close_session", "create_session", "execute_script", "get_property", "get_telemetry", "list_aircraft",
        "set_initial_conditions", "set_property", "step", "trim")]
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
    for k, (tool, args) in enumerate(calls):
        shown = {a: v for a, v in args.items() if not a.startswith("_")}
        events += event(f"c{k}", f"mcp__{SERVER}__{tool}", shown, json.dumps(_payload(tool, args), indent=2))
    events.append({"type": "result", "subtype": "success", "is_error": False, "num_turns": len(calls)})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _codex(calls):
    events = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"}]
    for k, (tool, args) in enumerate(calls):
        shown = {a: v for a, v in args.items() if not a.startswith("_")}
        item = {"id": f"m{k}", "type": "mcp_tool_call", "server": SERVER, "tool": tool, "arguments": shown,
                "status": "completed", "error": None,
                "result": {"content": [{"type": "text", "text": json.dumps(_payload(tool, args))}],
                           "structured_content": None}}
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


def _run(tmp_path, stream, reference, answer=None, codex=False):
    results, instances = tmp_path / "out", tmp_path / "instances"
    task_out = results / TASK_ID
    outputs = task_out / f"{INSTANCE_ID}__b1.outputs"
    outputs.mkdir(parents=True)
    ref = instances / INSTANCE_ID / "reference"
    ref.mkdir(parents=True)
    ref.joinpath("reference.json").write_text(json.dumps(reference))
    outputs.joinpath("result.json").write_text(json.dumps(answer or TOOL))
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


def test_genuine_b1_run_passes_every_check(tmp_path, reference):
    row = _run(tmp_path, _stream(_b1_calls()), reference)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert row["checks"]["tool_correct"]["per_call"]["telemetry"]["status"] == "SKIP"


def test_telemetry_only_and_create_session_initial_conditions_pass(tmp_path, reference):
    rounded = {k: round(v, 2) if k != "sim_time_s" else v for k, v in TOOL.items()}
    row = _run(tmp_path, _stream(_b1_calls(read="telemetry", ic_in_create=True)), reference, answer=rounded)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


def test_chunked_steps_pass_with_a_warning_on_inputs(tmp_path, reference):
    row = _run(tmp_path, _stream(_b1_calls(step_chunks=(5.0, 5.0))), reference)
    assert row["verdict"] == "PASS"
    assert _status(row)["tool_correct"] == "WARN" and _status(row)["tool_chain"] == "PASS"


def test_steps_on_another_session_break_the_chain(tmp_path, reference):
    calls = _b1_calls()
    calls = [(t, {**a, "session_id": "0123456789ab"}) if t == "step" else (t, a) for t, a in calls]
    row = _run(tmp_path, _stream(calls), reference)
    assert _status(row)["tool_chain"] == "FAIL" and "step←create" in row["checks"]["tool_chain"]["detail"]


def test_no_state_read_is_an_uncalled_requirement(tmp_path, reference):
    row = _run(tmp_path, _stream(_b1_calls(read="none")), reference)
    assert row["failure"] == "tool_called"
    assert "get_telemetry|get_property" in row["checks"]["tool_called"]["detail"]


def test_answers_must_be_copied_from_a_tool(tmp_path, reference):
    recomputed = {**TOOL, "altitude_ft": FLOWN["altitude_ft"]}  # right within tolerance, but not returned
    row = _run(tmp_path, _stream(_b1_calls()), reference, answer=recomputed)
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "altitude_ft" in row["checks"]["answer_from_tool"]["detail"]


def test_trim_is_flagged_for_review(tmp_path, reference):
    calls = _b1_calls()
    calls.insert(5, ("trim", {"session_id": SID, "mode": "longitudinal"}))
    row = _run(tmp_path, _stream(calls), reference)
    assert _status(row)["no_bypass"] == "WARN" and "mcp__jsbsim__trim" in row["checks"]["no_bypass"]["detail"]


@pytest.mark.parametrize("command,status", [
    ("python3 -c 'import jsbsim; print(jsbsim.__version__)'", "FAIL"),
    ("pip install jsbsim==1.3.1", "FAIL"),
    ("uv run --with jsbsim python fly.py", "FAIL"),
    ("jsbsim --script=scripts/c1722.xml", "FAIL"),
    ("~/mcp/jsbsim/.venv/bin/python -c 'print(1)'", "FAIL"),
    ("ls ~/mcp/jsbsim/jsbsim_data/aircraft", "WARN"),
    ("python3 -c \"import json; json.dump({'altitude_ft': 1}, open('result.json','w'))\"", "PASS"),
])
def test_shell_access_to_jsbsim(tmp_path, reference, command, status):
    row = _run(tmp_path, _stream(_b1_calls(), extra=[("Bash", {"command": command})]), reference)
    assert _status(row)["no_bypass"] == status


def test_codex_run_passes_without_reading_telemetry(tmp_path, reference):
    row = _run(tmp_path, _codex(_b1_calls()), reference, codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


# --- verify_run helpers -------------------------------------------------------

def test_same_link_compares_strings_exactly_and_numbers_to_print_precision():
    assert verify.values.same_link("b03a88ab5beb", "b03a88ab5beb")
    assert not verify.values.same_link("b03a88ab5beb", "b03a88ab5bec")
    assert verify.values.same_link("123456789012", 123456789012)
    assert verify.values.same_link([1.0, 2.0], [1.0, 2.0 + 1e-13])
    assert not verify.values.same_link(None, None) and not verify.values.same_link("", "")
    assert not verify.values.same_link({"a": 1}, {"a": 1})


def test_requirements_group_optional_calls():
    CallSpec = verify.spec.CallSpec
    spec = verify.spec.Spec(2, "s", "r.json", "p.json", answers=(), calls=(
        CallSpec("a", "A"), CallSpec("b", "B", optional=True, group="g"),
        CallSpec("c", "C", optional=True, group="g"), CallSpec("d", "D", optional=True)))
    assert [[cs.name for cs in req] for req in spec.requirements()] == [["a"], ["b", "c"]]
