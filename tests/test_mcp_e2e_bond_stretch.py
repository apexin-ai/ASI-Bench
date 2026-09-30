"""Offline checks for the pyscf bond-stretch MCP E2E task and verify_run schema 2 (no MCP server, no PySCF)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E_TASKS = ROOT / "examples/mcp-e2e-tasks"
TASK_DIR = E2E_TASKS / "mcp_e2e/pyscf_bond_stretch"
TASK_ID = "mcp_e2e.pyscf_bond_stretch"
INSTANCE_ID = f"{TASK_ID}__seed31415"

LENGTHS = [0.994, 1.0711428571428572, 1.1482857142857144, 1.2254285714285715,
           1.3025714285714287, 1.3797142857142859, 1.456857142857143, 1.534]
ENERGIES = [-91.6717701093, -91.6752210554, -91.6690843271, -91.6571264203,
            -91.6419519384, -91.6252847402, -91.6082226815, -91.5914488702]
REFERENCE = {"smiles": "C#N", "atom1_idx": 0, "atom2_idx": 2, "start_dist": 0.994, "end_dist": 1.534,
             "num_points": 8, "bond_lengths": LENGTHS, "energies_hartree": ENERGIES,
             "min_bond_length": LENGTHS[1], "min_energy_hartree": ENERGIES[1]}
SCAN_ARGS = {"smiles_string": "C#N", "atom1_idx": 0, "atom2_idx": 2, "start_dist": 0.994, "end_dist": 1.534,
             "num_points": 8, "basis": "sto-3g"}
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAA+gAAAJYCAYAAADxHswl"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_gt = _load(TASK_DIR / "generate_gt.py", "mcp_e2e_bond_generate_gt")
scorer = _load(TASK_DIR / "custom_scorer.py", "mcp_e2e_bond_custom_scorer")
verify = _load(ROOT / "scripts/mcp/e2e/verify_run.py", "mcp_e2e_bond_verify_run")


# --- task ---------------------------------------------------------------------

def test_cases_are_deterministic_and_within_ranges():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415)
    seen = set()
    for seed in range(200):
        case = generate_gt.build_case(seed)
        bond = next(b for b in generate_gt.BONDS if (b[0], b[2], b[3]) ==
                    (case["smiles"], case["atom1_idx"], case["atom2_idx"]))
        r_eq = bond[5]
        assert 0.85 * r_eq - 1e-3 <= case["start_dist"] <= 0.95 * r_eq + 1e-3
        assert 1.25 * r_eq - 1e-3 <= case["end_dist"] <= 1.45 * r_eq + 1e-3
        assert 5 <= case["num_points"] <= 9 and case["basis"] == "sto-3g"
        seen.add(case["bond"])
    assert len(seen) == len({b[4] for b in generate_gt.BONDS})


def test_only_rigid_molecules_are_used():
    # Flexible molecules have several UFF minima, so the unseeded server geometry
    # would not reproduce the reference (scripts/mcp/e2e/README.md).
    assert {b[0] for b in generate_gt.BONDS} <= {"O", "N", "F", "C#N", "C=O", "CF"}


def test_prompts_name_tools_only_at_b1_and_forbid_local_work():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert "result.json" in text and "Do not install, import or run PySCF" in text
        assert "not locally" in text
        assert ("{{molecule}}" in text) == (level in ("b1", "b2"))
        named = ("`run_bond_stretch_calculation_mcp`" in text, "`plot_energy_scan_image_mcp`" in text)
        assert named == ((True, True) if level == "b1" else (False, False))
        assert "mcp__" not in text and "Claude" not in text


# --- scorer -------------------------------------------------------------------

def _dirs(tmp_path, prediction, reference=REFERENCE):
    pred, ref = tmp_path / "pred", tmp_path / "ref"
    pred.mkdir(parents=True)
    ref.mkdir(parents=True)
    if prediction is not None:
        (pred / "result.json").write_text(json.dumps(prediction))
    if reference is not None:
        (ref / "reference.json").write_text(json.dumps(reference))
    return pred, ref


def _answer(**overrides):
    return {"molecule": "hydrogen cyanide", "bond_lengths": LENGTHS, "energies_hartree": ENERGIES,
            "min_bond_length": LENGTHS[1], "min_energy_hartree": ENERGIES[1], **overrides}


ENERGY_CFG = {"weight": 80, "full_score_tol": 1e-5, "zero_score_tol": 1e-3, "grid_tol": 1e-6}


def test_scorers_give_full_credit_to_the_reference(tmp_path):
    pred, ref = _dirs(tmp_path, _answer())
    assert scorer.ScanSchema().score(pred, ref, {"weight": 1}).passed
    assert scorer.ScanEnergies().score(pred, ref, dict(ENERGY_CFG)).score == 80
    assert scorer.ScanMinimum().score(pred, ref, {"weight": 20}).score == 20


def test_energy_credit_is_log_linear_on_the_worst_point(tmp_path):
    energies = list(ENERGIES)
    energies[3] += 1e-4
    detail = scorer.ScanEnergies().score(*_dirs(tmp_path, _answer(energies_hartree=energies)), dict(ENERGY_CFG))
    assert detail.score == pytest.approx(40) and not detail.passed


def test_wrong_grid_or_minimum_scores_zero(tmp_path):
    shifted = [v + 0.01 for v in LENGTHS]
    detail = scorer.ScanEnergies().score(*_dirs(tmp_path / "g", _answer(bond_lengths=shifted)), dict(ENERGY_CFG))
    assert detail.score == 0 and "grid" in detail.message
    short = scorer.ScanEnergies().score(*_dirs(tmp_path / "s", _answer(bond_lengths=LENGTHS[:3],
                                                                      energies_hartree=ENERGIES[:3])), dict(ENERGY_CFG))
    assert short.score == 0
    wrong_min = scorer.ScanMinimum().score(*_dirs(tmp_path / "m", _answer(min_bond_length=LENGTHS[0],
                                                                          min_energy_hartree=ENERGIES[0])), {"weight": 20})
    assert wrong_min.score == 0


def test_submission_and_evaluator_failures_are_separated(tmp_path):
    bad = scorer.ScanEnergies().score(*_dirs(tmp_path / "b", _answer(energies_hartree=ENERGIES[:-1])), dict(ENERGY_CFG))
    assert bad.score == 0 and "scorer_internal_error" not in bad.details
    missing = scorer.ScanSchema().score(*_dirs(tmp_path / "n", None), {"weight": 1})
    assert not missing.passed and "scorer_internal_error" not in missing.details
    no_ref = scorer.ScanMinimum().score(*_dirs(tmp_path / "r", _answer(), reference=None), {"weight": 20})
    assert no_ref.details["scorer_internal_error"] is True
    assert no_ref.details["failure_kind"] == "missing_evaluator_input"


# --- verify_run schema 2 ----------------------------------------------------------

SCAN = "mcp__pyscf__run_bond_stretch_calculation_mcp"
PLOT = "mcp__pyscf__plot_energy_scan_image_mcp"


def _stream(scan_args=SCAN_ARGS, plot_input=None, plot=True, plot_result=None, bash=()):
    scan_result = json.dumps({"bond_lengths": LENGTHS, "energies": ENERGIES}, indent=2)
    events = [{"type": "system", "subtype": "init", "mcp_servers": [{"name": "pyscf", "status": "connected"}],
               "tools": ["Bash", "Read", "Write", SCAN, PLOT]}]
    for i, command in enumerate(bash):
        events.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"b{i}", "name": "Bash", "input": {"command": command}}]}})
    events += [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "s1", "name": SCAN, "input": scan_args}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "s1", "is_error": False,
                                                  "content": [{"type": "text", "text": scan_result}]}]}},
    ]
    if plot:
        events += [
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "p1", "name": PLOT,
                                                           "input": plot_input or {"bond_lengths": LENGTHS,
                                                                                   "energies": ENERGIES}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "p1", "is_error": False,
                                                      "content": plot_result or [{"type": "image", "source": {
                                                          "type": "base64", "media_type": "image/png",
                                                          "data": PNG_B64}}]}]}},
        ]
    events.append({"type": "result", "subtype": "success", "is_error": False, "num_turns": 4})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _persist_like_run(stream):
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator
    from ai4sci_bench.trajectory.claude_extractor import extract_from_jsonl

    orchestrator = object.__new__(BenchmarkOrchestrator)
    persisted = "".join(json.dumps(orchestrator._redact_raw_prompt_fields(json.loads(line))) + "\n"
                        for line in stream.splitlines() if line.strip())
    return persisted, [step.to_dict() for step in extract_from_jsonl(stream, INSTANCE_ID).steps]


def _run(tmp_path, stream, answer=None, edit_trajectory=None):
    results, instances = tmp_path / "out", tmp_path / "instances"
    task_out = results / TASK_ID
    outputs = task_out / f"{INSTANCE_ID}__b1.outputs"
    outputs.mkdir(parents=True)
    ref = instances / INSTANCE_ID / "reference"
    ref.mkdir(parents=True)
    ref.joinpath("reference.json").write_text(json.dumps(REFERENCE))
    outputs.joinpath("result.json").write_text(json.dumps(answer or _answer()))
    persisted, steps = _persist_like_run(stream)
    if edit_trajectory:
        edit_trajectory(steps)
    stdout, traj = f"{INSTANCE_ID}__b1.agent_stdout.jsonl", f"{INSTANCE_ID}__b1.trajectory.json"
    task_out.joinpath(stdout).write_text(persisted)
    task_out.joinpath(traj).write_text(json.dumps(steps))
    result = {"task_id": TASK_ID, "instance_id": INSTANCE_ID, "prompt_level": "b1", "status": "completed",
              "agent_output": {"raw_stdout_file": stdout, "trajectory_file": traj,
                               "persisted_outputs": {"dir": outputs.name}}}
    path = task_out / f"{INSTANCE_ID}__b1.json"
    path.write_text(json.dumps(result))
    return verify.verify_one(path, result, instances, E2E_TASKS)


def _status(row):
    return {name: check["status"] for name, check in row["checks"].items()}


def test_extractor_keeps_image_results_observable():
    _, steps = _persist_like_run(_stream())
    plot_result = [s for s in steps if s["step_type"] == "tool_result"][-1]
    assert plot_result["content"] == ""  # image bytes are not copied into the trajectory
    assert plot_result["metadata"]["content_types"] == ["image"]
    assert plot_result["metadata"]["image_media_types"] == ["image/png"]


def test_genuine_scan_and_plot_run_passes(tmp_path):
    row = _run(tmp_path, _stream(bash=["cat data/scan.json"]))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert list(_status(row)) == list(verify.CHECK_ORDER)


def test_numeric_inputs_passed_as_strings_still_match(tmp_path):
    args = {**SCAN_ARGS, "start_dist": "0.994", "num_points": "8"}
    assert _status(_run(tmp_path, _stream(scan_args=args)))["tool_correct"] == "PASS"


def test_missing_plot_call_fails(tmp_path):
    row = _run(tmp_path, _stream(plot=False, bash=["python3 -c 'import matplotlib'"]))
    assert row["failure"] == "tool_called"
    assert _status(row)["tool_chain"] == "FAIL" and _status(row)["no_bypass"] == "WARN"


def test_plot_with_retyped_data_breaks_the_chain(tmp_path):
    rounded = {"bond_lengths": LENGTHS, "energies": [round(e, 4) for e in ENERGIES]}
    row = _run(tmp_path, _stream(plot_input=rounded))
    assert row["failure"] == "tool_chain" and _status(row)["tool_correct"] == "PASS"


def test_plot_without_image_fails_tool_correct(tmp_path):
    row = _run(tmp_path, _stream(plot_result=[{"type": "text", "text": "plot saved"}]))
    assert row["failure"] == "tool_correct"
    assert row["checks"]["tool_correct"]["per_call"]["plot"]["status"] == "FAIL"


def test_old_trajectories_without_block_types_only_warn(tmp_path):
    def strip(steps):
        for step in steps:
            step["metadata"].pop("content_types", None)
            step["metadata"].pop("image_media_types", None)
    row = _run(tmp_path, _stream(), edit_trajectory=strip)
    assert row["verdict"] == "PASS" and _status(row)["tool_correct"] == "WARN"


def test_answer_must_copy_tool_values(tmp_path):
    row = _run(tmp_path, _stream(), answer=_answer(energies_hartree=[round(e, 6) for e in ENERGIES]))
    assert row["failure"] == "answer_from_tool"


def test_wrong_scan_inputs_fail(tmp_path):
    row = _run(tmp_path, _stream(scan_args={**SCAN_ARGS, "atom2_idx": 1}))
    # energies returned are still the reference ones here, so only the inputs differ
    assert row["checks"]["tool_correct"]["per_call"]["scan"]["status"] == "WARN"


def test_schema_1_specs_are_normalised():
    spec = json.loads((E2E_TASKS / "mcp_e2e/pyscf_rhf_energy/e2e_check.json").read_text())
    norm = verify.normalize_spec(spec)
    assert [c["tool"] for c in norm["calls"]] == ["pyscf_rhf_energy"]
    assert norm["answers"][0]["from_call"] == "pyscf_rhf_energy"
    assert verify.normalize_spec(norm) is norm
