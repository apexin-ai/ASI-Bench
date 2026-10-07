"""gpaw_mos2_bandgap (MoS2 monolayer: relax -> convergence gate -> bands -> verify ->
artefacts, all on one run_id): generator, scorer, verifier scenarios. No MCP server,
no GPAW and no ASE at import time.

Unlike the other fake tasks, this one's DFT reference cannot be recomputed here:
GPAW lives only inside the server's pinned conda environment, so ``generate_gt.MEASURED``
is a table measured against that server by ``scripts/mcp/e2e/measure_gpaw_table.py``.
The behavioural tests below therefore run on :data:`SAMPLE_MEASURED`, a small synthetic
table engineered to exercise every branch (both convergence tolerances, all three
verify_run verdicts, params_verified true and false). Two further tests read the real
table instead: one asserts it covers the whole instance grid, the other that the
derivations here reproduce what the server itself answered (the ``measured_*`` fields).
They skip while the table is still empty.
"""
import json

import pytest

from . import support
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.gpaw_mos2_bandgap")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id
SERVER = "gpaw"
TOOLS = support.setup.load_manifest()[SERVER]["expected_tools"]

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify
_status = support.statuses

# The measured table as committed, captured before any test patches it.
REAL_MEASURED = dict(generate_gt.MEASURED)
_OMIT = object()

# The built-in 2H-MoS2 monolayer, as ase.build.mx2(a=3.18, thickness=3.17, vacuum=10.0)
# builds it (checked against ASE 3.29.0 in test_mx2_structure_matches_the_hardcoded_cell).
CELL = {
    "symbols": ["Mo", "S", "S"],
    "lengths": [3.18, 3.18, 23.17],
    "angles": [90.0, 90.0, 120.00000000000001],
    "scaled": [[0.0, 0.0, 0.5],
               [0.6666666667, 0.3333333333, 0.5684074234],
               [0.6666666667, 0.3333333333, 0.4315925766]],
}

# --------------------------------------------------------------------------
# A synthetic measurement, engineered so that every branch is reachable
# --------------------------------------------------------------------------
# k-grid sweep: delta(10->15) fails both tolerances, delta(15->25) passes 5.0 but not
# 0.3, delta(25->35) passes 0.3. So tol 5.0 recommends density 15 and tol 0.3 recommends
# 25, which makes params_verified false exactly for (kpts_density 15, tol 0.3).
KPTS_SWEEP = [
    {"kpts_density": 10.0, "energy_ev": -22.0, "gap_ev": 1.7, "kpts": [3, 3, 1]},
    {"kpts_density": 15.0, "energy_ev": -22.024, "gap_ev": 1.67, "kpts": [6, 6, 1]},
    {"kpts_density": 25.0, "energy_ev": -22.03, "gap_ev": 1.674, "kpts": [9, 9, 1]},
    {"kpts_density": 35.0, "energy_ev": -22.0306, "gap_ev": 1.675, "kpts": [12, 12, 1]},
    {"kpts_density": 45.0, "energy_ev": -22.03075, "gap_ev": 1.6752, "kpts": [15, 15, 1]},
]
# ecut sweep: the first point already settles, so the recommendation is 300 eV and every
# instance cutoff (350-500) clears it.
ECUT_SWEEP = [
    {"ecut_ev": 300, "energy_ev": -22.1, "gap_ev": 1.674},
    {"ecut_ev": 400, "energy_ev": -22.07, "gap_ev": 1.6754},
    {"ecut_ev": 500, "energy_ev": -22.06, "gap_ev": 1.6756},
    {"ecut_ev": 600, "energy_ev": -22.055, "gap_ev": 1.6757},
    {"ecut_ev": 800, "energy_ev": -22.0545, "gap_ev": 1.67575},
]


def _point(ecut: int, kd: float) -> dict:
    """One grid point: the cutoff moves the energies, the density the step count."""
    return {
        "relax_total_energy_ev": -22.0 - ecut / 1000 - kd / 100000,
        "relax_max_force_ev_per_a": 0.0066 + ecut / 1e7,
        "relax_n_steps": 3 if kd >= 25 else 4,
        "relax_kpts": [9, 9, 1] if kd >= 25 else [6, 6, 1],
        "scf_total_energy_ev": -22.07 - ecut / 1000 - kd / 100000,
        "fermi_ev": -1.55 - ecut / 100000,
        # |gap - 1.66| is about 0.0154 eV: a clear fail at gap_tol 0.005, warn at
        # 0.012 and pass at 0.3, every branch far from its boundary.
        "band_gap_ev": 1.6754 + ecut / 1e7,
        # Constant over the real grid too: a relaxed 2H-MoS2 monolayer has a direct
        # K->K gap, which is why the bands scorer carries only 10 of 100 points.
        "gap_type": "direct",
        "vbm_label": "K",
        "cbm_label": "K",
        "kpts_scf": [9, 9, 1] if kd >= 25 else [6, 6, 1],
        "band_path": "GMKG",
        # The chain leaves more than REQUIRED_RUN_ARTIFACTS: GPAW writes a text log
        # per calculation, and the fixed convergence sweep runs ten of them.
        "artifact_count": 24,
        "measured_at_tol_mev_per_atom": 5.0,
        "measured_recommended_ecut_ev": 300,
        "measured_recommended_kpts_density": 15.0,
        "measured_converged": True,
        "measured_params_verified": True,
        "measured_band_gap_ref": 1.66,
        "measured_verdict_by_gap_tol": {"0.005": "fail", "0.3": "pass"},
        "measured_verify_labels_by_gap_tol": {
            "0.005": ["convergence_gate:pass", "relax_convergence:pass", "structure_drift:pass",
                      "band_gap_vs_mp:fail", "gap_character:info"],
            "0.3": ["convergence_gate:pass", "relax_convergence:pass", "structure_drift:pass",
                    "band_gap_vs_mp:pass", "gap_character:info"]},
        "ecut_sweep": ECUT_SWEEP,
        "kpts_sweep": KPTS_SWEEP,
    }


SAMPLE_MEASURED = {(ecut, kd): _point(ecut, kd)
                   for ecut in generate_gt.ECUTS for kd in generate_gt.KPTS_DENSITIES}

# Instance 31415: ecut 450, density 25, tolerance 5.0 meV/atom, gap tolerance 0.3 eV —
# the all-clear branch (gate converged, parameters verified, verdict "pass").
CASE = generate_gt.build_case(support.SEED)


# The real builder, captured before the autouse fixture replaces it.
REAL_STRUCTURE = generate_gt.mos2_structure


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Every test runs against the synthetic table and the recorded cell, so the module
    neither measures DFT nor needs ASE. test_mx2_structure_matches_the_hardcoded_cell is
    what ties CELL back to ase.build.mx2, and the generator test puts the real builder
    back for one genuine end-to-end run."""
    monkeypatch.setattr(generate_gt, "MEASURED", SAMPLE_MEASURED)
    monkeypatch.setattr(generate_gt, "mos2_structure", lambda: CELL)


def _reference(case=None) -> dict:
    case = case or CASE
    return {**case, **generate_gt.reference(case)}


def _answer(reference=None, **over) -> dict:
    ref = reference or REFERENCE
    data = {key: ref[key] for key in
            ("relax_total_energy_ev", "relax_max_force_ev_per_a", "relax_n_steps", "relax_kpts",
             "scf_total_energy_ev", "fermi_ev", "band_gap_ev", "gap_type", "vbm_label",
             "cbm_label", "params_verified", "recommended_ecut_ev", "recommended_kpts_density",
             "converged", "verdict", "artifact_count")}
    data.update(over)
    return data


_ANSWER_TEMPLATE: dict = {}


def _png(width=640, height=480) -> bytes:
    """A PNG header the scorer accepts, padded past its minimum size."""
    return support.png_bytes(width, height) + b"\0" * scorer.PNG_MIN_BYTES


def _dirs(tmp_path, prediction, reference=_OMIT, figures=None):
    pred, ref = support.score_dirs(tmp_path, prediction,
                                   REFERENCE if reference is _OMIT else reference)
    for name, blob in (figures if figures is not None else
                       {"bands.png": _png(), "dos.png": _png()}).items():
        (pred / name).write_bytes(blob)
    return pred, ref


def _total(pred, ref) -> float:
    return TASK.total(pred, ref)


# The reference of instance 31415 under SAMPLE_MEASURED, used by the scorer and
# verifier scenarios. Built once here rather than in every test.
def _build_reference() -> dict:
    saved = generate_gt.MEASURED, generate_gt.mos2_structure
    generate_gt.MEASURED, generate_gt.mos2_structure = SAMPLE_MEASURED, lambda: CELL
    try:
        return {**CASE, **generate_gt.reference(CASE)}
    finally:
        generate_gt.MEASURED, generate_gt.mos2_structure = saved


REFERENCE = _build_reference()
_ANSWER_TEMPLATE.update(_answer())
RUN_ID = "20260103-101500_mos2_1a2b"


# --------------------------------------------------------------------------
# The instance grid
# --------------------------------------------------------------------------

def test_cases_are_deterministic_and_varied():
    assert generate_gt.build_case(support.SEED) == generate_gt.build_case(support.SEED)
    cases = [generate_gt.build_case(seed) for seed in range(300)]
    assert {c["ecut"] for c in cases} == set(generate_gt.ECUTS)
    assert {c["kpts_density"] for c in cases} == set(generate_gt.KPTS_DENSITIES)
    assert {c["tol_mev_per_atom"] for c in cases} == set(generate_gt.TOL_MEV_PER_ATOM)
    assert {c["gap_tol_ev"] for c in cases} == set(generate_gt.GAP_TOL_EV)
    # Only (ecut, density) costs a measurement; the other two are free, so the
    # instance space is much larger than the measured grid.
    assert len({(c["ecut"], c["kpts_density"]) for c in cases}) == 6
    assert len({tuple(c[k] for k in ("ecut", "kpts_density", "tol_mev_per_atom",
                                     "gap_tol_ev")) for c in cases}) == 3 * 2 * 2 * 2
    # 350 eV was measured and dropped: its relaxation stops on the force threshold.
    assert 350 not in generate_gt.ECUTS
    assert all(c["use_builtin"] is True and c["query"] == "MoS2" for c in cases)
    assert CASE["ecut"] == 500 and CASE["kpts_density"] == 25.0


# --------------------------------------------------------------------------
# The policy functions the generator reimplements
# --------------------------------------------------------------------------

def test_mx2_structure_matches_the_hardcoded_cell():
    """Needs ASE (the version the task's runtime pins). Everything else here works from
    CELL, so the rest of the module runs without ASE installed."""
    pytest.importorskip("ase")
    built = REAL_STRUCTURE()
    assert built["symbols"] == CELL["symbols"]
    assert built["lengths"] == pytest.approx(CELL["lengths"], abs=1e-12)
    assert built["angles"] == pytest.approx(CELL["angles"], abs=1e-12)
    for got, want in zip(built["scaled"], CELL["scaled"]):
        assert got == pytest.approx(want, abs=1e-9)


def test_auto_kpts_follows_the_hexagonal_policy():
    kpts = lambda density: generate_gt.auto_kpts(  # noqa: E731
        CELL["lengths"], CELL["angles"], CELL["scaled"], density)
    # 15/3.18 = 4.7 -> 5 -> rounded up to a multiple of three; 25/3.18 = 7.9 -> 8 -> 9.
    assert kpts(15.0) == (6, 6, 1)
    assert kpts(25.0) == (9, 9, 1)
    assert kpts(10.0) == (3, 3, 1) and kpts(45.0) == (15, 15, 1)
    # 20 A of vacuum along c, 2.1 A of atom span in plane.
    assert generate_gt.vacuum_axes(CELL["lengths"], CELL["scaled"]) == (False, False, True)
    # A cubic cell with atoms everywhere is not a slab and is not rounded to threes.
    cubic = ([4.0, 4.0, 4.0], [90.0, 90.0, 90.0], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    assert generate_gt.vacuum_axes(cubic[0], cubic[2]) == (False, False, False)
    assert generate_gt.auto_kpts(*cubic, 20.0) == (5, 5, 5)


def test_convergence_policy_recomputes_the_recommendation():
    at = lambda tol: generate_gt.convergence_policy(ECUT_SWEEP, KPTS_SWEEP, tol)  # noqa: E731
    loose, strict = at(5.0), at(0.3)
    assert loose["recommended_kpts_density"] == 15.0 and loose["converged"] is True
    assert strict["recommended_kpts_density"] == 25.0 and strict["converged"] is True
    assert loose["recommended_ecut_ev"] == strict["recommended_ecut_ev"] == 300
    # A tolerance no point meets falls back to the ceiling of the sweep, not converged.
    tight = at(1e-6)
    assert tight["recommended_kpts_density"] == 45.0 and tight["converged"] is False
    assert tight["kpts_converged"] is False and tight["ecut_converged"] is True
    # The deltas are |dE| per atom in meV and |dgap| in eV to the next point.
    deltas = generate_gt.sweep_deltas(KPTS_SWEEP)
    assert deltas[0]["delta_mev_per_atom"] == pytest.approx(8.0)
    assert deltas[0]["delta_gap_ev"] == pytest.approx(0.03)
    assert deltas[-1] == {"delta_mev_per_atom": None, "delta_gap_ev": None}


def test_params_verified_needs_a_passed_gate_and_large_enough_parameters():
    gate = generate_gt.convergence_policy(ECUT_SWEEP, KPTS_SWEEP, 0.3)   # recommends 25
    assert generate_gt.params_verified(gate, 450, 25.0) is True
    assert generate_gt.params_verified(gate, 450, 15.0) is False         # density too coarse
    assert generate_gt.params_verified(gate, 200, 25.0) is False         # cutoff below 300
    assert generate_gt.params_verified({**gate, "converged": False}, 450, 25.0) is False


@pytest.mark.parametrize("gap_tol,verdict,gap_status", [
    (0.3, "pass", "pass"), (0.012, "pass_with_warnings", "warn"), (0.005, "fail", "fail")])
# 0.012 is not in GAP_TOL_EV (its warn band is too narrow to survive a GPAW release);
# the state machine still has to implement the branch, so it is exercised here.
def test_verify_state_machine_branches_on_the_gap_tolerance(gap_tol, verdict, gap_status):
    out = generate_gt.verify_checks(
        gate_converged=True, relax_converged=True, max_force=0.0066, drift_a=0.0,
        band_gap_ev=1.6754, band_gap_ref=1.66, gap_type="direct", gap_tol_ev=gap_tol)
    assert out["verdict"] == verdict
    assert dict((c["check"], c["status"]) for c in out["checks"])["band_gap_vs_mp"] == gap_status
    assert [c["check"] for c in out["checks"]] == [
        "convergence_gate", "relax_convergence", "structure_drift", "band_gap_vs_mp",
        "gap_character"]
    assert out["blocking_failures"] == (["band_gap_vs_mp"] if verdict == "fail" else [])


def test_verify_state_machine_warns_on_an_unconverged_gate_and_fails_on_a_bad_relaxation():
    warned = generate_gt.verify_checks(
        gate_converged=False, relax_converged=True, max_force=0.0066, drift_a=0.0,
        band_gap_ev=1.6754, band_gap_ref=1.66, gap_type="direct", gap_tol_ev=0.3)
    assert warned["verdict"] == "pass_with_warnings"
    failed = generate_gt.verify_checks(
        gate_converged=True, relax_converged=False, max_force=0.9, drift_a=0.0,
        band_gap_ev=1.6754, band_gap_ref=1.66, gap_type="direct", gap_tol_ev=0.3)
    assert failed["verdict"] == "fail" and failed["blocking_failures"] == ["relax_convergence"]
    # Without a Materials Project reference the gap check is skipped, not failed.
    skipped = generate_gt.verify_checks(
        gate_converged=True, relax_converged=True, max_force=0.0066, drift_a=0.0,
        band_gap_ev=1.6754, band_gap_ref=None, gap_type="direct", gap_tol_ev=0.3)
    assert skipped["verdict"] == "pass"


def test_a_gap_on_a_branch_boundary_is_refused(monkeypatch):
    """A verdict that a 2-3 meV gpaw release drift would flip is not a ground truth."""
    assert 0.012 not in generate_gt.GAP_TOL_EV          # the warn band is only ~6 meV wide
    assert generate_gt.gap_branch_margin(1.6754, 1.66, 0.012) < generate_gt.GAP_BRANCH_MARGIN_EV
    for gap_tol in generate_gt.GAP_TOL_EV:
        assert generate_gt.gap_branch_margin(1.6754, 1.66, gap_tol) \
            > generate_gt.GAP_BRANCH_MARGIN_EV
    on_edge = {point: {**data, "band_gap_ev": generate_gt.BAND_GAP_REF + CASE["gap_tol_ev"]}
               for point, data in SAMPLE_MEASURED.items()}       # |gap - ref| == gap_tol
    monkeypatch.setattr(generate_gt, "MEASURED", on_edge)
    with pytest.raises(generate_gt.GenerationError, match="branch boundary"):
        generate_gt.reference(CASE)


# --------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------

def test_reference_is_derived_from_the_measured_table():
    measured = SAMPLE_MEASURED[(CASE["ecut"], CASE["kpts_density"])]
    for key in ("relax_total_energy_ev", "relax_max_force_ev_per_a", "relax_n_steps",
                "scf_total_energy_ev", "fermi_ev", "band_gap_ev", "gap_type", "vbm_label",
                "cbm_label", "band_path", "artifact_count"):
        assert REFERENCE[key] == measured[key], key
    # derived, not measured
    assert REFERENCE["recommended_kpts_density"] == 25.0      # tolerance 0.3 meV/atom
    assert REFERENCE["converged"] is True and REFERENCE["params_verified"] is True
    assert REFERENCE["verdict"] == "fail"                     # gap tolerance 0.005 eV
    assert REFERENCE["verify_check_labels"][0] == "convergence_gate:pass"
    assert REFERENCE["run_artifact_names"] == list(generate_gt.REQUIRED_RUN_ARTIFACTS)
    # The listing is longer than the required set, so the verifier matches a superset.
    assert len(REFERENCE["run_artifact_names"]) < REFERENCE["artifact_count"]


def test_a_strict_tolerance_can_leave_the_parameters_unverified():
    case = {**CASE, "kpts_density": 15.0, "tol_mev_per_atom": 0.3}
    ref = _reference(case)
    assert ref["recommended_kpts_density"] == 25.0
    assert ref["params_verified"] is False and ref["converged"] is True
    # params_verified is the run's own flag, not one of verify_run's checks.
    assert ref["verdict"] == _reference({**case, "kpts_density": 25.0})["verdict"]


def test_an_unmeasured_grid_point_fails_loudly(monkeypatch):
    monkeypatch.setattr(generate_gt, "MEASURED",
                        {k: v for k, v in SAMPLE_MEASURED.items() if k != (500, 25.0)})
    with pytest.raises(generate_gt.GenerationError, match="no measured DFT"):
        generate_gt.reference(CASE)


def test_a_table_disagreeing_with_the_k_grid_policy_fails_loudly(monkeypatch):
    point = {**SAMPLE_MEASURED[(500, 25.0)], "relax_kpts": [7, 7, 1]}
    monkeypatch.setattr(generate_gt, "MEASURED", {**SAMPLE_MEASURED, (500, 25.0): point})
    with pytest.raises(generate_gt.GenerationError, match="auto_kpts"):
        generate_gt.reference(CASE)


def test_generate_writes_the_instance(tmp_path, monkeypatch):
    """The one end-to-end generation, with the real ASE builder rather than CELL."""
    pytest.importorskip("ase")
    monkeypatch.setattr(generate_gt, "mos2_structure", REAL_STRUCTURE)
    meta = generate_gt.generate(tmp_path, {"seed": support.SEED})
    assert meta["input_files"] == ["calculation.json"]
    data = json.loads((tmp_path / "data/calculation.json").read_text())
    assert data["ecut"] == 500 and data["copied_files"] == ["bands.png", "dos.png"]
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    assert ref["band_gap_ev"] == REFERENCE["band_gap_ev"]
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{" not in text and "bands.png, dos.png" in text
        # B4 names no parameter: it points at the file instead.
        assert ("500" in text) == (level != "b4"), level


# --------------------------------------------------------------------------
# The real measured table (skipped while it is still being measured)
# --------------------------------------------------------------------------

GRID = {(ecut, kd) for ecut in generate_gt.ECUTS for kd in generate_gt.KPTS_DENSITIES}


def test_measured_table_is_either_absent_or_complete():
    """A partly measured table would silently restrict which seeds can be generated."""
    assert set(SAMPLE_MEASURED) == GRID
    assert not REAL_MEASURED or set(REAL_MEASURED) == GRID


def test_real_table_agrees_with_the_derivations_the_generator_makes():
    """The server answered the gate itself at the tolerance the measurement used; the
    pure functions here must reproduce that, or the derived instances are fiction."""
    if not REAL_MEASURED:
        pytest.skip("MEASURED is still empty; run scripts/mcp/e2e/measure_gpaw_table.py")
    for point, data in sorted(REAL_MEASURED.items()):
        gate = generate_gt.convergence_policy(data["ecut_sweep"], data["kpts_sweep"],
                                              data["measured_at_tol_mev_per_atom"])
        assert gate["recommended_ecut_ev"] == data["measured_recommended_ecut_ev"], point
        assert gate["recommended_kpts_density"] == data["measured_recommended_kpts_density"], point
        assert gate["converged"] is data["measured_converged"], point
        assert generate_gt.params_verified(gate, point[0], point[1]) \
            is data["measured_params_verified"], point
        assert len(generate_gt.REQUIRED_RUN_ARTIFACTS) <= data["artifact_count"], point
        assert data["measured_band_gap_ref"] == generate_gt.BAND_GAP_REF, point
        for gap_tol in generate_gt.GAP_TOL_EV:
            derived = generate_gt.verify_checks(
                gate_converged=gate["converged"], relax_converged=True,
                max_force=data["relax_max_force_ev_per_a"], drift_a=0.0,
                band_gap_ev=data["band_gap_ev"], band_gap_ref=generate_gt.BAND_GAP_REF,
                gap_type=data["gap_type"], gap_tol_ev=gap_tol)
            key = repr(gap_tol)
            assert derived["verdict"] == data["measured_verdict_by_gap_tol"][key], (point, gap_tol)
            assert [f"{c['check']}:{c['status']}" for c in derived["checks"]] \
                == data["measured_verify_labels_by_gap_tol"][key], (point, gap_tol)


def test_real_table_keeps_every_instance_off_a_verify_branch_boundary():
    if not REAL_MEASURED:
        pytest.skip("MEASURED is still empty; run scripts/mcp/e2e/measure_gpaw_table.py")
    for point, data in sorted(REAL_MEASURED.items()):
        for gap_tol in generate_gt.GAP_TOL_EV:
            margin = generate_gt.gap_branch_margin(data["band_gap_ev"],
                                                   generate_gt.BAND_GAP_REF, gap_tol)
            assert margin >= generate_gt.GAP_BRANCH_MARGIN_EV, (point, gap_tol, margin)


# --------------------------------------------------------------------------
# Task files
# --------------------------------------------------------------------------

def test_reference_carries_every_key_the_checks_use():
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    assert keys <= set(REFERENCE), sorted(keys - set(REFERENCE))
    assert spec["server_tools"] == TOOLS
    # Every later call is linked to the run the first one opened.
    linked = {cs["name"]: cs["inputs_from_call"] for cs in spec["calls"] if cs.get("inputs_from_call")}
    assert set(linked) == {"relax", "convergence", "bands", "verify", "artifacts"}
    assert all(link == {"call": "fetch", "map": {"run_id": "run_id"}} for link in linked.values())
    # The one-call shortcut is a bypass, not merely suspicious.
    assert spec["bypass_tools"] == {"mcp__gpaw__run_verified_workflow": ""}
    # Locating the run directory puts the server's path into a shell command, so a bare
    # "gpaw" must not be a pattern (see generate_gt.py's module docstring).
    import re
    for pattern in spec["bypass_patterns"] + spec["suspicious_patterns"]:
        for allowed in ("find $HOME -type d -name 20260103-101500_mos2_1a2b",
                        "cp ~/mcp/gpaw/runs/20260103-101500_mos2_1a2b/bands.png .",
                        "ls ~/mcp/gpaw/runs"):
            assert not re.search(pattern, allowed), (pattern, allowed)
    # Every argument the prompts pin is checked against the reference.
    inputs = {cs["name"]: set(cs.get("inputs_from_reference", {})) for cs in spec["calls"]}
    assert inputs["relax"] == {"ecut", "kpts_density", "fmax", "max_steps", "engine"}
    assert inputs["bands"] == {"ecut", "kpts_density", "npoints", "window_ev", "engine"}
    assert inputs["convergence"] == {"tol_mev_per_atom", "engine"}


def test_prompts_name_the_server_only_at_b1_b2_and_keep_the_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert ("`gpaw` MCP server" in text) == (level in ("b1", "b2"))
        assert ("`fetch_structure` tool" in text) == (level == "b1")
        # The forbidden shortcut and the ordering rule bind at every level.
        assert "`run_verified_workflow`" in text
        assert "convergence check before the band-structure calculation" in text
        assert "no `import gpaw`" in text and "Do not modify `data/calculation.json`" in text
        assert "full precision" in text and "one tool call at a time on a single run" in text
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text


def test_b3_and_b4_differ_only_in_the_parameters_they_name():
    """b4 drops the parameter line; the background and the whole contract stay the same."""
    b3 = (TASK_DIR / "prompt_b3.md").read_text()
    b4 = (TASK_DIR / "prompt_b4.md").read_text()
    background = lambda text: [l for l in text.splitlines() if l.startswith("Background:")]  # noqa: E731
    assert background(b3) == background(b4) and background(b3)
    tail = lambda text: text[text.index("## Output"):]                                       # noqa: E731
    assert tail(b3) == tail(b4)
    # b3 states the parameters, b4 points at the file instead.
    assert "{{ecut}}" in b3 and "{{" + "ecut}}" not in b4


def test_task_meta_pins_the_runtime_to_ase():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == ["ase==3.29.0"]
    assert [f["name"] for f in meta["output"]["files"]] == ["result.json", "bands.png", "dos.png"]


def test_scoring_weights_put_the_cutoff_dependent_numbers_above_the_band_gap():
    scoring = {item["scorer"]: item["weight"] for item in TASK.eval_config()["scoring"]}
    assert sum(scoring.values()) == 100
    assert scoring["gpaw_e2e_energies"] > scoring["gpaw_e2e_bands"]


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def test_tool_values_score_full(tmp_path):
    assert _total(*_dirs(tmp_path, _answer())) == pytest.approx(100.0)


def test_a_neighbouring_cutoff_scores_zero_on_the_energies(tmp_path):
    """Reading another instance's run is the realistic way to be wrong: the gap barely
    moves, the total energies move by ~1e-2 relative, far outside the credit range."""
    other = SAMPLE_MEASURED[(400, 25.0)]
    wrong = _answer(relax_total_energy_ev=other["relax_total_energy_ev"],
                    scf_total_energy_ev=other["scf_total_energy_ev"],
                    fermi_ev=other["fermi_ev"])
    detail = _scored(tmp_path, wrong, "gpaw_e2e_energies")
    assert detail.score == 0.0 and not detail.passed


def test_rounding_the_energies_loses_credit(tmp_path):
    rounded = _answer(scf_total_energy_ev=round(REFERENCE["scf_total_energy_ev"], 4))
    detail = _scored(tmp_path, rounded, "gpaw_e2e_energies")
    assert 0.0 < detail.score < 45.0


def test_recalling_the_band_gap_is_worth_little(tmp_path):
    """~1.67 eV for a PBE MoS2 monolayer is a published number, so the band-gap scorer
    is the one a model can partly reach without calling anything. It is 20 of 100
    points and the recalled value is not even inside its credit range; the
    cutoff-dependent energies, worth 35, go to zero."""
    recalled = _answer(relax_total_energy_ev=-22.0, scf_total_energy_ev=-22.0,
                       fermi_ev=-1.5, band_gap_ev=1.67)
    assert _scored(tmp_path, recalled, "gpaw_e2e_energies").score == 0.0
    bands = _scored(tmp_path / "b", recalled, "gpaw_e2e_bands")
    assert bands.score == pytest.approx(10.0 * 3 / 4)       # the three labels, not the number


def test_the_convergence_gate_is_all_or_nothing_per_field(tmp_path):
    wrong_verdict = "pass" if REFERENCE["verdict"] != "pass" else "fail"
    detail = _scored(tmp_path, _answer(verdict=wrong_verdict, converged=False),
                     "gpaw_e2e_convergence")
    assert detail.score == pytest.approx(20.0 * 3 / 5)
    assert sorted(detail.details["wrong"]) == ["converged", "verdict"]


def test_figures_must_be_real_pngs(tmp_path):
    small = {"bands.png": support.png_bytes(640, 480), "dos.png": _png()}
    detail = _scored(tmp_path, _answer(), "gpaw_e2e_artifacts", figures=small)
    assert detail.score == pytest.approx(10.0 * 2 / 3)
    assert "too small" in " ".join(detail.details["problems"])
    text = {"bands.png": b"not a png at all", "dos.png": _png()}
    detail = _scored(tmp_path / "b", _answer(), "gpaw_e2e_artifacts", figures=text)
    assert "not a PNG" in " ".join(detail.details["problems"])
    tiny = {"bands.png": support.png_bytes(80, 60) + b"\0" * scorer.PNG_MIN_BYTES, "dos.png": _png()}
    detail = _scored(tmp_path / "c", _answer(), "gpaw_e2e_artifacts", figures=tiny)
    assert "80x60" in " ".join(detail.details["problems"])


def test_png_header_reader():
    assert scorer.png_size(_png(640, 480)) == (640, 480)
    for bad in (b"", b"\x89PNG\r\n\x1a\n", b"x" * 40):
        with pytest.raises(scorer._PredictionError, match="not a PNG"):
            scorer.png_size(bad)


def test_credit_curve():
    assert scorer.credit(1e-9, 1e-6, 1e-4) == 1.0
    assert scorer.credit(1e-4, 1e-6, 1e-4) == 0.0
    assert 0.0 < scorer.credit(1e-5, 1e-6, 1e-4) < 1.0
    assert scorer.credit(float("inf"), 1e-6, 1e-4) == 0.0


def _scored(tmp_path, prediction, name, figures=None, reference=_OMIT):
    from ai4sci_bench.core.scorer import get_scorer
    item = next(i for i in TASK.eval_config()["scoring"] if i["scorer"] == name)
    pred, ref = _dirs(tmp_path, prediction, reference=reference, figures=figures)
    return get_scorer(name).score(pred, ref, {**item["config"], "weight": item["weight"]})


@pytest.mark.parametrize("prediction,figures", [
    (None, None),                                       # no result.json
    ("not json", None),
    ([], None),                                         # JSON, but not an object
    ({"band_gap_ev": 1.0}, None),                       # incomplete
    (dict(_ANSWER_TEMPLATE, relax_total_energy_ev="low"), None),
    (dict(_ANSWER_TEMPLATE, relax_n_steps=-1), None),
    (dict(_ANSWER_TEMPLATE, params_verified="yes"), None),
    (dict(_ANSWER_TEMPLATE, relax_kpts=[9, 9]), None),
    (dict(_ANSWER_TEMPLATE, band_gap_ev=float("nan")), None),
    (None, {}),                                         # nothing submitted at all
])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction, figures):
    """A malformed submission is an ordinary zero for every scorer that reads it, never
    an evaluator failure; the hard gate zeroes the instance as a whole."""
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, prediction, figures=figures)
    gate = get_scorer("gpaw_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in TASK.eval_config()["scoring"]:   # noqa: B007
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"],
                                                              "weight": item["weight"]})
        assert detail.score == 0.0, (item["scorer"], detail.message)
        assert not detail.details.get("scorer_internal_error"), item["scorer"]


def test_missing_figures_cost_only_the_artefact_points(tmp_path):
    """The gate is schema-only: an agent that drove the chain but never found the
    server's run directory loses ten points, not the instance."""
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, _answer(), figures={})
    gate = get_scorer("gpaw_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 1.0 and gate.passed
    assert _scored(tmp_path / "b", _answer(), "gpaw_e2e_energies", figures={}).score \
        == pytest.approx(45.0)
    artefacts = _scored(tmp_path / "c", _answer(), "gpaw_e2e_artifacts", figures={})
    assert artefacts.score == pytest.approx(10.0 / 3)      # artifact_count only
    assert _total(*_dirs(tmp_path / "d", _answer(), figures={})) == pytest.approx(100.0 - 2 * 10 / 3)


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, _answer(), reference=None)
    for item in TASK.eval_config()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"],
                                                              "weight": item["weight"]})
        assert detail.details["scorer_internal_error"]
        assert detail.details["failure_kind"] == "missing_evaluator_input"


# --------------------------------------------------------------------------
# Verifier scenarios
# --------------------------------------------------------------------------

def _payloads(run_id=RUN_ID, reference=None):
    """What the pinned server returns for this instance's chain."""
    ref = reference or REFERENCE
    return {
        "fetch": {"ok": True, "run_id": run_id, "source": "ase-mx2-builtin",
                  "formula": "MoS2 (monolayer 2H)", "n_atoms": 3, "mp_id": None,
                  "band_gap_ref": ref["band_gap_ref"], "query": ref["query"],
                  "cif": f"runs/{run_id}/structure.cif"},
        "relax": {"ok": True, "kpts": ref["relax_kpts"], "engine": "gpaw-pw",
                  "ecut_ev": ref["ecut"], "converged": True,
                  "n_steps": ref["relax_n_steps"],
                  "total_energy_ev": ref["relax_total_energy_ev"],
                  "max_force_ev_per_a": ref["relax_max_force_ev_per_a"]},
        "convergence": {"ok": True, "ecut_sweep": ECUT_SWEEP, "kpts_sweep": KPTS_SWEEP,
                        "recommended_ecut_ev": ref["recommended_ecut_ev"],
                        "recommended_kpts_density": ref["recommended_kpts_density"],
                        "ecut_converged": True, "kpts_converged": True,
                        "converged": ref["converged"], "tol_gap_ev": 0.02},
        "bands": {"ok": True, "kpts_scf": ref["relax_kpts"],
                  "total_energy_ev": ref["scf_total_energy_ev"], "fermi_ev": ref["fermi_ev"],
                  "band_gap_ev": ref["band_gap_ev"], "gap_type": ref["gap_type"],
                  "vbm": {"label": ref["vbm_label"], "band": 12, "energy_ev": -2.0},
                  "cbm": {"label": ref["cbm_label"], "band": 13, "energy_ev": -0.3},
                  "band_path": ref["band_path"], "params_verified": ref["params_verified"],
                  "bands_png": f"runs/{run_id}/bands.png", "dos_png": f"runs/{run_id}/dos.png"},
        "verify": {"ok": True, "verdict": ref["verdict"], "checks": ref["verify_checks"],
                   "blocking_failures": ref["blocking_failures"]},
        # The real listing carries GPAW's per-calculation text logs on top of the
        # artefacts the chain is required to leave, which is why the spec matches a superset.
        "artifacts": {"ok": True, "artifacts": [
            {"path": f"runs/{run_id}/{name}", "bytes": 1024}
            for name in [*ref["run_artifact_names"],
                         *(f"scf_{n}.txt" for n in range(ref["artifact_count"]
                                                        - len(ref["run_artifact_names"])))]]},
    }


def _calls(run_id=RUN_ID, reference=None, payloads=None, chain_run_id=None):
    """The B1 tool calls of this instance. ``chain_run_id`` breaks the chain."""
    ref = reference or REFERENCE
    data = payloads or _payloads(run_id, ref)
    passed = chain_run_id or run_id
    return [
        ("fetch_structure", {"query": ref["query"], "use_builtin": True}, data["fetch"]),
        ("relax_structure", {"run_id": passed, "ecut": ref["ecut"],
                             "kpts_density": ref["kpts_density"], "fmax": ref["fmax"],
                             "max_steps": ref["max_steps"], "engine": "gpaw"}, data["relax"]),
        ("check_convergence", {"run_id": passed, "tol_mev_per_atom": ref["tol_mev_per_atom"],
                               "engine": "gpaw"}, data["convergence"]),
        ("calc_band_dos", {"run_id": passed, "ecut": ref["ecut"],
                           "kpts_density": ref["kpts_density"], "npoints": ref["npoints"],
                           "window_ev": ref["window_ev"], "engine": "gpaw"}, data["bands"]),
        ("verify_run", {"run_id": passed, "gap_tol_ev": ref["gap_tol_ev"]}, data["verify"]),
        ("get_run_artifacts", {"run_id": passed}, data["artifacts"]),
    ]


def _stream(calls, extra=()):
    events = [claude.init(SERVER, ["Bash", "Read", "Write", "WebFetch", "WebSearch",
                                   *(f"mcp__{SERVER}__{t}" for t in TOOLS)])]
    for index, (name, arguments) in enumerate(extra):
        events += claude.call(f"x{index}", name, arguments, "ok")
    for index, (tool, arguments, payload) in enumerate(calls):
        events += claude.call(f"c{index}", f"mcp__{SERVER}__{tool}", arguments,
                              json.dumps(payload))
    events.append(claude.result(len(calls)))
    return jsonl(events)


def _codex_stream(calls):
    events = list(codex.START)
    for index, (tool, arguments, payload) in enumerate(calls):
        events += codex.mcp(f"m{index}", SERVER, tool, arguments, json.dumps(payload, indent=2),
                            structured=payload)
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream, answer, *, harness="claude", files=None, reference=None):
    figures = {"bands.png": _png(), "dos.png": _png()}
    return TASK.verify(tmp_path, stream, reference=reference or REFERENCE, answer=answer,
                       harness=harness, files=files if files is not None else figures)


def test_genuine_b1_run_passes_every_check(tmp_path):
    row = _run(tmp_path, _stream(_calls()), _answer())
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert row["tool_call_counts"][f"mcp__{SERVER}__calc_band_dos"] == 1


def test_codex_run_passes(tmp_path):
    row = _run(tmp_path, _codex_stream(_calls()), _answer(), harness="codex")
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


def test_every_later_call_must_carry_the_run_the_first_one_opened(tmp_path):
    """A second fetch_structure gives a second run_id; mixing runs is the realistic way
    to produce numbers that look right but describe a different calculation."""
    row = _run(tmp_path, _stream(_calls(chain_run_id="20260103-101501_mos2_9z9z")), _answer())
    assert _status(row)["tool_chain"] == "FAIL"
    assert row["verdict"] == "FAIL"


def test_the_one_call_workflow_shortcut_is_a_bypass(tmp_path):
    shortcut = [(f"mcp__{SERVER}__run_verified_workflow",
                 {"query": "MoS2", "use_builtin": True, "ecut": 450, "kpts_density": 25.0})]
    row = _run(tmp_path, _stream(_calls(), extra=shortcut), _answer())
    assert _status(row)["no_bypass"] == "FAIL"
    assert "run_verified_workflow" in row["checks"]["no_bypass"]["detail"]
    assert row["verdict"] == "FAIL"


def test_answers_must_be_copied_from_the_tool_results(tmp_path):
    answer = _answer(band_gap_ev=REFERENCE["band_gap_ev"] + 1e-9)
    row = _run(tmp_path, _stream(_calls()), answer)
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "band_gap_ev" in row["checks"]["answer_from_tool"]["detail"]


def test_the_recommendation_must_come_from_the_convergence_sweep(tmp_path):
    other = 15.0 if REFERENCE["recommended_kpts_density"] != 15.0 else 25.0
    row = _run(tmp_path, _stream(_calls()), _answer(recommended_kpts_density=other))
    assert _status(row)["answer_from_tool"] == "FAIL"


def test_the_realized_grid_must_come_from_the_relaxation(tmp_path):
    """A plausible guess: the requested density, not the grid the server realized."""
    row = _run(tmp_path, _stream(_calls()), _answer(relax_kpts=[8, 8, 1]))
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "relax_kpts" in row["checks"]["answer_from_tool"]["detail"]


def test_a_listing_missing_a_chain_artefact_fails_the_artefact_call(tmp_path):
    """A superset match still requires every artefact the chain must have produced."""
    payloads = _payloads()
    payloads["artifacts"] = {"ok": True, "artifacts": [{"path": "runs/x/bands.png", "bytes": 1}]}
    row = _run(tmp_path, _stream(_calls(payloads=payloads)), _answer())
    assert _status(row)["tool_correct"] == "FAIL"
    assert row["checks"]["tool_correct"]["per_call"]["artifacts"]["status"] == "FAIL"


def test_extra_files_in_the_listing_are_fine(tmp_path):
    """GPAW's text logs are not in REQUIRED_RUN_ARTIFACTS and must not fail the call."""
    row = _run(tmp_path, _stream(_calls()), _answer())
    assert row["checks"]["tool_correct"]["per_call"]["artifacts"]["status"] == "PASS"
    listing = json.loads((TASK_DIR / "e2e_check.json").read_text())
    assert next(c for c in listing["calls"]
                if c["name"] == "artifacts")["result"]["match"] == "superset"


def test_a_different_verification_report_fails_the_verify_call(tmp_path):
    payloads = _payloads()
    payloads["verify"] = {"ok": True, "verdict": "pass", "blocking_failures": [],
                          "checks": [{"check": "convergence_gate", "status": "pass"}]}
    row = _run(tmp_path, _stream(_calls(payloads=payloads)), _answer())
    assert row["checks"]["tool_correct"]["per_call"]["verify"]["status"] == "FAIL"


@pytest.mark.parametrize("command,status", [
    ("python3 -c 'import gpaw'", "FAIL"),
    ("python3 -c 'from ase.build import mx2'", "FAIL"),
    ("pip install gpaw", "FAIL"),
    ("conda install -c conda-forge gpaw gpaw-data", "FAIL"),
    ("uv run --with ase python relax.py", "FAIL"),
    ("uvx gpaw info", "FAIL"),
    ("~/mcp/gpaw/.venv/bin/python -c 'print(1)'", "FAIL"),
    ("python3 -m matmcp.server", "FAIL"),
    ("gpaw info", "FAIL"),
    ("pw.x < scf.in", "FAIL"),
    # Locating and copying the figures the server wrote is the task, not a bypass.
    ("find $HOME -maxdepth 5 -type d -name '20260103-101500_mos2_1a2b'", "PASS"),
    ("cp ~/mcp/gpaw/runs/20260103-101500_mos2_1a2b/bands.png .", "PASS"),
    ("ls ~/mcp/gpaw/runs", "PASS"),
    ("python3 -c \"import json; json.dump({}, open('result.json','w'))\"", "PASS"),
])
def test_shell_access_to_the_solver(tmp_path, command, status):
    row = _run(tmp_path, _stream(_calls(), extra=[("Bash", {"command": command})]), _answer())
    assert _status(row)["no_bypass"] == status, row["checks"]["no_bypass"]["detail"]


def test_web_lookup_of_the_band_gap_is_only_suspicious(tmp_path):
    row = _run(tmp_path, _stream(_calls(), extra=[("WebSearch", {"query": "MoS2 monolayer PBE band gap"})]),
               _answer())
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"
    assert "WebSearch" in row["checks"]["no_bypass"]["detail"]
    plain = _run(tmp_path / "b", _stream(_calls(), extra=[("WebFetch", {"url": "https://example.org/news"})]),
                 _answer())
    assert _status(plain)["no_bypass"] == "PASS"


def test_a_produced_script_importing_the_solver_is_a_bypass(tmp_path):
    row = _run(tmp_path, _stream(_calls()), _answer(),
               files={"bands.png": _png(), "dos.png": _png(),
                      "relax.py": "from gpaw import GPAW\nprint(1)\n"})
    assert _status(row)["no_bypass"] == "FAIL"

