"""e2e_smoke/servers/gpaw.py without GPAW: the k-grid policy, the convergence
recommendation, the gap analysis, the verifier state machine and the checks that
do not need a DFT reference.

The sweep rows are the ones the pinned server really returned for the built-in
2H-MoS2 monolayer (gpaw 26.7.0, aarch64), so the recommendation tested here is
the one the smoke reproduces in the live run.
"""
import json
import math

import pytest

from . import support
from .support import StubClient, png_bytes, report_statuses

g = support.smoke_module("gpaw")

# The built-in MoS2 monolayer: a = 3.18 A, c = 3.17 + 2*10 A of vacuum, gamma = 120.
MOS2_LENGTHS = [3.18, 3.18, 23.17]
MOS2_ANGLES = [90.0, 90.0, 120.0]
MOS2_SCALED = [[0.0, 0.0, 0.5],                     # Mo
               [1 / 3, 2 / 3, 0.5 + 3.17 / 2 / 23.17],
               [1 / 3, 2 / 3, 0.5 - 3.17 / 2 / 23.17]]

ECUT_SWEEP = [
    {"ecut_ev": 300, "energy_ev": -21.847853992109695, "gap_ev": 1.6785829310942835},
    {"ecut_ev": 400, "energy_ev": -22.072163530677866, "gap_ev": 1.6738722231343712},
    {"ecut_ev": 500, "energy_ev": -22.110899239393326, "gap_ev": 1.6735787769682646},
    {"ecut_ev": 600, "energy_ev": -22.121391464410276, "gap_ev": 1.6735481730731776},
    {"ecut_ev": 800, "energy_ev": -22.13168110046372, "gap_ev": 1.673554904433174},
]
KPTS_SWEEP = [
    {"kpts_density": 10.0, "kpts": [3, 3, 1], "energy_ev": -22.108757759329208, "gap_ev": 1.6377686737193333},
    {"kpts_density": 15.0, "kpts": [6, 6, 1], "energy_ev": -22.072163530677866, "gap_ev": 1.6738722231343712},
    {"kpts_density": 25.0, "kpts": [9, 9, 1], "energy_ev": -22.073441715568872, "gap_ev": 1.6754359249486146},
    {"kpts_density": 35.0, "kpts": [12, 12, 1], "energy_ev": -22.07348972143735, "gap_ev": 1.6755417555724441},
    {"kpts_density": 45.0, "kpts": [15, 15, 1], "energy_ev": -22.073491628270457, "gap_ev": 1.675549025229646},
]
FETCH = {"run_id": "20261003-083136_mos2_77bb", "query": "MoS2", "source": "ase-mx2-builtin",
         "formula": "MoS2 (monolayer 2H)", "n_atoms": 3, "mp_id": None, "band_gap_ref": 1.66,
         "note": "built-in builder; ref gap from MP mp-1023924"}
RELAX = {"converged": True, "n_steps": 3, "total_energy_ev": -22.07344177707024,
         "max_force_ev_per_a": 0.006676059350922604, "kpts": [9, 9, 1], "ecut_ev": 400}
SUMMARY = {"params_verified": True, "total_energy_ev": -22.073441715568872,
           "fermi_ev": -1.5523279503446843, "band_gap_ev": 1.6754359249486146,
           "gap_type": "direct", "kpts_scf": [9, 9, 1], "ecut_ev": 400, "band_path": "GMKG"}


def rpc(payload):
    """The server's reply shape: one JSON text block plus the same dict as structuredContent."""
    return {"result": {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                       "structuredContent": payload, "isError": False}}


def session_for(responses, tmp_path, *, ref=None, state=None, chatter=0):
    """A Session with a stub client, enough for the checks that need no DFT reference."""
    report = support.runner.Report()
    call = support.runner.Caller(StubClient(responses, chatter=chatter), report)
    session = support.runner.Session(
        g.SMOKE, None, {}, {}, tmp_path, report, tmp_path, tmp_path, tmp_path, tmp_path, {})
    session.call = call
    session.state.update({"runs_dir": tmp_path / "runs", "ref": ref, **(state or {})})
    return session


# --------------------------------------------------------------------------
# k-grid policy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("density,grid", [(10.0, (3, 3, 1)), (15.0, (6, 6, 1)), (20.0, (6, 6, 1)),
                                          (25.0, (9, 9, 1)), (30.0, (9, 9, 1)), (35.0, (12, 12, 1)),
                                          (45.0, (15, 15, 1))])
def test_auto_kpts_hexagonal_slab_rounds_up_to_multiples_of_three(density, grid):
    assert g.auto_kpts_reference(MOS2_LENGTHS, MOS2_ANGLES, MOS2_SCALED, density) == grid


def test_auto_kpts_marks_only_the_vacuum_axis():
    assert g.vacuum_axes_reference(MOS2_LENGTHS, MOS2_SCALED) == (False, False, True)
    # Layered bulk: a long c axis that is filled with atoms is a real periodic direction.
    layered = [[0.0, 0.0, z] for z in (0.0, 0.17, 0.33, 0.5, 0.67, 0.83)]
    assert g.vacuum_axes_reference([2.8, 2.8, 14.0], layered) == (False, False, False)
    # The gap is measured wrapped around the cell.
    assert g.vacuum_axes_reference([3.0, 3.0, 20.0], [[0, 0, 0.95], [0, 0, 0.05]])[2] is True


def test_auto_kpts_cubic_cell_has_no_hexagonal_rule():
    cubic = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert g.auto_kpts_reference([4.0, 4.0, 4.0], [90.0, 90.0, 90.0], cubic, 25.0) == (6, 6, 6)
    assert g.auto_kpts_reference([40.0, 40.0, 40.0], [90.0, 90.0, 90.0], cubic, 25.0) == (1, 1, 1)


# --------------------------------------------------------------------------
# Convergence policy
# --------------------------------------------------------------------------

def test_convergence_reference_reproduces_the_servers_recommendation():
    got = g.convergence_reference(ECUT_SWEEP, KPTS_SWEEP, 3, 5.0)
    assert (got["recommended_ecut_ev"], got["recommended_kpts_density"]) == (300, 15.0)
    assert got["ecut_converged"] and got["kpts_converged"] and got["converged"]
    assert got["ecut_deltas"][0]["delta_mev_per_atom"] == pytest.approx(74.76984618939042)
    assert got["ecut_deltas"][-1] == {"delta_mev_per_atom": None, "delta_gap_ev": None}


def test_convergence_recommendation_tightens_with_the_tolerance():
    strict = g.convergence_reference(ECUT_SWEEP, KPTS_SWEEP, 3, 0.3)
    assert strict["recommended_kpts_density"] == 25.0        # 15 -> 25: dE(15) = 0.43 meV/atom
    assert strict["recommended_ecut_ev"] == 300              # ecut is judged on the gap only


def test_convergence_metallic_rows_fall_back_to_the_energy_criterion():
    metallic = [{"kpts_density": d, "energy_ev": e, "gap_ev": None}
                for d, e in ((10.0, -10.0), (15.0, -10.01), (25.0, -10.0101))]
    got = g.convergence_reference(ECUT_SWEEP, metallic, 1, 5.0)
    assert (got["recommended_kpts_density"], got["kpts_converged"]) == (15.0, True)


def test_convergence_sweep_ceiling_is_not_converged():
    drifting = [{"kpts_density": d, "energy_ev": e, "gap_ev": gap}
                for d, e, gap in ((10.0, -10.0, 1.0), (15.0, -11.0, 1.5), (25.0, -12.0, 2.0))]
    got = g.convergence_reference(ECUT_SWEEP, drifting, 1, 5.0)
    assert (got["recommended_kpts_density"], got["kpts_converged"], got["converged"]) == (25.0, False, False)


def test_deltas_match_flags_wrong_arithmetic():
    reference = g.sweep_deltas(KPTS_SWEEP, 3)
    rows = [{**row, **delta} for row, delta in zip(KPTS_SWEEP, reference)]
    assert g.deltas_match(rows, reference) == []
    rows[1]["delta_gap_ev"] = 0.5
    rows[2]["delta_mev_per_atom"] = None
    problems = g.deltas_match(rows, reference)
    assert len(problems) == 2 and "row 1 delta_gap_ev" in problems[0]


# --------------------------------------------------------------------------
# Gap analysis
# --------------------------------------------------------------------------

def test_gap_reference_direct_indirect_and_metallic():
    kpts = [[1 / 3, 1 / 3, 0.0], [0.0, 0.0, 0.0]]
    direct = g.gap_reference([[-3.0, -2.4067, -0.7312], [-3.1, -2.5, -0.5]], -1.5523, kpts)
    assert direct["gap"] == pytest.approx(1.6755) and direct["gap_type"] == "direct"
    assert (direct["vbm"]["label"], direct["vbm"]["band"]) == ("K", 1)
    assert (direct["cbm"]["label"], direct["cbm"]["energy_ev"]) == ("K", -0.7312)
    indirect = g.gap_reference([[-2.4, -0.9], [-2.0, -0.6]], -1.5, kpts)
    assert indirect["gap_type"] == "indirect"
    assert (indirect["vbm"]["label"], indirect["cbm"]["label"]) == ("Γ", "K")
    metallic = g.gap_reference([[-2.0, -1.0]], -0.5, [[0.0, 0.0, 0.0]])
    assert metallic == {"gap": None, "gap_type": "metallic", "vbm": None, "cbm": None}


def test_gap_reference_keeps_the_first_extremum_and_rounds_the_edges():
    kpts = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    got = g.gap_reference([[-2.0, 1.000049], [-2.0, 1.0]], 0.0, kpts)
    assert got["vbm"]["label"] == "Γ" and got["cbm"]["label"] == "M"      # ties keep the first k-point
    assert got["vbm"]["energy_ev"] == -2.0 and got["cbm"]["energy_ev"] == 1.0


@pytest.mark.parametrize("k,label", [([0.0, 0.0, 0.0], "Γ"), ([1 / 3, 1 / 3, 0.0], "K"),
                                     ([-1 / 3, -1 / 3, 0.0], "K"), ([1 / 3, -1 / 3, 0.0], "K'"),
                                     ([0.5, 0.5, 0.0], "M"), ([0.0, 0.0, 0.5], "A"),
                                     ([1 / 3, 1 / 3, 0.5], "H"), ([0.2, 0.1, 0.0], "[0.20,0.10,0.00]")])
def test_klabel_reference(k, label):
    assert g.klabel_reference(k) == label


def test_klabel_tolerance_is_six_hundredths():
    assert g.klabel_reference([0.05, 0.05, 0.0]) == "Γ"
    assert g.klabel_reference([0.07, 0.0, 0.0]) == "[0.07,0.00,0.00]"


# --------------------------------------------------------------------------
# verify_run state machine
# --------------------------------------------------------------------------

@pytest.mark.parametrize("gap_tol,verdict,gap_status", [(0.3, "pass", "pass"),
                                                        (0.012, "pass_with_warnings", "warn"),
                                                        (0.005, "fail", "fail")])
def test_verify_reference_gap_tolerance_state_machine(gap_tol, verdict, gap_status):
    conv = {"converged": True}
    got = g.verify_reference(FETCH, RELAX, SUMMARY, conv, 0.020, gap_tol)
    assert got["verdict"] == verdict
    assert dict(got["checks"])["band_gap_vs_mp"] == gap_status
    assert [name for name, _ in got["checks"]] == ["convergence_gate", "relax_convergence",
                                                   "structure_drift", "band_gap_vs_mp", "gap_character"]
    assert got["blocking_failures"] == (["band_gap_vs_mp"] if gap_status == "fail" else [])


def test_verify_reference_without_results_fails_closed():
    got = g.verify_reference(FETCH, None, None, None, None, 0.3)
    assert got["verdict"] == "fail"
    assert got["checks"] == [("relax_convergence", "fail"), ("band_gap_vs_mp", "fail")]


def test_verify_reference_warns_on_gate_drift_and_metastability():
    got = g.verify_reference({**FETCH, "is_stable": False}, RELAX, SUMMARY,
                             {"converged": False}, 0.8, 0.3)
    assert got["verdict"] == "pass_with_warnings" and got["blocking_failures"] == []
    assert dict(got["checks"]) == {"convergence_gate": "warn", "relax_convergence": "pass",
                                   "structure_drift": "warn", "band_gap_vs_mp": "pass",
                                   "gap_character": "info", "mp_stability": "warn"}


def test_verify_reference_skips_the_gap_check_without_a_reference():
    got = g.verify_reference({**FETCH, "band_gap_ref": None}, RELAX, SUMMARY, None, 0.02, 0.3)
    assert dict(got["checks"])["band_gap_vs_mp"] == "skip" and got["verdict"] == "pass"


def test_verify_reference_fails_on_an_unconverged_relaxation():
    got = g.verify_reference(FETCH, {**RELAX, "max_force_ev_per_a": 0.4}, SUMMARY, None, 0.02, 0.3)
    assert got["verdict"] == "fail" and got["blocking_failures"] == ["relax_convergence"]


# --------------------------------------------------------------------------
# Result handling
# --------------------------------------------------------------------------

def test_payload_prefers_structured_content_and_in_band_error_detection():
    assert g.payload(rpc({"ok": True, "run_id": "x"})["result"]) == {"ok": True, "run_id": "x"}
    text_only = {"content": [{"type": "text", "text": json.dumps({"ok": True})}], "isError": False}
    assert g.payload(text_only) == {"ok": True}
    assert g.in_band_error(rpc({"ok": False, "error": "ValueError: no such run"})["result"]) == \
        "ok=false, ValueError: no such run"
    assert g.in_band_error(rpc({"ok": True})["result"]) is None
    assert g.in_band_error({"content": [{"type": "text", "text": "not json"}]}) is None


def test_tool_ok_fails_on_in_band_errors_and_on_disagreeing_blocks(tmp_path):
    session = session_for({"relax_structure": rpc({"ok": False, "error": "ValueError: boom",
                                                   "traceback_tail": ["..."]})}, tmp_path)
    assert g.tool_ok(session.call, "relax", "relax_structure", {}) is None
    assert report_statuses(session.report) == {"relax": "FAIL"}
    assert "boom" in session.report.checks[0]["detail"]

    mismatch = {"result": {"content": [{"type": "text", "text": json.dumps({"ok": True, "a": 1})}],
                           "structuredContent": {"ok": True, "a": 2}, "isError": False}}
    session = session_for({"verify_run": mismatch}, tmp_path)
    assert g.tool_ok(session.call, "verify", "verify_run", {}) == {"ok": True, "a": 2}
    assert report_statuses(session.report) == {"verify[text == structuredContent]": "FAIL"}


def test_run_id_regex_accepts_server_ids_and_rejects_traversal():
    assert g.RUN_ID_RE.fullmatch("20261003-083136_mos2_77bb")
    assert g.RUN_ID_RE.fullmatch("20261003-083136_mp-1434_0a1b")
    assert not g.RUN_ID_RE.fullmatch("../etc")
    assert not g.RUN_ID_RE.fullmatch("20261003-083136_mos2")


def test_require_reports_a_missing_artefact(tmp_path):
    report = support.runner.Report()
    assert g.require(report, tmp_path / "relaxed.cif", "calc_band_dos[input]") is False
    (tmp_path / "relaxed.cif").write_text("data")
    assert g.require(report, tmp_path / "relaxed.cif", "calc_band_dos[input]") is True
    assert report_statuses(report) == {"calc_band_dos[input]": "FAIL"}


# --------------------------------------------------------------------------
# Checks that need no DFT reference
# --------------------------------------------------------------------------

def test_fetch_variants_warns_when_the_query_is_ignored(tmp_path):
    responses = {"fetch_structure": lambda args: rpc(
        {"ok": True, **FETCH, "query": args["query"]} if args.get("use_builtin")
        else {"ok": False, "error": "ValueError: no MP_API_KEY"})}
    session = session_for(responses, tmp_path)
    run = g.check_fetch_variants(session)
    assert run.run_id == FETCH["run_id"]
    assert report_statuses(session.report) == {
        "fetch_structure[use_builtin ignores query]": "WARN",
        "fetch_structure[MP query without credentials]": "WARN"}


def test_fetch_variants_fails_when_a_credential_free_mp_query_succeeds(tmp_path):
    responses = {"fetch_structure": lambda args: rpc({"ok": True, **FETCH, "formula": "Si"})}
    session = session_for(responses, tmp_path)
    g.check_fetch_variants(session)
    statuses = report_statuses(session.report)
    assert statuses["fetch_structure[use_builtin ignores query]"] == "FAIL"
    assert statuses["fetch_structure[MP query without credentials]"] == "FAIL"


def _gate_response(run_id="20261003-090000_mos2_abcd"):
    return {"ok": True, "run_id": run_id, "query": "MoS2", "status": "rejected_unconverged",
            "root_cause": "convergence gate never opened (sweep ceiling or relax not converged)",
            "n_attempts": 1, "attempts": [{"attempt": 0}],
            "final_params": {"ecut_ev": 200, "kpts_density": 3.0},
            "message": "收敛门未开启：未产出能带/DOS/带隙等最终结果；run 目录内中间产物仅供取证，禁止作为结果引用。"}


def test_workflow_gate_passes_when_no_results_were_produced(tmp_path):
    run_dir = tmp_path / "runs" / "20261003-090000_mos2_abcd"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("relax.json").write_text("{}")
    session = session_for({"run_verified_workflow": rpc(_gate_response())}, tmp_path)
    g.check_workflow_gate(session)
    assert report_statuses(session.report) == {
        "run_verified_workflow[gate rejects ecut=200 kd=3]": "PASS"}


@pytest.mark.parametrize("leftover,extra", [
    ("summary.json", {}),
    ("bands.png", {}),
    (None, {"band_gap_ev": 1.67}),
])
def test_workflow_gate_fails_when_the_gate_leaks_results(tmp_path, leftover, extra):
    run_dir = tmp_path / "runs" / "20261003-090000_mos2_abcd"
    run_dir.mkdir(parents=True)
    if leftover:
        run_dir.joinpath(leftover).write_bytes(png_bytes(640, 480) if leftover.endswith(".png") else b"{}")
    session = session_for({"run_verified_workflow": rpc({**_gate_response(), **extra})}, tmp_path)
    g.check_workflow_gate(session)
    assert list(report_statuses(session.report).values()) == ["FAIL"]


def test_workflow_gate_fails_without_the_chinese_rejection_notice(tmp_path):
    (tmp_path / "runs" / "20261003-090000_mos2_abcd").mkdir(parents=True)
    session = session_for({"run_verified_workflow": rpc({**_gate_response(), "message": "ok"})}, tmp_path)
    g.check_workflow_gate(session)
    assert list(report_statuses(session.report).values()) == ["FAIL"]


class FakeRef:
    """GpawRef stand-in: fixed grids and restart files, no GPAW."""

    def __init__(self, grids, ibz):
        self.grids = grids
        self.ibz = list(ibz)

    def read(self, path):
        return path

    def kpts_for(self, atoms, density):
        return self.grids[density]

    def from_gpw(self, gpw):
        return {"kpts_ibz": [[0.0, 0.0, 0.0]] * self.ibz.pop(0), "gap": 1.67, "gap_type": "direct",
                "eigvals": [], "fermi_ev": -1.55}


def test_unverified_params_warns_about_the_note_and_the_shared_restart_file(tmp_path):
    run_id = "20261003-091000_mos2_beef"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    run_dir.joinpath("structure.cif").write_text("cif")
    run_dir.joinpath("gs.gpw").write_bytes(b"gpw")
    note = "no convergence gate passed at these parameters — treat results as unverified"
    small = {"ok": True, "run_id": run_id, "params_verified": False, "kpts_scf": [3, 3, 1],
             "verification_note": note}
    big = {"ok": True, "run_id": run_id, "params_verified": False, "kpts_scf": [9, 9, 1],
           "verification_note": note}
    replies = iter([rpc(small), rpc(big)])
    run_dir.joinpath("summary.json").write_text(json.dumps(
        {k: v for k, v in small.items() if k not in ("ok", "verification_note")}))
    responses = {"fetch_structure": rpc({"ok": True, **FETCH, "run_id": run_id}),
                 "calc_band_dos": lambda args: next(replies)}
    session = session_for(responses, tmp_path,
                          ref=FakeRef({g.KD_UNVERIFIED: (3, 3, 1)}, ibz=[3, 12]))
    g.check_unverified_params(session)
    assert report_statuses(session.report) == {
        "calc_band_dos[no gate, kd=10]": "PASS",
        "calc_band_dos[verification_note is not persisted]": "WARN",
        "calc_band_dos[gs.gpw is shared and overwritten]": "WARN"}


def test_unverified_params_fails_when_the_restart_file_does_not_change(tmp_path):
    run_id = "20261003-091000_mos2_beef"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    for name in ("structure.cif", "gs.gpw"):
        run_dir.joinpath(name).write_text("x")
    reply = {"ok": True, "run_id": run_id, "params_verified": False, "kpts_scf": [3, 3, 1],
             "verification_note": "no convergence gate passed at these parameters"}
    run_dir.joinpath("summary.json").write_text(json.dumps({"params_verified": False}))
    session = session_for({"fetch_structure": rpc({"ok": True, **FETCH, "run_id": run_id}),
                           "calc_band_dos": rpc(reply)}, tmp_path,
                          ref=FakeRef({g.KD_UNVERIFIED: (3, 3, 1)}, ibz=[3, 3]))
    g.check_unverified_params(session)
    assert report_statuses(session.report)["calc_band_dos[gs.gpw is shared and overwritten]"] == "FAIL"


@pytest.mark.parametrize("by_tool,status", [
    ({"calc_band_dos": 123, "run_verified_workflow": 123}, "WARN"),
    ({}, "PASS"),
    ({"relax_structure": 7}, "WARN"),
])
def test_check_stdout_attributes_the_pollution(tmp_path, by_tool, status):
    session = session_for({}, tmp_path)
    session.call.stdout_by_tool.update(by_tool)
    g.check_stdout(session)
    assert report_statuses(session.report) == {"calc_band_dos[stdout pollution]": status}
    if status == "WARN" and "calc_band_dos" in by_tool:
        assert "band_eigs" in session.report.checks[0]["detail"]


def test_smoke_declares_the_manifest_server_and_its_packages():
    assert g.SMOKE.server == "gpaw" and g.SMOKE.prepare is not None
    assert {"gpaw", "ase", "fastmcp"} <= set(g.SMOKE.packages)
    assert g.SMOKE.call_timeout >= 3600         # a loaded host must not time out a correct call
    entry = support.setup.load_manifest()["gpaw"]
    assert entry["install"] == "conda-explicit"
    assert entry["launch"]["env"]["OMP_NUM_THREADS"] == "1"          # references assume one thread
    assert math.isclose(g.MOS2_BUILDER["a"], 3.18)


def test_tool_ok_records_the_wall_clock_time_of_every_call(tmp_path):
    g.TIMINGS.clear()
    session = session_for({"verify_run": rpc({"ok": True, "verdict": "pass"})}, tmp_path)
    g.tool_ok(session.call, "verify_run[gap_tol_ev=0.3]", "verify_run", {})
    assert list(g.TIMINGS) == ["call verify_run[gap_tol_ev=0.3]"]
    assert g.TIMINGS["call verify_run[gap_tol_ev=0.3]"] >= 0
    assert g.report_fields(session)["seconds_by_step"] == g.TIMINGS
