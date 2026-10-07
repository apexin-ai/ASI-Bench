"""qe_si_bandstructure (silicon: pseudopotential index -> suggested SCF grid -> band
structure on that grid -> listing of the run -> the listed band file): generator,
measurement script, scorer and verifier scenarios. No MCP server and no Quantum
ESPRESSO at import time.

The DFT reference cannot be recomputed here: pw.x and bands.x live only inside the
server's pinned conda environment, so ``generate_gt.MEASURED`` is a table measured
against that server by ``scripts/mcp/e2e/measure_qe_table.py``. The tests read the
committed table: that it covers the instance grid, that the generator's derivations
agree with what the server answered, that neighbouring instances are far enough apart
for the scorer's tolerances, and the guards that refuse a table contradicting them.
"""
import itertools
import json
import re

import pytest

from . import support
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.qe_si_bandstructure")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id
SERVER = "quantum_espresso"
TOOLS = support.setup.load_manifest()[SERVER]["expected_tools"]

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
measure = support.load(support.BUNDLE / "measure_qe_table.py", "mcp_e2e_measure_qe_table")
verify = support.verify
_status = support.statuses

MEASURED = generate_gt.MEASURED
GRID = list(itertools.product(generate_gt.ECUTWFC, generate_gt.KSPACING, generate_gt.NPOINTS_BAND))
CASE = generate_gt.build_case(support.SEED)
REFERENCE = {**CASE, **generate_gt.reference(CASE)}
WORKFLOW_ID = "bands_1a2b3c4d"
HOME = "/home/e2e"
RUN_DIR = f"{HOME}/mcp/quantum_espresso/qe_calculations/{WORKFLOW_ID}"
ANSWER_KEYS = ("si_pseudopotential", "scf_kpoints", "total_energy_eV", "fermi_energy_eV",
               "band_gap_eV", "vbm_eV", "cbm_eV", "is_direct_gap", "n_bands", "n_kpoints",
               "path_length")


def _answer(reference=None, **over) -> dict:
    ref = reference or REFERENCE
    return {**{key: ref[key] for key in ANSWER_KEYS}, **over}


def _case(ecutwfc, kspacing, npoints) -> dict:
    return {"structure": "Si", "ecutwfc": ecutwfc, "kspacing": kspacing, "npoints_band": npoints,
            "result_file": "result.json"}


def _patched_point(monkeypatch, **changes) -> tuple:
    """MEASURED with the instance-31415 point changed; returns that point's key."""
    point = (CASE["ecutwfc"], REFERENCE["scf_kpoints"][0], CASE["npoints_band"])
    points = dict(MEASURED["points"])
    points[point] = {**points[point], **changes}
    monkeypatch.setattr(generate_gt, "MEASURED", {**MEASURED, "points": points})
    return point


# --------------------------------------------------------------------------
# The instance grid and the k-grid policy
# --------------------------------------------------------------------------

def test_cases_are_deterministic_and_cover_the_grid():
    assert generate_gt.build_case(support.SEED) == CASE
    seen = {(c["ecutwfc"], c["kspacing"], c["npoints_band"])
            for c in map(generate_gt.build_case, range(400))}
    assert seen == set(GRID)


def test_instance_31415():
    assert (CASE["ecutwfc"], CASE["kspacing"], CASE["npoints_band"]) == (30, 0.04, 30)
    assert REFERENCE["scf_kpoints"] == [7, 7, 7]
    assert REFERENCE["si_pseudopotential"] == "Si_ONCV_PBE-1.2.upf"
    assert REFERENCE["total_energy_eV"] == -214.45630404636253
    assert REFERENCE["fermi_energy_eV"] == 6.4911
    assert (REFERENCE["vbm_eV"], REFERENCE["cbm_eV"]) == (6.237, 6.836)
    assert REFERENCE["is_direct_gap"] is False and REFERENCE["n_bands"] == 16
    assert REFERENCE["n_kpoints"] == 30 and REFERENCE["path_length"] == 4.9721


def test_grid_rule_matches_the_server_and_stays_off_the_ceiling():
    assert {k: generate_gt.kgrid(k)[0] for k in generate_gt.KSPACING} == {0.06: 5, 0.04: 7, 0.03: 9}
    assert generate_gt.si_cell_length() == pytest.approx(3.8395898218429, abs=1e-12)
    # ties go to the larger odd count, as upstream's snapping does
    assert [generate_gt.snap_odd(n) for n in (1, 2, 4, 6, 8, 30)] == [1, 3, 5, 7, 9, 21]
    on_an_integer = 1.0 / (generate_gt.si_cell_length() * 7.0)
    with pytest.raises(generate_gt.GenerationError, match="too close to an integer"):
        generate_gt.kgrid(on_an_integer)


def test_cutoffs_keep_the_norm_conserving_dual():
    assert max(generate_gt.ECUTWFC) * 4 <= generate_gt.ECUTRHO_RY == MEASURED["si_cutoff_hints_ry"][1]
    with pytest.raises(generate_gt.GenerationError, match="dual of 4"):
        generate_gt.reference({**CASE, "ecutwfc": 40})


# --------------------------------------------------------------------------
# The measured table
# --------------------------------------------------------------------------

def test_measured_table_covers_exactly_the_instance_grid():
    want = {(e, generate_gt.kgrid(k)[0], n) for e, k, n in GRID}
    assert set(MEASURED["points"]) == want
    assert MEASURED["suggested_grid_by_kspacing"] == {repr(k): generate_gt.kgrid(k)
                                                      for k in generate_gt.KSPACING}
    # The measurement script sweeps the same grid the generator draws from.
    assert (measure.ECUTWFC, measure.KSPACING, measure.NPOINTS_BAND) == \
        (generate_gt.ECUTWFC, generate_gt.KSPACING, generate_gt.NPOINTS_BAND)


@pytest.mark.parametrize("ecutwfc,kspacing,npoints", GRID)
def test_every_grid_point_generates(ecutwfc, kspacing, npoints):
    ref = generate_gt.reference(_case(ecutwfc, kspacing, npoints))
    assert ref["n_kpoints"] == npoints and ref["band_gap_eV"] > 0


def test_what_moves_with_which_parameter():
    """The energies follow (cutoff, grid), the band edges also the path sampling; the
    valence-band maximum sits at Gamma, which every path contains."""
    points = MEASURED["points"]
    for (e, g, n), row in points.items():
        for other_n in generate_gt.NPOINTS_BAND:
            same_scf = points[(e, g, other_n)]
            assert same_scf["total_energy_eV"] == row["total_energy_eV"]
            assert same_scf["fermi_energy_eV"] == row["fermi_energy_eV"]
            assert same_scf["vbm_eV"] == row["vbm_eV"]
    assert len({row["total_energy_eV"] for row in points.values()}) == 9
    assert len({row["cbm_eV"] for row in points.values()}) >= 20
    assert {row["path_length"] for row in points.values()} == {4.9721}


def test_a_copied_neighbour_scores_nothing():
    """Two different instances differ by at least twice the zero-credit tolerance on every
    scored energy they do not share, so another instance's numbers earn nothing."""
    zero = {item["scorer"]: item["config"].get("zero_score_tol")
            for item in TASK.eval_config()["scoring"]}
    rows = list(MEASURED["points"].values())
    for field, scorer_name in (("total_energy_eV", "qe_e2e_energies"), ("fermi_energy_eV", "qe_e2e_energies"),
                               ("band_gap_eV", "qe_e2e_band_edges"), ("vbm_eV", "qe_e2e_band_edges"),
                               ("cbm_eV", "qe_e2e_band_edges")):
        values = sorted({row[field] for row in rows})
        closest = min(b - a for a, b in zip(values, values[1:]))
        assert closest >= 2 * zero[scorer_name] - 1e-12, (field, closest)


@pytest.mark.parametrize("changes,match", [
    ({"pseudo_copied": ["Si_ONCV_PBE-1.0.upf"]}, "the run copied"),
    ({"n_bands": 8}, "n_bands"),
    ({"read_n_kpoints": 29}, "n_kpoints"),
    ({"file_names": ["bands.dat", "scf.out"]}, "the run left"),
    ({"is_metal": True}, "no band gap"),
    ({"band_gap_eV": 0.5}, "cbm_eV - vbm_eV"),
])
def test_an_inconsistent_record_is_refused(monkeypatch, changes, match):
    _patched_point(monkeypatch, **changes)
    with pytest.raises(generate_gt.GenerationError, match=match):
        generate_gt.reference(CASE)


def test_an_unmeasured_point_and_a_disagreeing_suggestion_are_refused(monkeypatch):
    with pytest.raises(generate_gt.GenerationError, match="no measured run"):
        generate_gt.reference({**CASE, "npoints_band": 35})
    grids = {**MEASURED["suggested_grid_by_kspacing"], repr(CASE["kspacing"]): [9, 9, 9]}
    monkeypatch.setattr(generate_gt, "MEASURED", {**MEASURED, "suggested_grid_by_kspacing": grids})
    with pytest.raises(generate_gt.GenerationError, match="server suggested"):
        generate_gt.reference(CASE)


def test_generate_writes_the_instance(tmp_path):
    meta = generate_gt.generate(tmp_path, {"seed": support.SEED})
    assert meta["input_files"] == ["calculation.json"]
    data = json.loads((tmp_path / "data/calculation.json").read_text())
    assert data == CASE
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    assert ref == json.loads(json.dumps(REFERENCE))
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{" not in text
        if level != "b4":
            assert "`ecutwfc` = 30 Ry" in text and "`kspacing` = 0.04" in text


# --------------------------------------------------------------------------
# The measurement script
# --------------------------------------------------------------------------

def _records() -> list[dict]:
    """The committed table in the shape measure_chain records it."""
    out = []
    for (ecut, grid, npoints), row in MEASURED["points"].items():
        out.append({"ecutwfc": ecut, "grid": [grid] * 3, "npoints_band": npoints,
                    "bands": {k: row[k] for k in ("total_energy_eV", "fermi_energy_eV", "band_gap_eV",
                                                  "is_metal", "is_direct_gap", "vbm_eV", "cbm_eV",
                                                  "n_bands", "n_kpoints")},
                    "file_names": row["file_names"], "pseudo_copied": row["pseudo_copied"],
                    "read_bands": {"n_bands": row["n_bands"], "n_kpoints": row["read_n_kpoints"],
                                   "path_length": row["path_length"]}})
    return out


def _table() -> dict:
    kgrids = {float(k): v for k, v in MEASURED["suggested_grid_by_kspacing"].items()}
    pick = {"filename": MEASURED["si_pseudopotential"], "ecutwfc_Ry": MEASURED["si_cutoff_hints_ry"][0],
            "ecutrho_Ry": MEASURED["si_cutoff_hints_ry"][1]}
    return measure.table_from(_records(), kgrids, pick)


def test_the_literal_round_trips_to_the_committed_table():
    """Pasting the script's literal reproduces MEASURED exactly (replace wholesale)."""
    table = _table()
    assert table == MEASURED
    namespace: dict = {}
    exec(measure.render_literal(table, ["header"]), namespace)            # noqa: S102
    assert namespace["MEASURED"] == MEASURED


def test_check_mode_reports_every_difference():
    table = _table()
    assert measure.compare(table, MEASURED) == []
    point = next(iter(table["points"]))
    drifted = {**table, "si_pseudopotential": "Si_ONCV_PBE-1.0.upf",
               "points": {**table["points"], point: {**table["points"][point], "cbm_eV": 7.0}}}
    problems = measure.compare(drifted, MEASURED)
    assert len(problems) == 2
    assert any("si_pseudopotential" in p for p in problems) and any("cbm_eV" in p for p in problems)


def test_spread_reports_the_closest_instances():
    spread = measure.spread(_table())
    assert spread["path_length"] == {"distinct": 1, "min_gap": None, "min_relative_gap": None}
    assert spread["total_energy_eV"]["distinct"] == 9
    assert spread["cbm_eV"]["min_gap"] == pytest.approx(1e-3, abs=1e-9)


def test_measure_chain_reads_the_listed_band_file(tmp_path):
    folder = tmp_path / WORKFLOW_ID
    (folder / "pseudo").mkdir(parents=True)
    (folder / "pseudo" / "Si_ONCV_PBE-1.2.upf").write_text("x")
    gnu = str(folder / "bands.dat.gnu")
    payloads = _payloads(output_dir=str(folder))
    client = support.StubClient({
        "qe_workflow_bandstructure": support.rpc_json(payloads["bands"]),
        "qe_list_files": support.rpc_json({**payloads["files"], "band_files": [gnu]}),
        "qe_read_bands": support.rpc_json(payloads["read"]),
    })
    record = measure.measure_chain(client, 30, [7, 7, 7], 30, {})
    assert [name for name, _ in client.calls] == ["qe_workflow_bandstructure", "qe_list_files", "qe_read_bands"]
    assert client.calls[0][1] == {"structure": "Si", "kpoints": "7,7,7", "ecutwfc": 30, "npoints_band": 30}
    assert client.calls[2][1] == {"output_dir": gnu}
    assert record["read_bands"]["path_length"] == 4.9721 and record["pseudo_copied"] == ["Si_ONCV_PBE-1.2.upf"]
    assert record["file_names"] == sorted(generate_gt.RUN_FILES)


def test_measure_refuses_an_in_band_error():
    client = support.StubClient({"qe_workflow_bandstructure": support.rpc_json(
        {"success": False, "step_failed": "scf", "error": "boom"})})
    with pytest.raises(measure.MeasureError, match="success=false"):
        measure.measure_chain(client, 30, [7, 7, 7], 30, {})


# --------------------------------------------------------------------------
# Task files
# --------------------------------------------------------------------------

def _spec() -> dict:
    return json.loads((TASK_DIR / "e2e_check.json").read_text())


def test_reference_carries_every_key_the_checks_use():
    spec = _spec()
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    assert keys <= set(REFERENCE), sorted(keys - set(REFERENCE))
    assert spec["server_tools"] == sorted(TOOLS)
    assert [cs["tool"] for cs in spec["calls"]] == list(generate_gt.MCP_TOOLS)
    links = {cs["name"]: cs.get("inputs_from_call") for cs in spec["calls"]}
    assert links["pseudos"] is None and links["grid"] is None
    assert links["bands"] == {"call": "grid", "extract": "kpoint_grid", "args": ["kpoints"], "match": "member"}
    assert links["files"] == {"call": "bands", "map": {"output_dir": "output_dir"}}
    assert links["read"] == {"call": "files", "extract": "categorized_paths", "args": ["output_dir"],
                             "match": "member"}
    assert sorted(a["prediction_key"] for a in spec["answers"]) == sorted(ANSWER_KEYS[:2] + ANSWER_KEYS[2:7]
                                                                          + ("n_bands", "n_kpoints", "path_length"))


def test_the_paths_a_link_compares_survive_persistence():
    """The two path links only PASS because the trajectory keeps ``output_dir``."""
    from ai4sci_bench.core.trajectory import KEY_ARG_NAMES
    assert "output_dir" in KEY_ARG_NAMES


def test_prompts_name_the_tools_only_at_b1_b2_and_keep_the_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert ("`quantum_espresso` MCP server" in text) == (level in ("b1", "b2"))
        assert ("`qe_workflow_bandstructure`" in text) == (level in ("b1", "b2"))
        assert ("`qe_list_files` tool" in text) == (level == "b1")
        assert "Quantum ESPRESSO is available **only** through the MCP server" in text
        assert "Do not modify `data/calculation.json`" in text and "full precision" in text
        assert "not with shell commands" in text
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text
        for key in ANSWER_KEYS:
            assert f'"{key}"' in text, (level, key)
    b1 = (TASK_DIR / "prompt_b1.md").read_text()
    assert "path of the **file**, not of the directory" in b1          # D2, spelled out at b1


def test_b3_and_b4_share_the_background_and_the_contract():
    b3 = (TASK_DIR / "prompt_b3.md").read_text()
    b4 = (TASK_DIR / "prompt_b4.md").read_text()
    background = lambda text: [l for l in text.splitlines() if l.startswith("Background:")]  # noqa: E731
    assert background(b3) == background(b4) and background(b3)
    tail = lambda text: text[text.index("## Output"):]                                       # noqa: E731
    assert tail(b3) == tail(b4)
    assert "{{ecutwfc}}" in b3 and "{{" + "ecutwfc}}" not in b4


def test_task_meta_needs_no_packages():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == []
    assert [f["name"] for f in meta["output"]["files"]] == ["result.json"]


def test_scoring_weights_put_the_computed_numbers_first():
    scoring = {item["scorer"]: item["weight"] for item in TASK.eval_config()["scoring"]}
    assert sum(scoring.values()) == 100
    assert scoring["qe_e2e_energies"] + scoring["qe_e2e_band_edges"] == 70
    assert max(scoring[k] for k in ("qe_e2e_grid", "qe_e2e_band_file", "qe_e2e_pseudopotential")) == 10


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def _scored(tmp_path, prediction, name, reference=None):
    from ai4sci_bench.core.scorer import get_scorer
    item = next(i for i in TASK.eval_config()["scoring"] if i["scorer"] == name)
    pred, ref = support.score_dirs(tmp_path, prediction, reference or REFERENCE)
    return get_scorer(name).score(pred, ref, {**item["config"], "weight": item["weight"]})


def test_tool_values_score_full(tmp_path):
    assert TASK.total(*support.score_dirs(tmp_path, _answer(), REFERENCE)) == pytest.approx(100.0)


def test_another_instance_scores_zero_on_what_it_does_not_share(tmp_path):
    """Instance (30, 9x9x9, 30): same cutoff and path, another grid."""
    other = MEASURED["points"][(30, 9, 30)]
    wrong = _answer(**{k: other[k] for k in ("total_energy_eV", "fermi_energy_eV", "band_gap_eV",
                                             "vbm_eV", "cbm_eV")})
    assert _scored(tmp_path, wrong, "qe_e2e_energies").score == 0.0
    assert _scored(tmp_path / "b", wrong, "qe_e2e_band_edges").score == 0.0


def test_six_decimals_score_full_and_three_lose_credit(tmp_path):
    six = _answer(total_energy_eV=round(REFERENCE["total_energy_eV"], 6))
    assert _scored(tmp_path, six, "qe_e2e_energies").score == pytest.approx(35.0)
    three = _answer(total_energy_eV=round(REFERENCE["total_energy_eV"], 3))
    detail = _scored(tmp_path / "b", three, "qe_e2e_energies")
    assert 17.5 <= detail.score < 35.0


def test_exact_fields_are_all_or_nothing(tmp_path):
    detail = _scored(tmp_path, _answer(scf_kpoints=[11, 11, 11]), "qe_e2e_grid")
    assert detail.score == 0.0 and "scf_kpoints" in detail.message
    detail = _scored(tmp_path / "b", _answer(n_kpoints=40, is_direct_gap=True), "qe_e2e_band_file")
    assert detail.score == pytest.approx(10.0 * 2 / 4)
    # A file name: case and version both matter.
    for name in ("si_oncv_pbe-1.2.upf", "Si_ONCV_PBE-1.0.upf"):
        assert _scored(tmp_path / name, _answer(si_pseudopotential=name), "qe_e2e_pseudopotential").score == 0.0


def test_credit_curve():
    assert scorer.credit(1e-9, 1e-6, 5e-4) == 1.0
    assert scorer.credit(5e-4, 1e-6, 5e-4) == 0.0
    assert 0.0 < scorer.credit(1e-5, 1e-6, 5e-4) < 1.0
    assert scorer.credit(float("inf"), 1e-6, 5e-4) == 0.0


@pytest.mark.parametrize("prediction", [
    None, "not json", [], {"band_gap_eV": 0.5},
    _answer(total_energy_eV="low"), _answer(n_bands=-1), _answer(is_direct_gap="no"),
    _answer(scf_kpoints=[7, 7]), _answer(scf_kpoints="7,7,7"), _answer(cbm_eV=float("nan")),
    _answer(si_pseudopotential=""),
])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = support.score_dirs(tmp_path, prediction, REFERENCE)
    gate = get_scorer("qe_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in TASK.eval_config()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.score == 0.0, (item["scorer"], detail.message)
        assert not detail.details.get("scorer_internal_error"), item["scorer"]


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = support.score_dirs(tmp_path, _answer(), None)
    for item in TASK.eval_config()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.details["scorer_internal_error"]
        assert detail.details["failure_kind"] == "missing_evaluator_input"


# --------------------------------------------------------------------------
# Verifier scenarios
# --------------------------------------------------------------------------

def _payloads(reference=None, output_dir=RUN_DIR, si_file=None) -> dict:
    """What the pinned server returns for this instance's chain (abridged where the
    verifier does not look)."""
    ref = reference or REFERENCE
    n = ref["n_kpoints"]
    distances = [round(ref["path_length"] * i / (n - 1), 4) for i in range(n)]
    pick = si_file or ref["si_pseudopotential"]
    files = {f"{key}_files": [] for key in ("band", "dos", "pdos", "input", "output", "other")}
    for name in generate_gt.RUN_FILES:
        key = ("band" if name.endswith(".gnu") else "input" if name.endswith(".in")
               else "output" if name.endswith(".out") else "other")
        files[f"{key}_files"].append(f"{output_dir}/{name}")
    return {
        "pseudos": {"success": True, "library": "SG15 ONCV", "n_elements": 3, "elements": ["O", "Si", "Ag"],
                    "details": {"O": {"filename": "O_ONCV_PBE-1.0.upf", "ecutwfc_Ry": 40, "ecutrho_Ry": 160},
                                "Si": {"filename": pick, "ecutwfc_Ry": 30, "ecutrho_Ry": 120},
                                "Ag": {"filename": "Ag_ONCV_PBE-1.2.upf", "ecutwfc_Ry": 40,
                                       "ecutrho_Ry": 160}}},
        "grid": {"success": True, "kpoints": ref["scf_kpoints"], "total_kpoints_approx": 343,
                 "method": f"kspacing={ref['kspacing']} 1/Å", "cell_lengths_angstrom": ["3.84"] * 3},
        "bands": {"workflow_id": WORKFLOW_ID, "output_dir": output_dir,
                  "total_energy_eV": ref["total_energy_eV"], "fermi_energy_eV": ref["fermi_energy_eV"],
                  "success": True, "band_gap_eV": ref["band_gap_eV"], "is_metal": False,
                  "is_direct_gap": ref["is_direct_gap"], "vbm_eV": ref["vbm_eV"], "cbm_eV": ref["cbm_eV"],
                  "n_bands": ref["n_bands"], "n_kpoints": n, "high_symmetry_points": {"G": [0.0, 0.0, 0.0]},
                  "eigenvalues_eV": [[-5.7, 6.2]] * n, "kpoints": [[0.0, 0.0, 0.0]] * n,
                  "bands_file": f"{output_dir}/bands.dat"},
        "files": {"success": True, "directory": output_dir, **files},
        "read": {"success": True, "n_bands": ref["n_bands"], "n_kpoints": n, "k_distances": distances,
                 "bands": [[-5.7] * n] * ref["n_bands"], "raw_data": "    0.0000   -5.7007\n",
                 "plot_instruction": "To plot: use k_distances as x-axis, each band as y-values."},
    }


def _calls(reference=None, payloads=None, kpoints=None, list_dir=None, read_path=None, before=()):
    ref = reference or REFERENCE
    data = payloads or _payloads(ref)
    grid = ",".join(map(str, ref["scf_kpoints"])) if kpoints is None else kpoints
    return [
        ("qe_list_pseudopotentials", {}, data["pseudos"]),
        ("qe_suggest_kpoints", {"structure": "Si", "kspacing": ref["kspacing"]}, data["grid"]),
        ("qe_workflow_bandstructure", {"structure": "Si", "kpoints": grid, "ecutwfc": ref["ecutwfc"],
                                       "npoints_band": ref["npoints_band"]}, data["bands"]),
        ("qe_list_files", {"output_dir": list_dir or data["bands"]["output_dir"]}, data["files"]),
        *before,
        ("qe_read_bands", {"output_dir": read_path or data["files"]["band_files"][0]}, data["read"]),
    ]


def _stream(calls, extra=()):
    events = [claude.init(SERVER, ["Bash", "Read", "Write", "WebFetch", "WebSearch",
                                   *(f"mcp__{SERVER}__{t}" for t in TOOLS)])]
    for index, (name, arguments) in enumerate(extra):
        events += claude.call(f"x{index}", name, arguments, "ok")
    for index, (tool, arguments, payload) in enumerate(calls):
        events += claude.call(f"c{index}", f"mcp__{SERVER}__{tool}", arguments, json.dumps(payload))
    events.append(claude.result(len(calls)))
    return jsonl(events)


def _codex_stream(calls):
    events = list(codex.START)
    for index, (tool, arguments, payload) in enumerate(calls):
        events += codex.mcp(f"m{index}", SERVER, tool, arguments, json.dumps(payload, indent=2),
                            structured=payload)
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream, answer=None, **kwargs):
    return TASK.verify(tmp_path, stream, reference=REFERENCE,
                       answer=_answer() if answer is None else answer, **kwargs)


def test_genuine_b1_run_passes_every_check(tmp_path):
    row = _run(tmp_path, _stream(_calls()))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}, row["checks"]


def test_codex_run_passes(tmp_path):
    row = _run(tmp_path, _codex_stream(_calls()), harness="codex")
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}, row["checks"]


@pytest.mark.parametrize("spelling", ["7 7 7", "7x7x7", "7, 7, 7", "7"])
def test_any_spelling_of_the_suggested_grid_is_the_chain(tmp_path, spelling):
    """qe-mcp reads ``kpoints`` as one string; the agent may space or separate it freely."""
    row = _run(tmp_path, _stream(_calls(kpoints=spelling)))
    assert _status(row)["tool_chain"] == "PASS", row["checks"]["tool_chain"]


def test_a_grid_the_server_did_not_suggest_breaks_the_chain(tmp_path):
    row = _run(tmp_path, _stream(_calls(kpoints="9,9,9")))
    assert _status(row)["tool_chain"] == "FAIL"
    assert "bands←grid" in row["checks"]["tool_chain"]["detail"]


def test_listing_another_run_breaks_the_chain(tmp_path):
    row = _run(tmp_path, _stream(_calls(list_dir=RUN_DIR.replace(WORKFLOW_ID, "bands_99999999"))))
    assert _status(row)["tool_chain"] == "FAIL"
    assert "files←bands" in row["checks"]["tool_chain"]["detail"]


def test_reading_a_file_the_listing_did_not_return_breaks_the_chain(tmp_path):
    row = _run(tmp_path, _stream(_calls(read_path=f"{HOME}/elsewhere/bands.dat.gnu")))
    assert _status(row)["tool_chain"] == "FAIL"
    assert "read←files" in row["checks"]["tool_chain"]["detail"]


def test_reading_the_directory_first_is_recoverable(tmp_path):
    """D2: the natural first call passes the directory and gets an in-band error; the
    retry with the listed file is the chain."""
    failed = ("qe_read_bands", {"output_dir": RUN_DIR},
              {"success": False, "error": f"[Errno 21] Is a directory: '{RUN_DIR}'"})
    row = _run(tmp_path, _stream(_calls(before=[failed])))
    assert row["verdict"] == "PASS", row["checks"]
    assert row["tool_call_counts"][f"mcp__{SERVER}__qe_read_bands"] == 2


def test_without_the_raw_paths_the_path_links_are_unobservable_not_wrong(tmp_path):
    """The persisted log has the host paths scrubbed. A trajectory written before
    ``output_dir`` joined KEY_ARG_NAMES has no raw copy, so the two path links cannot be
    checked: a coverage gap (WARN), never a FAIL."""
    def drop_output_dir(steps):
        for step in steps:
            (step.get("metadata") or {}).get("key_args", {}).pop("output_dir", None)

    row = _run(tmp_path, _stream(_calls()), edit_trajectory=drop_output_dir)
    assert _status(row)["tool_chain"] == "WARN", row["checks"]["tool_chain"]
    assert "scrubbed" in row["checks"]["tool_chain"]["detail"]
    assert row["verdict"] != "FAIL"


def test_another_pseudopotential_pick_fails_the_index_call(tmp_path):
    """D1: a host whose directory order picks another Si file is not the measured host."""
    payloads = _payloads(si_file="Si_ONCV_PBE-1.0.upf")
    row = _run(tmp_path, _stream(_calls(payloads=payloads)), _answer(si_pseudopotential="Si_ONCV_PBE-1.0.upf"))
    assert row["checks"]["tool_correct"]["per_call"]["pseudos"]["status"] == "FAIL"
    assert _status(row)["answer_from_tool"] == "FAIL"


def test_a_listing_missing_a_run_file_fails_the_listing_call(tmp_path):
    payloads = _payloads()
    payloads["files"]["output_files"] = []
    row = _run(tmp_path, _stream(_calls(payloads=payloads)))
    assert row["checks"]["tool_correct"]["per_call"]["files"]["status"] == "FAIL"


def test_answers_must_be_copied_from_the_tool_results(tmp_path):
    row = _run(tmp_path, _stream(_calls()), _answer(cbm_eV=REFERENCE["cbm_eV"] + 1e-7))
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "cbm_eV" in row["checks"]["answer_from_tool"]["detail"]
    row = _run(tmp_path / "b", _stream(_calls()), _answer(path_length=4.97))
    assert "path_length" in row["checks"]["answer_from_tool"]["detail"]
    assert _status(row)["answer_from_tool"] == "FAIL"


@pytest.mark.parametrize("command,status", [
    ("python3 -c 'from ase.build import bulk'", "FAIL"),
    ("pip install ase", "FAIL"),
    ("conda install -c conda-forge qe", "FAIL"),
    ("micromamba create -n x -c conda-forge quantum-espresso", "FAIL"),
    ("apt-get install quantum-espresso", "FAIL"),
    ("~/mcp/quantum_espresso/.venv/bin/pw.x -i scf.in", "FAIL"),
    ("pw.x < scf.in > scf.out", "FAIL"),
    ("python3 -m qe_mcp.server", "FAIL"),
    ("cat ~/mcp/quantum_espresso/qe_calculations/bands_1a2b3c4d/bands.dat.gnu", "WARN"),
    ("pip install numpy", "WARN"),
    # Looking at the run directory or writing the answer is not a bypass.
    ("ls ~/mcp/quantum_espresso/qe_calculations/bands_1a2b3c4d", "PASS"),
    ("python3 -c \"import json; json.dump({'n_bands': 16}, open('result.json','w'))\"", "PASS"),
    ("cat data/calculation.json", "PASS"),
])
def test_shell_access_to_the_solver(tmp_path, command, status):
    row = _run(tmp_path, _stream(_calls(), extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == status, row["checks"]["no_bypass"]["detail"]


def test_a_pasted_tool_payload_is_not_a_bypass(tmp_path):
    """An agent may paste a tool's reply into a script to pick values out of it."""
    pasted = "python3 - <<'EOF'\nimport json\ndata = " + json.dumps(_payloads()["read"]) + \
             "\nprint(max(data['k_distances']))\nEOF"
    row = _run(tmp_path, _stream(_calls(), extra=[("Bash", {"command": pasted})]))
    assert _status(row)["no_bypass"] == "PASS", row["checks"]["no_bypass"]["detail"]


def test_web_lookup_is_only_suspicious(tmp_path):
    row = _run(tmp_path, _stream(_calls(), extra=[("WebSearch", {"query": "silicon PBE band gap"})]))
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"
    plain = _run(tmp_path / "b", _stream(_calls(), extra=[("WebFetch", {"url": "https://example.org/news"})]))
    assert _status(plain)["no_bypass"] == "PASS"


def test_a_produced_script_importing_ase_is_a_bypass(tmp_path):
    row = _run(tmp_path, _stream(_calls()), files={"bands.py": "from ase.build import bulk\n"})
    assert _status(row)["no_bypass"] == "FAIL"


def test_patterns_do_not_match_the_server_directory():
    for pattern in _spec()["bypass_patterns"]:
        for allowed in (f"ls {RUN_DIR}", "find ~/mcp/quantum_espresso -name '*.gnu'",
                        f"cat {RUN_DIR}/scf.in"):
            assert not re.search(pattern, allowed), (pattern, allowed)
