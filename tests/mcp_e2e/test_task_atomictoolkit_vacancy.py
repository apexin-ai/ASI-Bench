"""atomictoolkit_vacancy (fcc cell -> supercell -> vacancy -> two EMT single points, a
structure analysis and a bulk-modulus fit): generator, scorer, verifier scenarios.
No MCP server and no ASE at import time."""
import json
import math

import pytest

from . import support
from .conftest import SENTINEL_HOME
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.atomictoolkit_vacancy")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id
SERVER = "atomictoolkit"
TOOLS = support.setup.load_manifest()[SERVER]["expected_tools"]

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify
_status = support.statuses

# Seed 31415 (Pd, a = 3.882 A, 3x3x3, vacancy 16, strain 0.02) as generated with ASE 3.29.0 on
# 2026-10-07; the pinned server answered the same chain with the same numbers bit for bit.
CASE = {"element": "Pd", "crystal_system": "fcc", "lattice_constant": 3.882, "supercell": 3,
        "vacancy_index": 16, "strain_max": 0.02, "calculator": "emt",
        "unit_cell_file": "unit_cell.extxyz", "supercell_file": "supercell.extxyz",
        "vacancy_file": "vacancy.extxyz", "analysis_dir": "analysis"}
REFERENCE = {
    **CASE,
    "n_atoms_unit_cell": 4,
    "n_atoms_perfect": 108,
    "n_atoms_vacancy": 107,
    "energy_perfect_eV": -0.023567484444418696,
    "energy_vacancy_eV": 0.9246658707987856,
    "vacancy_formation_energy_eV": 0.9480151377946449,
    "coordination_average": 17.83177570093458,
    "coordination_average_skin0": 11.88785046728972,
    "bulk_modulus_GPa": 177.9097120211918,
    "unit_cell_volume_A3": 58.50144496800002,
    "operation_supercell": "supercell",
    "operation_vacancy": "vacancy",
    "ase_version": "3.29.0",
    "mcp_tools": generate_gt.MCP_TOOLS,
}
ANSWER_KEYS = ("n_atoms_perfect", "energy_perfect_eV", "energy_vacancy_eV", "vacancy_formation_energy_eV",
               "coordination_average", "bulk_modulus_GPa")


def _answer(**over):
    data = {key: REFERENCE[key] for key in ANSWER_KEYS}
    data.update(over)
    return data


def _dirs(tmp_path, prediction, reference=REFERENCE):
    return support.score_dirs(tmp_path, prediction, reference)


def _total(pred, ref):
    return TASK.total(pred, ref)


def _artifacts(**paths):
    """``with_downloadable_artifacts``: one entry per existing file path in the result (and a
    preview page per structure file); the ids are fresh uuid4s, never chained."""
    out = []
    for k, (label, path) in enumerate(paths.items()):
        name = path.rsplit("/", 1)[-1]
        out.append({"label": label, "id": f"art_{k:032x}", "artifact_type": "structure",
                    "mimeType": "application/octet-stream", "filepath": path,
                    "download_url": f"/artifacts/art_{k:032x}/{name}"})
    return out


def _calls(ws, *, unit=None, supercell=None, vacancy=None, sp_inputs=None, analyze_input=None,
           elastic_input=None, n=3, energies=None, coordination=None, bulk_modulus=None):
    """The B1 calls of this instance as the pinned server answers them (forces, stress, RDF
    data and previews trimmed), for an agent working in ``ws``."""
    unit = unit or f"{ws}/unit_cell.extxyz"
    supercell = supercell or f"{ws}/supercell.extxyz"
    vacancy = vacancy or f"{ws}/vacancy.extxyz"
    sp_inputs = sp_inputs or (supercell, vacancy)
    analyze_input = analyze_input or vacancy
    elastic_input = elastic_input or unit
    n_atoms = 4 * n ** 3
    e_perfect, e_vacancy = energies or (REFERENCE["energy_perfect_eV"], REFERENCE["energy_vacancy_eV"])
    a = CASE["lattice_constant"]
    cell = [[a * n, 0.0, 0.0], [0.0, a * n, 0.0], [0.0, 0.0, a * n]]

    def manipulate(op, kwargs, src, dst, atoms):
        return ("manipulate_structure_workflow",
                {"input_filepath": src, "operation": op, "operation_kwargs": kwargs, "output_filepath": dst},
                {"status": "success", "operation": op, "input_filepath": src, "filepath": dst, "format": "extxyz",
                 "num_atoms": atoms, "formula": f"Pd{atoms}", "cell": cell,
                 "artifacts": _artifacts(input_filepath=src, filepath=dst)})

    def single_point(path, energy, atoms):
        return ("single_point_workflow", {"input_filepath": path, "calculator_name": "emt"},
                {"input_filepath": path, "energy": energy, "forces": [[0.0, 0.0, 0.0]] * 2,
                 "stress": [0.0] * 6, "calculator_requested": "emt", "calculator_used": "emt",
                 "calculator_fallbacks": [], "num_atoms": atoms, "artifacts": _artifacts(input_filepath=path)})

    return [
        ("build_structure_workflow",
         {"formula": "Pd", "crystal_system": "fcc", "lattice_constant": a, "output_filepath": unit},
         {"filepath": unit, "format": "extxyz", "formula": "Pd4", "num_atoms": 4,
          "cell": [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
          "symmetry": {"spacegroup": "Fm-3m", "crystal_system": "cubic", "point_group": "m-3m"},
          "artifacts": _artifacts(filepath=unit)}),
        manipulate("supercell", {"size": [n, n, n]}, unit, supercell, n_atoms),
        manipulate("vacancy", {"index": CASE["vacancy_index"]}, supercell, vacancy, n_atoms - 1),
        single_point(sp_inputs[0], e_perfect, n_atoms),
        single_point(sp_inputs[1], e_vacancy, n_atoms - 1),
        ("analyze_structure_workflow", {"filepath": analyze_input, "output_dir": f"{ws}/analysis"},
         {"filepath": analyze_input, "format": "extxyz",
          "info": {"formula": "Pd107", "num_atoms": n_atoms - 1, "spacegroup": "Pm-3m"},
          "analysis": {"summary": {"input_filepath": analyze_input, "num_atoms": n_atoms - 1,
                                   "coordination": {"cutoff": None, "factor": 1.2,
                                                    "average": coordination or REFERENCE["coordination_average"],
                                                    "by_element": {"Pd": REFERENCE["coordination_average"]}}},
                       "outputs": {"summary_json": f"{ws}/analysis/structure_summary.json"}},
          "artifacts": _artifacts(filepath=analyze_input)}),
        ("estimate_elastic_workflow",
         {"input_filepath": elastic_input, "calculator_name": "emt", "strain_max": CASE["strain_max"]},
         {"input_filepath": elastic_input, "samples": [], "bulk_modulus_GPa": bulk_modulus or REFERENCE["bulk_modulus_GPa"],
          "volume_A3": REFERENCE["unit_cell_volume_A3"], "calculator_requested": "emt", "calculator_used": "emt",
          "calculator_fallbacks": [], "method": "isotropic E(strain) quadratic fit",
          "artifacts": _artifacts(input_filepath=elastic_input)}),
    ]


def _stream(calls, extra=()):
    """Claude Code stream-json. Every tool declares a bare ``object`` outputSchema, so Claude
    shows the business dict itself (no ``result`` wrapper)."""
    events = [claude.init(SERVER, ["Bash", "Read", "Write", "WebFetch", "WebSearch",
                                   *(f"mcp__{SERVER}__{t}" for t in TOOLS)])]
    for k, (name, args) in enumerate(extra):
        events += claude.call(f"x{k}", name, args, "ok")
    for k, (tool, args, payload) in enumerate(calls):
        events += claude.call(f"c{k}", f"mcp__{SERVER}__{tool}", args, json.dumps(payload))
    events.append(claude.result(len(calls)))
    return jsonl(events)


def _codex(calls):
    events = list(codex.START)
    for k, (tool, args, payload) in enumerate(calls):
        events += codex.mcp(f"m{k}", SERVER, tool, args, json.dumps(payload), structured=payload)
    events.append(codex.done())
    return jsonl(events)


def _ws(tmp_path):
    """The agent's working directory as ``persist_like_run`` scrubs it (``<workspace>``)."""
    return str(tmp_path / "ws")


def _run(tmp_path, stream, answer, *, codex=False):
    return TASK.verify(tmp_path, stream, reference=REFERENCE, answer=answer,
                       harness="codex" if codex else "claude")


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def test_cases_are_deterministic_and_varied():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415) == CASE
    cases = [generate_gt.build_case(s) for s in range(300)]
    assert {c["element"] for c in cases} == set(generate_gt.ELEMENTS) and "Al" not in generate_gt.ELEMENTS
    assert {c["supercell"] for c in cases} == {2, 3}
    assert {c["strain_max"] for c in cases} == set(generate_gt.STRAIN_MAX)
    assert len({(c["element"], c["lattice_constant"]) for c in cases}) > 200
    for c in cases:
        a0 = generate_gt.ELEMENTS[c["element"]]["a0"]
        assert abs(c["lattice_constant"] / a0 - 1) <= generate_gt.STRAIN_RANGE + 1e-3
        assert 0 <= c["vacancy_index"] < 4 * c["supercell"] ** 3
        assert c["calculator"] == "emt" and c["vacancy_file"] == "vacancy.extxyz"


def test_cases_stay_clear_of_neighbour_shells():
    """A neighbour shell within 0.02 A of a coordination cutoff would let rounding decide the
    count; the generator skips those lattice constants (Pd at +2 % sits 0.003 A from one)."""
    for c in (generate_gt.build_case(s) for s in range(300)):
        r_cov = generate_gt.ELEMENTS[c["element"]]["r_cov"]
        assert generate_gt.clear_of_shells(c["lattice_constant"], r_cov), c
    assert not generate_gt.clear_of_shells(3.933, 1.39)              # Pd: second shell at the D4 cutoff 3.936
    with_skin, physical = generate_gt.cutoffs(1.32)
    assert with_skin == pytest.approx(3.768) and physical == pytest.approx(3.168)


def test_reference_matches_the_recorded_one(tmp_path):
    """Needs ASE (the version the task's runtime pins)."""
    pytest.importorskip("ase")
    generate_gt.generate(tmp_path, {"seed": 31415})
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    for key, value in REFERENCE.items():
        if isinstance(value, float):
            assert ref[key] == pytest.approx(value, abs=1e-12, rel=1e-12), key
        else:
            assert ref[key] == value, key
    assert json.loads((tmp_path / "data/calculation.json").read_text()) == CASE
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{" not in text
        assert ("3.882" in text) == (level in ("b1", "b2", "b3"))


def test_reference_carries_every_key_the_checks_use():
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    assert keys <= set(REFERENCE)
    assert spec["server_tools"] == TOOLS
    links = {cs["name"]: (cs["inputs_from_call"]["call"], cs["inputs_from_call"]["args"])
             for cs in spec["calls"] if cs.get("inputs_from_call")}
    assert links == {"supercell": ("build", ["input_filepath"]), "vacancy": ("supercell", ["input_filepath"]),
                     "sp_perfect": ("supercell", ["input_filepath"]), "sp_vacancy": ("vacancy", ["input_filepath"]),
                     "analyze": ("vacancy", ["filepath"]), "elastic": ("build", ["input_filepath"])}
    assert all(cs["inputs_from_call"]["extract"] == "output_file" for cs in spec["calls"] if cs.get("inputs_from_call"))
    # the derived formation energy is the scorer's business: no tool returns it
    assert "vacancy_formation_energy_eV" not in {a["prediction_key"] for a in spec["answers"]}


def test_prompts_name_the_server_only_at_b1_b2_and_keep_the_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert ("`atomictoolkit` MCP server" in text) == (level in ("b1", "b2"))
        assert ("`manipulate_structure_workflow`" in text) == (level in ("b1", "b2"))
        assert '`size`' in text and '`index`' in text and "silently ignores" in text      # D10, every level
        assert "unrelaxed" in text and "absolute path" in text and "`calculator_name` = `emt`" in text
        assert "E_vac = E_vacancy − E_perfect × (N − 1) / N" in text
        assert "no `import ase`" in text and "Do not modify `data/calculation.json`" in text
        assert "full precision" in text


def test_task_meta_pins_the_runtime_to_ase():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == ["ase==3.29.0"]
    assert [f["name"] for f in meta["output"]["files"]] == ["result.json"]


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------

def test_tool_values_score_full(tmp_path):
    assert _total(*_dirs(tmp_path, _answer())) == pytest.approx(100.0)
    # numbers written as strings, or the formation energy recomputed in another order
    e_vac = REFERENCE["energy_vacancy_eV"] - REFERENCE["energy_perfect_eV"] * 107 / 108
    answer = _answer(energy_perfect_eV=str(REFERENCE["energy_perfect_eV"]), vacancy_formation_energy_eV=e_vac)
    assert _total(*_dirs(tmp_path / "b", answer)) == pytest.approx(100.0)


def test_wrong_procedures_lose_their_credit(tmp_path):
    # operation_kwargs {"repeat": ...} is ignored (D10): the default 2x2x2 supercell of 32 atoms
    small = _answer(n_atoms_perfect=32, energy_perfect_eV=-0.006982958353848545,
                    vacancy_formation_energy_eV=0.948021841332288)
    total = _total(*_dirs(tmp_path / "a", small))
    assert 50.0 < total < 70.0                       # energies and atom count gone, formation energy partial
    # the bulk modulus fitted over another strain range (0.01 instead of 0.02)
    assert _total(*_dirs(tmp_path / "b", _answer(bulk_modulus_GPa=177.49235995393028))) == pytest.approx(80.0)
    # the physical coordination (ASE neighbour list without skin) instead of the server's count
    physical = _answer(coordination_average=REFERENCE["coordination_average_skin0"])
    assert _total(*_dirs(tmp_path / "c", physical)) == pytest.approx(85.0)


def test_rounded_values_lose_part_of_the_credit(tmp_path):
    rounded = _answer(energy_vacancy_eV=round(REFERENCE["energy_vacancy_eV"], 4))
    assert 85.0 < _total(*_dirs(tmp_path, rounded)) < 100.0


def test_credit_curve():
    assert scorer.credit(1e-7, 1e-6, 1e-3) == 1.0 and scorer.credit(1e-3, 1e-6, 1e-3) == 0.0
    assert scorer.credit(math.inf, 1e-6, 1e-3) == 0.0
    assert scorer.credit(10 ** -4.5, 1e-6, 1e-3) == pytest.approx(0.5)


@pytest.mark.parametrize("prediction", [
    None, "not json", [], {"n_atoms_perfect": 108},
    _answer(energy_perfect_eV=None), _answer(bulk_modulus_GPa="stiff"), _answer(n_atoms_perfect=107.5),
    _answer(coordination_average=float("nan")), _answer(n_atoms_perfect=True),
])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    """A broken result.json is an ordinary zero for every scorer (never an evaluator failure),
    and the hard gate zeroes the instance as a whole."""
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, prediction)
    gate = get_scorer("atomictoolkit_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in TASK.eval_config()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error"), item["scorer"]


@pytest.mark.parametrize("reference", [None, {"energy_perfect_eV": 1.0}, {**REFERENCE, "n_atoms_perfect": "many"}])
def test_missing_reference_is_an_evaluator_failure(tmp_path, reference):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, _answer(), reference=reference)
    for item in TASK.eval_config()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.details["scorer_internal_error"] and detail.details["failure_kind"] == "missing_evaluator_input"


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------

def test_genuine_b1_run_passes_every_check(tmp_path):
    row = _run(tmp_path, _stream(_calls(_ws(tmp_path))), _answer())
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert row["tool_call_counts"]["mcp__atomictoolkit__single_point_workflow"] == 2


def test_codex_run_passes(tmp_path):
    row = _run(tmp_path, _codex(_calls(_ws(tmp_path))), _answer(), codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


def test_paths_scrubbed_in_the_persisted_log_still_chain_by_file_name(tmp_path):
    """``input_filepath`` / ``filepath`` are not trajectory key args: the persisted log is all
    that records them, as ``<workspace>/<name>``, while the trajectory restores the results'
    real paths. The ``output_file`` link compares file names, so the chain still holds."""
    stream = _stream(_calls(_ws(tmp_path)))
    persisted, steps = support.persist_like_run(stream, "claude", tmp_path=tmp_path)
    assert "<workspace>/supercell.extxyz" in persisted and _ws(tmp_path) not in persisted
    assert not any("input_filepath" in (s.get("metadata", {}).get("key_args") or {}) for s in steps)
    row = _run(tmp_path, stream, _answer())
    assert _status(row)["tool_chain"] == "PASS" and row["verdict"] == "PASS"


def test_relative_paths_into_the_server_checkout_still_chain(tmp_path):
    """Relative names land in the server's cwd, the MCP checkout under the home directory
    (``<home>/mcp/atomictoolkit/...`` once persisted). The prompt asks for absolute paths in the
    working directory, but the chain is still the server's own files."""
    checkout = f"{SENTINEL_HOME}/mcp/atomictoolkit"
    row = _run(tmp_path, _stream(_calls(checkout)), _answer())
    assert row["verdict"] == "PASS" and _status(row)["tool_chain"] == "PASS"


def test_paths_outside_the_workspace_are_a_coverage_gap(tmp_path):
    """A path outside the workspace and home is persisted as a bare ``<abs_path>``: no file name
    is left to compare, so the link is unobservable (WARN), not broken."""
    row = _run(tmp_path, _stream(_calls("/opt/scratch")), _answer())
    assert _status(row)["tool_chain"] == "WARN" and row["verdict"] == "PASS"
    assert "scrubbed" in row["checks"]["tool_chain"]["detail"]


def test_a_structure_the_agent_wrote_itself_breaks_the_chain(tmp_path):
    """The defect energy computed on a file no manipulate call wrote (e.g. re-typed by the agent)."""
    ws = _ws(tmp_path)
    calls = _calls(ws, sp_inputs=(f"{ws}/supercell.extxyz", f"{ws}/my_vacancy.extxyz"),
                   analyze_input=f"{ws}/my_vacancy.extxyz")
    row = _run(tmp_path, _stream(calls), _answer())
    assert _status(row)["tool_chain"] == "FAIL" and row["verdict"] == "FAIL"
    assert "analyze←vacancy" in row["checks"]["tool_chain"]["detail"]


def test_the_ignored_kwarg_gives_the_wrong_supercell(tmp_path):
    """D10: ``{"repeat": [3, 3, 3]}`` is silently ignored, so the server builds 2x2x2."""
    ws = _ws(tmp_path)
    calls = _calls(ws, n=2, energies=(-0.006982958353848545, 1.0))
    row = _run(tmp_path, _stream(calls), _answer(n_atoms_perfect=32, energy_perfect_eV=-0.006982958353848545,
                                                 energy_vacancy_eV=1.0))
    status = _status(row)
    assert status["tool_correct"] == "FAIL" and status["answer_from_tool"] == "FAIL" and row["verdict"] == "FAIL"
    per_call = row["checks"]["tool_correct"]["per_call"]
    assert per_call["supercell"]["status"] == "FAIL" and per_call["build"]["status"] == "PASS"


def test_answers_must_be_copied_from_tool_results(tmp_path):
    """Inside the scorer's tolerance but not what single_point_workflow returned."""
    answer = _answer(energy_vacancy_eV=REFERENCE["energy_vacancy_eV"] + 1e-8)
    row = _run(tmp_path, _stream(_calls(_ws(tmp_path))), answer)
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "energy_vacancy_eV" in row["checks"]["answer_from_tool"]["detail"]


def test_the_formation_energy_is_scored_not_traced(tmp_path):
    """No tool returns E_vac, so the verifier does not trace it; the scorer grades it."""
    answer = _answer(vacancy_formation_energy_eV=0.95)
    assert _run(tmp_path, _stream(_calls(_ws(tmp_path))), answer)["verdict"] == "PASS"
    assert _total(*_dirs(tmp_path / "s", answer)) < 80.0


def test_unavailable_tools_tried_first_do_not_matter(tmp_path):
    """optimize_structure_workflow refuses a plain call (D2) and the deprecated build_structure
    always fails (D1); an agent that tries them and then follows the chain still passes."""
    stream_events = [claude.init(SERVER, [f"mcp__{SERVER}__{t}" for t in TOOLS])]
    stream_events += claude.call("o1", f"mcp__{SERVER}__optimize_structure_workflow",
                                 {"input_filepath": f"{_ws(tmp_path)}/vacancy.extxyz"},
                                 "Tool 'optimize_structure_workflow' requires task-augmented execution", True)
    stream_events += claude.call("o2", f"mcp__{SERVER}__build_structure", {"formula": "Pd"},
                                 "Error calling tool 'build_structure': 'FunctionTool' object is not callable", True)
    body = _stream(_calls(_ws(tmp_path))).splitlines()[1:]
    row = _run(tmp_path, jsonl(stream_events) + "\n".join(body) + "\n", _answer())
    assert row["verdict"] == "PASS" and set(_status(row).values()) == {"PASS"}


@pytest.mark.parametrize("command,status", [
    ("python3 -c 'import ase'", "FAIL"),
    ("python3 -c 'from ase.calculators.emt import EMT'", "FAIL"),
    ("python3 -m ase gui vacancy.extxyz", "FAIL"),
    ("pip install ase", "FAIL"),
    ("uv run --with ase python fit.py", "FAIL"),
    ("uvx --from ase ase info", "FAIL"),
    ("pip install pymatgen", "FAIL"),
    ("~/mcp/atomictoolkit/.venv/bin/python -c 'print(1)'", "FAIL"),
    ("python3 -c 'import mcp_atomictoolkit'", "FAIL"),
    ("ls ~/mcp/atomictoolkit", "WARN"),
    ("python3 -c \"print(0.9246658707987856 - (-0.023567484444418696) * 107 / 108)\"", "PASS"),
    ("cat data/calculation.json", "PASS"),
])
def test_shell_access_to_the_backend(tmp_path, command, status):
    row = _run(tmp_path, _stream(_calls(_ws(tmp_path)), extra=[("Bash", {"command": command})]), _answer())
    assert _status(row)["no_bypass"] == status


def test_web_lookup_of_the_physics_is_only_suspicious(tmp_path):
    row = _run(tmp_path, _stream(_calls(_ws(tmp_path)),
                                 extra=[("WebSearch", {"query": "Pd vacancy formation energy EMT"})]), _answer())
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"
    plain = _run(tmp_path / "b", _stream(_calls(_ws(tmp_path / "b")),
                                         extra=[("WebFetch", {"url": "https://example.org/weather"})]), _answer())
    assert _status(plain)["no_bypass"] == "PASS"
