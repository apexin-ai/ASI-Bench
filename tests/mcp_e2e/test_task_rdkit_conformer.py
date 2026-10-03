"""rdkit_conformer (SMILES -> seeded ETKDG conformer -> SDF, plus Gasteiger charge extremes):
generator, scorer, verifier scenarios. No MCP server and no RDKit at import time."""
import base64
import json
import math
import pickle

import pytest

from . import support
from .support import ROOT, claude, codex, jsonl

TASK = support.Task("mcp_e2e.rdkit_conformer")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id
SERVER = "rdkit"
TOOLS = support.setup.load_manifest()[SERVER]["expected_tools"]

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify
_status = support.statuses

# Seed 31415 (salbutamol, random_seed 36888) as generated with RDKit 2025.3.1 on 2026-10-02.
SMILES = "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"
REFERENCE = {
    "molecule": "salbutamol",
    "smiles": SMILES,
    "random_seed": 36888,
    "output_file": "conformer.sdf",
    "mol_pickle":
        "gASVIgEAAAAAAACMEXJka2l0LkNoZW0ucmRjaGVtlIwDTW9slJOUQ/nvvq3eAAAAABAAAAACAAAAAAAAABEAAAARAAAAgAEGAGAA"
        "AAABAwYAIAAAAAQGAGAAAAABAwYAYAAAAAEDBwBgAAAAAgEGAGAAAAACAgYAYAAAAAMBCABgAAAAAQEGQCgAAAADBAZAaAAAAAMD"
        "AQZAaAAAAAMDAQZAKAAAAAMECABoAAAAAwEBBkAoAAAAAwQGAGAAAAACAggAYAAAAAEBBkBoAAAAAwMBCwABAAECAAEDAAEEAAQF"
        "AAUGAAYHAAYIAAgJaAwJCmgMCgtoDAsMIAsNaAwNDgAODwANEGgMEAhoDEIBAAAABggQDQsKCRcEAAAAAAAAABaUhZRSlH2UhZRi"
        "Lg==",
    "embedded_mol_pickle":
        "gASV9wEAAAAAAACMEXJka2l0LkNoZW0ucmRjaGVtlIwDTW9slJOUQssBAADvvq3eAAAAABAAAAACAAAAAAAAABEAAAARAAAAgAEG"
        "AGAAAAABAwYAIAAAAAQGAGAAAAABAwYAYAAAAAEDBwBgAAAAAgEGAGAAAAACAgYAYAAAAAMBCABgAAAAAQEGQCgAAAADBAZAaAAA"
        "AAMDAQZAaAAAAAMDAQZAKAAAAAMECABoAAAAAwEBBkAoAAAAAwQGAGAAAAACAggAYAAAAAEBBkBoAAAAAwMBCwABAAECAAEDAAEE"
        "AAQFAAUGAAYHAAYIAAgJaAwJCmgMCgtoDAsMIAsNaAwNDgAODwANEGgMEAhoDEIBAAAABggQDQsKCRfWAAAAAQAAAAEAAAAAEUZY"
        "fUD8XKY9rRkBQL9WhEDm9cS8ADwEP0/bfUCMErM/i5+CvOBJsEDJUva+TNUSPvnKRUBFjFK/YNHTvaG35j+2rn2+vPECPr3vKj+M"
        "bYa/p/gAv7SnXj8Ew46/jSLzv+XKGL91FKS+6dVaviuGKr+9Syg/3ARJP8Ou8L9EVKc/+BiBP+pIP8AJy4A/nLyMPrdZhsDNLNU/"
        "MoMCP6gHO8DW3D89QrE0vycBg8AKAKy++UvGv8J8ncCkF5G/L1NBvwKs3b/K7iG/ugl4vxaUhZRSlH2UhZRiLg==",
    "atoms": [
        ["C", 3.958512783050537, 0.08123204112052917, 2.0171921253204346],
        ["C", 4.135589122772217, -0.02404303476214409, 0.51654052734375],
        ["C", 3.966510534286499, 1.399003505706787, -0.01594521664083004],
        ["C", 5.5090179443359375, -0.4811003506183624, 0.14339178800582886],
        ["N", 3.0905134677886963, -0.8224528431892395, -0.10342669486999512],
        ["C", 1.8024789094924927, -0.24773678183555603, 0.12787526845932007],
        ["C", 0.6677206158638, -1.050218105316162, -0.5037941336631775],
        ["O", 0.8697464466094971, -1.1153264045715332, -1.8994919061660767],
        ["C", -0.5968459248542786, -0.32046857476234436, -0.2137066274881363],
        ["C", -0.6661097407341003, 0.65740567445755, 0.7852303981781006],
        ["C", -1.8803333044052124, 1.3072590827941895, 1.0085744857788086],
        ["C", -2.9888253211975098, 1.006196141242981, 0.2748764753341675],
        ["O", -4.198451519012451, 1.665429711341858, 0.5098143815994263],
        ["C", -2.922342300415039, 0.046841464936733246, -0.7058297395706177],
        ["C", -4.09389066696167, -0.3359377980232239, -1.5491935014724731],
        ["O", -4.921479225158691, -1.1335339546203613, -0.7551755309104919],
        ["C", -1.731811761856079, -0.6325498819351196, -0.9688984155654907]
    ],
    "conf_id": 0,
    "max_partial_charge": 0.12060718497834182,
    "min_partial_charge": -0.5075837293484645,
    "sdf_file": "conformer.sdf",
    "rdkit_version": "2025.03.1",
}


def _sdf(atoms=None, name="conformer.sdf"):
    """A V2000 molfile record in the layout RDKit writes (4 decimals)."""
    atoms = atoms if atoms is not None else REFERENCE["atoms"]
    lines = [REFERENCE["molecule"], "     RDKit          3D", "",
             f"{len(atoms):3d}{len(atoms):3d}  0  0  0  0  0  0  0  0999 V2000"]
    lines += [f"{x:10.4f}{y:10.4f}{z:10.4f} {s:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
              for s, x, y, z in atoms]
    lines += ["M  END", "$$$$"]
    return "\n".join(lines) + "\n"


def _answer(**over):
    data = {"molecule": "salbutamol", "conf_id": 0,
            "max_partial_charge": REFERENCE["max_partial_charge"],
            "min_partial_charge": REFERENCE["min_partial_charge"]}
    data.update(over)
    return data


def _dirs(tmp_path, prediction, reference=REFERENCE, sdf=None):
    pred, ref = support.score_dirs(tmp_path, prediction, reference)
    (pred / "conformer.sdf").write_text(sdf if sdf is not None else _sdf())
    return pred, ref


def _eval():
    return TASK.eval_config()


def _total(pred, ref):
    return TASK.total(pred, ref)


def _calls(mol_pickle=None, embedded=None, sdf_path="/tmp/ws/conformer.sdf", smiles=SMILES,
           seed=None, qmax=None, qmin=None):
    """The B1 calls of this instance, as the pinned server answered them."""
    mol_pickle = REFERENCE["mol_pickle"] if mol_pickle is None else mol_pickle
    embedded = REFERENCE["embedded_mol_pickle"] if embedded is None else embedded
    seed = REFERENCE["random_seed"] if seed is None else seed
    qmax = REFERENCE["max_partial_charge"] if qmax is None else qmax
    qmin = REFERENCE["min_partial_charge"] if qmin is None else qmin
    return [
        ("smiles_to_mol", {"smiles": smiles}, {"result": mol_pickle}),
        ("EmbedMolecule", {"p_mol": mol_pickle, "params": {"randomSeed": seed}},
         {"conf_id": 0, "mol": embedded}),
        ("mol_to_sdf", {"pmol": embedded, "file_dir": "/tmp/ws", "filename": "conformer.sdf"},
         {"result": sdf_path}),
        ("MaxPartialCharge", {"smiles": smiles}, {"result": qmax}),
        ("MinPartialCharge", {"smiles": smiles}, {"result": qmin}),
    ]


def _stream(calls, extra=(), structured=True):
    """Claude Code stream-json. smiles_to_mol returns a plain string, so Claude shows the
    structuredContent wrapper; EmbedMolecule returns a pydantic object, so it shows the object."""
    events = [claude.init(SERVER, ["Bash", "Read", "Write", "WebFetch", "WebSearch",
                                   *(f"mcp__{SERVER}__{t}" for t in TOOLS)])]
    for k, (name, args) in enumerate(extra):
        events += claude.call(f"x{k}", name, args, "ok")
    for k, (tool, args, payload) in enumerate(calls):
        shown = json.dumps({"result": payload["result"]} if "result" in payload else payload,
                           indent=None if structured else 2)
        events += claude.call(f"c{k}", f"mcp__{SERVER}__{tool}", args, shown)
    events.append(claude.result(len(calls)))
    return jsonl(events)


def _codex(calls):
    events = list(codex.START)
    for k, (tool, args, payload) in enumerate(calls):
        shown = json.dumps(payload, indent=2)
        events += codex.mcp(f"m{k}", SERVER, tool, args, shown, structured=payload)
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream, answer, *, sdf=None, codex=False, files=None):
    files = {"conformer.sdf": sdf if sdf is not None else _sdf()} if files is None else files
    return TASK.verify(tmp_path, stream, reference=REFERENCE, answer=answer,
                       harness="codex" if codex else "claude", files=files)


def test_cases_are_deterministic_and_varied():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415)
    cases = [generate_gt.build_case(s) for s in range(60)]
    assert {c["molecule"] for c in cases} == set(generate_gt.MOLECULES)
    assert {c["smiles"] for c in cases} == set(generate_gt.MOLECULES.values())
    assert len({c["random_seed"] for c in cases}) > 40
    assert all(c["output_file"] == "conformer.sdf" for c in cases)
    assert generate_gt.build_case(31415)["molecule"] == REFERENCE["molecule"]
    assert generate_gt.build_case(31415)["random_seed"] == REFERENCE["random_seed"]


def test_reference_matches_the_recorded_one(tmp_path):
    """Needs RDKit (the same version the task's runtime pins)."""
    pytest.importorskip("rdkit")
    generate_gt.generate(tmp_path, {"seed": 31415})
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    assert ref["mol_pickle"] == REFERENCE["mol_pickle"]
    assert ref["embedded_mol_pickle"] == REFERENCE["embedded_mol_pickle"]
    assert ref["conf_id"] == REFERENCE["conf_id"] == 0
    assert ref["atoms"] == REFERENCE["atoms"]
    assert ref["max_partial_charge"] == pytest.approx(REFERENCE["max_partial_charge"], abs=1e-12)
    assert json.loads((tmp_path / "data/molecule.json").read_text())["smiles"] == SMILES
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{" not in text
        assert ("salbutamol" in text) == (level in ("b1", "b2", "b3"))


def test_reference_carries_every_key_the_checks_use():
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    assert keys <= set(REFERENCE)
    assert spec["server_tools"] == TOOLS
    linked = {cs["name"]: cs["inputs_from_call"] for cs in spec["calls"] if cs.get("inputs_from_call")}
    assert linked["embed"]["map"] == {"p_mol": "*"}          # smiles_to_mol returns one bare string
    assert linked["write"]["map"] == {"pmol": "mol"}


def test_prompts_name_the_server_only_at_b1_b2_and_keep_the_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert ("`rdkit` MCP server" in text) == (level in ("b1", "b2"))
        assert ("`smiles_to_mol`" in text) == (level == "b1")
        assert "no `import rdkit`" in text and "Do not modify `data/molecule.json`" in text
        assert "verbatim" in text and "full precision" in text
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text
    assert "`EmbedMolecule`" in (TASK_DIR / "prompt_b1.md").read_text()


def test_task_meta_pins_the_runtime_to_rdkit():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == ["rdkit==2025.3.1"]
    assert [f["name"] for f in meta["output"]["files"]] == ["result.json", "conformer.sdf"]


def test_molfile_parser_reads_what_rdkit_writes():
    atoms = scorer.parse_molfile(_sdf())
    assert len(atoms) == len(REFERENCE["atoms"])
    assert atoms[0][0] == "C" and atoms[4][0] == "N"
    assert atoms[1][1:] == pytest.approx(REFERENCE["atoms"][1][1:], abs=1e-4)
    truncated = "\n".join(_sdf().splitlines()[:6])          # counts line and two atom lines
    for bad, match in (("no counts line here", "V2000"), (_sdf()[:20], "V2000"),
                       ("\n\n\n  2  1  0  0  0  0  0  0  0  0999 V2000\n", "announces"),
                       (truncated, "lines after it"),
                       ("\n\n\n  x  1  0  0  0  0  0  0  0  0999 V2000\n", "bad counts line")):
        with pytest.raises(scorer._PredictionError, match=match):
            scorer.parse_molfile(bad)


def test_tool_values_score_full(tmp_path):
    assert _total(*_dirs(tmp_path, _answer())) == pytest.approx(100.0)
    # the same conformer written with 5 decimals or through a >3D record still scores full
    more = [[s, x + 1e-6, y, z] for s, x, y, z in REFERENCE["atoms"]]
    assert _total(*_dirs(tmp_path / "b", _answer(), sdf=_sdf(more))) == pytest.approx(100.0)


def test_wrong_conformer_loses_the_conformer_credit(tmp_path):
    # another seed: ETKDG frames are not aligned, so atoms move by Angstroms
    shifted = [[s, x + 0.7, y - 1.3, z + 2.1] for s, x, y, z in REFERENCE["atoms"]]
    assert _total(*_dirs(tmp_path / "a", _answer(), sdf=_sdf(shifted))) == pytest.approx(40.0)
    # hydrogens added (AddHs) or a different molecule: the atom list no longer matches
    added = [[*REFERENCE["atoms"][0]], *[[ "H", x, y, z] for _, x, y, z in REFERENCE["atoms"]]]
    assert _total(*_dirs(tmp_path / "b", _answer(), sdf=_sdf(added))) == pytest.approx(40.0)
    other = [[s if s != "N" else "C", x, y, z] for s, x, y, z in REFERENCE["atoms"]]
    assert _total(*_dirs(tmp_path / "c", _answer(), sdf=_sdf(other))) == pytest.approx(40.0)


def test_rounded_charges_lose_part_of_the_credit(tmp_path):
    rounded = _answer(max_partial_charge=round(REFERENCE["max_partial_charge"], 3))
    total = _total(*_dirs(tmp_path, rounded))
    assert 60.0 < total < 100.0
    assert _total(*_dirs(tmp_path / "b", _answer(max_partial_charge=REFERENCE["max_partial_charge"] + 0.01)))         == pytest.approx(80.0)


def test_coordinate_and_charge_credit_curves():
    assert scorer.credit(1e-5, 2e-4, 0.1) == 1.0 and scorer.credit(0.1, 2e-4, 0.1) == 0.0
    assert 0.0 < scorer.credit(1e-2, 2e-4, 0.1) < 1.0
    assert scorer.credit(1e-9, 1e-6, 1e-3) == 1.0 and scorer.credit(1e-3, 1e-6, 1e-3) == 0.0
    assert math.isinf(scorer.coordinate_error([("C", 0, 0, 0)], [("C", 0, 0, 0), ("O", 0, 0, 0)]))
    assert scorer.coordinate_error([("C", 1, 0, 0)], [("C", 0, 0, 0)]) == pytest.approx(1.0)


@pytest.mark.parametrize("prediction,sdf,broken", [
    (None, None, "result.json"), ("not json", None, "result.json"), ([], None, "result.json"),
    ({"molecule": "x"}, None, "result.json"),
    (_answer(max_partial_charge=None), None, "result.json"),
    (_answer(min_partial_charge="low"), None, "result.json"),
    (_answer(max_partial_charge=float("nan")), None, "result.json"),
    (_answer(), "not a molfile", "conformer.sdf"), (_answer(), "", "conformer.sdf"),
    (None, "", "both"),
])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction, sdf, broken):
    """A broken artefact is an ordinary zero for the scorers that read it (never an
    evaluator failure); the other artefact keeps its own credit, and the hard gate
    zeroes the instance as a whole.
    """
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, prediction, sdf=sdf)
    gate = get_scorer("rdkit_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in _eval()["scoring"]:
        config = {**item["config"], "weight": item["weight"]}
        detail = get_scorer(item["scorer"]).score(pred, ref, config)
        reads = {config.get("pred_file"), config.get("sdf_file")} - {None}
        affected = broken == "both" or broken in reads
        assert (detail.score == 0.0) is affected, (item["scorer"], broken, detail)
        assert not detail.details.get("scorer_internal_error")


def test_conf_id_is_checked_from_the_run_not_by_the_scorer(tmp_path):
    """The scorers read the two charges and the SDF only. conf_id has no reference-free
    meaning in a file, so that it came from the embedding is checked from the run
    artefacts (test_conf_id_must_come_from_the_embedding)."""
    assert _total(*_dirs(tmp_path, _answer(conf_id="zero"))) == pytest.approx(100.0)


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, _answer(), reference=None)
    for item in _eval()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.details["scorer_internal_error"] and detail.details["failure_kind"] == "missing_evaluator_input"


@pytest.mark.parametrize("structured", [True, False])
def test_genuine_b1_run_passes_every_check(tmp_path, structured):
    row = _run(tmp_path, _stream(_calls(), structured=structured), _answer())
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert row["tool_call_counts"]["mcp__rdkit__EmbedMolecule"] == 1


def test_codex_run_passes(tmp_path):
    row = _run(tmp_path, _codex(_calls()), _answer(), codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}


def _scrubbed_calls(calls):
    """The same calls as the persisted log shows them: every path-like run inside the
    base64 pickles replaced by <abs_path>, in the tool results and in the agent's inputs."""
    def scrub(value):
        if isinstance(value, str):
            return verify.values.scrub_host_paths(value)
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value
    return [(tool, scrub(args), scrub(payload)) for tool, args, payload in calls]


def test_pickles_scrubbed_in_the_persisted_log_still_chain(tmp_path):
    """Absolute-path-like runs inside a base64 pickle are scrubbed in the saved stdout; the
    unscrubbed rest must still match, or every rdkit chain would look broken."""
    stream = _stream(_calls())
    persisted = support.persist_like_run(stream, "claude", tmp_path=tmp_path)[0]
    assert verify.values.SCRUBBED in persisted and REFERENCE["mol_pickle"] in persisted
    row = _run(tmp_path, stream, _answer())
    assert _status(row)["tool_chain"] == "PASS" and row["verdict"] == "PASS"


def _with_path_run(pickle_b64, at=40):
    """A base64 pickle carrying an absolute-path-like run, as other instances' do: the
    sanitizer only rewrites a ``/a/b`` run whose first slash does not follow a base64
    character, so the run is spliced in after a ``+``."""
    spliced = pickle_b64[:at] + "+/aa/bbbb/cc" + pickle_b64[at:]
    assert verify.values.SCRUBBED in verify.values.scrub_host_paths(spliced)
    return spliced


def test_a_fully_scrubbed_pickle_still_chains(tmp_path):
    """Instance 31415's own pickles survive persistence; other instances' carry a path-like
    run and come back as ``<abs_path>``, so the portable fix is exercised on a spec whose
    reference carries such a pickle too. Here nothing is left unscrubbed (the trajectory is
    taken from the already-scrubbed log), so the path mol_to_sdf returned is a coverage gap."""
    mol = _with_path_run(REFERENCE["mol_pickle"])
    embedded = _with_path_run(REFERENCE["embedded_mol_pickle"])
    ref = {**REFERENCE, "mol_pickle": mol, "embedded_mol_pickle": embedded}
    calls = _calls(mol_pickle=mol, embedded=embedded)
    row = TASK.verify(tmp_path, _stream(_scrubbed_calls(calls)), reference=ref, answer=_answer(),
                      harness="claude", files={"conformer.sdf": _sdf()})
    per_call = row["checks"]["tool_correct"]["per_call"]
    assert [per_call[name]["status"] for name in ("parse", "embed")] == ["PASS", "PASS"], row["checks"]
    assert per_call["write"]["status"] == "WARN" and "not observable" in per_call["write"]["detail"]
    assert _status(row)["tool_chain"] == "PASS" and row["verdict"] == "PASS"


def test_real_references_of_other_seeds_carry_pickles_the_sanitizer_rewrites(tmp_path):
    """Needs RDKit: why the comparison has to tolerate scrubbing at all. Kept free of
    ``verify_one`` so the golden snapshot does not depend on RDKit being installed."""
    pytest.importorskip("rdkit")
    ref = generate_gt.reference(generate_gt.build_case(3))                  # ibuprofen
    assert all(verify.values.SCRUBBED in verify.values.scrub_host_paths(ref[key])
               for key in ("mol_pickle", "embedded_mol_pickle"))


def test_other_seed_breaks_the_chain_and_the_answer_check(tmp_path):
    """Embedding without the seed: another conformer, chains still match (the agent passed the
    tool's own string on) but the returned molecule and the SDF are not the reference ones."""
    other = "gASW" + REFERENCE["embedded_mol_pickle"][4:]   # not a valid pickle, but a different string
    calls = _calls(embedded=other)
    calls[2] = ("mol_to_sdf", {"pmol": REFERENCE["embedded_mol_pickle"], "file_dir": "/tmp/ws",
                               "filename": "conformer.sdf"}, {"result": "/tmp/ws/conformer.sdf"})
    shifted = [[s, x + 0.9, y, z] for s, x, y, z in REFERENCE["atoms"]]
    row = _run(tmp_path, _stream(calls), _answer(), sdf=_sdf(shifted))
    status = _status(row)
    assert status["tool_correct"] == "FAIL" and status["tool_chain"] == "FAIL"
    assert row["verdict"] == "FAIL"


def test_answers_must_be_copied_from_tool_results(tmp_path):
    """Inside the scorer's tolerance but above what ``values.copied`` accepts as a copy:
    correct enough to score, still not the value MaxPartialCharge returned."""
    answer = _answer(max_partial_charge=REFERENCE["max_partial_charge"] + 1e-7)
    row = _run(tmp_path, _stream(_calls()), answer)
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "max_partial_charge" in row["checks"]["answer_from_tool"]["detail"]


def test_conf_id_must_come_from_the_embedding(tmp_path):
    row = _run(tmp_path, _stream(_calls()), _answer(conf_id=1))
    assert _status(row)["answer_from_tool"] == "FAIL"


def test_written_sdf_must_stay_a_molfile_of_the_reference_molecule(tmp_path):
    """A hand-written SDF (e.g. re-typed coordinates) still has to be the seeded conformer."""
    retyped = _sdf([[s, x + 0.05, y, z] for s, x, y, z in REFERENCE["atoms"]])
    row = _run(tmp_path, _stream(_calls()), _answer(), sdf=retyped)
    assert row["verdict"] == "PASS"                                  # 0.05 Angstrom is inside the credit range
    assert 40.0 < _total(*_dirs(tmp_path / "b", _answer(), sdf=retyped)) < 100.0


@pytest.mark.parametrize("command,status", [
    ("python3 -c 'import rdkit'", "FAIL"),
    ("python3 -c 'from rdkit import Chem'", "FAIL"),
    ("pip install rdkit", "FAIL"),
    ("conda install -c conda-forge rdkit", "FAIL"),
    ("uv run --with rdkit python embed.py", "FAIL"),
    ("uvx rdkit-pypi", "FAIL"),
    ("~/mcp/rdkit/.venv/bin/python -c 'print(1)'", "FAIL"),
    ("python3 -m rdkit_mcp.register_tools", "FAIL"),
    ("obabel conformer.sdf -O out.xyz", "FAIL"),
    ("ls ~/mcp/rdkit", "WARN"),
    ("python3 -c \"import json; json.dump({}, open('result.json','w'))\"", "PASS"),
])
def test_shell_access_to_rdkit(tmp_path, command, status):
    row = _run(tmp_path, _stream(_calls(), extra=[("Bash", {"command": command})]), _answer())
    assert _status(row)["no_bypass"] == status


def test_batch_helpers_are_only_suspicious(tmp_path):
    calls = [("batch_map", {"tool_name": "MaxPartialCharge", "inputs": [{"smiles": SMILES}]},
              {"results": []}), *_calls()]
    row = _run(tmp_path, _stream(calls), _answer())
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"
    assert "batch_map" in row["checks"]["no_bypass"]["detail"]


def test_web_lookup_of_the_chemistry_is_only_suspicious(tmp_path):
    row = _run(tmp_path, _stream(_calls(), extra=[("WebFetch", {"url": "https://example.org/rdkit"})]), _answer())
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"
    assert "WebFetch" in row["checks"]["no_bypass"]["detail"]
    plain = _run(tmp_path / "b", _stream(_calls(), extra=[("WebFetch", {"url": "https://example.org/weather"})]),
                 _answer())
    assert _status(plain)["no_bypass"] == "PASS"


def test_a_produced_script_importing_rdkit_is_a_bypass(tmp_path):
    row = _run(tmp_path, _stream(_calls()), _answer(),
               files={"conformer.sdf": _sdf(), "embed.py": "import rdkit\nprint(1)\n"})
    assert _status(row)["no_bypass"] == "FAIL"


def test_pickles_are_never_treated_as_numbers():
    """The reference carries base64 pickles; the verifier must not read them as values."""
    assert verify.values.numbers(REFERENCE["mol_pickle"]) is None
    assert verify.values.number(REFERENCE["conf_id"]) == 0.0
    assert verify.values.field(REFERENCE, "mol_pickle") == REFERENCE["mol_pickle"]
