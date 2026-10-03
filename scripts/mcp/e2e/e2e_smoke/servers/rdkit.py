"""Direct (agent-free) E2E smoke test for the pinned tandemai rdkit-mcp-server tools.

Run with the server's own virtualenv (it provides RDKit and Pillow for the references)::

    ~/mcp/rdkit/.venv/bin/python scripts/mcp/e2e/smoke.py rdkit --config ~/mcp/rdkit.mcp.json

No network is needed. The server is a thin wrapper around RDKit, so the
references are RDKit itself, called here, in this process, outside the server:
every tool result is compared with the documented RDKit function applied to a
molecule parsed in the smoke process, plus a few values that do not depend on
RDKit at all (hand-computed masses, valence-electron counts, formulae, Ertl
TPSA contributions, ring and stereocentre counts). What L1 proves is that the
wrapper passes arguments through, returns the right quantity and serialises
it faithfully — not that RDKit is right.

Molecules travel between tools as base64 Python pickles (``p_mol``/``pmol``);
RDKit pickles conformers as float32, so coordinates are compared to 1e-5 Å.
Files are written into a scratch directory given as ``file_dir``.

L1: real tools/call of all 74 tools:
      descriptors   45 single-SMILES descriptor tools on six drug-like
                    molecules vs RDKit, options (includeHs, strict), hand
                    anchors, compute_descriptors with all 36 names
      structure     SMILES/SMARTS <-> pickle, scaffolds, MMPA fragments,
                    Tanimoto (vs Morgan generator bit sets), substructure
      3D / 2D       EmbedMolecule / EmbedMultipleConfs with a seed vs ETKDGv3
                    with the schema's documented parameters, Compute2DCoords
      files         SDF/PDB write -> read (path and contents), PNG images
                    pixel-identical to RDKit's own rendering
      batch_map     values, per-item errors, fail_fast, prefixed tool names
      plus validation errors and a coverage check (every listed tool called)

Upstream defects that do not make a correctly used tool wrong are WARN: a
recognised defect matched exactly (tools that can never be called, outputs
that always fail validation, properties lost in the pickle, parameters that
crash, unordered batch results, unconfined or broken file names, explicit
hydrogens dropped, client pickles loaded as arbitrary objects, ignored
arguments). Wrong values for correct inputs, or anything that matches
neither the correct answer nor the recognised defect, are FAIL.
"""
from __future__ import annotations

import base64
import json
import math
import pickle
import re
import struct
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..client import text_of
from ..helpers import png_size
from ..runner import Caller, Report, Session, Smoke, check_rejected

# --------------------------------------------------------------------------
# Inputs, anchors, tolerances
# --------------------------------------------------------------------------

MOLECULES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "spiro[4.5]decane": "C1CCC2(C1)CCCCC2",
    "N-methylacetamide": "CC(=O)NC",
}
ASPIRIN = MOLECULES["aspirin"]
SALICYLIC_ACID = "Oc1ccccc1C(=O)O"
INVALID_SMILES = "C1CC(("

# Single-SMILES descriptor tools -> RDKit module holding the function of the same
# name (called with the molecule only, i.e. RDKit's defaults).
DESCRIPTORS_MODULE = ("ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
                      "HeavyAtomMolWt", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge",
                      "MinPartialCharge", "MolWt", "NumRadicalElectrons", "NumValenceElectrons")
RDMOLDESCRIPTORS = ("CalcChi0v", "CalcChi1v", "CalcChi2v", "CalcChi3v", "CalcChi4v", "CalcKappa1",
                    "CalcKappa2", "CalcKappa3", "CalcMolFormula", "CalcNumAliphaticRings", "CalcNumAmideBonds",
                    "CalcNumAromaticRings", "CalcNumHBA", "CalcNumHBD", "CalcNumHeavyAtoms",
                    "CalcNumHeteroatoms", "CalcNumHeterocycles", "CalcNumLipinskiHBA", "CalcNumLipinskiHBD",
                    "CalcNumRings", "CalcNumSaturatedCarbocycles", "CalcNumSaturatedRings", "CalcNumSpiroAtoms",
                    "CalcNumUnspecifiedAtomStereoCenters", "CalcTPSA")
# Tools with options: tool -> (option, values); RDKit function of the same name minus nothing.
OPTION_TOOLS = {"CalcLabuteASA": ("includeHs", (True, False)),
                "CalcCrippenDescriptors": ("includeHs", (True, False)),
                "CalcNumRotatableBonds": ("strict", (True, False))}

# Values that do not come from RDKit: (tool, smiles, expected, tolerance, why).
_H, _C, _O = 1.008, 12.011, 15.999                              # IUPAC conventional atomic weights
_H1, _O16 = 1.00782503223, 15.99491461957                       # AME2020 isotope masses, 12C = 12
ANCHORS = (
    ("MolWt", "CCO", 2 * _C + 6 * _H + _O, 1e-9, "C2H6O from conventional atomic weights"),
    ("HeavyAtomMolWt", "CCO", 2 * _C + _O, 1e-9, "C2O, hydrogens excluded"),
    ("ExactMolWt", "CCO", 2 * 12.0 + 6 * _H1 + _O16, 1e-6, "12C2 1H6 16O monoisotopic mass"),
    ("NumValenceElectrons", "CCO", 2 * 4 + 6 * 1 + 6, 0, "2 C x 4 + 6 H x 1 + O x 6"),
    ("NumValenceElectrons", MOLECULES["caffeine"], 8 * 4 + 10 + 4 * 5 + 2 * 6, 0, "C8H10N4O2"),
    ("NumRadicalElectrons", "[CH3]", 1, 0, "methyl radical"),
    ("CalcMolFormula", ASPIRIN, "C9H8O4", None, "aspirin"),
    ("CalcMolFormula", MOLECULES["caffeine"], "C8H10N4O2", None, "caffeine"),
    ("CalcMolFormula", MOLECULES["ibuprofen"], "C13H18O2", None, "ibuprofen"),
    ("CalcTPSA", ASPIRIN, 2 * 17.07 + 9.23 + 20.23, 1e-9, "Ertl contributions: 2 C=O, ester O, OH"),
    ("CalcNumRings", MOLECULES["caffeine"], 2, 0, "purine"),
    ("CalcNumAromaticRings", MOLECULES["caffeine"], 2, 0, "purine"),
    ("CalcNumHeterocycles", MOLECULES["caffeine"], 2, 0, "purine"),
    ("CalcNumSpiroAtoms", MOLECULES["spiro[4.5]decane"], 1, 0, "one spiro centre"),
    ("CalcNumSaturatedCarbocycles", MOLECULES["spiro[4.5]decane"], 2, 0, "cyclopentane + cyclohexane"),
    ("CalcNumAmideBonds", MOLECULES["N-methylacetamide"], 1, 0, "one C(=O)N"),
    ("CalcNumUnspecifiedAtomStereoCenters", MOLECULES["ibuprofen"], 1, 0, "racemic alpha carbon"),
    ("CalcNumHeavyAtoms", ASPIRIN, 13, 0, "C9O4"),
    ("CalcNumLipinskiHBD", ASPIRIN, 1, 0, "one OH"),
    ("CalcNumLipinskiHBA", ASPIRIN, 4, 0, "four O"),
)
# All names compute_descriptors accepts (its schema's Literal list, a subset of
# rdMolDescriptors.Properties.GetAvailableProperties()).
DESCRIPTOR_NAMES = ("exactmw", "amw", "lipinskiHBA", "lipinskiHBD", "NumRotatableBonds", "NumHBD", "NumHBA",
                    "NumHeavyAtoms", "NumAtoms", "NumHeteroatoms", "NumAmideBonds", "FractionCSP3", "NumRings",
                    "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings", "NumHeterocycles",
                    "NumAromaticHeterocycles", "NumSaturatedHeterocycles", "NumAliphaticHeterocycles",
                    "NumSpiroAtoms", "NumBridgeheadAtoms", "NumAtomStereoCenters",
                    "NumUnspecifiedAtomStereoCenters", "labuteASA", "tpsa", "CrippenClogP", "CrippenMR", "chi0v",
                    "chi1v", "chi2v", "chi3v", "chi4v", "kappa1", "kappa2", "kappa3")
# Tools whose Python signature is (*args, **kwargs): the generated schema requires
# fields "args" and "kwargs", and no JSON value can be a molecule.
ARGS_KWARGS_TOOLS = ("CalcFractionCSP3", "CalcPBF", "GetUSR", "GetUSRScore")
PROPERTY_TOOLS = (("SetProp", "label", "aspirin"), ("SetIntProp", "count", -3), ("SetBoolProp", "flag", True),
                  ("SetDoubleProp", "score", 0.25), ("SetUnsignedProp", "id", 7),
                  ("UpdatePropertyCache", "cache", 1))
EMBED_SEED = 42
CONFS_SEED, CONFS_N, CONFS_PRUNE = 7, 6, 0.5
TANIMOTO_CASES = ((ASPIRIN, SALICYLIC_ACID, 2, 2048), (ASPIRIN, SALICYLIC_ACID, 1, 64),
                  (MOLECULES["ibuprofen"], ASPIRIN, 3, 1024), (ASPIRIN, ASPIRIN, 2, 2048))
FRAGMENT_CUTS = (1, 2, 3)
REL_TOL = 1e-12            # same library, same machine: values survive JSON exactly; leave room for printing
COORD_TOL = 1e-5           # Å; RDKit pickles conformer coordinates as float32
SDF_TOL = 1e-4             # Å; molfile coordinates have 4 decimals
PDB_TOL = 1e-3             # Å; PDB coordinates have 3 decimals
TANIMOTO_DECIMALS = 4      # the tool rounds its result
FAIL_FAST_REPEATS = 12


# --------------------------------------------------------------------------
# Pure helpers (no RDKit; unit-tested offline)
# --------------------------------------------------------------------------

def value_of(result: dict | None):
    """The value of a tools/call result: FastMCP's ``structuredContent`` (``{"result": x}``
    for plain return types, the model itself for pydantic returns); text JSON as fallback."""
    if result is None:
        return None
    data = result.get("structuredContent")
    if isinstance(data, dict):
        return data["result"] if set(data) == {"result"} else data
    text = text_of(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def deviation(got, expected) -> float:
    """Largest relative/absolute difference between nested numbers; inf on any mismatch
    of type, length or non-numeric value."""
    if isinstance(expected, bool) or isinstance(got, bool):
        return 0.0 if got is expected else math.inf
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        if math.isnan(expected) and math.isnan(got):
            return 0.0
        return abs(got - expected) / max(1.0, abs(expected))
    if isinstance(expected, (list, tuple)) and isinstance(got, (list, tuple)):
        if len(got) != len(expected):
            return math.inf
        return max((deviation(g, e) for g, e in zip(got, expected)), default=0.0)
    return 0.0 if got == expected else math.inf


def is_error(result: dict | None, *markers: str) -> bool:
    """isError=true and the text contains every marker."""
    return bool(result) and result.get("isError") is True and all(m in text_of(result) for m in markers)


def classify(result: dict | None, *, correct: Callable[[dict], str | None],
             defect: Callable[[dict], str | None]) -> tuple[str, str]:
    """Three-state probe: ``correct(result)`` returns a detail if the tool behaved correctly
    (PASS), ``defect(result)`` one if the result is exactly the recognised defect (WARN);
    anything else is FAIL."""
    if result is None:
        return "FAIL", "no result"
    detail = correct(result)
    if detail is not None:
        return "PASS", detail
    detail = defect(result)
    if detail is not None:
        return "WARN", detail
    return "FAIL", f"neither correct nor the known defect: isError={result.get('isError')} {text_of(result)[:200]!r}"


def inside(path: str | Path, root: Path) -> bool:
    """Whether ``path`` (after resolving ``..``) lies inside ``root``."""
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def batch_outputs(payload: dict) -> list[tuple[bool, Any, Any, str | None]]:
    """(ok, input, value, error) per batch_map result item. An ok item's ``output`` is the
    inner tool's raw FastMCP ``call_tool`` return: [content blocks, structured content]."""
    items = []
    for item in payload.get("results", []):
        value = None
        output = item.get("output")
        if item.get("ok") and isinstance(output, list) and len(output) == 2 and isinstance(output[1], dict):
            structured = output[1]
            value = structured["result"] if set(structured) == {"result"} else structured
        items.append((bool(item.get("ok")), item.get("input"), value, item.get("error")))
    return items


TIMESTAMP_NAME = re.compile(r"mol_\d{8}_\d{6}\.png")


def default_image_names(names: list[str]) -> tuple[str, str]:
    """Classify the default file names of two quick MolToImage calls: second-resolution
    timestamps are the recognised defect whether or not the two calls collided."""
    if names[0] == names[1]:
        return "WARN", (f"both calls wrote {names[0]}: default names are mol_<YYYYmmdd_HHMMSS>.png, so the "
                        "second image overwrote the first")
    if all(TIMESTAMP_NAME.fullmatch(n) for n in names):
        return "WARN", (f"{names}: default names are second-resolution timestamps; two images in the same "
                        "second overwrite each other")
    return "PASS", f"{names}"


def fail_fast_outcome(items: list[tuple], invalid: str) -> tuple[str, str]:
    """batch_map(fail_fast=True) over [invalid, valid]: stopping at the first error in input
    order reports only the invalid item (PASS). The recognised defect stops at the first
    error in *completion* order, which asyncio.as_completed does not tie to input order:
    any ok items followed by the invalid one (WARN)."""
    def is_invalid(item):
        return not item[0] and isinstance(item[1], dict) and item[1].get("smiles") == invalid

    if len(items) == 1 and is_invalid(items[0]):
        return "PASS", "only the first (invalid) input reported"
    if items and is_invalid(items[-1]) and all(item[0] for item in items[:-1]):
        return "WARN", (f"{len(items)} items, the error last: fail_fast stops at the first error in completion "
                        "order (asyncio.as_completed), not input order, so what is reported varies between calls")
    return "FAIL", f"unexpected items {items}"


def property_outcome(props: dict, key: str, value, same_molecule: bool, extra: str = "") -> tuple[str, str]:
    """Properties of the molecule a Set*Prop tool returned: the property set (PASS); no
    properties at all on an otherwise unchanged molecule — RDKit pickles drop properties by
    default — is the recognised defect (WARN); anything else FAIL."""
    if key in props and props[key] == value and type(props[key]) is type(value):
        return "PASS", f"{key}={props[key]!r}"
    if not props and same_molecule:
        return "WARN", (f"property lost: the returned pickle has no properties — RDKit pickles drop properties "
                        f"unless pickle options are set, so the Set*Prop tools have no effect on later calls{extra}")
    return "FAIL", f"props {props}, same molecule {same_molecule}"


def schema_defaults(tool: dict, model: str) -> dict:
    """Documented defaults of a pydantic model in a tool's inputSchema ``$defs``."""
    props = tool.get("inputSchema", {}).get("$defs", {}).get(model, {}).get("properties", {})
    return {name: spec["default"] for name, spec in props.items() if "default" in spec}


# --------------------------------------------------------------------------
# RDKit references (imported lazily: the offline tests run without RDKit)
# --------------------------------------------------------------------------

class Ref:
    """RDKit called directly, in the smoke process."""

    def __init__(self) -> None:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import (
            Descriptors,
            Draw,
            rdDepictor,
            rdDistGeom,
            rdFingerprintGenerator,
            rdMMPA,
            rdMolDescriptors,
        )
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp.*")
        self.Chem, self.DataStructs, self.Descriptors, self.Draw = Chem, DataStructs, Descriptors, Draw
        self.rdDepictor, self.rdDistGeom, self.rdMMPA = rdDepictor, rdDistGeom, rdMMPA
        self.rdMolDescriptors, self.MurckoScaffold = rdMolDescriptors, MurckoScaffold
        self.fpgen = rdFingerprintGenerator

    def mol(self, smiles: str):
        mol = self.Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"reference cannot parse {smiles!r}")
        return mol

    def canon(self, smiles: str) -> str:
        return self.Chem.MolToSmiles(self.mol(smiles))

    def descriptor(self, tool: str, smiles: str, **options):
        mol = self.mol(smiles)
        if tool in DESCRIPTORS_MODULE:
            return getattr(self.Descriptors, tool)(mol)
        function = getattr(self.rdMolDescriptors, tool)
        if tool == "CalcNumRotatableBonds":
            return function(mol, options.get("strict", True))
        value = function(mol, **options)
        return list(value) if isinstance(value, tuple) else value

    def properties(self, smiles: str, names) -> list[float]:
        return list(self.rdMolDescriptors.Properties(list(names)).ComputeProperties(self.mol(smiles)))

    def rotatable_strict_linkages(self, smiles: str) -> int:
        return self.rdMolDescriptors.CalcNumRotatableBonds(
            self.mol(smiles), self.rdMolDescriptors.NumRotatableBondsOptions.StrictLinkages)

    def tanimoto(self, a: str, b: str, radius: int, bits: int) -> float:
        gen = self.fpgen.GetMorganGenerator(radius=radius, fpSize=bits)
        x, y = (set(gen.GetFingerprint(self.mol(s)).GetOnBits()) for s in (a, b))
        return len(x & y) / len(x | y)

    def murcko(self, smiles: str) -> str:
        return self.MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)

    def generic(self, smiles: str) -> str:
        return self.Chem.MolToSmiles(self.MurckoScaffold.MakeScaffoldGeneric(self.mol(smiles)))

    def fragments(self, smiles: str, max_cuts: int) -> set:
        out = set()
        for core, side in self.rdMMPA.FragmentMol(self.mol(smiles), maxCuts=max_cuts, resultsAsMols=False):
            out.add((self.canon(core) if core else None, self.canon(side)))
        return out

    def substruct(self, smiles: str, query_smarts: str | None = None, query_smiles: str | None = None,
                  chirality: bool = False):
        query = self.Chem.MolFromSmarts(query_smarts) if query_smarts else self.mol(query_smiles)
        mol = self.mol(smiles)
        return mol.HasSubstructMatch(query, useChirality=chirality), tuple(mol.GetSubstructMatch(query))

    def embed_params(self, documented: dict, **overrides):
        params = self.rdDistGeom.ETKDGv3()
        for name, value in {**documented, **overrides}.items():
            if name != "coordMap" and hasattr(params, name):
                setattr(params, name, value)
        return params

    def embed(self, smiles: str, params):
        mol = self.mol(smiles)
        conf = self.rdDistGeom.EmbedMolecule(mol, params)
        return conf, mol

    def embed_multiple(self, smiles: str, n: int, params):
        mol = self.mol(smiles)
        return list(self.rdDistGeom.EmbedMultipleConfs(mol, n, params)), mol

    def coords2d(self, smiles: str):
        mol = self.mol(smiles)
        self.rdDepictor.Compute2DCoords(mol)
        return mol

    def decode(self, blob: str):
        return pickle.loads(base64.b64decode(blob))

    def etkdg_v3_defaults(self, names) -> dict:
        params = self.rdDistGeom.ETKDGv3()
        return {n: getattr(params, n) for n in names if hasattr(params, n)}

    def image_pixels(self, mol, size: tuple[int, int], **kwargs) -> bytes:
        return self.Draw.MolToImage(mol, size=size, **kwargs).convert("RGB").tobytes()

    def drawn_file_pixels(self, mol, size: tuple[int, int], scratch: Path) -> bytes:
        """Pixels of Draw.MolToFile's PNG (a different drawing path from MolToImage)."""
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / "reference.png"
        self.Draw.MolToFile(mol, str(path), size=size)
        return file_pixels(path)[1]

    def grid_pixels(self, smiles_matrix, sub_size, legends) -> bytes:
        mols = [[self.mol(s) for s in row] for row in smiles_matrix]
        image = self.Draw.MolsMatrixToGridImage(mols, subImgSize=tuple(sub_size), legendsMatrix=legends)
        return image.convert("RGB").tobytes()


def file_pixels(path: str | Path) -> tuple[tuple[int, int], bytes]:
    from PIL import Image
    with Image.open(path) as image:
        return image.size, image.convert("RGB").tobytes()


def max_coord_diff(a, b) -> float:
    """Largest coordinate difference between two conformers with the same atom order."""
    if a.GetNumAtoms() != b.GetNumAtoms():
        return math.inf
    pa, pb = a.GetPositions(), b.GetPositions()
    return float(max((abs(x - y) for ra, rb in zip(pa, pb) for x, y in zip(ra, rb)), default=0.0))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _value(call: Caller, check: str, tool: str, arguments: dict):
    result = call(check, tool, arguments)
    return None if result is None else value_of(result)


def _compare(report: Report, name: str, pairs: list[tuple[str, Any, Any]], tol: float = REL_TOL) -> None:
    """``pairs``: (label, got, expected); one PASS/FAIL line for all of them."""
    worst, bad = 0.0, []
    for label, got, expected in pairs:
        dev = deviation(got, expected)
        worst = max(worst, dev)
        if dev > tol:
            bad.append(f"{label}: got {got!r}, expected {expected!r}")
    if bad:
        report.add("L1", name, "FAIL", "; ".join(bad)[:400])
    else:
        report.add("L1", name, "PASS", f"{len(pairs)} value(s), max deviation {worst:.1e}", max_deviation=worst)


def check_descriptors(call: Caller, report: Report, ref: Ref) -> None:
    for tool in DESCRIPTORS_MODULE + RDMOLDESCRIPTORS:
        pairs = []
        for label, smiles in MOLECULES.items():
            got = _value(call, f"{tool}[{label}]", tool, {"smiles": smiles})
            if got is not None:
                pairs.append((label, got, ref.descriptor(tool, smiles)))
        if len(pairs) == len(MOLECULES):
            _compare(report, f"{tool} vs RDKit ({len(MOLECULES)} molecules)", pairs)
    for tool, (option, values) in OPTION_TOOLS.items():
        pairs = []
        for label, smiles in MOLECULES.items():
            got = _value(call, f"{tool}[{label}]", tool, {"smiles": smiles})
            if got is not None:
                pairs.append((f"{label} default", got, ref.descriptor(tool, smiles)))
            for value in values:
                got = _value(call, f"{tool}[{label}, {option}={value}]", tool, {"smiles": smiles, option: value})
                if got is not None:
                    pairs.append((f"{label} {option}={value}", got, ref.descriptor(tool, smiles, **{option: value})))
        _compare(report, f"{tool} vs RDKit (default and {option}={'/'.join(map(str, values))})", pairs)


def check_anchors(call: Caller, report: Report) -> None:
    for tool, smiles, expected, tol, why in ANCHORS:
        got = _value(call, f"{tool}[{smiles} anchor]", tool, {"smiles": smiles})
        if got is None:
            continue
        ok = got == expected if tol is None else (isinstance(got, (int, float)) and abs(got - expected) <= tol)
        report.add("L1", f"{tool}({smiles}) = {expected:g} ({why})" if tol is not None
                   else f"{tool}({smiles}) = {expected} ({why})", "PASS" if ok else "FAIL", f"got {got!r}")


def check_rotatable_description(call: Caller, report: Report, ref: Ref) -> None:
    """The ``strict`` option is described as handling ring-system linkages (RDKit's
    StrictLinkages), but True maps to RDKit's Strict."""
    smiles = "c1ccccc1-c1ccccc1"
    got = _value(call, "CalcNumRotatableBonds[biphenyl strict]", "CalcNumRotatableBonds",
                 {"smiles": smiles, "strict": True})
    strict = ref.descriptor("CalcNumRotatableBonds", smiles, strict=True)
    linkages = ref.rotatable_strict_linkages(smiles)
    if got is None:
        return
    status = "PASS" if got == linkages else ("WARN" if got == strict else "FAIL")
    report.add("L1", "CalcNumRotatableBonds strict = ring linkages (as described)", status,
               f"biphenyl strict=True -> {got}; RDKit Strict {strict}, StrictLinkages {linkages}"
               + ("" if status == "PASS" else ": the description promises ring-linkage handling, "
                  "the option is RDKit's plain Strict"))


def check_compute_descriptors(call: Caller, report: Report, ref: Ref) -> None:
    smiles = list(MOLECULES.values())
    got = _value(call, "compute_descriptors[all names]", "compute_descriptors",
                 {"smiles_list": smiles, "descriptor_names": list(DESCRIPTOR_NAMES)})
    if got is not None:
        _compare(report, f"compute_descriptors vs Properties.ComputeProperties ({len(DESCRIPTOR_NAMES)} names x "
                         f"{len(smiles)} molecules)", [("rows", got, [ref.properties(s, DESCRIPTOR_NAMES)
                                                                      for s in smiles])])
    check_rejected(call, "compute_descriptors[unknown name]", "compute_descriptors",
                   {"smiles_list": ["CCO"], "descriptor_names": ["no_such_descriptor"]})
    names = ["amw", "tpsa"]
    batch = ["CCO", INVALID_SMILES, ASPIRIN]
    expected = [ref.properties("CCO", names), ref.properties(ASPIRIN, names)]
    result = call("compute_descriptors[invalid SMILES in the list]", "compute_descriptors",
                  {"smiles_list": batch, "descriptor_names": names}, allow_error=True)

    def correct(r):
        rows = value_of(r)
        if r.get("isError"):
            return f"rejected: {text_of(r)[:100]!r}"
        if isinstance(rows, list) and len(rows) == 3 and rows[1] is None:
            return "invalid entry kept as null"
        return None

    def defect(r):
        rows = value_of(r)
        if not r.get("isError") and isinstance(rows, list) and deviation(rows, expected) <= REL_TOL:
            return ("the invalid SMILES is dropped silently: 2 rows for 3 inputs, so later rows no longer line "
                    "up with smiles_list")
        return None

    status, detail = classify(result, correct=correct, defect=defect)
    report.add("L1", "compute_descriptors[invalid SMILES in the list]", status, detail)


def args_kwargs_outcome(tool: str, result: dict | None, required: list[str], description: str,
                        fraction: float) -> tuple[str, str]:
    """A (*args, **kwargs) tool called with a SMILES. Only CalcFractionCSP3 has a SMILES-only
    correct answer (``fraction``); PBF/USR need a 3D molecule, so any non-error result of
    theirs is unexplained (FAIL) until reviewed."""
    note = " (its description is CalcNumHeavyAtoms' docstring)" if "heavy atoms" in description.lower() else ""

    def correct(r):
        if tool == "CalcFractionCSP3" and not r.get("isError") and deviation(value_of(r), fraction) <= REL_TOL:
            return f"returned {value_of(r)!r}"
        return None

    def defect(r):
        if required == ["args", "kwargs"] and is_error(r, "args", "Field required"):
            return (f"unusable: declared as (*args, **kwargs), so the schema requires {required} and no JSON "
                    f"value can carry a molecule{note}")
        return None

    return classify(result, correct=correct, defect=defect)


def check_args_kwargs_tools(call: Caller, report: Report, ref: Ref, tools: dict[str, dict]) -> None:
    fraction = ref.rdMolDescriptors.CalcFractionCSP3(ref.mol(ASPIRIN))
    for tool in ARGS_KWARGS_TOOLS:
        required = sorted(tools[tool].get("inputSchema", {}).get("required", []))
        result = call(f"{tool}[smiles]", tool, {"smiles": ASPIRIN}, allow_error=True)
        status, detail = args_kwargs_outcome(tool, result, required, tools[tool].get("description") or "", fraction)
        report.add("L1", f"{tool} callable with a SMILES", status, detail)


def check_oxidation_numbers(call: Caller, report: Report) -> None:
    result = call("CalcOxidationNumbers[ethanol]", "CalcOxidationNumbers", {"smiles": "CCO"}, allow_error=True)
    expected = [-3, -1, -2]                                   # CH3, CH2OH, O (heavy atoms)
    status, detail = classify(
        result,
        correct=lambda r: None if r.get("isError") or value_of(r) != expected else f"{value_of(r)}",
        defect=lambda r: ("always fails: RDKit's CalcOxidationNumbers returns None (it sets atom properties) "
                          "and the tool declares a float result") if is_error(r, "valid number") else None)
    report.add("L1", "CalcOxidationNumbers(ethanol)", status, detail)


def check_invalid_inputs(call: Caller, report: Report) -> None:
    for tool in ("MolWt", "CalcMolFormula", "CalcTPSA", "smiles_to_mol", "MurckoScaffoldSmilesFromSmiles",
                 "MakeScaffoldGeneric", "FragmentMol"):
        check_rejected(call, f"{tool}[invalid SMILES]", tool, {"smiles": INVALID_SMILES})
    check_rejected(call, "TanimotoSimilarity[invalid SMILES]", "TanimotoSimilarity",
                   {"smiles1": "CCO", "smiles2": INVALID_SMILES})
    check_rejected(call, "smarts_to_mol[invalid SMARTS]", "smarts_to_mol", {"smarts": "[C"})
    check_rejected(call, "FragmentMol[invalid pattern]", "FragmentMol", {"smiles": ASPIRIN, "pattern": "[C"})
    check_rejected(call, "MolWt[wrong argument name]", "MolWt", {"smile": "CCO"})
    check_rejected(call, "MolWt[unknown extra argument]", "MolWt", {"smiles": "CCO", "no_such_option": 1},
                   on_accept=lambda r: ("WARN", (f"accepted, returns {value_of(r)!r}: unknown arguments are silently "
                                                 "ignored, so a misspelt option goes unnoticed")))


def check_scaffolds_fragments_similarity(call: Caller, report: Report, ref: Ref) -> None:
    pairs = []
    for label, smiles in MOLECULES.items():
        got = _value(call, f"MurckoScaffoldSmilesFromSmiles[{label}]", "MurckoScaffoldSmilesFromSmiles",
                     {"smiles": smiles})
        if got is not None:
            pairs.append((label, got, ref.murcko(smiles)))
    _compare(report, "MurckoScaffoldSmilesFromSmiles vs RDKit", pairs)
    got = _value(call, "MurckoScaffoldSmilesFromSmiles[aspirin anchor]", "MurckoScaffoldSmilesFromSmiles",
                 {"smiles": ASPIRIN})
    report.add("L1", "MurckoScaffoldSmilesFromSmiles(aspirin) = benzene", "PASS" if got == "c1ccccc1" else "FAIL",
               f"got {got!r}")
    pairs = []
    for label, smiles in MOLECULES.items():
        got = _value(call, f"MakeScaffoldGeneric[{label}]", "MakeScaffoldGeneric", {"smiles": smiles})
        if got is not None:
            pairs.append((label, got, ref.generic(smiles)))
    _compare(report, "MakeScaffoldGeneric vs RDKit (applied to the whole molecule, as RDKit does)", pairs)

    for cuts in FRAGMENT_CUTS:
        smiles = MOLECULES["ibuprofen"]
        got = _value(call, f"FragmentMol[maxCuts={cuts}]", "FragmentMol", {"smiles": smiles, "maxCuts": cuts})
        if got is None:
            continue
        server = {(ref.canon(c) if c else None, ref.canon(s)) for c, s in got}
        expected = ref.fragments(smiles, cuts)
        ok = server == expected and len(got) == len(expected)
        report.add("L1", f"FragmentMol(ibuprofen, maxCuts={cuts}) vs rdMMPA", "PASS" if ok else "FAIL",
                   f"{len(got)} fragmentations" + ("" if ok else f"; reference has {len(expected)}"))

    pairs = []
    for a, b, radius, bits in TANIMOTO_CASES:
        got = _value(call, f"TanimotoSimilarity[r={radius}, {bits} bits]", "TanimotoSimilarity",
                     {"smiles1": a, "smiles2": b, "radius": radius, "nBits": bits})
        if got is not None:
            pairs.append((f"r={radius} {bits} bits", got, round(ref.tanimoto(a, b, radius, bits), TANIMOTO_DECIMALS)))
    _compare(report, "TanimotoSimilarity vs Morgan generator bit sets (rounded to 4 decimals, radius/nBits honoured)",
             pairs)


def check_mol_io(call: Caller, report: Report, ref: Ref) -> dict[str, str]:
    """SMILES/SMARTS <-> pickle; returns pickles for later checks."""
    pickles: dict[str, str] = {}
    pairs = []
    for label, smiles in {**MOLECULES, "salicylic acid": SALICYLIC_ACID}.items():
        blob = _value(call, f"smiles_to_mol[{label}]", "smiles_to_mol", {"smiles": smiles})
        if not isinstance(blob, str):
            continue
        pickles[label] = blob
        try:
            decoded = ref.Chem.MolToSmiles(ref.decode(blob))
        except Exception as exc:                                # noqa: BLE001 - any decode failure is a FAIL
            decoded = f"<undecodable: {exc}>"
        back = _value(call, f"mol_to_smiles[{label}]", "mol_to_smiles", {"pmol": blob})
        pairs.append((f"{label} decoded", decoded, ref.canon(smiles)))
        pairs.append((f"{label} mol_to_smiles", back, ref.canon(smiles)))
    _compare(report, "smiles_to_mol -> pickle -> mol_to_smiles round trip (canonical SMILES)", pairs)

    for smarts in ("C(=O)[OX2H1]", "[#6]~[#8]", "c1ccccc1"):
        blob = _value(call, f"smarts_to_mol[{smarts}]", "smarts_to_mol", {"smarts": smarts})
        if isinstance(blob, str):
            pickles[f"smarts {smarts}"] = blob
            expected = ref.Chem.MolToSmarts(ref.Chem.MolFromSmarts(smarts))
            got = ref.Chem.MolToSmarts(ref.decode(blob))
            report.add("L1", f"smarts_to_mol({smarts})", "PASS" if got == expected else "FAIL",
                       f"decoded query {got}")

    explicit = "[H]OC([H])([H])C([H])([H])[H]"
    blob = _value(call, "smiles_to_mol[explicit hydrogens]", "smiles_to_mol", {"smiles": explicit})
    if isinstance(blob, str):
        atoms = ref.decode(blob).GetNumAtoms()
        status = "PASS" if atoms == 9 else ("WARN" if atoms == 3 else "FAIL")
        report.add("L1", "smiles_to_mol keeps explicit hydrogens", status,
                   f"{atoms} atoms for {explicit}" + ("" if status != "WARN" else
                   ": [H] atoms are removed (MolFromSmiles default) and no tool adds hydrogens, so every 3D "
                   "structure the server produces is heavy-atom only"))

    foreign = base64.b64encode(pickle.dumps({"not": "a molecule"})).decode()
    result = call("mol_to_smiles[pickled dict]", "mol_to_smiles", {"pmol": foreign}, allow_error=True)
    status, detail = classify(
        result,
        correct=lambda r: f"rejected before unpickling: {text_of(r)[:120]!r}"
        if r.get("isError") and "dict" not in text_of(r) else None,
        defect=lambda r: ("the server ran pickle.loads on client data and passed the reconstructed dict to RDKit: "
                          "every p_mol/pmol argument is an arbitrary pickle, so a crafted value executes code in the "
                          "server process") if is_error(r, "dict") else None)
    report.add("L1", "p_mol is not unpickled blindly", status, detail)
    check_rejected(call, "mol_to_smiles[not base64]", "mol_to_smiles", {"pmol": "!!not-base64"})
    return pickles


def _documented_embed_defaults(tools: dict[str, dict]) -> dict:
    return schema_defaults(tools["EmbedMolecule"], "EmbedParameters")


def check_embedding(call: Caller, report: Report, ref: Ref, tools: dict[str, dict],
                    pickles: dict[str, str]) -> str | None:
    """Returns the seeded aspirin conformer pickle for the file checks."""
    documented = _documented_embed_defaults(tools)
    rdkit_defaults = ref.etkdg_v3_defaults(documented)
    differ = {k: (documented[k], rdkit_defaults[k]) for k in rdkit_defaults if documented[k] != rdkit_defaults[k]}
    not_rdkit = sorted(set(documented) - set(rdkit_defaults) - {"coordMap"})
    report.add("L1", "EmbedParameters defaults = RDKit ETKDGv3()", "WARN" if differ or not_rdkit else "PASS",
               "; ".join([f"{k}: schema {a}, ETKDGv3 {b}" for k, (a, b) in differ.items()]
                         + ([f"not ETKDGv3 attributes: {not_rdkit}"] if not_rdkit else []))
               + (" — harmless while molecules carry no hydrogens" if set(differ) <= {"onlyHeavyAtomsForRMS"}
                  and differ else ""), differences={k: list(v) for k, v in differ.items()})

    blob = pickles.get("aspirin")
    if blob is None:
        return None
    embedded = None
    first = _value(call, f"EmbedMolecule[seed {EMBED_SEED}]", "EmbedMolecule",
                   {"p_mol": blob, "params": {"randomSeed": EMBED_SEED}})
    second = _value(call, f"EmbedMolecule[seed {EMBED_SEED} again]", "EmbedMolecule",
                    {"p_mol": blob, "params": {"randomSeed": EMBED_SEED}})
    conf, expected = ref.embed(ASPIRIN, ref.embed_params(documented, randomSeed=EMBED_SEED))
    if isinstance(first, dict) and isinstance(first.get("mol"), str):
        embedded = first["mol"]
        mol = ref.decode(embedded)
        diff = max_coord_diff(mol.GetConformer(), expected.GetConformer()) if mol.GetNumConformers() else math.inf
        ok = first.get("conf_id") == conf == 0 and diff <= COORD_TOL and mol.GetConformer().Is3D()
        report.add("L1", f"EmbedMolecule(aspirin, randomSeed={EMBED_SEED}) vs ETKDGv3", "PASS" if ok else "FAIL",
                   f"conf_id {first.get('conf_id')} (ref {conf}), max |dxyz| {diff:.1e} Å (float32 pickle)",
                   max_coord_diff=diff)
        same = isinstance(second, dict) and second.get("mol") == first.get("mol")
        report.add("L1", "EmbedMolecule repeatable with a seed", "PASS" if same else "FAIL",
                   "identical pickles" if same else "different conformers for the same seed")
    variant = _value(call, "EmbedMolecule[useRandomCoords]", "EmbedMolecule",
                     {"p_mol": blob, "params": {"randomSeed": 11, "useRandomCoords": True}})
    conf, expected = ref.embed(ASPIRIN, ref.embed_params(documented, randomSeed=11, useRandomCoords=True))
    if isinstance(variant, dict) and isinstance(variant.get("mol"), str):
        diff = max_coord_diff(ref.decode(variant["mol"]).GetConformer(), expected.GetConformer())
        report.add("L1", "EmbedMolecule params passed through (useRandomCoords, seed 11)",
                   "PASS" if diff <= COORD_TOL else "FAIL", f"max |dxyz| {diff:.1e} Å")
    plain = _value(call, "EmbedMolecule[default params]", "EmbedMolecule", {"p_mol": blob})
    if isinstance(plain, dict) and isinstance(plain.get("mol"), str):
        mol = ref.decode(plain["mol"])
        conf3d = mol.GetConformer() if mol.GetNumConformers() else None
        bonds = [math.dist(conf3d.GetAtomPosition(b.GetBeginAtomIdx()), conf3d.GetAtomPosition(b.GetEndAtomIdx()))
                 for b in mol.GetBonds()] if conf3d else []
        ok = plain.get("conf_id") == 0 and conf3d is not None and conf3d.Is3D() and bonds \
            and all(1.1 <= d <= 1.6 for d in bonds)
        report.add("L1", "EmbedMolecule without params (randomSeed -1, not reproducible)", "PASS" if ok else "FAIL",
                   f"conf_id {plain.get('conf_id')}, bond lengths {min(bonds, default=0):.3f}-"
                   f"{max(bonds, default=0):.3f} Å")

    multi = _value(call, "EmbedMultipleConfs", "EmbedMultipleConfs",
                   {"p_mol": pickles.get("ibuprofen", blob), "numConfs": CONFS_N,
                    "params": {"randomSeed": CONFS_SEED, "pruneRmsThresh": CONFS_PRUNE}})
    ids, expected = ref.embed_multiple(MOLECULES["ibuprofen"], CONFS_N,
                                       ref.embed_params(documented, randomSeed=CONFS_SEED, pruneRmsThresh=CONFS_PRUNE))
    if isinstance(multi, dict) and isinstance(multi.get("mol"), str):
        mol = ref.decode(multi["mol"])
        diffs = [max_coord_diff(mol.GetConformer(i), expected.GetConformer(i)) for i in ids] \
            if mol.GetNumConformers() == len(ids) else [math.inf]
        ok = multi.get("conf_ids") == ids and multi.get("num_confs") == len(ids) and max(diffs) <= COORD_TOL
        report.add("L1", f"EmbedMultipleConfs(ibuprofen, {CONFS_N}, seed {CONFS_SEED}, prune {CONFS_PRUNE} Å) vs "
                         "ETKDGv3", "PASS" if ok else "FAIL",
                   f"conf_ids {multi.get('conf_ids')} (ref {ids}), max |dxyz| {max(diffs):.1e} Å")

    flat = _value(call, "Compute2DCoords", "Compute2DCoords", {"p_mol": blob})
    if isinstance(flat, str):
        mol = ref.decode(flat)
        expected = ref.coords2d(ASPIRIN)
        diff = max_coord_diff(mol.GetConformer(), expected.GetConformer()) if mol.GetNumConformers() else math.inf
        ok = diff <= COORD_TOL and not mol.GetConformer().Is3D()
        report.add("L1", "Compute2DCoords(aspirin) vs rdDepictor", "PASS" if ok else "FAIL",
                   f"max |dxy| {diff:.1e} Å, 3D flag {mol.GetConformer().Is3D() if mol.GetNumConformers() else None}")
    return embedded


def check_files(call: Caller, report: Report, ref: Ref, files: Path, embedded: str | None,
                pickles: dict[str, str]) -> dict[str, str]:
    """SDF/PDB writers and readers; returns written paths for the image checks."""
    paths: dict[str, str] = {}
    if embedded is None:
        return paths
    mol3d = ref.decode(embedded)
    canon = ref.canon(ASPIRIN)

    path = _value(call, "mol_to_sdf", "mol_to_sdf", {"pmol": embedded, "file_dir": str(files), "filename": "aspirin"})
    if isinstance(path, str):
        expected = files / "aspirin.sdf"
        written = ref.Chem.MolFromMolFile(path) if Path(path).is_file() else None
        diff = max_coord_diff(written.GetConformer(), mol3d.GetConformer()) if written else math.inf
        ok = Path(path) == expected and written is not None and diff <= SDF_TOL
        report.add("L1", "mol_to_sdf writes the conformer (.sdf appended)", "PASS" if ok else "FAIL",
                   f"{path}; max |dxyz| {diff:.1e} Å vs the pickle")
        if ok:
            paths["sdf"] = path
    path = _value(call, "mol_to_pdb", "mol_to_pdb",
                  {"pmol": embedded, "file_dir": str(files), "filename": "aspirin.pdb"})
    if isinstance(path, str):
        written = ref.Chem.MolFromPDBFile(path) if Path(path).is_file() else None
        diff = max_coord_diff(written.GetConformer(), mol3d.GetConformer()) if written else math.inf
        ok = Path(path) == files / "aspirin.pdb" and diff <= PDB_TOL
        report.add("L1", "mol_to_pdb writes the conformer", "PASS" if ok else "FAIL",
                   f"{path}; max |dxyz| {diff:.1e} Å")
        if ok:
            paths["pdb"] = path

    readers = []
    if "sdf" in paths:
        text = Path(paths["sdf"]).read_text(encoding="utf-8")
        readers += [("sdf_to_mol", {"sdf_path": paths["sdf"]}, SDF_TOL, ref.Chem.MolFromMolFile(paths["sdf"])),
                    ("sdf_contents_to_mol", {"sdf_contents": text}, SDF_TOL, ref.Chem.MolFromMolFile(paths["sdf"]))]
    if "pdb" in paths:
        text = Path(paths["pdb"]).read_text(encoding="utf-8")
        readers += [("pdb_to_mol", {"pdb_path": paths["pdb"]}, PDB_TOL, ref.Chem.MolFromPDBFile(paths["pdb"])),
                    ("pdb_contents_to_mol", {"pdb_contents": text}, PDB_TOL, ref.Chem.MolFromPDBFile(paths["pdb"]))]
    for tool, arguments, tol, expected in readers:
        blob = _value(call, tool, tool, arguments)
        if not isinstance(blob, str):
            continue
        mol = ref.decode(blob)
        diff = max_coord_diff(mol.GetConformer(), expected.GetConformer()) if mol.GetNumConformers() else math.inf
        smiles, ref_smiles = ref.Chem.MolToSmiles(mol), ref.Chem.MolToSmiles(expected)
        ok = smiles == ref_smiles and diff <= COORD_TOL
        report.add("L1", f"{tool} vs RDKit reading the same file", "PASS" if ok else "FAIL",
                   f"{smiles}; max |dxyz| {diff:.1e} Å" + ("" if tool.startswith("pdb") or smiles == canon
                                                          else f" (aspirin is {canon})"))
    for tool, key in (("sdf_to_mol", "sdf_path"), ("pdb_to_mol", "pdb_path")):
        check_rejected(call, f"{tool}[missing file]", tool, {key: str(files / "missing.file")})
    check_rejected(call, "sdf_contents_to_mol[garbage]", "sdf_contents_to_mol", {"sdf_contents": "not a molfile"})

    # Default file names are the canonical SMILES; '/' (E/Z bonds) is a path separator.
    blob = pickles.get("trans-2-butene")
    result = call("mol_to_sdf[default filename, C/C=C/C]", "mol_to_sdf", {"pmol": blob, "file_dir": str(files)},
                  allow_error=True) if blob else None
    status, detail = classify(
        result,
        correct=lambda r: f"wrote {value_of(r)}" if not r.get("isError") and inside(str(value_of(r)), files)
        and Path(str(value_of(r))).is_file() else None,
        defect=lambda r: ("fails: the default file name is the SMILES, and '/' in E/Z SMILES is taken as a "
                          "directory") if is_error(r, "Bad output file") else None)
    report.add("L1", "mol_to_sdf default file name for C/C=C/C", status, detail)

    result = call("mol_to_sdf[filename ../]", "mol_to_sdf",
                  {"pmol": embedded, "file_dir": str(files), "filename": "../outside.sdf"}, allow_error=True)
    status, detail = classify(
        result,
        correct=lambda r: f"rejected: {text_of(r)[:100]!r}" if r.get("isError")
        else (f"kept inside file_dir: {value_of(r)}" if inside(str(value_of(r)), files) else None),
        defect=lambda r: (f"wrote {Path(str(value_of(r))).resolve()} outside file_dir: filename is not confined "
                          "(any absolute file_dir is accepted anyway)")
        if not r.get("isError") and Path(str(value_of(r))).resolve().is_file() else None)
    report.add("L1", "mol_to_sdf filename confined to file_dir", status, detail)
    return paths


def _png_check(report: Report, name: str, path, expected_path: Path | None, size: tuple[int, int],
               pixels: bytes | None) -> None:
    if not isinstance(path, str) or not Path(path).is_file():
        report.add("L1", name, "FAIL", f"no file: {path!r}")
        return
    data = Path(path).read_bytes()
    try:
        width_height = png_size(data)
    except (ValueError, struct.error) as exc:
        report.add("L1", name, "FAIL", f"not a PNG: {exc}")
        return
    got_size, got_pixels = file_pixels(path)
    problems = []
    if expected_path is not None and Path(path) != expected_path:
        problems.append(f"path {path} != {expected_path}")
    if width_height != size or got_size != size:
        problems.append(f"size {width_height} != {size}")
    if pixels is not None and got_pixels != pixels:
        problems.append("pixels differ from RDKit's own rendering")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else f"{size[0]}x{size[1]} PNG, pixel-identical to RDKit")


def check_images(call: Caller, report: Report, ref: Ref, files: Path, pickles: dict[str, str],
                 paths: dict[str, str]) -> None:
    blob = pickles.get("aspirin")
    if blob is None:
        return
    mol = ref.mol(ASPIRIN)
    path = _value(call, "MolToImage[pmol]", "MolToImage",
                  {"file_dir": str(files), "pmol": blob, "filename": "aspirin.png", "size": [400, 250]})
    _png_check(report, "MolToImage(pmol, 400x250)", path, files / "aspirin.png", (400, 250),
               ref.image_pixels(mol, (400, 250)))
    path = _value(call, "MolToImage[highlight]", "MolToImage",
                  {"file_dir": str(files), "pmol": blob, "filename": "hl.png", "highlightAtoms": [0, 1, 2],
                   "highlightBonds": [0, 1], "highlightColor": [0, 0.5, 1]})
    _png_check(report, "MolToImage(highlightAtoms/Bonds/Color)", path, files / "hl.png", (300, 300),
               ref.image_pixels(mol, (300, 300), highlightAtoms=(0, 1, 2), highlightBonds=(0, 1),
                                highlightColor=(0, 0.5, 1)))
    for kind in ("sdf", "pdb"):
        if kind in paths:
            path = _value(call, f"MolToImage[{kind}_path]", "MolToImage",
                          {"file_dir": str(files), f"{kind}_path": paths[kind], "filename": f"from_{kind}.png"})
            source = (ref.Chem.MolFromMolFile if kind == "sdf" else ref.Chem.MolFromPDBFile)(paths[kind])
            _png_check(report, f"MolToImage({kind}_path)", path, files / f"from_{kind}.png", (300, 300),
                       ref.image_pixels(source, (300, 300)))
    path = _value(call, "MolToFile", "MolToFile",
                  {"file_dir": str(files), "filename": "drawn", "pmol": blob, "width": 320, "height": 200})
    _png_check(report, "MolToFile(320x200, .png appended)", path, files / "drawn.png", (320, 200),
               ref.drawn_file_pixels(mol, (320, 200), files.parent / "reference"))
    matrix = [[ASPIRIN, SALICYLIC_ACID, "CCO"], ["c1ccccc1", "CC(=O)O", "N"]]
    legends = [["a", "b", "c"], ["d", "e", "f"]]
    path = _value(call, "MolsMatrixToGridImage", "MolsMatrixToGridImage",
                  {"molsMatrix": matrix, "subImgSize": [150, 120], "legendsMatrix": legends,
                   "file_dir": str(files), "filename": "grid.png"})
    _png_check(report, "MolsMatrixToGridImage(2x3, 150x120, legends)", path, files / "grid.png", (450, 240),
               ref.grid_pixels(matrix, (150, 120), legends))
    check_rejected(call, "MolToImage[pmol and sdf_path]", "MolToImage",
                   {"file_dir": str(files), "pmol": blob, "sdf_path": paths.get("sdf", str(files / "x.sdf"))})
    check_rejected(call, "MolToImage[no molecule]", "MolToImage", {"file_dir": str(files)})
    check_rejected(call, "MolsMatrixToGridImage[no filename]", "MolsMatrixToGridImage",
                   {"molsMatrix": [["CCO"]], "file_dir": str(files)})

    for label, extra, marker in (
            ("useSVG=true", {"useSVG": True}, "'str' object has no attribute 'save'"),
            ("returnPNG=true", {"returnPNG": True}, "'bytes' object has no attribute 'save'"),
            ("highlightAtomListsMatrix as declared (list of lists of int)",
             {"highlightAtomListsMatrix": [[0, 1]]}, "has no attribute '__iter__'")):
        result = call(f"MolsMatrixToGridImage[{label}]", "MolsMatrixToGridImage",
                      {"molsMatrix": [["CCO", "CC"]], "file_dir": str(files), "filename": "probe.png", **extra},
                      allow_error=True)
        status, detail = classify(
            result,
            correct=lambda r: f"wrote {value_of(r)}" if not r.get("isError") else None,
            defect=lambda r, m=marker: f"crashes: {text_of(r)[:120]!r}" if is_error(r, m) else None)
        report.add("L1", f"MolsMatrixToGridImage {label}", status, detail)

    # Without filename, MolToImage names the file after the current second.
    first = call("MolToImage[default filename 1]", "MolToImage", {"file_dir": str(files), "pmol": blob})
    second = call("MolToImage[default filename 2]", "MolToImage",
                  {"file_dir": str(files), "pmol": pickles.get("caffeine", blob)})
    if first is not None and second is not None:
        names = [Path(str(value_of(r))).name for r in (first, second)]
        status, detail = default_image_names(names)
        report.add("L1", "MolToImage default file names are unique", status, detail)


def check_substructure(call: Caller, report: Report, ref: Ref, pickles: dict[str, str]) -> None:
    acid = pickles.get("smarts C(=O)[OX2H1]")
    carbon_oxygen = pickles.get("smarts [#6]~[#8]")
    aspirin, ethanol = pickles.get("aspirin"), pickles.get("ethanol")
    if not all((acid, aspirin, ethanol)):
        return
    pairs = []
    for label, mol, query, smiles, smarts in (("aspirin / COOH", aspirin, acid, ASPIRIN, "C(=O)[OX2H1]"),
                                              ("ethanol / COOH", ethanol, acid, "CCO", "C(=O)[OX2H1]"),
                                              ("ethanol / C~O", ethanol, carbon_oxygen, "CCO", "[#6]~[#8]")):
        got = _value(call, f"HasSubstructMatch[{label}]", "HasSubstructMatch", {"p_mol": mol, "p_query": query})
        pairs.append((label, got, ref.substruct(smiles, query_smarts=smarts)[0]))
    chiral = {name: _value(call, f"smiles_to_mol[{name}]", "smiles_to_mol", {"smiles": s})
              for name, s in (("L-alanine", "C[C@@H](N)C(=O)O"), ("D-alanine", "C[C@H](N)C(=O)O"))}
    if all(isinstance(v, str) for v in chiral.values()):
        for flag in (False, True):
            got = _value(call, f"HasSubstructMatch[use_chirality={flag}]", "HasSubstructMatch",
                          {"p_mol": chiral["L-alanine"], "p_query": chiral["D-alanine"], "use_chirality": flag})
            pairs.append((f"L/D-alanine use_chirality={flag}", got,
                          ref.substruct("C[C@@H](N)C(=O)O", query_smiles="C[C@H](N)C(=O)O", chirality=flag)[0]))
    _compare(report, "HasSubstructMatch vs RDKit (incl. use_chirality)", pairs)

    hydroxyl = _value(call, "smarts_to_mol[[OX2H]]", "smarts_to_mol", {"smarts": "[OX2H]"})
    if isinstance(hydroxyl, str):
        got = _value(call, "GetSubstructMatch[one-atom query]", "GetSubstructMatch",
                     {"p_mol": aspirin, "p_query": hydroxyl})
        expected = list(ref.substruct(ASPIRIN, query_smarts="[OX2H]")[1])
        report.add("L1", "GetSubstructMatch(aspirin, [OX2H]) vs RDKit", "PASS" if got == expected else "FAIL",
                   f"got {got}, RDKit {expected}")
    for label, query, smarts in (("three-atom match", acid, "C(=O)[OX2H1]"),
                                 ("no match", pickles.get("smarts c1ccccc1"), "c1ccccc1")):
        mol, smiles = (aspirin, ASPIRIN) if label != "no match" else (ethanol, "CCO")
        expected = list(ref.substruct(smiles, query_smarts=smarts)[1])
        result = call(f"GetSubstructMatch[{label}]", "GetSubstructMatch", {"p_mol": mol, "p_query": query},
                      allow_error=True)
        status, detail = classify(
            result,
            correct=lambda r, e=expected: f"{value_of(r)}" if not r.get("isError") and value_of(r) == e else None,
            defect=lambda r, e=expected: (f"fails for a correct match {e}: the output is declared Tuple[int] "
                                          "(exactly one atom index)") if is_error(r, "validation error") else None)
        report.add("L1", f"GetSubstructMatch {label}", status, detail)


def check_properties(call: Caller, report: Report, ref: Ref, files: Path, tools: dict[str, dict],
                     pickles: dict[str, str]) -> None:
    blob = pickles.get("aspirin")
    if blob is None:
        return
    canon = ref.canon(ASPIRIN)
    for tool, key, value in PROPERTY_TOOLS:
        result = call(f"{tool}[{key}]", tool, {"p_mol": blob, "key": key, "value": value})
        if result is None:
            continue
        mol = ref.decode(value_of(result))
        extra = ""
        if tool == "UpdatePropertyCache":
            params = sorted(tools[tool].get("inputSchema", {}).get("properties", {}))
            extra = f"; UpdatePropertyCache takes {params} and is SetUnsignedProp under another name"
        status, detail = property_outcome(mol.GetPropsAsDict(), key, value, ref.Chem.MolToSmiles(mol) == canon, extra)
        report.add("L1", f"{tool} property survives in the returned molecule", status, detail)
    labelled = _value(call, "SetProp[for SDF]", "SetProp", {"p_mol": blob, "key": "label", "value": "aspirin"})
    if isinstance(labelled, str):
        path = _value(call, "mol_to_sdf[after SetProp]", "mol_to_sdf",
                      {"pmol": labelled, "file_dir": str(files), "filename": "labelled.sdf"})
        if isinstance(path, str) and Path(path).is_file():
            has = "> <label>" in Path(path).read_text(encoding="utf-8")
            report.add("L1", "SetProp -> mol_to_sdf writes the property", "PASS" if has else "WARN",
                       "data field present" if has else "no '> <label>' data field: the property was lost in the "
                       "pickle")


def check_batch_map(call: Caller, report: Report, ref: Ref) -> None:
    series = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC"]
    payload = _value(call, "batch_map[MolWt]", "batch_map",
                     {"tool_name": "MolWt", "inputs": [{"smiles": s} for s in series + [INVALID_SMILES]]})
    if isinstance(payload, dict):
        items = batch_outputs(payload)
        by_input = {i["smiles"]: (ok, v, e) for ok, i, v, e in items if isinstance(i, dict)}
        values = [(s, by_input.get(s, (False, None, None))[1], ref.descriptor("MolWt", s)) for s in series]
        bad = by_input.get(INVALID_SMILES, (True, None, None))
        ok = len(items) == len(series) + 1 and all(deviation(g, e) <= REL_TOL for _, g, e in values) \
            and bad[0] is False and "Invalid SMILES" in str(bad[2])
        report.add("L1", "batch_map(MolWt) values and per-item error (matched by input)", "PASS" if ok else "FAIL",
                   f"{len(items)} items; invalid entry ok={bad[0]} error={str(bad[2])[:60]!r}; each output is the "
                   "inner call_tool return [content blocks, structured content]")
    payload = _value(call, "batch_map[include_input=false]", "batch_map",
                     {"tool_name": "MolWt", "inputs": [{"smiles": s} for s in series], "include_input": False})
    if isinstance(payload, dict):
        got = [v for _, _, v, _ in batch_outputs(payload)]
        expected = [ref.descriptor("MolWt", s) for s in series]
        in_order = deviation(got, expected) <= REL_TOL
        permuted = sorted(got, key=lambda x: (x is None, x)) == sorted(expected) and not in_order
        status = "PASS" if in_order else ("WARN" if permuted else "FAIL")
        report.add("L1", "batch_map results in input order", status,
                   "in order" if in_order else (f"completion order {[round(v, 2) for v in got]}: with "
                                                "include_input=false the results cannot be matched to the inputs"
                                                if permuted else f"values {got} vs {expected}"))
    outcomes = []
    for attempt in range(FAIL_FAST_REPEATS):       # the order is arbitrary: one lucky call proves nothing
        payload = _value(call, f"batch_map[fail_fast #{attempt + 1}]", "batch_map",
                         {"tool_name": "MolWt", "inputs": [{"smiles": INVALID_SMILES}, {"smiles": "C"}],
                          "fail_fast": True, "concurrency": 1})
        if isinstance(payload, dict):
            outcomes.append(fail_fast_outcome(batch_outputs(payload), INVALID_SMILES))
    if outcomes:
        worst = next((o for status in ("FAIL", "WARN") for o in outcomes if o[0] == status), outcomes[0])
        counts = Counter(status for status, _ in outcomes)
        report.add("L1", "batch_map fail_fast stops at the first error (input order)", worst[0],
                   f"{dict(counts)} over {len(outcomes)} calls; {worst[1]}")
    for name in ("rdkit_MolWt", "mcp__rdkit__MolWt"):
        payload = _value(call, f"batch_map[{name}]", "batch_map", {"tool_name": name, "inputs": [{"smiles": "CCO"}]})
        if isinstance(payload, dict):
            items = batch_outputs(payload)
            ok = len(items) == 1 and items[0][0] and deviation(items[0][2], ref.descriptor("MolWt", "CCO")) <= REL_TOL
            report.add("L1", f"batch_map resolves prefixed tool name {name}", "PASS" if ok else "FAIL",
                       f"{items[:1]}")
    check_rejected(call, "batch_map[unknown tool]", "batch_map", {"tool_name": "NoSuchTool", "inputs": [{}]})
    check_rejected(call, "batch_map[concurrency=0]", "batch_map",
                   {"tool_name": "MolWt", "inputs": [{"smiles": "C"}], "concurrency": 0})


def check_coverage(call: Caller, report: Report, expected: list[str]) -> None:
    called = set(call.stdout_by_tool)
    missing = sorted(set(expected) - called)
    report.add("L1", "every listed tool was called", "FAIL" if missing else "PASS",
               f"missing {missing}" if missing else f"{len(expected)} tools")


def run_l1(session: Session) -> None:
    call, report = session.call, session.report
    ref = Ref()
    tools = {t["name"]: t for t in session.client.list_tools()}
    files = session.tmp / "files" / "out"
    files.mkdir(parents=True)

    check_descriptors(call, report, ref)
    check_anchors(call, report)
    check_rotatable_description(call, report, ref)
    check_compute_descriptors(call, report, ref)
    check_args_kwargs_tools(call, report, ref, tools)
    check_oxidation_numbers(call, report)
    check_invalid_inputs(call, report)
    check_scaffolds_fragments_similarity(call, report, ref)
    pickles = check_mol_io(call, report, ref)
    pickles["trans-2-butene"] = _value(call, "smiles_to_mol[C/C=C/C]", "smiles_to_mol", {"smiles": "C/C=C/C"})
    embedded = check_embedding(call, report, ref, tools, pickles)
    paths = check_files(call, report, ref, files, embedded, pickles)
    check_images(call, report, ref, files, pickles, paths)
    check_substructure(call, report, ref, pickles)
    check_properties(call, report, ref, files, tools, pickles)
    check_batch_map(call, report, ref)
    check_coverage(call, report, session.entry["expected_tools"])


SMOKE = Smoke(
    server="rdkit",
    run_l1=run_l1,
    packages=("mcp", "rdkit", "pillow", "pydantic", "numpy"),
)
