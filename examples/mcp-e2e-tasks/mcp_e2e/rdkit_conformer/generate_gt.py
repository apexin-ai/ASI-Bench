"""Instance generator for the rdkit conformer MCP E2E fake task.

A seeded RNG picks a small drug-like molecule and an embedding seed. Through the
rdkit MCP server the agent must turn the SMILES into a molecule
(``smiles_to_mol``), generate one 3D conformer with ETKDGv3 and that seed
(``EmbedMolecule``), write it to ``conformer.sdf`` in its working directory
(``mol_to_sdf``) and report the most positive and most negative Gasteiger
partial charges (``MaxPartialCharge`` / ``MinPartialCharge``).

Molecules travel between the tools as base64 Python pickles; the agent has to
pass each one on verbatim. The reference is computed here with RDKit directly
(never through the server), mirroring what the tools document:

* the molecule pickle = ``pickle.dumps(Chem.MolFromSmiles(smiles))`` with the
  server's pickle protocol (4, Python 3.12's default), base64-encoded;
* ``EmbedMolecule`` with no other parameters = ETKDGv3 with ``randomSeed``
  (the tool's documented ``EmbedParameters`` defaults equal ETKDGv3's for a
  single conformer; checked by the server's L1 smoke). No hydrogens are added:
  the server has no AddHs tool, so the conformer is heavy-atom only;
* coordinates after a pickle round trip (RDKit pickles conformers as float32);
* Gasteiger charges from ``Descriptors.MaxPartialCharge`` / ``MinPartialCharge``.

With the same RDKit build the pickles and coordinates are bit-identical to the
server's (VM aarch64, all ten molecules, 2026-10-02), so the verifier can
compare the returned pickles exactly. SMILES are chosen without ``/`` or
``\\`` so that no input string looks like a host path in persisted logs.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs RDKit 2025.3.1: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import pickle
import random
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "molecule.json", "description": "molecule name, SMILES, embedding seed, output SDF file name"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "conf_id, max_partial_charge, min_partial_charge"},
    {"name": "conformer.sdf", "description": "the seeded 3D conformer written by mol_to_sdf"},
]
DEFAULT_PARAMS = {"seed": 0}

MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "benzocaine": "CCOC(=O)c1ccc(N)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    "nicotine": "CN1CCCC1c1cccnc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "phenacetin": "CCOc1ccc(NC(C)=O)cc1",
    "procaine": "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "salbutamol": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
}
SDF_FILE = "conformer.sdf"
SEED_RANGE = (1, 100_000)
PICKLE_PROTOCOL = 4                  # the server's interpreter (Python 3.12) default


def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    name = sorted(MOLECULES)[rng.randrange(len(MOLECULES))]
    return {"molecule": name, "smiles": MOLECULES[name], "random_seed": rng.randrange(*SEED_RANGE),
            "output_file": SDF_FILE}


def encode(mol) -> str:
    return base64.b64encode(pickle.dumps(mol, protocol=PICKLE_PROTOCOL)).decode("ascii")


def reference(case: dict) -> dict:
    import rdkit
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdDistGeom

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(case["smiles"])
    if mol is None:
        raise RuntimeError(f"cannot parse {case['smiles']!r}")
    parsed = encode(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = case["random_seed"]
    conf_id = rdDistGeom.EmbedMolecule(mol, params)
    if conf_id != 0:
        raise RuntimeError(f"embedding failed for seed {case['random_seed']}")
    embedded = encode(mol)
    mol = pickle.loads(base64.b64decode(embedded))            # float32 coordinates, as the server returns them
    atoms = [[atom.GetSymbol(), *map(float, mol.GetConformer().GetAtomPosition(atom.GetIdx()))]
             for atom in mol.GetAtoms()]
    fresh = Chem.MolFromSmiles(case["smiles"])
    return {
        "mol_pickle": parsed,
        "embedded_mol_pickle": embedded,
        "conf_id": conf_id,
        "atoms": atoms,
        "max_partial_charge": float(Descriptors.MaxPartialCharge(fresh)),
        "min_partial_charge": float(Descriptors.MinPartialCharge(fresh)),
        "sdf_file": SDF_FILE,
        "rdkit_version": rdkit.__version__,
    }


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key in ("molecule", "smiles", "random_seed"):
            text = text.replace("{{" + key + "}}", str(case[key]))
        (output_dir / f"prompt_{level}.md").write_text(text, encoding="utf-8")


def generate(output_dir: Path, params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    ref_dir = output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    case = build_case(int(p["seed"]))
    ref = reference(case)
    numbers = [ref["max_partial_charge"], ref["min_partial_charge"], *(c for a in ref["atoms"] for c in a[1:])]
    if not all(math.isfinite(v) for v in numbers):
        raise RuntimeError("non-finite reference")
    (data_dir / "molecule.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    reference_doc = {**case, **ref, "mcp_tools": ["smiles_to_mol", "EmbedMolecule", "mol_to_sdf",
                                                  "MaxPartialCharge", "MinPartialCharge"]}
    (ref_dir / "reference.json").write_text(json.dumps(reference_doc, indent=2) + "\n", encoding="utf-8")

    render_prompts(Path(__file__).resolve().parent, output_dir, case)
    meta = {
        "params_used": p,
        "input_files": [s["name"] for s in INPUT_SPEC],
        "reference_files": ["reference.json"],
        "generation_time_seconds": round(time.time() - t0, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, json.loads(args.params)), indent=2))
