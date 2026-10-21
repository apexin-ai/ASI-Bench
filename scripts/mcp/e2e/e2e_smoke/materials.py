"""Shared material smoke checks; business errors must never count as success."""
from __future__ import annotations

import json
import math
import subprocess

from .client import text_of

SI2 = {"symbols": ["Si", "Si"], "positions": [[0, 0, 0], [1.35, 1.35, 1.35]],
       "cell": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]], "pbc": [True, True, True]}


def payload(result):
    if result is None:
        return {"error": "missing response"}
    if result.get("isError"):
        return {"error": text_of(result)}
    if "structuredContent" in result:
        value = result["structuredContent"]
        return value["result"] if isinstance(value, dict) and set(value) == {"result"} else value
    try:
        return json.loads(text_of(result))
    except ValueError:
        return text_of(result)


def in_band_error(result):
    value = payload(result)
    if isinstance(value, dict) and value.get("error"):
        return str(value["error"])
    return None


def load_model(session):
    name = "load_model succeeds"
    value = payload(session.call(name, "load_model", {"model_name": session.args.model_name, "device": "cpu"}))
    ok = isinstance(value, str) and value.startswith("Successfully loaded ")
    session.report.add("L1", name, "PASS" if ok else "FAIL", str(value)[:500])
    return ok


# This runs in a separate process in the server venv and calls the official
# backend directly, without importing an MCP server, wrapper or this adapter.
REFERENCE = '''
import json, sys
from ase import Atoms
kind, model, request, output = sys.argv[1:]
atoms = Atoms(**json.load(open(request)))
if kind == 'chgnet':
    import matgl
    from matgl.ext.ase import PESCalculator
    atoms.calc = PESCalculator(potential=matgl.load_model(model), device='cpu', stress_unit='eV/A3')
elif kind == 'mace':
    from mace.calculators import mace_mp
    model = {'MACE-OMAT-0-small': 'small-omat-0', 'MACE-OMAT-0-medium': 'medium-omat-0'}.get(model, model)
    atoms.calc = mace_mp(model=model, device='cpu')
else:
    import os
    from mattersim.forcefield import MatterSimCalculator
    atoms.calc = MatterSimCalculator(load_path=os.environ.get('MATTERSIM_LOAD_PATH') or model, device='cpu')
json.dump({'energy': float(atoms.get_potential_energy()), 'forces': atoms.get_forces().tolist(),
           'stress': atoms.get_stress().tolist()}, open(output, 'w'))
'''


def prediction(session, model):
    name = "predict_structure agrees with independent ASE backend"
    result = payload(session.call(name, "predict_structure", {"structure_data": SI2, **(
        {"device": "cpu"} if session.smoke.server == "mattersim" else {})}))
    if not isinstance(result, dict) or result.get("error"):
        session.report.add("L1", name, "FAIL", str(result)[:500])
        return
    request, output = session.tmp / "structure.json", session.tmp / "reference.json"
    request.write_text(json.dumps(SI2))
    try:
        proc = subprocess.run([session.server["command"], "-c", REFERENCE, session.smoke.server,
                               model, str(request), str(output)], cwd=session.cwd, env=session.env,
                              text=True, capture_output=True, timeout=600)
        if proc.returncode:
            raise ValueError(proc.stderr[-1500:])
        ref = json.loads(output.read_text())
        keys = {"energy": "energy_eV", "forces": "forces_eV_per_A", "stress": "stress_eV_per_A3"}
        actual = {k: result[keys[k] if session.smoke.server == "mattersim" else k] for k in keys}
        def flat(v):
            return [x for item in v for x in flat(item)] if isinstance(v, list) else [float(v)]
        diffs = {}
        for key in keys:
            a, b = flat(actual[key]), flat(ref[key])
            if len(a) != len(b) or not all(math.isfinite(x) for x in a + b):
                raise ValueError(f"invalid {key} shape or nonfinite values")
            diffs[key] = max(abs(x-y) for x, y in zip(a, b))
        ok = all(v < 1e-5 for v in diffs.values())
        session.report.add("L1", name, "PASS" if ok else "FAIL", f"max abs errors: {diffs}",
                           actual=actual, reference=ref)
    except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
        session.report.add("L1", name, "FAIL", str(exc))
