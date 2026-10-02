"""Instance generator for the jsbsim engine-run MCP E2E fake task.

A seeded RNG picks a short powered-flight scenario for the JSBSim c172x model:
seven initial conditions, an engine start plus mixture and throttle settings,
and a duration. The reference flies the same scenario with the JSBSim Python
module directly, independently of the MCP server code: its own initial-
condition properties (``ic/h-sl-ft``, ``ic/vc-kts`` with an exact ft/s -> kt
conversion, ...), ``run_ic()``, the property writes in order, then
round(duration / dt) frames at the default dt of 1/60 s.

The aircraft data comes from the jsbsim wheel (``jsbsim.get_default_root_dir()``);
for c172x its aircraft/engine/systems files are identical to the
``jsbsim_data/`` tree of the pinned server checkout, so nothing is vendored.
JSBSim is deterministic: the server and this reference agree to ~1e-3 ft
(the server converts ft/s to kt with 0.592484, see scripts/mcp/e2e/e2e_smoke/servers/jsbsim.py).

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs jsbsim: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "scenario.json", "description": "aircraft, initial conditions, property settings and duration"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "altitude_ft, airspeed_kt, thrust_lbs, sim_time_s after the run"},
]
DEFAULT_PARAMS = {"seed": 0}

AIRCRAFT = "c172x"
DT = 1.0 / 60.0
FPS_PER_KT = 6076.115485564304 / 3600.0      # ft/s per knot (international nautical mile)
# (place, latitude deg, longitude deg)
LOCATIONS = [("San Francisco Bay", 37.62, -122.38), ("Seattle", 47.45, -122.31),
             ("Los Angeles basin", 33.94, -118.41), ("Denver plains", 39.86, -104.67),
             ("Chicago", 41.98, -87.90)]
# What the agent must report and the JSBSim property that holds it.
QUANTITIES = {
    "altitude_ft": "position/h-sl-ft",
    "airspeed_kt": "velocities/vc-kts",
    "thrust_lbs": "propulsion/engine[0]/thrust-lbs",
    "sim_time_s": "simulation/sim-time-sec",
}


def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    place, lat, lon = LOCATIONS[rng.randrange(len(LOCATIONS))]
    initial_conditions = {
        "altitude_ft": float(rng.randrange(3000, 6001, 100)),
        "latitude_deg": lat,
        "longitude_deg": lon,
        "airspeed_fps": round(rng.uniform(150.0, 185.0), 2),
        "heading_deg": float(rng.randrange(0, 360, 5)),
        "pitch_deg": float(rng.choice([0, 1, 2, 3])),
        "roll_deg": 0.0,
    }
    settings = [
        {"path": "propulsion/set-running", "value": -1.0},
        {"path": "fcs/mixture-cmd-norm", "value": 1.0},
        {"path": "fcs/throttle-cmd-norm", "value": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])},
    ]
    return {"aircraft": AIRCRAFT, "place": place, "initial_conditions": initial_conditions,
            "settings": settings, "duration_s": float(rng.choice([6, 8, 10])), "timestep_s": DT,
            "units": {"altitude": "ft above mean sea level", "airspeed": "calibrated, kt",
                      "thrust": "lbf", "time": "s"}}


def ic_properties(ic: dict) -> dict[str, float]:
    """JSBSim's own initial-condition properties for the scenario keys."""
    return {"ic/h-sl-ft": ic["altitude_ft"], "ic/lat-geod-deg": ic["latitude_deg"],
            "ic/long-gc-deg": ic["longitude_deg"], "ic/vc-kts": ic["airspeed_fps"] / FPS_PER_KT,
            "ic/psi-true-deg": ic["heading_deg"], "ic/theta-deg": ic["pitch_deg"], "ic/phi-deg": ic["roll_deg"]}


def fly(case: dict, root: str | None = None) -> dict:
    os.environ["JSBSIM_DEBUG"] = "0"          # read by FGFDMExec: no console chatter
    import jsbsim

    fdm = jsbsim.FGFDMExec(root or jsbsim.get_default_root_dir())
    fdm.set_dt(case["timestep_s"])
    if not fdm.load_model(case["aircraft"]):
        raise RuntimeError(f"could not load {case['aircraft']}")
    for path, value in ic_properties(case["initial_conditions"]).items():
        fdm.set_property_value(path, float(value))
    fdm.run_ic()
    for setting in case["settings"]:
        fdm.set_property_value(setting["path"], float(setting["value"]))
    for _ in range(max(1, round(case["duration_s"] / case["timestep_s"]))):
        fdm.run()
    values = {key: float(fdm.get_property_value(path)) for key, path in QUANTITIES.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError(f"non-finite reference state: {values}")
    if values["thrust_lbs"] <= 0.0:
        raise RuntimeError("engine not running in the reference; scenario is wrong")
    if values["altitude_ft"] < 500.0:
        raise RuntimeError(f"aircraft too low at the end ({values['altitude_ft']:.0f} ft)")
    return {**values, "jsbsim_version": jsbsim.__version__}


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key in ("aircraft", "place"):
            text = text.replace("{{" + key + "}}", str(case[key]))
        text = text.replace("{{duration_s}}", f"{case['duration_s']:g}")
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
    flown = fly(case)
    (data_dir / "scenario.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    reference = {**case, **flown, "dt_s": DT, "frames": max(1, round(case["duration_s"] / DT)),
                 "property_paths": dict(QUANTITIES), "altitude_path": QUANTITIES["altitude_ft"],
                 **{f"setting{k}_{f}": s[f] for k, s in enumerate(case["settings"]) for f in ("path", "value")},
                 "mcp_tools": ["create_session", "set_initial_conditions", "set_property", "step",
                               "get_telemetry", "get_property"]}
    (ref_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

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
