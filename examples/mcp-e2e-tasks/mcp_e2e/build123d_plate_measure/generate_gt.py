"""Instance generator for the build123d plate MCP E2E fake task.

A seeded RNG picks a rectangular cover plate: filleted corners, a central bore
and a bolt circle. Through the build123d MCP server the agent must model it
(``execute``), gate it (``validate``), measure it (``measure``), read its hole
table (``find_holes`` / ``find_hole_patterns``), export STEP + STL
(``export``), check the written STEP by importing it back
(``import_cad_file``) and report the numbers in ``result.json``.

Everything scored has a **closed form**, so this generator needs neither
build123d nor OpenCascade — only the standard library:

* cross-section ``A``, ``∫y²dA`` and ``∫x²dA`` of the plate outline minus the
  bore, minus four corner slivers (square − quarter disc), minus every bolt
  hole (disc + Steiner term);
* ``volume = A·t``; ``area = 2A + t·(perimeter + π·bore_d + n·π·bolt_d)``,
  where the outline perimeter is the four straight runs plus one full circle of
  the corner radius;
* ``Izz = t(∫y²+∫x²)dA``, ``Ixx = t∫y²dA + A·t³/12``, ``Iyy`` likewise; with a
  ``material`` the server scales the volume inertia by the density, so the
  reference is in g·mm² like its answer;
* ``mass = volume · density``, with the server's documented preset densities.

Checked against the pinned server on three shapes (2026-10-03): every value
agrees to its 4-decimal rounding (≤5e-5), the exported mesh is watertight and
its volume is 2e-5 above the exact solid.

The export goes to a fixed ``/tmp`` stem (the server may only write under its
own cwd or the temporary directory, and its ``export`` does not create missing
parent directories). Repetitions of the same instance therefore share the
stem; that is harmless, because the geometry is deterministic and ``export``
writes atomically, so a concurrent reader sees one complete file either way.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Only the standard library: generate with ``asibench generate --sandbox none``
or ``--sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "plate.json", "description": "part name, plate dimensions, bore and bolt circle, "
                                          "material and the export stem"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "measured volume, area, mass, Izz, the re-imported "
                                           "volume, bbox, hole table and gate"},
    {"name": "part.step", "description": "the STEP written by the server's export tool"},
    {"name": "part.stl", "description": "the STL written by the server's export tool"},
]
DEFAULT_PARAMS = {"seed": 0}

# Plate names (a part name also becomes the STEP product label).
PARTS = ("bearing_cover", "gearbox_cover", "inspection_plate", "manifold_flange",
         "motor_mount", "pump_cover", "sensor_bracket", "valve_plate")
PLATE_X = (76.0, 80.0, 86.0, 90.0, 96.0, 100.0, 108.0)
PLATE_Y = (54.0, 58.0, 62.0, 66.0, 70.0, 76.0, 84.0)
PLATE_Z = (8.0, 9.0, 10.0, 11.0, 12.0, 14.0)
CORNER_R = (5.0, 6.0, 8.0, 10.0)
BORE_D = (14.0, 16.0, 18.0, 20.0, 24.0)
BOLT_N = (4, 6, 8)
BOLT_D = (5.0, 6.0, 8.0)
# The server's documented material presets (g/cm³), build123d_mcp/tools/measure.py.
DENSITIES = {"steel": 7.85, "stainless": 8.00, "aluminum": 2.70, "brass": 8.50,
             "titanium": 4.43, "pla": 1.24}
MATERIALS = tuple(sorted(DENSITIES))

STEP_FILE = "part.step"
STL_FILE = "part.stl"
EXPORT_DIR = "/tmp"
BORE_CLEARANCE = 4.0        # mm of material between the bore and a bolt hole
EDGE_CLEARANCE = 6.0        # mm of material between a bolt hole and the plate edge
BOLT_GAP = 2.0              # mm between neighbouring bolt holes
MCP_TOOLS = ("execute", "validate", "measure", "find_holes", "find_hole_patterns",
             "export", "import_cad_file")


# --------------------------------------------------------------------------
# The instance
# --------------------------------------------------------------------------

def build_case(seed: int) -> dict:
    """A plate whose bore, bolt circle and corner radius always fit."""
    rng = random.Random(seed)
    for _ in range(200):
        case = {
            "part": PARTS[rng.randrange(len(PARTS))],
            "plate_x": PLATE_X[rng.randrange(len(PLATE_X))],
            "plate_y": PLATE_Y[rng.randrange(len(PLATE_Y))],
            "plate_z": PLATE_Z[rng.randrange(len(PLATE_Z))],
            "corner_radius": CORNER_R[rng.randrange(len(CORNER_R))],
            "bore_d": BORE_D[rng.randrange(len(BORE_D))],
            "bolt_count": BOLT_N[rng.randrange(len(BOLT_N))],
            "bolt_d": BOLT_D[rng.randrange(len(BOLT_D))],
            "material": MATERIALS[rng.randrange(len(MATERIALS))],
        }
        low = case["bore_d"] + case["bolt_d"] + 2 * BORE_CLEARANCE
        high = min(case["plate_x"], case["plate_y"]) - case["bolt_d"] - 2 * EDGE_CLEARANCE
        choices = [d for d in range(int(low) + (int(low) % 2), int(high) + 1, 2)
                   if d * math.sin(math.pi / case["bolt_count"]) >= case["bolt_d"] + BOLT_GAP]
        if not choices or case["corner_radius"] > min(case["plate_x"], case["plate_y"]) / 4:
            continue
        case["bolt_circle_d"] = float(choices[rng.randrange(len(choices))])
        case["export_stem"] = f"{EXPORT_DIR}/asibench-b123d-{seed}-{case['part']}"
        case["step_file"] = STEP_FILE
        case["stl_file"] = STL_FILE
        return case
    raise RuntimeError(f"no valid plate for seed {seed}")


def bolt_positions(case: dict) -> list[tuple[float, float]]:
    """Bolt-hole centres, as ``PolarLocations(bolt_circle_d / 2, n)`` places them."""
    radius = case["bolt_circle_d"] / 2
    return [(radius * math.cos(2 * math.pi * k / case["bolt_count"]),
             radius * math.sin(2 * math.pi * k / case["bolt_count"]))
            for k in range(case["bolt_count"])]


# --------------------------------------------------------------------------
# Closed-form reference
# --------------------------------------------------------------------------

def section_moments(case: dict) -> tuple[float, float, float]:
    """``(A, ∫y²dA, ∫x²dA)`` of the cross-section about its centroid."""
    x, y = case["plate_x"], case["plate_y"]
    area, jx, jy = x * y, x * y**3 / 12, x**3 * y / 12

    bore = case["bore_d"] / 2
    area -= math.pi * bore**2
    jx -= math.pi * bore**4 / 4
    jy -= math.pi * bore**4 / 4

    f = case["corner_radius"]
    if f:
        # one corner: the square [cx, cx+f] × [cy, cy+f] minus the quarter disc
        # centred at (cx, cy); the four corners are symmetric.
        cx, cy = x / 2 - f, y / 2 - f
        square_a = f * f
        square_jy = ((cx + f) ** 3 - cx**3) / 3 * f
        square_jx = ((cy + f) ** 3 - cy**3) / 3 * f
        quarter_a = math.pi * f * f / 4
        # polar about (cx, cy), θ ∈ [0, π/2]: ∫cos dθ = 1, ∫cos²dθ = π/4
        quarter_jy = cx**2 * quarter_a + 2 * cx * f**3 / 3 + f**4 / 4 * math.pi / 4
        quarter_jx = cy**2 * quarter_a + 2 * cy * f**3 / 3 + f**4 / 4 * math.pi / 4
        area -= 4 * (square_a - quarter_a)
        jx -= 4 * (square_jx - quarter_jx)
        jy -= 4 * (square_jy - quarter_jy)

    bolt = case["bolt_d"] / 2
    for bx, by in bolt_positions(case):
        area -= math.pi * bolt**2
        jx -= math.pi * bolt**4 / 4 + math.pi * bolt**2 * by**2     # Steiner
        jy -= math.pi * bolt**4 / 4 + math.pi * bolt**2 * bx**2
    return area, jx, jy


def reference(case: dict) -> dict:
    area2d, jx, jy = section_moments(case)
    t = case["plate_z"]
    f = case["corner_radius"]
    volume = area2d * t
    straight = 2 * (case["plate_x"] - 2 * f) + 2 * (case["plate_y"] - 2 * f)
    perimeter = straight + 2 * math.pi * f                     # four quarter arcs = one circle
    surface = (2 * area2d
               + t * (perimeter + math.pi * case["bore_d"]
                      + case["bolt_count"] * math.pi * case["bolt_d"]))
    density = DENSITIES[case["material"]] / 1000.0             # g/mm³
    stem = Path(case["export_stem"]).name
    return {
        "volume_mm3": volume,
        # the STEP written and read back must report the same exact solid
        "reimported_volume_mm3": volume,
        "surface_area_mm2": surface,
        "mass_g": volume * density,
        # measure(material=...) returns mass moments in g·mm²
        "izz_g_mm2": t * (jx + jy) * density,
        "ixx_g_mm2": (t * jx + area2d * t**3 / 12) * density,
        "iyy_g_mm2": (t * jy + area2d * t**3 / 12) * density,
        "section_area_mm2": area2d,
        "bbox_mm": [case["plate_x"], case["plate_y"], case["plate_z"]],
        "center_of_mass_mm": [0.0, 0.0, 0.0],
        "hole_count": case["bolt_count"] + 1,                  # bolt circle plus the bore
        "bolt_pattern_count": 1,
        "bolt_hole_diameter_mm": case["bolt_d"],
        "bolt_positions_mm": [[round(bx, 6), round(by, 6)] for bx, by in bolt_positions(case)],
        "face_count": 11 + case["bolt_count"],                 # 2 + 4 walls + 4 fillets + bore + bolts
        "n_solids": 1,
        "passes_gate": True,
        "density_g_cm3": DENSITIES[case["material"]],
        "export_files": sorted([f"{stem}.step", f"{stem}.stl"]),
        "mcp_tools": list(MCP_TOOLS),
    }


# --------------------------------------------------------------------------
# Framework entry point
# --------------------------------------------------------------------------

def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key, value in case.items():
            text = text.replace("{{" + key + "}}", str(value))
        if "{{" in text:
            raise RuntimeError(f"unresolved placeholder in prompt_{level}.md")
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
    numbers = [v for v in ref.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not all(math.isfinite(v) for v in numbers) or ref["volume_mm3"] <= 0:
        raise RuntimeError("non-finite or empty reference")

    (data_dir / "plate.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    (ref_dir / "reference.json").write_text(json.dumps({**case, **ref}, indent=2) + "\n",
                                            encoding="utf-8")
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
