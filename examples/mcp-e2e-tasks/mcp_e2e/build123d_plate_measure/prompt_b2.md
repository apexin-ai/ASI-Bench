# Bolted cover plate: model, gate, measure and export via the build123d MCP server

`data/plate.json` describes one part, `{{part}}`: a rectangular plate {{plate_x}} × {{plate_y}} × {{plate_z}} mm with all four vertical edges filleted at radius {{corner_radius}} mm, a central through bore of diameter {{bore_d}} mm, and {{bolt_count}} through holes of diameter {{bolt_d}} mm on a bolt circle of diameter {{bolt_circle_d}} mm. The material is `{{material}}`.

## Method

Model the plate in the `build123d` MCP server's session and register it under the part name `{{part}}`, then take every number below from that server's own tools:

- build the solid by subtracting the bore and the bolt circle from the filleted box (a polar pattern places the bolt holes);
- run the validity gate before measuring, and measure with the material so that the answer carries the mass and mass moments;
- read the hole table and the bolt-circle diameter from the feature recognisers rather than from the input file;
- export STEP and STL in one call from the registered object, to the stem `{{export_stem}}`. The server may only write under its own working directory or the temporary directory, so export there and copy the two files into the current working directory afterwards, unchanged;
- confirm the written STEP by importing it back into the session and comparing its volume with the measured one.

## Output

Three files must end up in the current working directory:

- `{{step_file}}` and `{{stl_file}}`: the STEP and STL written by the MCP server (not by you), copied unchanged.
- `result.json`, written by you:

```json
{
  "part": "{{part}}",
  "volume_mm3": <float>,
  "surface_area_mm2": <float>,
  "mass_g": <float>,
  "izz_g_mm2": <float>,
  "reimported_volume_mm3": <float>,
  "bbox_mm": [<float>, <float>, <float>],
  "hole_count": <int>,
  "bolt_hole_diameter_mm": <float>,
  "n_solids": <int>,
  "passes_gate": <true or false>
}
```

- `volume_mm3`, `surface_area_mm2`, `mass_g`: the solid's volume (mm³), surface area (mm²) and mass (g) as measured.
- `izz_g_mm2`: the `Izz` component of the inertia tensor as measured with the material (g·mm²).
- `reimported_volume_mm3`: the volume reported when the exported STEP is imported back into the session — read the file back and report what that import says, which is what proves the written file holds the solid.
- `bbox_mm`: the bounding-box sizes along X, Y and Z (mm), in that order.
- `hole_count`: how many holes the recogniser finds in the part.
- `bolt_hole_diameter_mm`: the diameter of **one bolt hole** of the pattern, in mm, as the feature recogniser reports it — the hole itself, not the diameter of the circle the holes sit on.
- `n_solids`, `passes_gate`: from the validity gate.

Copy each value exactly as the tools return it (full precision); do not round, convert or recompute them.

## Rules

- build123d and its CAD kernel are available **only** through the MCP server attached to this session. Do not install, import or run build123d, CadQuery, OCP/OpenCascade, trimesh, numpy-stl, FreeCAD, OpenSCAD or any other CAD or mesh library yourself (no `pip`/`uv`/`conda` installs, no `import build123d`, no Python interpreter or files belonging to an MCP server), and do not estimate values from memory or by hand.
- Do not write, edit, convert or regenerate `{{step_file}}` or `{{stl_file}}`; they must be the files the server exported, copied unchanged.
- Build exactly the plate described above: the bore and the bolt circle are through holes, and the fillets are on the four vertical edges only.
- Do not modify `data/plate.json`.
