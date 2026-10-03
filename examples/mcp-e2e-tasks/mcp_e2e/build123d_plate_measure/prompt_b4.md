# Bolted cover plate: model, gate, measure and export

Build the cover plate specified in `data/plate.json` as a CAD solid, check it against a validity gate, measure its mass properties and hole table for the material given there, and hand the geometry over as STEP and STL. Use the CAD modelling tools available in this session; register the part under its name, export to the stem given in the file (the server writes only under its own working directory or the temporary directory) and copy both files into the current working directory unchanged.

Background: a validity gate asks whether the result is a single well-formed, watertight, manifold solid — the condition a STEP or STL consumer needs. Mass properties follow from the geometry and the material density; the inertia tensor's `Izz` is the mass moment about the plate's own Z axis. Feature recognisers read holes and bolt circles back out of the finished solid instead of trusting the construction parameters, and a tessellated STL approximates curved faces, so its volume differs slightly from the exact solid.

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
  "bbox_mm": [<float>, <float>, <float>],
  "hole_count": <int>,
  "bolt_hole_diameter_mm": <float>,
  "n_solids": <int>,
  "passes_gate": <true or false>
}
```

- `volume_mm3`, `surface_area_mm2`, `mass_g`: the solid's volume (mm³), surface area (mm²) and mass (g) as measured.
- `izz_g_mm2`: the `Izz` component of the inertia tensor as measured with the material (g·mm²).
- `bbox_mm`: the bounding-box sizes along X, Y and Z (mm), in that order.
- `hole_count`: how many holes the recogniser finds in the part.
- `bolt_hole_diameter_mm`: the diameter of the bolt-circle holes (mm).
- `n_solids`, `passes_gate`: from the validity gate.

Copy each value exactly as the tools return it (full precision); do not round, convert or recompute them.

## Rules

- build123d and its CAD kernel are available **only** through the MCP server attached to this session. Do not install, import or run build123d, CadQuery, OCP/OpenCascade, trimesh, numpy-stl, FreeCAD, OpenSCAD or any other CAD or mesh library yourself (no `pip`/`uv`/`conda` installs, no `import build123d`, no Python interpreter or files belonging to an MCP server), and do not estimate values from memory or by hand.
- Do not write, edit, convert or regenerate `{{step_file}}` or `{{stl_file}}`; they must be the files the server exported, copied unchanged.
- Build exactly the plate described in the file: the bore and the bolt circle are through holes, and the fillets are on the four vertical edges only.
- Do not modify `data/plate.json`.
