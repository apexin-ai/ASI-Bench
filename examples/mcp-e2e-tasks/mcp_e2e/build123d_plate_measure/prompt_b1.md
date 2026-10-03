# Bolted cover plate: model, gate, measure and export via the build123d MCP server

`data/plate.json` describes one part, `{{part}}`: a rectangular plate {{plate_x}} × {{plate_y}} × {{plate_z}} mm with all four vertical edges filleted at radius {{corner_radius}} mm, a central through bore of diameter {{bore_d}} mm, and {{bolt_count}} through holes of diameter {{bolt_d}} mm on a bolt circle of diameter {{bolt_circle_d}} mm. The material is `{{material}}`.

## Steps

1. Read `data/plate.json`.
2. Call the `execute` tool of the `build123d` MCP server with this code, which builds the plate and registers it under the part name:

   ```python
   from build123d import *
   plate_x = {{plate_x}}
   plate_y = {{plate_y}}
   plate_z = {{plate_z}}
   corner_radius = {{corner_radius}}
   bore_d = {{bore_d}}
   bolt_d = {{bolt_d}}
   bolt_circle_d = {{bolt_circle_d}}
   with BuildPart() as part:
       Box(plate_x, plate_y, plate_z)
       fillet(part.edges().filter_by(Axis.Z), corner_radius)
       Cylinder(bore_d / 2, plate_z, mode=Mode.SUBTRACT)
       with PolarLocations(bolt_circle_d / 2, {{bolt_count}}):
           Cylinder(bolt_d / 2, plate_z, mode=Mode.SUBTRACT)
   show(part.part, '{{part}}')
   ```

3. Call the `validate` tool of the same server with `object_name` = `{{part}}`. Take `passes_gate` and `n_solids` from its answer; both must say the part is one valid solid before you continue.
4. Call the `measure` tool of the same server with `object_name` = `{{part}}` and `material` = `{{material}}`. Take `volume`, `area`, `mass_g`, the `Izz` component of `inertia` and the `xsize`/`ysize`/`zsize` of `bbox` from its answer.
5. Call the `find_holes` tool of the same server with `object_name` = `{{part}}` and take the hole `count`.
6. Call the `find_hole_patterns` tool of the same server with `object_name` = `{{part}}` and take the `diameter` of the bolt-circle holes.
7. Call the `export` tool of the same server with `filename` = `{{export_stem}}` (exactly that, no suffix), `format` = `step,stl` and `object_name` = `{{part}}`. The server writes `{{export_stem}}.step` and `{{export_stem}}.stl`; it may only write under its own working directory or the temporary directory, which is why the files go there and not straight into your working directory.
8. Call the `import_cad_file` tool of the same server with `path` = `{{export_stem}}.step` and `name` = `written_step`, and confirm that the volume it reports is the volume you measured in step 4 — this proves the written file holds the solid.
9. Copy `{{export_stem}}.step` to `{{step_file}}` and `{{export_stem}}.stl` to `{{stl_file}}` in the current working directory. Copy them unchanged; do not edit or convert them.
10. Write `result.json` as described below.

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
- Build exactly the plate described above: the bore and the bolt circle are through holes, and the fillets are on the four vertical edges only.
- Do not modify `data/plate.json`.
