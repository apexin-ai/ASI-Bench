# Reflection spectrum of a dielectric grating via the s4 MCP server

`data/grating.json` describes a one-dimensional dielectric grating on a substrate: the period (`period_um`), the medium the light comes from (`superstrate`), a grating layer of thickness `grating.thickness_um` made of {{ridge}} ridges of width `grating.ridge_width_um` separated by grooves of `grating.groove_material` (the ridges run along y, are uniform along y and repeat with the period along x, one ridge centred at x = 0), and the `substrate` below. Every material is given by its refractive index `n`. A {{polarization}}-polarised plane wave is incident from the superstrate at `illumination.theta_deg` from the normal, in the xz plane; the simulation uses `n_harmonics` Fourier harmonics and sweeps the vacuum wavelength linearly over `sweep` (`wavelength_start_um` to `wavelength_stop_um` inclusive, `wavelength_points` points).

## Steps

1. Read `data/grating.json`.
2. Call the `simulate_stack_spectrum` tool provided by the `s4` MCP server **once** with:
   - `period` = `period_um`;
   - `materials`: one entry per material name, `{"n": <index>}` (the substrate, the groove material and the ridge material);
   - `layers`, ordered **bottom to top** (substrate first, incidence medium last):
     1. `{"name": "substrate", "thickness": 1.0, "material": <substrate name>}`,
     2. `{"name": "grating", "thickness": grating.thickness_um, "material": <groove material name>, "pattern": {"material": <ridge material name>, "halfwidths": [ridge_width_um / 2, period_um / 2], "center": [0, 0]}}`,
     3. `{"name": "superstrate", "thickness": 1.0, "material": <superstrate name>}`
     (the first and last layers are semi-infinite, so their thickness does not matter; `halfwidths[1] = period_um / 2` makes the ridge fill the whole cell along y);
   - `incidence_layer` = `"superstrate"`, `substrate_layer` = `"substrate"`;
   - `polarization` = `illumination.polarization`, `theta_deg` = `illumination.theta_deg`, `n_harmonics` = `n_harmonics`;
   - `wavelength_start` = `sweep.wavelength_start_um`, `wavelength_stop` = `sweep.wavelength_stop_um`, `wavelength_points` = `sweep.wavelength_points`;
   - `include_plot` = `false`.
3. From the returned `wavelength`, `R` and `T` arrays take the values described below.
4. Write `result.json`.

## Output

Write `result.json` in the current working directory:

```json
{
  "R_at_report": <float>,
  "T_at_report": <float>,
  "R_max": <float>,
  "wavelength_at_R_max_um": <float>
}
```

- `R_at_report`, `T_at_report`: reflectance and transmittance at the sweep point `report_point_index` (0-based; its wavelength is `report_wavelength_um`).
- `R_max`: the largest reflectance over the whole sweep; `wavelength_at_R_max_um`: the sweep wavelength (µm) where it occurs.

Copy each value exactly as the simulator returns it (full precision); do not round, interpolate or recompute them.

## Rules

- The RCWA solver (S4) is available **only** through the MCP server attached to this session. Do not install, import or run S4 or any other electromagnetic solver yourself (no `pip`/`uv` installs, no RCWA/FMM/TMM code of your own, no Python interpreter, library or source file belonging to the MCP server), and do not estimate values from memory, formulas or by hand.
- Simulate exactly the structure, illumination, number of harmonics and wavelength sweep given in `data/grating.json` (all lengths in µm), in one simulation over the full sweep.
- Do not modify `data/grating.json`.
