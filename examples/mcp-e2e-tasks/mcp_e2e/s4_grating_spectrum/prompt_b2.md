# Reflection spectrum of a dielectric grating via the s4 MCP server

`data/grating.json` describes a one-dimensional dielectric grating on a substrate: the period (`period_um`), the medium the light comes from (`superstrate`), a grating layer of thickness `grating.thickness_um` made of {{ridge}} ridges of width `grating.ridge_width_um` separated by grooves of `grating.groove_material` (the ridges run along y, are uniform along y and repeat with the period along x, one ridge centred at x = 0), and the `substrate` below. Every material is given by its refractive index `n`. A {{polarization}}-polarised plane wave is incident from the superstrate at `illumination.theta_deg` from the normal, in the xz plane; the simulation uses `n_harmonics` Fourier harmonics and sweeps the vacuum wavelength linearly over `sweep` (`wavelength_start_um` to `wavelength_stop_um` inclusive, `wavelength_points` points).

Using the `simulate_stack_spectrum` tool of the `s4` MCP server, simulate this grating over the sweep (without a plot) and read the reflectance and transmittance spectra.

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
