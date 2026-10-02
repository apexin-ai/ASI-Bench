# Reflection spectrum of a dielectric grating

`data/grating.json` describes a one-dimensional dielectric grating on a substrate: the period (`period_um`), the medium the light comes from (`superstrate`), a grating layer of thickness `grating.thickness_um` made of {{ridge}} ridges of width `grating.ridge_width_um` separated by grooves of `grating.groove_material` (the ridges run along y, are uniform along y and repeat with the period along x, one ridge centred at x = 0), and the `substrate` below. Every material is given by its refractive index `n`. A {{polarization}}-polarised plane wave is incident from the superstrate at `illumination.theta_deg` from the normal, in the xz plane; the simulation uses `n_harmonics` Fourier harmonics and sweeps the vacuum wavelength linearly over `sweep` (`wavelength_start_um` to `wavelength_stop_um` inclusive, `wavelength_points` points).

Compute the reflectance and transmittance spectra of this grating with rigorous coupled-wave analysis. Use the RCWA MCP tools available in this session.

Background: RCWA (the Fourier modal method) treats structures that are periodic in the lateral plane and layered along z: each layer is uniform or patterned in-plane, the fields are expanded in a finite number of Fourier harmonics, and the result depends on that number, so use exactly the given one. A one-dimensional grating can be described on a two-dimensional lattice as a rectangle that spans the whole cell in the direction along the ridges. TM polarisation means the electric field lies in the plane of incidence (xz).

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
