# Reflection spectrum of a dielectric grating

Compute the reflectance and transmittance spectra of the grating described in `data/grating.json` (geometry, materials, illumination, number of Fourier harmonics, wavelength sweep, point to report) with rigorous coupled-wave analysis. Use the RCWA MCP tools available in this session.

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
