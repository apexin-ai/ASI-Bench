# MoS2 monolayer: relax, gate the parameters and compute the band structure via the gpaw MCP server

`data/calculation.json` describes one plane-wave DFT calculation on a {{structure}}: cutoff `ecut` = {{ecut}} eV, k-point density `kpts_density` = {{kpts_density}}, convergence tolerance `tol_mev_per_atom` = {{tol_mev_per_atom}} meV/atom and band-gap tolerance `gap_tol_ev` = {{gap_tol_ev}} eV.

Every call below goes to the `gpaw` MCP server and uses the **same** `run_id` — the one the first call returns. The order matters.

## Steps

1. Read `data/calculation.json`.
2. Call the `fetch_structure` tool with `query` = `{{query}}` and `use_builtin` = `true`. Keep the `run_id` it returns; every later call takes it as `run_id`.
3. Call the `relax_structure` tool with that `run_id`, `ecut` = {{ecut}}, `kpts_density` = {{kpts_density}}, `fmax` = {{fmax}}, `max_steps` = {{max_steps}} and `engine` = `{{engine}}`. Take `total_energy_ev`, `max_force_ev_per_a`, `n_steps` and `kpts` from its answer.
4. Call the `check_convergence` tool with that `run_id`, `tol_mev_per_atom` = {{tol_mev_per_atom}} and `engine` = `{{engine}}`. Take `recommended_ecut_ev`, `recommended_kpts_density` and `converged` from its answer. This step must come **before** step 5: it is what lets the ground state be marked as verified, and it also rewrites the run's shared restart file.
5. Call the `calc_band_dos` tool with that `run_id`, `ecut` = {{ecut}}, `kpts_density` = {{kpts_density}}, `npoints` = {{npoints}}, `window_ev` = {{window_ev}} and `engine` = `{{engine}}`. Take `total_energy_ev`, `fermi_ev`, `band_gap_ev`, `gap_type`, `params_verified` and the `label` of both `vbm` and `cbm` from its answer.
6. Call the `verify_run` tool with that `run_id` and `gap_tol_ev` = {{gap_tol_ev}}. Take `verdict` from its answer.
7. Call the `get_run_artifacts` tool with that `run_id`. It lists every file the run left, with a path relative to the server's own directory. Count them, and use the listing to locate the files you have to hand back.
8. Copy {{copied_files}} out of the run directory into the current working directory. The listing's paths are relative to the server's own working directory, not to yours, so locate the run directory on disk first (it is named after the `run_id`). Copy the files unchanged; do not edit, regenerate or re-plot them.
9. Write `result.json` as described below.

## Output

Three files must end up in the current working directory:

- {{copied_files}}: the two figures the MCP server wrote (not you), copied unchanged. The artefact listing reports paths relative to the **server's own working directory**, which is not this one, so resolve them to the real location on disk — the run directory is named after the `run_id` — and copy the files from there. These figures exist nowhere else: the tools return their paths, never their contents.
- `result.json`, written by you:

```json
{
  "relax_total_energy_ev": <float>,
  "relax_max_force_ev_per_a": <float>,
  "relax_n_steps": <int>,
  "relax_kpts": [<int>, <int>, <int>],
  "scf_total_energy_ev": <float>,
  "fermi_ev": <float>,
  "band_gap_ev": <float>,
  "gap_type": "<direct or indirect>",
  "vbm_label": "<string>",
  "cbm_label": "<string>",
  "params_verified": <true or false>,
  "recommended_ecut_ev": <int>,
  "recommended_kpts_density": <float>,
  "converged": <true or false>,
  "verdict": "<string>",
  "artifact_count": <int>
}
```

- `relax_total_energy_ev`, `relax_max_force_ev_per_a`, `relax_n_steps`, `relax_kpts`: the total energy (eV), the largest residual force (eV/Å), the number of optimiser steps and the realized Monkhorst–Pack grid, all from the relaxation.
- `scf_total_energy_ev`, `fermi_ev`, `band_gap_ev`: the total energy (eV), the Fermi level (eV) and the band gap (eV) of the ground state computed on the relaxed structure.
- `gap_type`: whether the gap is direct or indirect.
- `vbm_label`, `cbm_label`: the high-symmetry labels of the valence-band maximum and the conduction-band minimum, exactly as reported (they may be Greek letters).
- `params_verified`: whether the ground state was computed at parameters that cleared the convergence gate.
- `recommended_ecut_ev`, `recommended_kpts_density`, `converged`: the cutoff and k-point density the convergence sweep recommends at this tolerance, and whether the sweep converged.
- `verdict`: the overall verdict of the verification step.
- `artifact_count`: how many files the run left, as the artefact listing reports.

Copy each value exactly as the tools return it (full precision); do not round, convert or recompute them.

## Rules

- GPAW and ASE are available **only** through the MCP server attached to this session. Do not install, import or run GPAW, ASE, Quantum ESPRESSO, pymatgen or any other electronic-structure or materials package yourself (no `pip`/`uv`/`conda`/`micromamba` installs, no `import gpaw`, no Python interpreter or files belonging to an MCP server), and do not estimate values from memory or by hand.
- Drive the chain one tool call at a time on a single run. Do **not** use the `run_verified_workflow` tool: it performs the whole chain in one call, which is exactly what this task is not asking for.
- Run the convergence check before the band-structure calculation, as in the steps above.
- Use exactly the parameters in `data/calculation.json` for every call.
- Do not write, edit, convert, regenerate or re-plot {{copied_files}}; they must be the files the server wrote, copied unchanged.
- Do not modify `data/calculation.json`.
