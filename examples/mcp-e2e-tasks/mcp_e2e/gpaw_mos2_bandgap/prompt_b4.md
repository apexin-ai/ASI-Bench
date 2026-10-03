# MoS2 monolayer: relaxation, convergence gate and band structure

Carry out the plane-wave DFT study specified in `data/calculation.json` on a {{structure}}, using the electronic-structure modelling tools available in this session, and report the result. Everything belongs to a single run: the first call opens one and hands back its identifier, which every later call needs. The file gives every parameter the calculation must use; take them from it rather than choosing your own.

Background: a plane-wave calculation is only as good as its cutoff and its k-point sampling, so the usual discipline is to sweep both, take the recommendation that falls inside a chosen energy tolerance, and only then trust a band structure computed at or above those settings — a ground state computed before that sweep is recorded as unverified. The sweep also recomputes the ground state several times in the run's own directory, so running it after the band calculation would leave the run's restart file describing a different calculation than the figures do. The verification step finally compares the computed gap with a reference value at the tolerance you give it, and reports a verdict over all its checks.

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
- Run the convergence check before the band-structure calculation.
- Use exactly the parameters in `data/calculation.json` for every call.
- Do not write, edit, convert, regenerate or re-plot {{copied_files}}; they must be the files the server wrote, copied unchanged.
- Do not modify `data/calculation.json`.
