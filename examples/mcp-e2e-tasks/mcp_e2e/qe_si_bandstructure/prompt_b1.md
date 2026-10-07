# Silicon band structure via the quantum_espresso MCP server

`data/calculation.json` describes one Quantum ESPRESSO band-structure calculation on bulk silicon (the server's built-in structure `{{structure}}`): wavefunction cutoff `ecutwfc` = {{ecutwfc}} Ry, k-point spacing `kspacing` = {{kspacing}} 1/Å for the SCF grid, and `npoints_band` = {{npoints_band}} points along the band path.

Every call below goes to the `quantum_espresso` MCP server. Each step uses what the previous one returned.

## Steps

1. Read `data/calculation.json`.
2. Call the `qe_list_pseudopotentials` tool. Take the `filename` of the `Si` entry under `details`: the pseudopotential file the server will use for silicon.
3. Call the `qe_suggest_kpoints` tool with `structure` = `{{structure}}` and `kspacing` = {{kspacing}}. Take the grid it returns as `kpoints` (three integers).
4. Call the `qe_workflow_bandstructure` tool with `structure` = `{{structure}}`, `ecutwfc` = {{ecutwfc}}, `npoints_band` = {{npoints_band}} and `kpoints` = the grid from step 3 written as one string of three comma-separated integers (for example `"5,5,5"`). Take `total_energy_eV`, `fermi_energy_eV`, `band_gap_eV`, `vbm_eV`, `cbm_eV`, `is_direct_gap`, `n_bands` and `output_dir` from its answer.
5. Call the `qe_list_files` tool with `output_dir` = the `output_dir` from step 4, exactly as returned. Its `band_files` entry lists the band file of the run (a path ending in `.gnu`).
6. Call the `qe_read_bands` tool with `output_dir` = that band-file path from step 5, exactly as listed. Despite the parameter's name, it needs the path of the **file**, not of the directory. Take `n_kpoints` and the largest value of `k_distances` (the length of the band path) from its answer.
7. Write `result.json` as described below.

## Output

Write `result.json` in the current working directory:

```json
{
  "si_pseudopotential": "<file name>",
  "scf_kpoints": [<int>, <int>, <int>],
  "total_energy_eV": <float>,
  "fermi_energy_eV": <float>,
  "band_gap_eV": <float>,
  "vbm_eV": <float>,
  "cbm_eV": <float>,
  "is_direct_gap": <true or false>,
  "n_bands": <int>,
  "n_kpoints": <int>,
  "path_length": <float>
}
```

- `si_pseudopotential`: the file name of the silicon pseudopotential, as the pseudopotential listing reports it.
- `scf_kpoints`: the SCF k-point grid suggested for this spacing, and used for the band-structure run.
- `total_energy_eV`, `fermi_energy_eV`: the SCF total energy and Fermi energy (eV) of the band-structure run.
- `band_gap_eV`, `vbm_eV`, `cbm_eV`, `is_direct_gap`, `n_bands`: the band gap, the valence-band maximum, the conduction-band minimum (all eV), whether the gap is direct, and the number of bands, as the band-structure run reports them.
- `n_kpoints`, `path_length`: the number of k-points in the band file and the largest k-distance in it, as the band-file reader reports them.

Copy each value exactly as the tools return it (full precision); do not round, convert or recompute them.

## Rules

- Quantum ESPRESSO is available **only** through the MCP server attached to this session. Do not install, import or run Quantum ESPRESSO (`pw.x`, `bands.x`), ASE, pymatgen, GPAW or any other electronic-structure or materials package yourself (no `pip`/`uv`/`conda`/`micromamba` installs, no Python interpreter or files belonging to an MCP server), and do not estimate values from memory or by hand.
- Use the tools named in the steps, in that order, one call at a time. Read the band file through the `qe_read_bands` tool, not with shell commands.
- Use exactly the parameters in `data/calculation.json`, and the k-point grid the server suggests for that spacing.
- Do not modify `data/calculation.json`.
