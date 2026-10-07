# Silicon band structure via the quantum_espresso MCP server

`data/calculation.json` describes one Quantum ESPRESSO band-structure calculation on bulk silicon (the server's built-in structure `{{structure}}`): wavefunction cutoff `ecutwfc` = {{ecutwfc}} Ry, k-point spacing `kspacing` = {{kspacing}} 1/Å for the SCF grid, and `npoints_band` = {{npoints_band}} points along the band path.

Using the tools of the `quantum_espresso` MCP server: look up which pseudopotential file the server uses for silicon (`qe_list_pseudopotentials`), get the SCF k-point grid it suggests for this spacing (`qe_suggest_kpoints`), run the band-structure workflow on that grid (`qe_workflow_bandstructure`), list the files of that run (`qe_list_files`) and read the run's band file (`qe_read_bands`).

Each step takes what the previous one returned: the suggested grid is the workflow's `kpoints` argument (one string of three comma-separated integers), the workflow's output directory is what gets listed, and the band file the listing returns is what gets read.

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
- Use the tools named above, in that order, one call at a time. Read the band file through the `qe_read_bands` tool, not with shell commands.
- Use exactly the parameters in `data/calculation.json`, and the k-point grid the server suggests for that spacing.
- Do not modify `data/calculation.json`.
