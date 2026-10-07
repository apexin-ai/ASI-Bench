# Silicon band structure

Carry out the band-structure calculation on bulk silicon specified in `data/calculation.json`, using the electronic-structure modelling tools available in this session, and report the result. The file gives every parameter the calculation must use; take them from it rather than choosing your own, and use the k-point grid the tools suggest for its spacing.

Background: a band structure is computed in two passes — a self-consistent (SCF) calculation on a uniform k-point grid that fixes the charge density, then a non-self-consistent pass along a path of high-symmetry lines whose eigenvalues are the bands — and the band edges are read off those eigenvalues relative to the SCF Fermi level. The SCF grid is chosen from a k-point spacing, and a plane-wave calculation needs a pseudopotential per element, which the calculation's numbers depend on. The run leaves its files in a directory of its own, among them the band file a plotting step would read: the distance along the path, and the energy of every band at every point.

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
- Drive the calculation one tool call at a time, each step using what the previous one returned. Read the band file through the server's band-file reading tool, not with shell commands.
- Use exactly the parameters in `data/calculation.json`, and the k-point grid the server suggests for that spacing.
- Do not modify `data/calculation.json`.
