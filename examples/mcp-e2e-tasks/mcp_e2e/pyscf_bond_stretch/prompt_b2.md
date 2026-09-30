# Bond-stretch scan and plot via the pyscf MCP server

`data/scan.json` describes a rigid bond-stretch scan of {{molecule}} (bond {{bond}}): the molecule's SMILES, the 0-based indices of the two bonded atoms (`atom1_idx`, `atom2_idx`; RDKit order, i.e. heavy atoms in SMILES order followed by the hydrogens, also listed in `atoms_in_order`), the first and last bond length in Angstrom (`start_dist`, `end_dist`) and the number of evenly spaced points (`num_points`).

Using the tools of the `pyscf` MCP server, run this rigid bond-stretch scan at the RHF/STO-3G level (only the second atom moves along the bond; all other atoms stay fixed), then plot energy versus bond length with the server's plotting tool, passing it the scan results unchanged.

## Output

Write `result.json` in the current working directory:

```json
{
  "molecule": "<molecule name>",
  "bond_lengths": [<float>, ...],
  "energies_hartree": [<float>, ...],
  "min_bond_length": <float>,
  "min_energy_hartree": <float>
}
```

- `bond_lengths` (Angstrom) and `energies_hartree` (Hartree) list every scan point in order, with full precision (at least 10 significant digits), exactly as returned by the tool.
- `min_bond_length` and `min_energy_hartree` are the scan point with the lowest energy.

## Rules

- PySCF is available **only** through the MCP server attached to this session. Do not install, import or run PySCF, RDKit or any other chemistry package yourself (no `pip`/`uv`/`conda` installs, no `import pyscf`), and do not estimate energies from memory or by hand.
- Create the plot with the server's plotting tool, not locally (no matplotlib or other plotting code).
- Do not modify `data/scan.json`.
