# Bond-stretch scan and plot via the pyscf MCP server

`data/scan.json` describes a rigid bond-stretch scan of {{molecule}} (bond {{bond}}): the molecule's SMILES, the 0-based indices of the two bonded atoms (`atom1_idx`, `atom2_idx`; RDKit order, i.e. heavy atoms in SMILES order followed by the hydrogens, also listed in `atoms_in_order`), the first and last bond length in Angstrom (`start_dist`, `end_dist`) and the number of evenly spaced points (`num_points`).

## Steps

1. Read `data/scan.json`.
2. Call the `run_bond_stretch_calculation_mcp` tool provided by the `pyscf` MCP server with `smiles_string`, `atom1_idx`, `atom2_idx`, `start_dist`, `end_dist` and `num_points` taken **verbatim** from the file, and `basis` = `"sto-3g"`. It returns JSON with `bond_lengths` and `energies` (RHF/STO-3G, Hartree).
3. Call the `plot_energy_scan_image_mcp` tool of the same server with the `bond_lengths` and `energies` returned in step 2, unchanged. It returns a PNG image of the energy curve.
4. Write `result.json` as described below.

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
