# Bond-stretch energy curve

Run the rigid bond-stretch scan described in `data/scan.json` (RHF/STO-3G; only the second atom of the bond moves, along the bond direction, all other atoms stay fixed) and plot the resulting energy curve. `atom1_idx`/`atom2_idx` are 0-based indices in RDKit atom order for the given SMILES (listed in `atoms_in_order`). Use the quantum-chemistry MCP tools available in this session for both the scan and the plot.

Background: a rigid (unrelaxed) bond scan changes one internuclear distance while freezing every other coordinate, so the curve is an upper bound to the relaxed potential-energy curve. Near equilibrium the energy is roughly quadratic in the bond length; restricted Hartree–Fock rises too steeply at long distances because a single closed-shell determinant cannot describe bond dissociation. Energies are in Hartree (1 Hartree ≈ 627.5095 kcal/mol).

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
