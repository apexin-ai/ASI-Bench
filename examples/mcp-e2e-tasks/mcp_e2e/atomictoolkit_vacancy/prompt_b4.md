# Vacancy in an fcc metal

Carry out the vacancy calculation specified in `data/calculation.json`, using the atomistic modelling tools available in this session, and report the result. The file gives every parameter the calculation must use (element, lattice constant, supercell size, the atom to remove, the strain range and the calculator); take them from it rather than choosing your own, and write the structure files and the analysis under the names it gives, as absolute paths inside the current working directory.

Background: the formation energy of a vacancy compares a supercell with one atom removed to the perfect crystal at the same number of atoms, `E_vac = E_vacancy − E_perfect × (N − 1) / N` for a perfect supercell of N atoms; unrelaxed, the remaining atoms stay at their ideal sites. A bulk modulus can be estimated from the curvature of the energy under small uniform strains of the cell, and the neighbours an analysis counts within a cutoff give the coordination number.

## Output

Write `result.json` in the current working directory:

```json
{
  "n_atoms_perfect": <int>,
  "energy_perfect_eV": <float>,
  "energy_vacancy_eV": <float>,
  "vacancy_formation_energy_eV": <float>,
  "coordination_average": <float>,
  "bulk_modulus_GPa": <float>
}
```

- `n_atoms_perfect`: the number of atoms N in the perfect supercell, as the supercell step reports it.
- `energy_perfect_eV`, `energy_vacancy_eV`: the EMT energies (eV) of the perfect and the defective supercell, as the single-point tool reports them.
- `vacancy_formation_energy_eV`: `E_vac = E_vacancy − E_perfect × (N − 1) / N`, computed from those two values and N.
- `coordination_average`: the average coordination number of the defective supercell, as the structure analysis reports it.
- `bulk_modulus_GPa`: the unit cell's bulk modulus over the given strain range, as the elastic tool reports it.

Copy each value exactly as the tools return it (full precision); do not round, convert or recompute them, and compute `vacancy_formation_energy_eV` in full precision too.

## Rules

- The structure tools and the EMT potential are available **only** through the MCP server attached to this session. Do not install, import or run ASE, pymatgen, ASAP, LAMMPS or any other atomistic package yourself (no `pip`/`uv`/`conda` installs, no `import ase`, no Python interpreter or files belonging to an MCP server), and do not estimate values from memory or by hand.
- The structure is **unrelaxed**: the atoms stay at their ideal lattice sites. Do not optimise or run molecular dynamics; this server's relaxation, MD and trajectory tools cannot return results in this setup.
- The structure-editing tool takes its parameters in `operation_kwargs`: `size` for a supercell and `index` for a vacancy. It silently ignores keys it does not know, so use exactly these.
- Always pass `calculator_name` = `emt`; the server's default (`auto`) tries other calculators first.
- Drive the calculation one tool call at a time, each step on the file the previous one wrote.
- Use exactly the parameters in `data/calculation.json`. Do not modify `data/calculation.json`.
