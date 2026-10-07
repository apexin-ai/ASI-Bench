# {{element}} vacancy via the atomictoolkit MCP server

`data/calculation.json` describes one unrelaxed vacancy calculation on fcc {{element}} with ASE's EMT potential: lattice constant `lattice_constant` = {{lattice_constant}} Å, a {{supercell}}×{{supercell}}×{{supercell}} supercell of the conventional cubic cell, the atom with index `vacancy_index` = {{vacancy_index}} removed, and a bulk-modulus fit over strains up to `strain_max` = {{strain_max}}.

Every call below goes to the `atomictoolkit` MCP server. Each step uses the file the previous one wrote. Pass every file path as an **absolute path inside the current working directory**, with the file names given in `data/calculation.json` (`{{unit_cell_file}}`, `{{supercell_file}}`, `{{vacancy_file}}`, and the directory `{{analysis_dir}}`).

## Steps

1. Read `data/calculation.json`.
2. Call the `build_structure_workflow` tool with `formula` = `{{element}}`, `crystal_system` = `fcc`, `lattice_constant` = {{lattice_constant}} and `output_filepath` = the absolute path of `{{unit_cell_file}}`. It writes the 4-atom conventional cubic cell; take the `filepath` it returns.
3. Call the `manipulate_structure_workflow` tool with `input_filepath` = the `filepath` from step 2, `operation` = `supercell`, `operation_kwargs` = `{"size": [{{supercell}}, {{supercell}}, {{supercell}}]}` and `output_filepath` = the absolute path of `{{supercell_file}}`. Take its `filepath` and `num_atoms` (the atom count N of the perfect supercell).
4. Call `manipulate_structure_workflow` again with `input_filepath` = the `filepath` from step 3, `operation` = `vacancy`, `operation_kwargs` = `{"index": {{vacancy_index}}}` and `output_filepath` = the absolute path of `{{vacancy_file}}`. Take its `filepath`.
5. Call the `single_point_workflow` tool with `input_filepath` = the `filepath` from step 3 and `calculator_name` = `emt`. Its `energy` (eV) is the perfect supercell's energy.
6. Call `single_point_workflow` with `input_filepath` = the `filepath` from step 4 and `calculator_name` = `emt`. Its `energy` (eV) is the defective supercell's energy.
7. Call the `analyze_structure_workflow` tool with `filepath` = the `filepath` from step 4 and `output_dir` = the absolute path of `{{analysis_dir}}`. Take `analysis.summary.coordination.average` from its answer.
8. Call the `estimate_elastic_workflow` tool with `input_filepath` = the `filepath` from step 2, `calculator_name` = `emt` and `strain_max` = {{strain_max}}. Take its `bulk_modulus_GPa`.
9. Compute the unrelaxed vacancy formation energy `E_vac = E_vacancy − E_perfect × (N − 1) / N` from the two energies and N, and write `result.json` as described below.

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
- `manipulate_structure_workflow` takes its parameters in `operation_kwargs`: `size` for a supercell and `index` for a vacancy. It silently ignores keys it does not know, so use exactly these.
- Always pass `calculator_name` = `emt`; the server's default (`auto`) tries other calculators first.
- Use the tools named in the steps, one call at a time, each on the file the previous one wrote.
- Use exactly the parameters in `data/calculation.json`. Do not modify `data/calculation.json`.
