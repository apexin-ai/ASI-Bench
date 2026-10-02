# Geometry optimisation and harmonic frequencies

Find the equilibrium structure of the molecule in `data/molecule.json` (starting geometry, charge, multiplicity, method, basis set) and report its energy, harmonic vibrational frequencies and zero-point energy. Use the quantum-chemistry MCP tools available in this session.

Background: harmonic vibrational frequencies are only meaningful at a stationary point of the potential-energy surface computed with the same method and basis set; at a minimum all of them are real. The zero-point vibrational energy is ½ Σ hν over the vibrational modes. Total energies are in Hartree (1 Hartree ≈ 627.5095 kcal/mol).

## Output

Write `result.json` in the current working directory:

```json
{
  "molecule": "<molecule name>",
  "final_energy_hartree": <float>,
  "frequencies_cm_inv": [<float>, ...],
  "zpe_hartree": <float>
}
```

- `final_energy_hartree`: total energy at the optimised geometry (Hartree).
- `frequencies_cm_inv`: every harmonic vibrational frequency at the optimised geometry (cm⁻¹), in the order the calculation returns them (ascending).
- `zpe_hartree`: the zero-point vibrational energy (Hartree).

Copy each value exactly as the calculation returns it (full precision); do not round, convert or compute them.

## Rules

- psi4 is available **only** through the MCP server attached to this session. Do not install, import or run psi4, PySCF or any other quantum-chemistry package yourself (no `pip`/`uv`/`conda` installs, no `import psi4`, no `psi4` command, no Python interpreter or files belonging to the MCP server), and do not estimate values from memory or by hand.
- Use exactly the `method` and `basis` from `data/molecule.json` for every calculation, with the given charge and multiplicity, and keep the optimiser's default (tight) convergence.
- Compute the frequencies at the optimised geometry, not at the starting geometry.
- Do not modify `data/molecule.json`.
