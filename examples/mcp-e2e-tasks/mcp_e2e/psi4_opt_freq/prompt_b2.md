# Geometry optimisation and harmonic frequencies via the psi4 MCP server

`data/molecule.json` describes a small closed-shell molecule ({{molecule}}): a distorted starting geometry `geometry_xyz` (one `Element x y z` line per atom, Angstrom), `charge`, `multiplicity`, the electronic-structure `method` ({{method}}, restricted Hartree–Fock) and the `basis` set ({{basis}}).

Using the tools of the `psi4` MCP server, optimise the geometry at the given level of theory, then compute the harmonic vibrational frequencies and the zero-point energy at the optimised geometry.

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
