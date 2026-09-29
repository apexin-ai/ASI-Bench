# Hartree–Fock energy

Compute the restricted Hartree–Fock total energy of the molecule described in `data/molecule.json`, using exactly the geometry and basis set given there. Use the quantum-chemistry MCP tools available in this session.

## Output

Write `result.json` in the current working directory:

```json
{"energy_hartree": <float>, "basis": "<basis>", "molecule": "<molecule name>"}
```

`energy_hartree` must be the RHF total energy in Hartree with full precision (at least 10 significant digits), exactly as returned by the tool.

## Rules

- PySCF is available **only** through the MCP server attached to this session. Do not install, import or run PySCF or any other quantum-chemistry package yourself (no `pip`/`uv`/`conda` installs, no `import pyscf`), and do not estimate the energy from memory or by hand.
- Do not modify `data/molecule.json`.
