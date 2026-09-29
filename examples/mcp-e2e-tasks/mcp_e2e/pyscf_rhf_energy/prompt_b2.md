# RHF total energy via the pyscf MCP server

`data/molecule.json` describes a closed-shell, neutral molecule ({{molecule}}) with its geometry (Angstrom, PySCF `atom` format) and basis set ({{basis}}).

Compute its restricted Hartree–Fock (RHF) total energy with the tools of the `pyscf` MCP server, using the geometry and basis exactly as given.

## Output

Write `result.json` in the current working directory:

```json
{"energy_hartree": <float>, "basis": "<basis>", "molecule": "<molecule name>"}
```

`energy_hartree` must be the RHF total energy in Hartree with full precision (at least 10 significant digits), exactly as returned by the tool.

## Rules

- PySCF is available **only** through the MCP server attached to this session. Do not install, import or run PySCF or any other quantum-chemistry package yourself (no `pip`/`uv`/`conda` installs, no `import pyscf`), and do not estimate the energy from memory or by hand.
- Do not modify `data/molecule.json`.
