# RHF total energy via the pyscf MCP server

`data/molecule.json` describes a closed-shell, neutral molecule ({{molecule}}): its Cartesian geometry in Angstrom as a PySCF `atom` string, and the basis set ({{basis}}).

## Steps

1. Read `data/molecule.json`.
2. Call the MCP tool `pyscf_rhf_energy` of the `pyscf` MCP server (in Claude Code it is named `mcp__pyscf__pyscf_rhf_energy`) with:
   - `atom`: the `atom` string from the file, passed **verbatim** (do not round, reorder or reformat coordinates);
   - `basis`: the `basis` string from the file.
3. The tool returns the RHF total energy in Hartree as text. Write it to `result.json` as described below.

## Output

Write `result.json` in the current working directory:

```json
{"energy_hartree": <float>, "basis": "<basis>", "molecule": "<molecule name>"}
```

`energy_hartree` must be the RHF total energy in Hartree with full precision (at least 10 significant digits), exactly as returned by the tool.

## Rules

- PySCF is available **only** through the MCP server attached to this session. Do not install, import or run PySCF or any other quantum-chemistry package yourself (no `pip`/`uv`/`conda` installs, no `import pyscf`), and do not estimate the energy from memory or by hand.
- Do not modify `data/molecule.json`.
