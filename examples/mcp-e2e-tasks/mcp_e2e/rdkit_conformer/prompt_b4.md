# Seeded 3D conformer and Gasteiger charges

Embed the molecule in `data/molecule.json` in 3D with ETKDG and the random seed given there, have the server save the conformer as `conformer.sdf` in the current working directory, and report its Gasteiger partial-charge extremes. Use the cheminformatics MCP tools available in this session.

Background: ETKDG distance-geometry embedding is stochastic; with a fixed random seed it is reproducible, and different seeds give different coordinates. Gasteiger charges are empirical partial atomic charges obtained by iterative partial equalisation of orbital electronegativity.

## Output

Two files must end up in the current working directory:

- `conformer.sdf`: the 3D conformer, written by the MCP server's SDF export (not by you).
- `result.json`, written by you:

```json
{
  "molecule": "<molecule name>",
  "conf_id": <int>,
  "max_partial_charge": <float>,
  "min_partial_charge": <float>
}
```

- `conf_id`: the conformer id returned by the embedding.
- `max_partial_charge` / `min_partial_charge`: the most positive and the most negative Gasteiger partial charge of the molecule (elementary charges).

Copy each value exactly as the tools return it (full precision); do not round, convert or compute them.

## Rules

- RDKit is available **only** through the MCP server attached to this session. Do not install, import or run RDKit or any other cheminformatics package yourself (no `pip`/`uv`/`conda` installs, no `import rdkit`, no Open Babel, no Python interpreter or files belonging to an MCP server), and do not estimate values from memory or by hand.
- Do not write, edit or convert `conformer.sdf` yourself; it must be the file the server writes.
- Molecules are passed between the tools as long encoded strings: pass each one on **verbatim**, never shortened or retyped.
- Embed exactly the molecule given by the SMILES (no added hydrogens) with the given random seed.
- Do not modify `data/molecule.json`.
