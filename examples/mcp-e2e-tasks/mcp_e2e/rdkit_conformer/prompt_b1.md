# Seeded 3D conformer and Gasteiger charges via the rdkit MCP server

`data/molecule.json` describes a small drug-like molecule ({{molecule}}): its `smiles` (`{{smiles}}`), the `random_seed` for the conformer embedding ({{random_seed}}) and the `output_file` name (`conformer.sdf`).

## Steps

1. Read `data/molecule.json`.
2. Call the `smiles_to_mol` tool provided by the `rdkit` MCP server with `smiles` copied verbatim from the file. It returns the molecule as an encoded string.
3. Call the `EmbedMolecule` tool of the same server with `p_mol` = the string returned in step 2 (verbatim) and `params` = `{"randomSeed": <random_seed from the file>}` (leave every other parameter at its default). Take `conf_id` and the embedded molecule string `mol` from its answer.
4. Call the `mol_to_sdf` tool of the same server with `pmol` = the `mol` string from step 3 (verbatim), `file_dir` = the absolute path of the current working directory (for example the output of `pwd`) and `filename` = `conformer.sdf`. The server writes `conformer.sdf` there.
5. Call the `MaxPartialCharge` and `MinPartialCharge` tools of the same server, each with `smiles` copied verbatim from the file.
6. Write `result.json` as described below.

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
