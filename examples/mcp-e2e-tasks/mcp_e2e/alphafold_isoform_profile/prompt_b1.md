# AlphaFold DB isoform profile via the alphafold_db MCP server

`data/profile.json` names one human protein ({{topic}}): its UniProt `accession`, the AlphaFold DB entry ids of its isoform models to rank (`entry_ids`) and an AlphaMissense score threshold (`am_threshold`).

## Steps

1. Read `data/profile.json`.
2. Call the `alphafold_get_prediction` tool of the `alphafold_db` MCP server with `qualifier` = `accession`. It returns one model per isoform; each carries `entryId`, `uniprotAccession` and `uniprotSequence`. Rank the entries listed in `entry_ids` by the number of residues in their `uniprotSequence`, longest first.
3. Call the `alphafold_get_summary` tool of the same server with the same `qualifier`. Its `uniprot_entry.sequence_length` is the length of the canonical sequence.
4. Call the `alphafold_get_annotations` tool of the same server with the same `qualifier`. Its `annotation` list holds a block of `type` `MUTAGEN` whose `regions` cover every residue of the canonical sequence: `start` (equal to `end`) is the 1-based residue position and `annotation_value` the mean AlphaMissense pathogenicity of substituting that residue, written as a string.
5. Write `result.json` as described below.

## Output

Write `result.json` in the current working directory:

```json
{
  "entry_ids_by_length_desc": ["<entryId>", ...],
  "longest_entry_id": "<entryId>",
  "canonical_length": <integer>,
  "max_am_residue": <integer>,
  "max_am_score": <number>,
  "n_am_above_threshold": <integer>
}
```

- `entry_ids_by_length_desc`: every entry id listed in `entry_ids`, longest model first. The length of an entry is the number of residues in its `uniprotSequence`; the listed lengths all differ. Leave out models that `entry_ids` does not list.
- `longest_entry_id`: the first entry of that ranking (it need not be the canonical `AF-<accession>-F1`).
- `canonical_length`: the length of the canonical sequence as the summary reports it (`uniprot_entry.sequence_length`).
- `max_am_residue`: the 1-based residue position with the highest AlphaMissense score in the canonical model's annotation (the highest score is unique).
- `max_am_score`: that score, as the number the tool reported.
- `n_am_above_threshold`: the number of residues whose score is strictly greater than `am_threshold`.

## Rules

- AlphaFold DB is available **only** through the MCP server attached to this session. Do not use web search or web fetch tools, `curl`/`wget`, Python HTTP libraries or any other network access; do not download model files or the AlphaMissense data, and do not install or import UniProt or AlphaFold clients (no `pip`/`uv` installs).
- Take every entry id, length and score from the tool results; do not answer from memory or from what you know about this protein.
- Do not modify `data/profile.json`.
