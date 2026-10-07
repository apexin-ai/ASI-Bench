# Isoform lengths and missense sensitivity of a protein

`data/profile.json` names one human protein ({{topic}}) by UniProt `accession`, lists the AlphaFold DB entries of its isoform models to rank (`entry_ids`) and gives an AlphaMissense score threshold (`am_threshold`).

Use the AlphaFold DB MCP tools available in this session to rank the listed isoform models by sequence length, read the canonical sequence length from the accession's summary, and profile the per-residue AlphaMissense scores of the canonical model: the residue with the highest score, and how many residues score above the threshold.

Background: AlphaFold DB names a model after the UniProt sequence it predicts, `AF-<accession>-F1` for the canonical isoform and `AF-<accession>-<n>-F1` for isoform `<accession>-<n>`; alternative splicing can make an isoform longer than the canonical sequence. AlphaMissense gives every possible single amino-acid substitution a pathogenicity score between 0 and 1; averaging the 19 substitutions at a position gives a per-residue score, which AlphaFold DB serves as an annotation of the canonical model, one region per residue numbered from 1.

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
