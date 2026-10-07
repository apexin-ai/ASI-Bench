# NCBI gene and protein card via the ncbi MCP server

`data/card.json` describes one human gene ({{topic}}): the official `symbol`, the `organism`, the Entrez Gene query to run (`term`) and the GI number of one of its RefSeq proteins (`protein_gi`).

Using the tools of the `ncbi` MCP server, run the gene query exactly as given in the file (it resolves to a single gene id), read the gene record of that id, and read the protein record of the GI number in the file. Report the identifiers below; the gene record carries the official symbol, the cytogenetic band, one genomic record with the chromosome accession and the organism's taxonomy id, and the protein record carries the accession with its version and the sequence length.

## Output

Write `result.json` in the current working directory:

```json
{
  "gene_id": "<digits>",
  "symbol": "<official gene symbol>",
  "maplocation": "<cytogenetic band>",
  "chr_accession": "<chromosome accession.version>",
  "protein_accver": "<protein accession.version>",
  "protein_len": <integer>,
  "taxid": <integer>
}
```

- `gene_id`: the single Entrez Gene id the query resolved to, as a string of digits.
- `symbol`: the official symbol the gene record reports (its `nomenclaturesymbol`).
- `maplocation`: the cytogenetic band of the gene record, exactly as reported (e.g. `17p13.1`).
- `chr_accession`: the `chraccver` of the gene's single genomic record (e.g. `NC_000017.11`).
- `protein_accver`: the protein record's `accessionversion`, **with** the version suffix (e.g. `NP_000537.3`).
- `protein_len`: the protein's length in amino-acid residues (`slen`), as an integer.
- `taxid`: the taxonomy id of the gene's organism (`organism.taxid`), as an integer.

## Rules

- NCBI is available **only** through the MCP server attached to this session. Do not use web search or web fetch tools, `curl`/`wget`, Python HTTP libraries or any other network access, and do not install or import NCBI clients (no `pip`/`uv` installs, no Biopython `Bio.Entrez`, no `entrez-direct`, no NCBI `datasets` command).
- Take every identifier and number from the tool results; do not answer from memory or from what you know about this gene.
- Do not modify `data/card.json`.
