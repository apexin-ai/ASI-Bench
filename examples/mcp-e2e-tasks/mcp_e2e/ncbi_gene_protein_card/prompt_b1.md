# NCBI gene and protein card via the ncbi MCP server

`data/card.json` describes one human gene ({{topic}}): the official `symbol`, the `organism`, the Entrez Gene query to run (`term`) and the GI number of one of its RefSeq proteins (`protein_gi`).

## Steps

1. Read `data/card.json`.
2. Call the `NCBIGene_search` tool of the `ncbi` MCP server with `term` taken **verbatim** from the file. It returns an E-utilities esearch payload; `esearchresult.idlist` holds the matching gene ids and the query is written so that exactly one gene matches.
3. Call the `NCBIGene_get_summary` tool of the same server with `id` = that single gene id. It returns an esummary payload in which `result.<id>` is the gene record, carrying `nomenclaturesymbol`, `maplocation`, a `genomicinfo` list with exactly one entry (whose `chraccver` is the chromosome accession) and `organism.taxid`.
4. Call the `NCBIProtein_get_summary` tool of the same server with `id` = `protein_gi` from the file. Its record carries `accessionversion` and `slen`.
5. Write `result.json` as described below.

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
