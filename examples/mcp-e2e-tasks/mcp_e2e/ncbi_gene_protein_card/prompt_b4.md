# Gene and protein identifier card

`data/card.json` names one human gene ({{topic}}) with the Entrez Gene query to run (`term`, which matches exactly one gene) and the GI number of one of its RefSeq proteins (`protein_gi`).

Use the NCBI MCP tools available in this session to find the gene, read its record, and read the protein record of that GI number, then report the identifier card below: the gene's id, official symbol, cytogenetic band, chromosome accession, organism taxonomy id, and the protein's accession with version and its residue count.

Background: an Entrez query combines fielded terms with `AND`, so `SYMBOL[Symbol] AND Homo sapiens[Organism]` restricts a symbol to one organism and normally matches a single gene. A search returns uids only; the descriptive record comes from a second, summary call, which keys its records by the uid that was requested. A gene record locates the gene on a reference-assembly sequence (`NC_...` accessions) and separately gives the cytogenetic band, the classical chromosome-arm notation such as `7q31.2`. A protein is addressable either by GI number, which is bound to one specific sequence version, or by an accession whose version suffix changes when the sequence is revised (`NP_` marks a RefSeq protein); its summary reports the length in amino-acid residues.

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
