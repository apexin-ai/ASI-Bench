# arXiv search and full-text snippets via the arxiv MCP server

`data/search.json` describes a literature search on {{topic}}: an arXiv query (`query`, with field prefixes such as `ti:` and `cat:`), a submission-date window (`date_from`, `date_to`), a sort order (`sort_by`, `sort_order`), a result limit (`limit`), a rule for picking one paper from the results (`selection_rule`), and the terms to look up in that paper's full text (`terms`, with `window_chars` and `max_snippets_per_term`).

## Steps

1. Read `data/search.json`.
2. Call the `ArXiv_search_papers` tool provided by the `arxiv` MCP server with `query`, `date_from`, `date_to`, `sort_by`, `sort_order` and `limit` taken **verbatim** from the file. It returns a list of papers with `title`, `authors`, `published` and `url` (`https://arxiv.org/abs/<id>v<n>`).
3. Apply `selection_rule` to the returned list to pick one paper.
4. Call the `ArXiv_get_pdf_snippets` tool of the same server once, with `arxiv_id` = the picked paper's ID and `terms`, `window_chars` and `max_snippets_per_term` taken verbatim from the file. It returns JSON with a `snippets` list; every snippet carries the `term` it matched.
5. Count the snippets per term and write `result.json` as described below.

## Output

Write `result.json` in the current working directory:

```json
{
  "arxiv_ids": ["<id>", ...],
  "selected_id": "<id>",
  "snippet_counts": {"<term>": <int>, ...}
}
```

- `arxiv_ids`: the arXiv IDs of **all** papers the search returned, in the order returned, without version suffix (e.g. `1103.0212`, not `1103.0212v1` or a URL).
- `selected_id`: the ID of the paper picked by the selection rule, in the same form.
- `snippet_counts`: for every term in `terms`, the number of snippets the snippet tool returned for that term (0 if none).

## Rules

- arXiv is available **only** through the MCP server attached to this session. Do not use web search or web fetch tools, `curl`/`wget`, Python HTTP libraries or any other network access, and do not install or import arXiv clients or PDF libraries (no `pip`/`uv` installs, no `import arxiv`).
- Take every ID and count from the tool results; do not answer from memory or by estimation.
- Do not modify `data/search.json`.
