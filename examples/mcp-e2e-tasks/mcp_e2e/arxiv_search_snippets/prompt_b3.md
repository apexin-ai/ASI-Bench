# Literature search with full-text evidence

Search arXiv as specified in `data/search.json` (query with field prefixes, submission-date window, sort order and result limit), pick one paper from the results with the file's `selection_rule`, and look up the file's `terms` in that paper's full text (with the given `window_chars` and `max_snippets_per_term`), counting the text snippets found for each term. Use the literature MCP tools available in this session for both the search and the full-text lookup.

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
