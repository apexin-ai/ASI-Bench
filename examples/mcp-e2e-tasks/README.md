# MCP end-to-end (E2E) fake tasks

Fake tasks whose only purpose is to prove that an agent, running inside the
normal `asibench run` → `asibench score` pipeline, **actually calls a specific
MCP tool and gets a correct result**. They are not benchmark tasks: they live
outside `tasks/`, have `status: test`, and are only discovered with
`--tasks-dir examples/mcp-e2e-tasks --include-test`.

Each task directory adds `e2e_check.json`, read by
`scripts/mcp/e2e/verify_run.py`, which checks the run artefacts (Claude Code
stream-json, Codex `exec --json` JSONL, or the normalised trajectory) for what
the scorer cannot see:

| Check | Meaning |
|---|---|
| `mcp_connected` | the MCP server reported `connected` and every required tool was offered to the agent; Codex has no such event, so there every required tool must have returned a result (WARN otherwise) |
| `tool_called` | the agent called every required `mcp__<server>__<tool>` (Codex `mcp_tool_call` items are reported under the same name) |
| `tool_correct` | per required tool, a successful call returned the expected result (value, JSON field, or an image of the expected media type); WARN if the inputs were not the reference inputs verbatim, or if the result type is not observable in older artefacts |
| `tool_chain` | only when configured: a call's inputs equal the result of an earlier call (e.g. the plot was drawn from the scan output, not retyped data) |
| `answer_from_tool` | each configured output-file value equals a value a tool returned and matches the reference |
| `no_bypass` | no Bash command or produced source file installs/imports the backend directly, and no listed non-MCP tool (e.g. `WebFetch` of arxiv.org) reached it (FAIL); broader matches and other web tool calls are WARN for review |

`asibench score` answers "is the number right"; `verify_run.py` answers "did
the number come from the MCP tool". An E2E pass needs both.

Evidence note: `asibench run` redacts the content of every user-role event in
the persisted `*.agent_stdout.jsonl` (prompt protection), which also blanks
tool results. The verifier therefore takes tool names, inputs, Bash commands
and the MCP init status from the stream, and fills tool results from
`*.trajectory.json` by `tool_call_id`. Image results are not copied into the
trajectory; the Claude extractor records `content_types` and
`image_media_types` in the tool_result metadata so that a returned PNG stays
observable.

Codex (`codex exec --json`, checked with codex-cli 0.159.2) reports each MCP
call as an `mcp_tool_call` item with `server`, `tool`, `arguments` and, on
`item.completed`, `result.content` (MCP blocks; images carry `mimeType`) or
`error`. These items are not user events, so the persisted JSONL keeps inputs
and results and the verifier needs no trajectory; the Codex extractor still
records the calls (`tool_call_id`, `content_types`, `image_media_types`, no
image data) so that the trajectory alone proves calls and results. The JSONL
has no list of connected servers or offered tools. Codex also calls its own
`list_mcp_resources` / `list_mcp_resource_templates` under the server's name;
they show up in the tool sequence and never count as a required tool. Shell
use is taken from `command_execution` items.

`e2e_check.json` schema 2 lists `calls` (each with `tool`, optional
`inputs_from_reference`, optional `inputs_from_call` for chaining, and a
`result` of format `number`, `json` + `key`, or `image` + `media_type`) and
`answers` (output-file key, the call and result key it must be copied from,
reference key and tolerance). Schema 1 (single `tool`, scalar answer) is still
accepted and normalised to schema 2.

Non-numeric results use a named extractor instead of `key`/`abs_tol`:
`"extract": "arxiv_ids"` (IDs from a search result list, without version,
prefix or URL) or `"term_counts"` (snippets per term), compared on canonical
values with `"match"`: `equal`, `subset` (every term of this call has the
reference count, for per-term calls) or `member` (the reference value is one
of the extracted values). `inputs_from_call` with `extract`, `args` and
`"match": "member"` passes when any listed argument of the consumer call is
one of the values extracted from the source call; an answer with
`"merge_calls": true` merges the extracted values of all calls first.
Optional `server_tools` lists the tools the server must offer (WARN if it
offers more), `bypass_tools` / `suspicious_tools` map non-MCP tool names to a
regex over the call input (FAIL / WARN).

## Tasks

| Task | MCP server | Tool | Reference |
|---|---|---|---|
| `mcp_e2e.pyscf_rhf_energy` | `pyscf` (`scripts/mcp/e2e/manifest.json`) | `pyscf_rhf_energy` | PySCF RHF computed in `generate_gt.py` |
| `mcp_e2e.pyscf_bond_stretch` | `pyscf` | `run_bond_stretch_calculation_mcp` → `plot_energy_scan_image_mcp` | seeded RDKit + UFF geometry, rigid stretch, PySCF RHF/STO-3G in `generate_gt.py` |
| `mcp_e2e.arxiv_search_snippets` | `arxiv` | `ArXiv_search_papers` → `ArXiv_get_pdf_snippets` | raw arXiv API query and the PDF converted with MarkItDown (server lockfile versions) in `generate_gt.py` |

`mcp_e2e.pyscf_rhf_energy`: a seed picks one of five small closed-shell
molecules and STO-3G or 6-31G, and perturbs every coordinate by up to ±0.02 Å so
the energy cannot be recalled. The agent must write `result.json` with
`energy_hartree`; full credit at |ΔE| ≤ 1e-6 Ha, zero at ≥ 1e-3 Ha, log-linear
in between. Only B1 names the tool; B3/B4 only say "use the MCP tools available".

`mcp_e2e.pyscf_bond_stretch`: a seed picks one of seven bonds in rigid
molecules (H₂O, NH₃, HF, HCN ×2, H₂CO, CH₃F), a start of 0.85–0.95 and an end
of 1.25–1.45 times the bond length, and 5–9 points. The agent must run the
scan tool, then call the plot tool with the scan result unchanged, and write
`result.json` with `bond_lengths`, `energies_hartree` and the lowest point.
Scoring: 80 for energies (full credit at max |ΔE| ≤ 1e-5 Ha, zero at ≥ 1e-3,
log-linear; the grid must match) and 20 for the minimum. The plot cannot be
scored from files (the image goes into the agent's context); `verify_run.py`
checks that the plot tool was called with the scan output and returned a PNG.
The server builds its geometry from an unseeded conformer, which for these
rigid molecules reproduces the reference to ~1e-7 Ha (checked against the
upstream code for 41 seeds). The tool ignores its `basis` argument, so the task
is fixed to STO-3G. This is the only L2 task for the remaining pyscf tools; the
others are covered by the L1 smoke only (`scripts/mcp/e2e/README.md`).

`mcp_e2e.arxiv_search_snippets`: a seed picks one of five curated searches
(fielded query, closed 2010–2013 submission-date window, 2–6 results, sorted
by submission date). The agent must search, pick the paper with the most
authors (ties: first in result order), call the snippet tool for two terms on
that paper with a per-term cap of 10, and write `result.json` with all result
IDs in order, the picked ID and the number of snippets per term (each term
occurs 2–6 times). Scoring: 40 for the ID list, 20 for the pick, 40 split over
the term counts. Counts rather than snippet text are scored because the
converted text of two-column PDFs merges words. Generation and the agent run
both need network access to arXiv; the reference is fetched at generation
time, so generate shortly before running (the windows are closed, but a paper
can still get a new PDF version). `verify_run.py` also checks that the snippet
call used a paper the search returned (`arxiv_id` or `pdf_url`), and flags
`WebFetch` of arxiv.org, shell HTTP access to arXiv and arXiv/PDF libraries as
bypass; any web search or fetch call is a WARN. With `--mcp-config` the
harnesses run in search mode, so these web tools are available to the agent.

Prompts must stay agent-neutral: name the MCP server and tool
(`pyscf_rhf_energy` of the `pyscf` server), never a harness-specific name such
as Claude Code's `mcp__pyscf__pyscf_rhf_energy` (see failure mode below; it is
also wrong for Codex).

## Results

### `mcp_e2e.arxiv_search_snippets` (Claude Code and Codex CLI)

2026-10-02, AWS Linux amd64, seed 31415 (gravitational waves: 5 results,
picked 1103.0576 with 19 authors, counts millisecond 5 / arecibo 3), one run
per level. Claude Code with `claude-opus-5-5`; Codex CLI with `gpt-5.6-sol`
through the custom gateway (`codex_home`), effort `medium`:

| Harness | Level | Local score | Verified PASS | Notes |
|---|---|---|---|---|
| Claude Code | B1–B4 | 4 × 100 | 4/4 | both tools found without being named at B3/B4 |
| Codex CLI | B1–B4 | 4 × 100 | 4/4 | both tools found without being named at B3/B4 |

All eight runs passed every verifier check with no WARN: the search returned
the reference IDs, the snippet call used the picked paper from the search
result (`tool_chain`) and returned the reference counts, and the answers equal
the tool-returned values. Although `--mcp-config` runs in search mode, neither
harness called a web search/fetch tool or reached arXiv from the shell, and
Codex's default MCP timeouts were enough for the PDF download and conversion.
One run per level shows the path works, not a pass rate.

### Codex CLI (both tasks)

2026-09-30, AWS Linux amd64, codex-cli 0.159.2, `gpt-5.6-sol` through a
custom gateway (`codex_home`), effort `medium`, seed 31415, one run per level:

| Task | Level | Local score | Verified PASS | Notes |
|---|---|---|---|---|
| `mcp_e2e.pyscf_rhf_energy` | B1–B4 | 4 × 100 | 4/4 | tool found without being named at B3/B4 |
| `mcp_e2e.pyscf_bond_stretch` | B1–B4 | 4 × 100 | 4/4 | both tools found without being named at B3/B4 |

All eight runs passed every verifier check with no WARN, from the persisted
Codex JSONL alone: required tools returned results (`mcp_connected`), the plot
call received the scan output unchanged (`tool_chain`) and returned an
`image/png` block, and the answers equal the tool-returned values. An earlier
B1 pass of both tasks gave the same result. The server's stdout prints and
Codex's default MCP timeouts caused no problems. This is one run per level
(the Claude Code rounds below used three), so it shows the path works, not a
pass rate.

### `mcp_e2e.pyscf_bond_stretch`

2026-09-30, AWS Linux amd64, Claude Code, `claude-opus-5-5`, seed 31415
(hydrogen cyanide, C–H 0.994–1.534 Å, 8 points), `--retries 3` per level:

| Level | Local score | Verified PASS | Notes |
|---|---|---|---|
| B1 | 3 × 100 | 3/3 | |
| B2 | 3 × 100 | 3/3 | |
| B3 | 3 × 100 | 3/3 | both tools found without being named |
| B4 | 3 × 100 | 3/3 | |

Every run called the scan tool with the reference inputs, passed its result
unchanged to the plot tool (`tool_chain`), received a PNG (`tool_correct` on
the image, from the trajectory's `image_media_types`) and copied the scan
values into `result.json`. No bypass or suspicious commands were flagged.

### `mcp_e2e.pyscf_rhf_energy`

2026-09-29, AWS Linux amd64, Claude Code 2.1.284, `claude-opus-5-5`, seed 31415
(water, 6-31G), `--retries 3` per level:

| Level | Verified PASS | Notes |
|---|---|---|
| B1 | 2/3 | prompt still contained the Claude-specific tool name; see below |
| B2 | 3/3 | |
| B3 | 3/3 | tool found without being named |
| B4 | 3/3 | |
| B1 (agent-neutral prompt) | 5/5 | re-generated instance, `--retries 5` |

In every run the server reported `connected` and all 7 tools were offered;
every tool call returned the reference energy within 1e-13 Ha. PySCF's stdout
line, the `cwd` config field and the `mcp__pyscf__*` tool allowlist caused no
problems. An earlier single pass of B1–B4 with the same model also failed only
at B1, with the same reasoning.

## Observed failure modes

**Agent believes an offered MCP tool is unavailable** (B1, 2 of 4 verified
runs; 0 of 12 at B2–B4). Sequence: the agent tries `ToolSearch` to "load" the
tool (ToolSearch is not offered in these runs, so the call errors), then runs
`claude mcp list` in Bash, which prints "No MCP servers configured" because the
harness passes servers with `--mcp-config` rather than a config file, and gives
up without writing output. The B1 prompt then named the tool as
`mcp__pyscf__pyscf_rhf_energy` "in Claude Code"; it now names only the server
and tool, after which B1 passed 5/5. `verify_run.py` shows this as `mcp_connected=PASS`,
`tool_called=FAIL` with the tool sequence and `toolsearch_offered`.

Related observation for formal runs: host-mode agents can start a nested
`claude` process and read their isolated harness home.

## Run (Linux host, unprivileged E2E user)

Prerequisites: the MCP server is installed and its smoke test passes
(`scripts/mcp/e2e/README.md`), and the agent CLI is logged in for this user.

```sh
cd ~/ASI-Bench

# 1. Instance + reference. PySCF (and RDKit for the bond scan) is needed only
#    here, in an isolated task venv. seed 31415 makes the instance id end in
#    __seed31415, as local scoring requires. Repeat for mcp_e2e.pyscf_bond_stretch.
uv run asibench generate --task mcp_e2e.pyscf_rhf_energy --params '{"seed": 31415}' \
  --sandbox task --tasks-dir examples/mcp-e2e-tasks --output-dir ~/e2e/instances

# 2. Agent run on the host; PySCF is reachable only through the MCP server.
#    For the bond scan use --tasks mcp_e2e.pyscf_bond_stretch.
uv run asibench run --agent claude_code_cli \
  --agent-config '{"model": "claude-opus-4-6", "permission_mode": "bypassPermissions"}' \
  --mcp-config ~/mcp/pyscf.mcp.json \
  --tasks mcp_e2e.pyscf_rhf_energy --include-test --tasks-dir examples/mcp-e2e-tasks \
  --instances-dir ~/e2e/instances --prompt-levels b1,b2,b3,b4 \
  --sandbox none --timeout 900 --output-dir ~/e2e/out-claude

# 3. Is the answer right?
uv run asibench score --repo seed31415 --results-dir ~/e2e/out-claude \
  --instances-dir ~/e2e/instances --tasks-dir examples/mcp-e2e-tasks

# 4. Did it come from the MCP tool?
python3 scripts/mcp/e2e/verify_run.py --results-dir ~/e2e/out-claude \
  --instances-dir ~/e2e/instances --tasks-dir examples/mcp-e2e-tasks
```

Codex CLI uses the same instances, scorer and verifier; only step 2 differs:

```sh
uv run asibench run --agent codex_cli \
  --agent-config '{"model": "gpt-5.6-sol", "effort": "medium"}' \
  --mcp-config ~/mcp/pyscf.mcp.json \
  --tasks mcp_e2e.pyscf_rhf_energy,mcp_e2e.pyscf_bond_stretch --include-test \
  --tasks-dir examples/mcp-e2e-tasks --instances-dir ~/e2e/instances \
  --prompt-levels b1,b2,b3,b4 --sandbox none --timeout 900 --output-dir ~/e2e/out-codex
```

With a custom gateway, keep `config.toml` (the `model_providers` entry) and
`auth.json` in a dedicated directory without other MCP servers, export
`CODEX_HOME` to it **and** pass it as `"codex_home": "/abs/path"` in
`--agent-config`: each run gets an isolated home that takes `auth.json` from
`$CODEX_HOME` but `config.toml` only from `codex_home`, so without the latter
the provider settings are dropped and Codex calls the default endpoint.

`--sandbox os` is intentionally rejected with `--mcp-config`; host runs have
no filesystem isolation, so use a dedicated unprivileged user. The agent can
still find PySCF elsewhere on the host (the MCP venv, the generation task venv)
— that is exactly what `no_bypass` detects.

arxiv task (needs `python3 scripts/mcp/e2e/setup.py arxiv` and network access
to arXiv for both generation and the run):

```sh
uv run asibench generate --task mcp_e2e.arxiv_search_snippets --params '{"seed": 31415}' \
  --sandbox task --tasks-dir examples/mcp-e2e-tasks --output-dir ~/e2e/instances

uv run asibench run --agent claude_code_cli \
  --agent-config '{"model": "claude-opus-4-6", "permission_mode": "bypassPermissions"}' \
  --mcp-config ~/mcp/arxiv.mcp.json \
  --tasks mcp_e2e.arxiv_search_snippets --include-test --tasks-dir examples/mcp-e2e-tasks \
  --instances-dir ~/e2e/instances --prompt-levels b1,b2,b3,b4 \
  --sandbox none --timeout 900 --output-dir ~/e2e/out-arxiv-claude
```

Score and verify as in steps 3–4; for Codex use `--agent codex_cli` with the
Codex `--agent-config` above.

## Adding a task

Copy the pyscf task layout: `task_meta.yaml` (`status: test`), `task_eval.yaml`,
`generate_gt.py` computing an independent reference, `custom_scorer.py`,
`prompt_b1..b4.md`, and `e2e_check.json` (schema 2: `calls` and `answers`,
see `pyscf_bond_stretch`; schema 1 for a single tool with a scalar answer, see
`pyscf_rhf_energy`; named extractors and web-tool bypass checks, see
`arxiv_search_snippets`) plus bypass patterns. Add the server to
`scripts/mcp/e2e/manifest.json` with a smoke test first.
