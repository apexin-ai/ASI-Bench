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

FastMCP tools with a return annotation declare an `outputSchema` and return
`structuredContent = {"result": ...}` next to the text block; Claude Code puts
that structured object, not the text, into the tool result (seen with the s4
server, mcp 1.30.0). The verifier unwraps it: a string inside is parsed as
JSON, a list of content blocks is reduced to its text first.

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
regex over the call input (FAIL / WARN); `suspicious_tools` also works for MCP
tools by their `mcp__<server>__<tool>` name (e.g. a tool the task forbids).

A call spec with `"optional": true` is judged only if the agent called it.
Optional specs that share a `"group"` form one requirement: at least one member
must be called (`tool_called`) and the best member counts for `tool_correct`,
e.g. reading the final state with either `get_telemetry` or `get_property`. A
numeric answer can take its value from several calls with `from_calls`
(`[{"call": ..., "result_key": ...}, ...]`); a source with `select` takes one
element of a list field: `{"reduce": "max"}` / `"min"`, `{"argmax_of": key}` /
`argmin_of` (the element at the extreme of another list field of the same
result), or `{"where_key": key, "equals_reference_key": ref}` (the element
where list field `key` equals a reference value, e.g. R at a given wavelength
of a returned spectrum). `inputs_from_call` values that
are not numbers, such as a `session_id`, must be identical strings.

## Tasks

| Task | MCP server | Tool | Reference |
|---|---|---|---|
| `mcp_e2e.pyscf_rhf_energy` | `pyscf` (`scripts/mcp/e2e/manifest.json`) | `pyscf_rhf_energy` | PySCF RHF computed in `generate_gt.py` |
| `mcp_e2e.pyscf_bond_stretch` | `pyscf` | `run_bond_stretch_calculation_mcp` → `plot_energy_scan_image_mcp` | seeded RDKit + UFF geometry, rigid stretch, PySCF RHF/STO-3G in `generate_gt.py` |
| `mcp_e2e.arxiv_search_snippets` | `arxiv` | `ArXiv_search_papers` → `ArXiv_get_pdf_snippets` | raw arXiv API query and the PDF converted with MarkItDown (server lockfile versions) in `generate_gt.py` |
| `mcp_e2e.jsbsim_engine_run` | `jsbsim` | `create_session` → (`set_initial_conditions`) → `set_property` ×3 → `step` → `get_property` or `get_telemetry`, one `session_id` | JSBSim 1.3.1 flown directly in `generate_gt.py` with JSBSim's own initial-condition properties |
| `mcp_e2e.s4_grating_spectrum` | `s4` | `simulate_stack_spectrum` (optional `check_engine_sanity`) | independent numpy 1D RCWA with S4's default formulation (Laurent's rule, same truncation) in `generate_gt.py` |

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

`mcp_e2e.jsbsim_engine_run`: a seed picks a short powered flight of the
JSBSim `c172x` (Cessna 172): all seven initial conditions (3000–6000 ft, one of
five locations, 150–185 ft/s calibrated, heading, pitch 0–3°, roll 0), the
engine start (`propulsion/set-running = -1`), mixture 1.0 and a throttle of
0.6–1.0, and a run time of 6, 8 or 10 s at the default dt of 1/60 s. The agent
must create one session, set the initial conditions, apply the settings in
order, step, and write `result.json` with altitude (`position/h-sl-ft`),
calibrated airspeed (`velocities/vc-kts`), thrust
(`propulsion/engine[0]/thrust-lbs`) and simulation time. Scoring: 35 altitude
(full credit within 0.05 ft), 35 airspeed (0.01 kt), 20 thrust (0.05 lbf), 10
time (1e-6 s), each zero at 10× its tolerance, log-linear in between. The
server's ft/s→kt factor and the two decimals of `get_telemetry` stay below
3.3e-4 ft over 300 seeds; applying the settings before the initial
conditions, the wrong engine property or one second too many costs ≥ 0.33 ft
(60 seeds). The prompts forbid `trim` and `execute_script` in domain terms
(both are broken upstream; see `scripts/mcp/e2e/README.md`), and
`verify_run.py` flags them for review. It also requires every `set_property`
and `step` call to use the `session_id` that `create_session` returned
(`tool_chain`), accepts the final state from either read tool, and treats
`import jsbsim`, installing it, the `jsbsim` CLI or the server's own Python as
bypass. No network is needed.

`mcp_e2e.s4_grating_spectrum`: a seed picks a lamellar dielectric grating on
glass (period 0.6–1.0 µm; Si₃N₄, Ta₂O₅, TiO₂ or Si ridges; fill 0.3–0.7;
height 0.1–0.4 µm), a TM plane wave at 5–20° from the superstrate, 21, 37 or
81 harmonics and a sweep of 11–21 wavelengths on a 5 nm grid just above the
period. The agent must run one `simulate_stack_spectrum` call (layers bottom
to top, the ridge as a rectangle spanning the cell along y, `include_plot`
false) and write `result.json` with R and T at a given sweep point, the
largest R and its wavelength. Scoring: 30 R, 30 T, 25 R_max (each full credit
within 1e-6, zero at 1e-3) and 15 for the wavelength (full within 1e-4 µm,
zero at 1e-3), log-linear in between. The reference is an independent 1D
RCWA in numpy: S4's default formulation uses Laurent's rule with circular
truncation, so these complete square-lattice shells keep exactly the x-axis
orders ±2/±3/±5 of a y-uniform grating, and the 1D RCWA with the same rule and
orders agrees with the server to ~1e-14 (checked by the L1 smoke and over 61
seeds). This makes the tool the only practical source: every instance is
selected so that the converged answer (Li's rule, ±40 orders) and the next
truncation both differ from the tool's R at the reported point by ≥ 1e-3,
i.e. a home-made RCWA scores zero there; the harmonic counts are never
equivalent to the server default (51 → ±4); the angle is never 0, so an
omitted `theta_deg` is visible; the maximum is unique and inside the sweep,
and grid points avoid Rayleigh anomalies. `verify_run.py` compares the
returned R spectrum with the reference, takes the answers from the returned
arrays (`select`: element at the reported wavelength, maximum, wavelength at
the maximum), judges an optional `check_engine_sanity` call, and treats
importing or installing S4 or other RCWA packages, loading `libS4.so` or the
server's own Python as bypass; RCWA-like code (`toeplitz`, `linalg.eig`) and
mentions of S4 or the server package are WARN for review. Generation needs
only numpy; the agent run needs the s4 server (x86-64 host, see
`scripts/mcp/e2e/README.md`). No network is needed.

Prompts must stay agent-neutral: name the MCP server and tool
(`pyscf_rhf_energy` of the `pyscf` server), never a harness-specific name such
as Claude Code's `mcp__pyscf__pyscf_rhf_energy` (see failure mode below; it is
also wrong for Codex).

## Results

### `mcp_e2e.jsbsim_engine_run` (Claude Code and Codex CLI)

2026-10-02, AWS Linux amd64, seed 31415 (Chicago, 5500 ft, 159.85 ft/s,
heading 285°, full throttle, 10 s), one run per level. Claude Code with
`claude-opus-5-5`; Codex CLI with `gpt-5.6-sol` through the custom gateway
(`codex_home`), effort `medium`:

| Harness | Level | Local score | Verified PASS | Notes |
|---|---|---|---|---|
| Claude Code | B1–B4 | 4 × 100 | 4/4 | tools found without being named at B3/B4 |
| Codex CLI | B1–B4 | 4 × 100 | 4/4 | tools found without being named at B3/B4 |

All eight runs passed every verifier check: one session created and every
`set_property`/`step` call used its `session_id` (`tool_chain`), the state was
read through the MCP tools and the answers equal tool-returned values within
the tolerances. With `JSBSIM_DEBUG=0` in the launch env neither client saw
JSBSim's stdout chatter. One run per level shows the path works, not a pass
rate.

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

jsbsim task (needs `python3 scripts/mcp/e2e/setup.py jsbsim`; no network):

```sh
uv run asibench generate --task mcp_e2e.jsbsim_engine_run --params '{"seed": 31415}' \
  --sandbox task --tasks-dir examples/mcp-e2e-tasks --output-dir ~/e2e/instances

uv run asibench run --agent claude_code_cli \
  --agent-config '{"model": "claude-opus-4-6", "permission_mode": "bypassPermissions"}' \
  --mcp-config ~/mcp/jsbsim.mcp.json \
  --tasks mcp_e2e.jsbsim_engine_run --include-test --tasks-dir examples/mcp-e2e-tasks \
  --instances-dir ~/e2e/instances --prompt-levels b1,b2,b3,b4 \
  --sandbox none --timeout 900 --output-dir ~/e2e/out-jsbsim-claude
```

Score and verify as in steps 3–4 (Codex as above). The server closes sessions
idle for 300 s, so a run with very long pauses between tool calls loses its
session.

s4 task (needs `python3 scripts/mcp/e2e/setup.py s4`, i.e. an x86-64 host with
AVX2/FMA/BMI2 and `libblas3 liblapack3`; generation needs only numpy; no
network):

```sh
uv run asibench generate --task mcp_e2e.s4_grating_spectrum --params '{"seed": 31415}' \
  --sandbox task --tasks-dir examples/mcp-e2e-tasks --output-dir ~/e2e/instances

uv run asibench run --agent claude_code_cli \
  --agent-config '{"model": "claude-opus-4-6", "permission_mode": "bypassPermissions"}' \
  --mcp-config ~/mcp/s4.mcp.json \
  --tasks mcp_e2e.s4_grating_spectrum --include-test --tasks-dir examples/mcp-e2e-tasks \
  --instances-dir ~/e2e/instances --prompt-levels b1,b2,b3,b4 \
  --sandbox none --timeout 900 --output-dir ~/e2e/out-s4-claude
```

Score and verify as in steps 3–4 (Codex as above).

## Adding a task

Copy the pyscf task layout: `task_meta.yaml` (`status: test`), `task_eval.yaml`,
`generate_gt.py` computing an independent reference, `custom_scorer.py`,
`prompt_b1..b4.md`, and `e2e_check.json` (schema 2: `calls` and `answers`,
see `pyscf_bond_stretch`; schema 1 for a single tool with a scalar answer, see
`pyscf_rhf_energy`; named extractors and web-tool bypass checks, see
`arxiv_search_snippets`; a stateful session chain with optional/grouped read
calls and multi-source answers, see `jsbsim_engine_run`; answers selected from
returned arrays, see `s4_grating_spectrum`) plus bypass patterns. Add the server to
`scripts/mcp/e2e/manifest.json` with a smoke test first.
