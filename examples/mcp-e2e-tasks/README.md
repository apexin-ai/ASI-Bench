# MCP E2E fake tasks

Fake tasks that prove an agent in the normal `asibench run` → `asibench score`
pipeline really calls a given MCP tool and uses its result. They are not
benchmark tasks: they sit outside `tasks/`, have `status: test`, and are only
found with `--tasks-dir examples/mcp-e2e-tasks --include-test`.

There are two separate questions:

- `asibench score`: is the answer right?
- [`scripts/mcp/e2e/verify_run.py`](../../scripts/mcp/e2e/verify_run.py): did
  the answer come from the MCP tool?

An E2E pass needs both. The verifier reads the run artefacts and the task's
`e2e_check.json`:

| Check | Passes when |
|---|---|
| `mcp_connected` | the server connected and offered every required tool (Codex reports neither, so it must show successful calls instead) |
| `tool_called` | every required tool was called |
| `tool_correct` | each required call returned the reference result (value, JSON field or image type) |
| `tool_chain` | a call's input is the output of an earlier call, where configured |
| `answer_from_tool` | each answer in the output file equals a value a tool returned |
| `no_bypass` | the agent did not reach the backend through other routes (imports, installs, web tools) |

## Tasks

| Task | Server | Tools | Reference in `generate_gt.py` |
|---|---|---|---|
| `mcp_e2e.pyscf_rhf_energy` | `pyscf` | `pyscf_rhf_energy` | PySCF RHF |
| `mcp_e2e.pyscf_bond_stretch` | `pyscf` | `run_bond_stretch_calculation_mcp` → `plot_energy_scan_image_mcp` (image) | seeded RDKit/UFF + PySCF |
| `mcp_e2e.arxiv_search_snippets` | `arxiv` | `ArXiv_search_papers` → `ArXiv_get_pdf_snippets` | raw arXiv API + MarkItDown (needs network; generate just before the run, papers can get new PDF versions) |
| `mcp_e2e.jsbsim_engine_run` | `jsbsim` | `create_session` → `set_*` → `step` → `get_property`/`get_telemetry` (one session) | JSBSim driven directly |
| `mcp_e2e.s4_grating_spectrum` | `s4` | `simulate_stack_spectrum` | numpy 1D RCWA |
| `mcp_e2e.psi4_opt_freq` | `psi4` | `optimize` → `frequency` at the returned geometry | PySCF DF-RHF + geomeTRIC |

How each task picks its instances, scores answers and keeps the tool the only
practical source of the numbers is described in its `generate_gt.py` and
`task_eval.yaml`. Server setup and quirks are in
[`scripts/mcp/e2e/README.md`](../../scripts/mcp/e2e/README.md).

## Results

All runs: AWS Linux amd64, seed 31415, `--sandbox none`. Claude Code used
`claude-opus-5-5`; Codex CLI used `gpt-5.6-sol` at effort `medium`. Each
cell gives local score / verifier PASS. One run per level shows that the path
works; it is not a pass rate.

| Task | Date | Claude Code B1–B4 | Codex CLI B1–B4 |
|---|---|---|---|
| `pyscf_rhf_energy` | 2026-09-29/30 | 3/3 per level (B1 2/3 → 5/5 after the prompt fix below) | 4 × 100, 4/4 |
| `pyscf_bond_stretch` | 2026-09-30 | 12 × 100, 12/12 (3 per level) | 4 × 100, 4/4 |
| `arxiv_search_snippets` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `jsbsim_engine_run` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `s4_grating_spectrum` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `psi4_opt_freq` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |

In every run at B3/B4 the agent found the tools without being told their names.

## Run

Prerequisites: the server is installed and its smoke test passes, and the agent
CLI is logged in for the E2E user. Example for one task; for the others, change
the task and the `--mcp-config` file.

```sh
cd ~/ASI-Bench
T=mcp_e2e.pyscf_rhf_energy; MCP=~/mcp/pyscf.mcp.json; OUT=~/e2e/out-claude

# 1. Instance + reference (seed 31415 is required for local scoring)
uv run asibench generate --task $T --params '{"seed": 31415}' \
  --sandbox task --tasks-dir examples/mcp-e2e-tasks --output-dir ~/e2e/instances

# 2. Agent run (host only: --sandbox os is rejected together with --mcp-config)
uv run asibench run --agent claude_code_cli \
  --agent-config '{"model": "claude-opus-5-5", "permission_mode": "bypassPermissions"}' \
  --mcp-config $MCP --tasks $T --include-test --tasks-dir examples/mcp-e2e-tasks \
  --instances-dir ~/e2e/instances --prompt-levels b1,b2,b3,b4 \
  --sandbox none --timeout 900 --output-dir $OUT

# 3. Score, 4. verify
uv run asibench score --repo seed31415 --results-dir $OUT \
  --instances-dir ~/e2e/instances --tasks-dir examples/mcp-e2e-tasks
python3 scripts/mcp/e2e/verify_run.py --results-dir $OUT \
  --instances-dir ~/e2e/instances --tasks-dir examples/mcp-e2e-tasks
```

For Codex, use `--agent codex_cli --agent-config '{"model": "gpt-5.6-sol",
"effort": "medium", "codex_home": "/abs/path"}'`. With a custom gateway, put
`config.toml` and `auth.json` in a directory of their own and point both
`CODEX_HOME` and `codex_home` at it. The isolated run home copies `config.toml`
only from `codex_home`; without it Codex falls back to the default endpoint.

Host runs have no filesystem isolation, so use a dedicated unprivileged user.
The agent can still find the backend library elsewhere on the host; catching
that is what `no_bypass` is for. `--mcp-config` also turns on web search
(Claude: WebSearch/WebFetch, Codex: web_search).

## Adding a task

1. Add the server to `scripts/mcp/e2e/manifest.json` and make its smoke test pass.
2. Copy the closest existing task. Keep `status: test` and seed 31415, and
   compute the reference independently of the server.
3. Write `e2e_check.json` (schema 2: `calls`, `answers`, bypass patterns). The
   format is documented in the `verify_run.py` docstring. Copy from an existing
   task:

   | Pattern | Task |
   |---|---|
   | single tool, scalar answer | `pyscf_rhf_energy` |
   | tool chain, image result | `pyscf_bond_stretch` |
   | non-numeric extractors, web bypass | `arxiv_search_snippets` |
   | session chain, optional/grouped calls | `jsbsim_engine_run` |
   | answers selected from returned arrays | `s4_grating_spectrum` |
   | nested keys, geometry chain | `psi4_opt_freq` |

4. Prompts name the server and the tool (e.g. "`pyscf_rhf_energy` of the
   `pyscf` server"). Never use a harness-specific name such as
   `mcp__pyscf__…`.

Why the prompt rule exists: when the B1 prompt used the name
`mcp__pyscf__pyscf_rhf_energy`, 2 of 4 Claude runs decided that the tool was
unavailable. They tried `ToolSearch`, then got "No MCP servers configured" from
`claude mcp list`, and gave up. After the prompt was changed to name only the
server and the tool, B1 passed 5/5.

## Evidence quirks

The verifier handles these; they matter only if you are debugging it.

- **Claude, tool results:** persisted stream-json redacts user events,
  including tool results. The verifier fills results in from
  `*.trajectory.json` by `tool_call_id`.
- **Claude, images:** trajectories never keep image data. The extractors
  record `content_types`/`image_media_types` instead.
- **Claude, structured output:** for FastMCP tools that declare an
  `outputSchema`, Claude Code shows `structuredContent` (`{"result": …}`) and
  not the text. The verifier unwraps it.
- **Codex:** each call is an `mcp_tool_call` item that keeps its inputs and
  results, but Codex does not list servers or offered tools. Its own
  `list_mcp_resources*` calls are not counted as required tools.
