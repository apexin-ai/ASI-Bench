# MCP end-to-end (E2E) fake tasks

Fake tasks whose only purpose is to prove that an agent, running inside the
normal `asibench run` → `asibench score` pipeline, **actually calls a specific
MCP tool and gets a correct result**. They are not benchmark tasks: they live
outside `tasks/`, have `status: test`, and are only discovered with
`--tasks-dir examples/mcp-e2e-tasks --include-test`.

Each task directory adds `e2e_check.json`, read by
`scripts/mcp/e2e/verify_run.py`, which checks the run artefacts (Claude Code
stream-json, or the normalised trajectory) for what the scorer cannot see:

| Check | Meaning |
|---|---|
| `mcp_connected` | the MCP server reported `connected` and the target tool was offered to the agent |
| `tool_called` | the agent called `mcp__<server>__<tool>` |
| `tool_correct` | a successful call returned the reference value; WARN if the tool inputs were not the reference inputs verbatim |
| `answer_from_tool` | the output file value equals a value the tool returned and matches the reference |
| `no_bypass` | no Bash command or produced source file installs/imports the backend directly (FAIL); broader matches are WARN for review |

`asibench score` answers "is the number right"; `verify_run.py` answers "did
the number come from the MCP tool". An E2E pass needs both.

Evidence note: `asibench run` redacts the content of every user-role event in
the persisted `*.agent_stdout.jsonl` (prompt protection), which also blanks
tool results. The verifier therefore takes tool names, inputs, Bash commands
and the MCP init status from the stream, and fills tool results from
`*.trajectory.json` by `tool_call_id`.

## Tasks

| Task | MCP server | Tool | Reference |
|---|---|---|---|
| `mcp_e2e.pyscf_rhf_energy` | `pyscf` (`scripts/mcp/e2e/manifest.json`) | `pyscf_rhf_energy` | PySCF RHF computed in `generate_gt.py` |

`mcp_e2e.pyscf_rhf_energy`: a seed picks one of five small closed-shell
molecules and STO-3G or 6-31G, and perturbs every coordinate by up to ±0.02 Å so
the energy cannot be recalled. The agent must write `result.json` with
`energy_hartree`; full credit at |ΔE| ≤ 1e-6 Ha, zero at ≥ 1e-3 Ha, log-linear
in between. Only B1 names the tool; B3/B4 only say "use the MCP tools available".

Prompts must stay agent-neutral: name the MCP server and tool
(`pyscf_rhf_energy` of the `pyscf` server), never a harness-specific name such
as Claude Code's `mcp__pyscf__pyscf_rhf_energy` (see failure mode below; it is
also wrong for Codex).

## Results

2026-09-29, AWS Linux amd64, Claude Code 2.1.284, `claude-opus-5-5`, seed 31415
(water, 6-31G), `--retries 3` per level:

| Level | Verified PASS | Notes |
|---|---|---|
| B1 | 2/3 | prompt still contained the Claude-specific tool name; see below |
| B2 | 3/3 | |
| B3 | 3/3 | tool found without being named |
| B4 | 3/3 | |

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
and tool. `verify_run.py` shows this as `mcp_connected=PASS`,
`tool_called=FAIL` with the tool sequence and `toolsearch_offered`.

Related observation for formal runs: host-mode agents can start a nested
`claude` process and read their isolated harness home.

## Run (Linux host, unprivileged E2E user)

Prerequisites: the MCP server is installed and its smoke test passes
(`scripts/mcp/e2e/README.md`), and the agent CLI is logged in for this user.

```sh
cd ~/ASI-Bench

# 1. Instance + reference. PySCF is needed only here, in an isolated task venv.
#    seed 31415 makes the instance id end in __seed31415, as local scoring requires.
uv run asibench generate --task mcp_e2e.pyscf_rhf_energy --params '{"seed": 31415}' \
  --sandbox task --tasks-dir examples/mcp-e2e-tasks --output-dir ~/e2e/instances

# 2. Agent run on the host; PySCF is reachable only through the MCP server.
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

`--sandbox os` is intentionally rejected with `--mcp-config`; host runs have
no filesystem isolation, so use a dedicated unprivileged user. The agent can
still find PySCF elsewhere on the host (the MCP venv, the generation task venv)
— that is exactly what `no_bypass` detects.

## Adding a task

Copy the pyscf task layout: `task_meta.yaml` (`status: test`), `task_eval.yaml`,
`generate_gt.py` computing an independent reference, `custom_scorer.py`,
`prompt_b1..b4.md`, and `e2e_check.json` naming the server, tool, reference
key, prediction file/key, tolerance and bypass patterns. Add the server to
`scripts/mcp/e2e/manifest.json` with a smoke test first.
