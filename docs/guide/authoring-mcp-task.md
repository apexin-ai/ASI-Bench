# Design a task that uses MCP

This guide is for task authors. It explains what you must prepare when a task
uses an MCP tool. Read [Contribute a task](authoring-a-task.md) first for the
normal task files, reference generation, scoring, and Portal submission flow.

## Decide whether MCP is part of the task

Use MCP only when the scientific objective depends on an operation or dataset
provided by the server. A task that merely mentions MCP is not an MCP task.
Define the required server and tool behavior in the prompt and output contract:
inputs, units, expected artifacts, and any multi-step tool dependencies.

Keep the task answer independently scorable. A successful MCP call does not
prove that the scientific answer is correct; the scorer still checks the final
output against a reference or structural contract.

## Prepare the server before writing the task

The MCP config tells the agent how to connect to a server. It does not install
one. The server must be available in one of these forms:

- **Instance stdio:** install the server and locked dependencies in the task
  runtime image. The config uses paths inside that image.
- **Internal sidecar:** provide a Docker image with a `HEALTHCHECK`, declare it
  under `runtime.services` in `task_meta.yaml`, and reference its internal
  hostname and port from the MCP config.

Do not put API keys, tokens, host absolute paths, host virtualenvs, Docker
sockets, or unpinned install commands in a task. Current OS sandbox sidecars
are internal and unauthenticated; remote authenticated MCP, GUI/licensed
software, and external egress allowlists need separate platform support.

For a new server, maintainers should also pin its source and dependencies,
check `initialize`/`tools/list`, and run a real `tools/call` against an
independent reference. A server becomes a reusable framework E2E integration
only after it is added to `scripts/mcp/e2e/manifest.json` with smoke tests;
that is separate from authoring one task.

## Add the normal task files

Create the task as usual. MCP does not add a required task file or a generic
`mcp: server-name` field to the task schema. The task normally contains:

- `task_meta.yaml`: metadata, input/output declarations, and optional
  `runtime.services` for a sidecar;
- `prompt_b1.md` through `prompt_b4.md`: tell the agent what scientific work to
  do and what it must write;
- `generate_gt.py`: deterministic inputs and reference generation;
- `task_eval.yaml` and, when needed, `custom_scorer.py`: final answer scoring.

`requires_tool` in `examples/mcp-tasks/` is an example convention. It does not
by itself install or inject an MCP server into a formal task.

## Write the MCP config

For an image-local server, use a container path and explicit deployment:

```json
{
  "mcpServers": {
    "science": {
      "command": "/opt/science/bin/python",
      "args": ["-m", "science_mcp.server"],
      "cwd": "/workspace",
      "sandbox": {"deployment": "instance"}
    }
  }
}
```

For a sidecar declared in `task_meta.yaml`:

```yaml
runtime:
  services:
    - name: science
      image: registry.example/science-mcp@sha256:<digest>
      port: 8765
```

```json
{
  "mcpServers": {
    "science": {
      "url": "http://science:8765/mcp",
      "sandbox": {"deployment": "service", "service": "science"}
    }
  }
}
```

Keep the config as an operator/run artifact when it contains machine-specific
paths or credentials. The task should name the expected server/tool behavior;
the runner supplies the environment-specific config.

For long Codex MCP calls, add `"tool_timeout_sec": 3600` inside the server
entry. Codex otherwise defaults to 60 seconds per call, which is insufficient
for CPU MatterGen generation. This optional field works in host and OS runs;
omit it when using Claude. The overall task/agent timeout must also allow the
computation to finish.

## Run and validate

Check the config, then run the task with a supported CLI agent:

```bash
asibench mcp check --config science-mcp.json

asibench run \
  --agent codex_cli \
  --agent-config '{"model":"your-model"}' \
  --mcp-config science-mcp.json \
  --sandbox os \
  --tasks domain.my_task \
  --instances-dir instances/ \
  --output-dir results/
```

The OS run validates every server with `initialize` and `tools/list` before the
agent starts. The CLI then connects to the server, exposes its tool schemas to
the model, and sends MCP `tools/call` requests when the model chooses a tool.
The framework does not ask the model to call a tool automatically and does not
install the server from the config.

MCP and built-in web search are separate. With the default restricted mode,
explicit MCP remains available while web search stays off. Add
`--tool-mode search` only when the task permits built-in web search/fetch.

Run the normal author checks as well:

```bash
asibench validate --pre-submit tasks/domain/my_task/
asibench difficulty-check --task domain.my_task \
  --agent codex_cli \
  --agent-config '{"model":"your-model"}' \
  --mcp-config science-mcp.json \
  --sandbox os
```

The MCP server must be available to the image or sidecar used by the
`difficulty-check`; a host-only config is not sufficient.

## Material-science servers

The framework E2E manifest currently provides four material-science IDs. Use
the same ID in a run-specific MCP config only when the task needs that model:

| ID | Server and model semantics | Typical tools |
| --- | --- | --- |
| `chgnet` | AtomisticSkills `matgl` server using a MatGL CHGNet-family PES model; it is not the original `chgnet` package | `predict_structure`, `relax_structure`, `run_md` |
| `mace` | AtomisticSkills MACE server; select and record the exact MACE checkpoint with `load_model` | `predict_structure`, `relax_structure`, `run_md` |
| `mattergen` | AtomisticSkills MatterGen structure generation; stochastic and usually GPU/checkpoint heavy | `generate_structures` |
| `mattersim` | ASI-Bench adapter around the official MatterSim package; set a pinned local checkpoint for offline runs | `predict_structure`, `relax_structure` |

The reusable setup pins the AtomisticSkills revision and installs the required
subproject (`venv/mlip` for `chgnet`/`mace`, `venv/mattergen` for
`mattergen`/`mattersim`), including the locked `cu126` extra. This build also
runs on CPU; GPU use requires a compatible NVIDIA driver. Setup fetches the
pinned MatterGen source resources omitted from its wheel and sets
`MATTERGEN_REPO` in its config. A task author normally supplies only the run MCP
config and task files; do not copy the E2E manifest into a task. For a local
server, the essential shape is:

```json
{
  "mcpServers": {
    "materials": {
      "command": "/opt/AtomisticSkills/venv/mlip/.venv/bin/python",
      "args": ["-m", "src.mcp_server.mace_server"],
      "cwd": "/opt/AtomisticSkills",
      "sandbox": {"deployment": "instance"}
    }
  }
}
```

Pin the server revision, Python environment, model name, device, and any
MatterSim checkpoint outside the prompt. In scoring, compare structure fields,
energies, forces, and units with tolerances; compare MatterGen properties and
metadata rather than generated bytes. Treat model download, GPU memory, and
checkpoint availability as runtime prerequisites and keep a deterministic
reference path independent of the server output.

Before using a deployment, run `scripts/mcp/e2e/smoke.py <id> --config
<id>.mcp.json` with that server's Python. Add `--load-model` for `chgnet`,
`mace`, or `mattersim` to compare a real prediction with a separate ASE backend;
add `--generate` for `mattergen` to generate and parse CIF files. Without these
flags, checkpoint execution is skipped and reported as a warning. MatterGen
CPU generation can take many minutes; explicitly set `batch_size` as well as
`num_structures` to avoid computing the upstream default batch of 10. These
direct checks do not certify agent execution or an OS sandbox image.

## Verify that the agent really used MCP (optional)

Ordinary scoring answers “is the output correct?”. If the task requirement also
says “the answer must come from MCP”, add an MCP evidence specification in the
same style as the E2E examples:

- required server and tool calls;
- expected tool inputs or calls chained from earlier results;
- which output fields must equal returned tool values;
- shell, import, install, or web routes that count as bypasses.

After the run, use the task's `e2e_check.json` with:

```bash
python3 scripts/mcp/e2e/verify_run.py \
  --results-dir results/ \
  --instances-dir instances/ \
  --tasks-dir tasks/
```

Use this only when tool provenance is part of the benchmark objective. The
verifier complements the scorer: `tool_called` does not mean `tool_correct`,
and a correct answer does not prove that it came from MCP.

## Author checklist

Before submitting, confirm:

- the task genuinely needs a named MCP operation;
- the server has a pinned, reproducible image/runtime;
- the config uses container paths or the declared internal service;
- no secret or host-specific path is committed;
- prompts specify inputs, outputs, units, and required artifacts;
- reference generation and scoring work without trusting the MCP server's own
  answer as the only reference;
- `mcp check`, `validate --pre-submit`, and OS difficulty-check pass;
- MCP evidence is added only if the task requires proof of tool use.

See [MCP in the OS sandbox](../mcp/os-sandbox.md) for deployment details and
[the MCP E2E README](../../examples/mcp-e2e-tasks/README.md) for verifier
examples.
