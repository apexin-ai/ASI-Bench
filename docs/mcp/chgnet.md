# CHGNet-family MCP deployment

The existing server is the third-party [AtomisticSkills MatGL server](https://github.com/learningmatter-mit/AtomisticSkills/blob/6afdd465a35088136e4875d7d2fbaefc633f4adf/src/mcp_server/matgl_server.py).
It runs MatGL's `CHGNet-PES-MatPES-PBE-1M-2026.9` checkpoint, **not the original
CederGroupHub `chgnet` package**. Do not use its results as interchangeable
reference values for the original package.

## Why the catalog config does not run directly

`asibench mcp init --servers chgnet_atomisticskills` exports an editable
configuration template with `/path/to/AtomisticSkills/...`. The catalog records
sources and command templates; it does not install repositories, dependencies
or weights. `scripts/mcp/e2e/e2e_smoke/servers/chgnet.py` is a test client, not
the server. The actual server is upstream `src.mcp_server.matgl_server`.

Use one of the installations below. Replacing the placeholder alone does not
install the server. The E2E id/config server name is `chgnet`; the catalog id is
`chgnet_atomisticskills`.

## Local installation

From the ASI-Bench checkout, with `uv`, Git and a C/C++ build toolchain installed:

```bash
python3 scripts/mcp/e2e/setup.py chgnet --root ~/mcp
~/mcp/atomisticskills_mlip/venv/mlip/.venv/bin/python \
  scripts/mcp/e2e/smoke.py chgnet --config ~/mcp/chgnet.mcp.json \
  --load-model --report ~/mcp/chgnet-smoke.json
asibench mcp check --config ~/mcp/chgnet.mcp.json
```

Setup clones revision `6afdd465a35088136e4875d7d2fbaefc633f4adf`, installs the
upstream `venv/mlip` environment with its frozen lock and `cu126` extra, then
writes a config with real absolute paths. CPU execution works; a GPU is not
required. The full upstream environment includes many scientific packages and
is large. The first local model load downloads weights; for reproducible
benchmarks use the pinned local checkpoint from the Docker recipe below.
A generated host config cannot be passed unchanged to an OS sandbox.

## OS sandbox installation

Build the reusable science image from the repository root:

```bash
docker build -t asibench-chgnet:20261031 \
  -f scripts/mcp/materials/Dockerfile.chgnet .
```

The recipe uses a digest-pinned Linux amd64 Python base, installs the same
locked upstream environment, downloads the three checkpoint files at a pinned
Hugging Face revision, verifies SHA-256 and verifies model loading as an
unprivileged user. Installation/downloads happen at build time, not when the
agent connects. Review upstream software/model licenses before redistribution.

Use this task Dockerfile:

```dockerfile
FROM asibench-chgnet:20261031
```

Reference it in the task's `runtime.dockerfile`, as described in the
[author guide](../guide/authoring-mcp-task.md), and pass:

```bash
asibench run ... --agent codex_cli --sandbox os \
  --mcp-config scripts/mcp/materials/mcp.chgnet.os.json
```

The normal framework overlay installs the selected agent CLI. The config uses
container paths and `sandbox.deployment=instance`: the agent and stdio server
share the instance, with `ATOMISTIC_WORKSPACE=/workspace`. It does not require
web search. There is no CHGNet HTTP sidecar in this recipe.

Tell the agent to call `load_model` **first**, with:

```json
{
  "model_name": "/opt/models/CHGNet-PES-MatPES-PBE-1M-2026.9",
  "device": "cpu"
}
```

Then call `predict_structure`, `relax_structure`, or `run_md` as needed.
**Use the absolute model path above, not a bare model name**: the framework
uses isolated HOME directories, so a cache populated in the build user's HOME
would be unavailable. `HF_HUB_OFFLINE=1` prevents fallback model downloads.
Other checkpoints, including the separate bandgap model, must be packaged
explicitly before offline use. The image does not certify every upstream tool
or model. Inspect business errors as well as MCP `isError`: upstream
`load_model` can return an error as text.

For reproducible task scoring specify the exact model, structure, units,
algorithm parameters and numerical tolerance. Energy is total eV, forces
are eV/angstrom, and stress is eV/angstrom³ in this server path. Use an
independent reference; tool availability alone does not establish correctness.

## Deployment acceptance

After building:

```bash
ASIBENCH_CHGNET_DOCKER=1 uv run --frozen pytest -q \
  tests/mcp_e2e/test_chgnet_deployment.py
```

This checks initialize/tools/list and real CPU energy/forces/stress prediction
in a network-disabled container with a fresh HOME, against an independent ASE
backend in another process. It does not invoke a paid model agent, verify GPU
execution, or certify the scientific accuracy of the pretrained potential.

Validated on 2026-10-31 (Linux amd64, CPU): image build passed; the four
deployment tests passed with networking disabled and fresh HOME. Energy, forces
and stress matched the separate ASE backend within `1e-5`. Full MCP E2E offline
regression: 1294 passed, 37 skipped. This deployment change did not rerun a
Codex model agent.
