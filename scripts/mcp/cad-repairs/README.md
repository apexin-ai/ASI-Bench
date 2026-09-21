# Opt-in CAD MCP compatibility repairs

Local patches, not upstream releases or bundled CAD software. They target the
exact public revisions in `manifest.json`; they do not alter the catalog or
operator configuration. Tested on macOS arm64, Python 3.12.14, MCP SDK 1.30.0.
Upstream licenses continue to apply. Review the patches before use.

## Reproduce

Choose a **new** directory, not an existing installation. From the ASI-Bench root:

```sh
export CAD_MCP_REPAIR_ROOT="$HOME/.local/share/asibench/cad-mcp-repaired-new"
python - <<'PY'
import json, os, pathlib, subprocess
root = pathlib.Path(os.environ['CAD_MCP_REPAIR_ROOT'])
assert not root.exists(), 'Choose a new directory; do not overwrite user files'
root.mkdir(parents=True)
manifest = json.loads(pathlib.Path('scripts/mcp/cad-repairs/manifest.json').read_text())
for entry in manifest['servers']:
    source = root / entry['id']
    subprocess.run(['git', 'clone', entry['repository'], str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', entry['revision']], check=True)
PY
python scripts/mcp/cad-repairs/apply.py --source-root "$CAD_MCP_REPAIR_ROOT"
for name in sketchup fusion360 cadquery; do
  uv venv --python 3.12.14 "$CAD_MCP_REPAIR_ROOT/$name/.venv"
  uv pip install --python "$CAD_MCP_REPAIR_ROOT/$name/.venv/bin/python" \
    -r "scripts/mcp/cad-repairs/$name-tested.txt"
done
for name in sketchup cadquery; do
  uv pip install --no-deps --python "$CAD_MCP_REPAIR_ROOT/$name/.venv/bin/python" \
    "$CAD_MCP_REPAIR_ROOT/$name"
done
"$CAD_MCP_REPAIR_ROOT/sketchup/.venv/bin/python" -m pytest \
  scripts/mcp/cad-repairs/test_smoke.py -q -s
```

Use a matching platform for the tested dependency snapshots. These files are
not cross-platform lockfiles and have no wheel hashes. Application fails closed
on wrong revisions, dirty tracked files, or patch checksum errors; all three
checkouts are preflighted before any patch is applied. Rerunning on patched
sources is intentionally rejected. The apply helper neither downloads nor
installs anything. If applying is interrupted, inspect each checkout; do not
reset existing user edits blindly.

The smoke suite runs installed packages from a temporary working directory
without `PYTHONPATH`, with a temporary HOME and no operator credentials. It
reserves SketchUp's fixed localhost:9876 port without listening; an occupied
port makes the test fail instead of connecting to a real extension. It makes
no SketchUp business calls. Fusion only generates source, never executes it;
CadQuery only queries the initially empty in-memory index and exercises
failure paths, without executing scripts or installing workspace packages.
CadQuery's cold native-library import can take time; each protocol test has a
60-second timeout. These tests are opt-in, not the default project test suite.

## Repaired launch commands

Replace `/path/to/cad-mcp-repaired` with the installed root:

| ID | Command | Arguments |
|---|---|---|
| sketchup | `/path/to/cad-mcp-repaired/sketchup/.venv/bin/sketchup-mcp` | none |
| fusion360 | `/path/to/cad-mcp-repaired/fusion360/.venv/bin/python` | `/path/to/cad-mcp-repaired/fusion360/src/main.py --mcp` |
| cadquery | `/path/to/cad-mcp-repaired/cadquery/.venv/bin/mcp-cadquery` | `--mode stdio` |

No extra `PYTHONPATH` is required. The original local `mcp-config.json` remains
unchanged: switch these three entries explicitly after reviewing the results.
Do not interpret these repairs as an end-to-end CAD certification or sandbox.
CadQuery still exposes upstream script execution, package installation, file
writes and GUI-launch tools; SketchUp exposes model modification and Ruby
evaluation. Only run trusted calls in an appropriately isolated workspace.

## Changes and limitations

- SketchUp: `FastMCP(description=...)` becomes `instructions=...`; pin the tested
  SDK. The existing lazy/offline-tolerant connection lifecycle is retained.
- Fusion 360: replace only the `--mcp` transport with the standard SDK server,
  map the existing registry to JSON Schema and validate inputs, return generated
  source as MCP text, propagate errors as `isError`. HTTP routes and the script
  generator are unchanged. The old `McpServer` class remains for compatibility,
  but is no longer used by `--mcp`.
- CadQuery: package-relative imports, Pydantic v2 model schemas and native typed
  field validation instead of the redundant deprecated root validator; standard
  SDK stdio transport forwards to existing handlers. Errors, absent results and
  `success: false` cannot become successful tool responses. Handler Python
  stdout is redirected to stderr. This is not an OS-level output/safety sandbox.
  HTTP/SSE remains the upstream legacy protocol and is **not** certified MCP.

See `docs/mcp/cad-mcp-repair-smoke.md` for the dated evidence and boundaries.
