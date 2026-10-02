#!/usr/bin/env python3
"""Install one pinned MCP server for end-to-end (E2E) testing.

Stdlib only. For the server ``<id>`` listed in ``manifest.json`` this:

1. clones the upstream repository into ``<root>/<id>`` (or reuses an existing
   clean checkout) and detaches at the pinned revision;
2. builds an isolated virtualenv with ``uv`` using the manifest's Python
   version (``UV_PYTHON`` from the caller's shell is ignored on purpose);
3. writes a portable ``<root>/<id>.mcp.json`` for ``asibench run --mcp-config``.

It never installs anything into the ASI-Bench environment, never touches
operator credentials and never runs business tool calls; use the matching
``smoke_<id>.py`` afterwards.

Usage::

    python3 scripts/mcp/e2e/setup.py pyscf [--root ~/mcp]
    python3 scripts/mcp/e2e/setup.py arxiv [--root ~/mcp]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
# Placeholder in launch args for the absolute checkout path, e.g. "{checkout}/main.py".
CHECKOUT = "{checkout}"


class SetupError(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise SetupError(f"Unsupported manifest schema: {document.get('schema_version')!r}")
    servers: dict[str, dict] = {}
    for entry in document["servers"]:
        sid = entry["id"]
        if sid in servers:
            raise SetupError(f"Duplicate manifest id: {sid}")
        if len(entry["revision"]) != 40:
            raise SetupError(f"{sid}: revision must be a full 40-char commit SHA")
        if entry["install"] != "uv-sync-frozen":
            raise SetupError(f"{sid}: unsupported install mode {entry['install']!r}")
        launch = entry["launch"]
        if Path(launch["command"]).is_absolute() or any(Path(a).is_absolute() for a in launch["args"]):
            raise SetupError(f"{sid}: launch paths must be relative to the checkout")
        if any(CHECKOUT in a and not a.startswith(CHECKOUT + "/") for a in launch["args"]):
            raise SetupError(f"{sid}: {CHECKOUT} may only prefix a path argument")
        env = launch.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise SetupError(f"{sid}: launch.env must map strings to strings")
        sync_args = entry.get("uv_sync_args", [])
        if not isinstance(sync_args, list) or not all(isinstance(a, str) and a.startswith("--") for a in sync_args) \
                or any(a.split("=", 1)[0] in {"--python", "--frozen"} for a in sync_args):
            raise SetupError(f"{sid}: uv_sync_args must be extra --flags (not --python/--frozen)")
        servers[sid] = entry
    return servers


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SetupError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}{proc.stderr}"
        )
    return proc.stdout.strip()


def git(checkout: Path, *args: str) -> str:
    return run(["git", "-C", str(checkout), *args])


def ensure_checkout(entry: dict, dest: Path) -> None:
    revision = entry["revision"]
    if dest.exists():
        if not (dest / ".git").is_dir():
            raise SetupError(f"{dest} exists but is not a git checkout; choose another --root")
        remote = git(dest, "remote", "get-url", "origin")
        if remote.rstrip("/").removesuffix(".git") != entry["repository"].rstrip("/").removesuffix(".git"):
            raise SetupError(f"{dest} points at {remote}, expected {entry['repository']}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", "--filter=blob:none", entry["repository"], str(dest)])
    # Refuse to silently discard local edits to tracked files.
    dirty = git(dest, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SetupError(f"{dest} has modified tracked files; inspect before reusing:\n{dirty}")
    if git(dest, "rev-parse", "HEAD") != revision:
        git(dest, "fetch", "--quiet", "origin")
        git(dest, "checkout", "--quiet", "--detach", revision)
    head = git(dest, "rev-parse", "HEAD")
    if head != revision:
        raise SetupError(f"{dest} HEAD {head} != pinned {revision}")


def build_env(entry: dict, dest: Path) -> Path:
    if shutil.which("uv") is None:
        raise SetupError("uv not found on PATH; install it first (https://docs.astral.sh/uv/)")
    env = {k: v for k, v in os.environ.items() if k not in {"UV_PYTHON", "VIRTUAL_ENV", "PYTHONPATH"}}
    run(["uv", "sync", "--frozen", *entry.get("uv_sync_args", []), "--python", entry["python"]], cwd=dest, env=env)
    python = dest / ".venv" / "bin" / "python"
    version = run([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
    if version != entry["python"]:
        raise SetupError(f"{entry['id']}: venv Python {version} != manifest {entry['python']}")
    return python


def render_config(entry: dict, dest: Path) -> dict:
    launch = entry["launch"]
    server = {
        "command": str(dest / launch["command"]),
        "args": [arg.replace(CHECKOUT, str(dest)) for arg in launch["args"]],
        "cwd": str(dest),
    }
    if launch.get("env"):
        server["env"] = dict(launch["env"])
    return {"mcpServers": {entry["id"]: server}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server", help="manifest id, e.g. pyscf")
    parser.add_argument("--root", default=os.environ.get("MCP_E2E_ROOT", "~/mcp"),
                        help="install root (default: $MCP_E2E_ROOT or ~/mcp)")
    args = parser.parse_args(argv)

    try:
        servers = load_manifest()
        if args.server not in servers:
            raise SetupError(f"Unknown server {args.server!r}; known: {', '.join(sorted(servers))}")
        entry = servers[args.server]
        root = Path(args.root).expanduser().resolve()
        dest = root / entry["id"]
        ensure_checkout(entry, dest)
        python = build_env(entry, dest)
        config_path = root / f"{entry['id']}.mcp.json"
        config_path.write_text(json.dumps(render_config(entry, dest), indent=2) + "\n", encoding="utf-8")
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nOK  {entry['id']} @ {entry['revision'][:12]} -> {dest}")
    print(f"    MCP config: {config_path}")
    if entry.get("smoke"):
        print("    Next, run the direct tools/call smoke test:")
        print(f"    {python} {HERE / entry['smoke']} --config {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
