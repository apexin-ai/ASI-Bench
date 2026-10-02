#!/usr/bin/env python3
"""Direct tools/call smoke test (L0/L1) of one pinned MCP server, outside any agent.

Run it with the server's own interpreter (its reference computations need the
server's libraries), after ``setup.py <id>`` wrote ``<root>/<id>.mcp.json``::

    ~/mcp/<id>/.venv/bin/python scripts/mcp/e2e/smoke.py <id> --config ~/mcp/<id>.mcp.json

``smoke.py <id> --help`` lists the server's extra options. The checks of
server ``<id>`` are documented in ``e2e_smoke/servers/<id>.py``; the shared
L0/L1 checks in ``e2e_smoke/runner.py``. Exit code 1 if any check FAILs.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e2e_smoke import runner  # noqa: E402


def load(server: str):
    """The ``SMOKE`` declaration of ``e2e_smoke/servers/<server>.py``."""
    if not (HERE / "e2e_smoke" / "servers" / f"{server}.py").is_file() or not server.isidentifier():
        known = sorted(p.stem for p in (HERE / "e2e_smoke" / "servers").glob("[a-z]*.py"))
        raise SystemExit(f"no smoke for {server!r}; known: {', '.join(known)}")
    return importlib.import_module(f"e2e_smoke.servers.{server}").SMOKE


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    return runner.main(load(argv[0]), argv[1:])


if __name__ == "__main__":
    sys.exit(main())
