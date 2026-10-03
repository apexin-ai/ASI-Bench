"""Package-local fixtures for the MCP E2E tests.

The orchestrator's persistence sanitizer replaces ``Path.home()`` with ``<home>``
*before* it collapses the remaining absolute paths to ``<abs_path>``
(``_build_path_replacements``). The fake agent logs in these tests contain
literal host paths — ``/home/e2e/mcp/...`` is the AWS E2E user's — so when the
tests run as a user whose home is one of those prefixes, the same fixture is
scrubbed to ``<home>/mcp/...`` instead of ``<abs_path>`` and assertions about
the scrubbed form flip. Pin the home directory to a sentinel no fixture can live
under, so the outcome (and the golden snapshot) is the same for every runner.
"""
from __future__ import annotations

import pytest

SENTINEL_HOME = "/nonexistent/asibench-mcp-e2e-home"


@pytest.fixture(autouse=True)
def pinned_home(monkeypatch):
    """Make ``Path.home()`` deterministic for the whole package."""
    monkeypatch.setenv("HOME", SENTINEL_HOME)
    monkeypatch.setenv("USERPROFILE", SENTINEL_HOME)   # Path.home() on Windows
