"""MCP E2E run verifier (L2), stdlib only; ``verify_run.py`` is the CLI.

  spec        e2e_check.json -> typed, strictly validated Spec
  extractors  named readers for non-numeric results (arXiv IDs, term counts)
  values      Selector reads and the shared comparators
  evidence    harness log parsers (Claude stream-json, Codex JSONL, trajectory)
  checks      the six checks, verify_one, iter_results
"""
from .checks import CHECK_ORDER, CHECKS, Context, iter_results, load_spec, verify_one
from .evidence import Evidence, ToolCall, load_evidence, parse_claude_stream, parse_codex_stream, parse_trajectory
from .spec import Spec, SpecError, normalize_spec, parse_spec

__all__ = [
    "CHECK_ORDER", "CHECKS", "Context", "Evidence", "Spec", "SpecError", "ToolCall",
    "iter_results", "load_evidence", "load_spec", "normalize_spec", "parse_claude_stream",
    "parse_codex_stream", "parse_spec", "parse_trajectory", "verify_one",
]
