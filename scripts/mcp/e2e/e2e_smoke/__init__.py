"""MCP E2E smoke (L0/L1), stdlib only apart from each server's reference libraries;
``smoke.py <id>`` is the CLI.

  client    minimal stdio MCP client that records non-JSON stdout lines
  runner    the shared run (handshake, generic checks, report), Report, Caller, Smoke, Session
  helpers   value helpers shared by several servers
  servers   one module per manifest id, defining SMOKE (references + server-specific checks)
"""
