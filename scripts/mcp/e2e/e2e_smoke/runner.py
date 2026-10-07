"""The shared smoke runner: everything a smoke run does that is not specific to one server.

A server module (``servers/<id>.py``) declares a :class:`Smoke` named
``SMOKE``; :func:`main` then

* L0: checks that the checkout is at the manifest revision and that the
  ``--config`` entry is exactly what ``setup.py`` renders from the manifest;
* starts the server with a minimal environment (:func:`server_env`) in a
  temporary HOME / cwd / TMPDIR, runs ``initialize`` and ``tools/list``
  (expected tools, stable across calls);
* L1: runs the server's ``run_l1`` and then the generic checks: an unknown
  tool is an error, the server is still alive, stdout carried only JSON-RPC,
  the cwd is untouched (apart from declared artefacts);
* runs the server's ``after`` hook (short-lived extra servers, see
  :meth:`Session.spawn`) and writes the JSON report.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .client import MCPError, StdioMCP, text_of

E2E_DIR = Path(__file__).resolve().parents[1]
PROXY_ENV = ("http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE")


# --------------------------------------------------------------------------
# Report and tool calls
# --------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, level: str, name: str, status: str, detail: str = "", **data) -> None:
        self.checks.append({"level": level, "name": name, "status": status, "detail": detail, **data})
        print(f"[{status:<4}] {level} {name}" + (f" — {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


def json_result(report: Report, check: str, result: dict | None, *, structured: bool = False):
    """Decoded JSON of a tools/call result; ``{"_isError": True, "_text": ...}`` for an
    isError result; None (with a FAIL) if the result is missing or not JSON.

    ``structured`` prefers a ``structuredContent`` object over the text block."""
    if result is None:
        return None
    if result.get("isError"):
        return {"_isError": True, "_text": text_of(result)}
    data = result.get("structuredContent") if structured else None
    if isinstance(data, dict):
        return data
    try:
        return json.loads(text_of(result))
    except json.JSONDecodeError:
        report.add("L1", check, "FAIL", f"result is not JSON: {text_of(result)[:200]!r}")
        return None


class Caller:
    """tools/call wrapper: records FAILs for transport/tool errors and stdout lines per tool.

    ``timeout`` (seconds) is for servers whose tools legitimately run longer than the
    client default — a slow host must not turn a correct result into a FAIL."""

    def __init__(self, client: StdioMCP, report: Report, *, timeout: float | None = None) -> None:
        self.client = client
        self.report = report
        self.timeout = timeout
        self.stdout_by_tool: Counter = Counter()

    def __call__(self, check: str, tool: str, arguments: dict, *, allow_error: bool = False,
                 timeout: float | None = None) -> dict | None:
        before = len(self.client.non_json_stdout)
        limit = timeout or self.timeout
        try:
            resp = self.client.call_tool(tool, arguments, **({"timeout": limit} if limit else {}))
        except MCPError as exc:
            self.report.add("L1", check, "FAIL", str(exc))
            return None
        finally:
            self.stdout_by_tool[tool] += len(self.client.non_json_stdout) - before
        if "error" in resp:
            self.report.add("L1", check, "FAIL", f"JSON-RPC error: {resp['error']}")
            return None
        result = resp["result"]
        if result.get("isError") and not allow_error:
            self.report.add("L1", check, "FAIL", f"tool error: {text_of(result)[:300]}")
            return None
        return result

    def json(self, check: str, tool: str, arguments: dict, *, allow_error: bool = False,
             structured: bool = False, timeout: float | None = None):
        """tools/call + :func:`json_result`."""
        return json_result(self.report, check,
                           self(check, tool, arguments, allow_error=allow_error, timeout=timeout),
                           structured=structured)


def check_rejected(call: Caller, name: str, tool: str, arguments: dict, *,
                   in_band: Callable[[dict], str | None] | None = None,
                   on_accept: Callable[[dict], tuple] | None = None) -> dict | None:
    """Invalid input must give isError=true (PASS). ``in_band(result)`` returns the message of
    an error reported in a normal result (WARN); anything else was accepted: FAIL, or
    ``on_accept(result) -> (status, detail[, data])``. Returns the result."""
    result = call(name, tool, arguments, allow_error=True)
    if result is None:
        return None
    text = text_of(result)
    if result.get("isError"):
        call.report.add("L1", name, "PASS", f"isError=true: {text[:120]!r}")
        return result
    message = in_band(result) if in_band else None
    if message:
        call.report.add("L1", name, "WARN", f"error returned in-band with isError=false: {message[:160]}")
    elif on_accept is not None:
        status, detail, *data = on_accept(result)
        call.report.add("L1", name, status, detail, **(data[0] if data else {}))
    else:
        call.report.add("L1", name, "FAIL", f"invalid input accepted: {text[:200]!r}")
    return result


def check_unknown_tool(client: StdioMCP, report: Report) -> None:
    try:
        response = client.call_tool("__nonexistent__", {})
        ok = "error" in response or response.get("result", {}).get("isError") is True
        report.add("L1", "unknown tool is an error", "PASS" if ok else "FAIL", "" if ok else f"got {response}")
    except MCPError as exc:
        report.add("L1", "unknown tool is an error", "FAIL", str(exc))


def package_versions(packages: tuple[str, ...]) -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in packages:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


# --------------------------------------------------------------------------
# Server processes
# --------------------------------------------------------------------------

def server_env(server: dict, home: Path, tmpdir: Path, *, extra: dict | None = None,
               pass_proxies: bool = False) -> dict:
    """Minimal server environment: no operator secrets or caller Python settings; the
    config's launch env is applied last. PATH is the launch command's directory, then
    ``setup.HOST_PATH``, where ``host_requirements.executables`` are checked."""
    venv_bin = str(Path(server["command"]).parent)
    env = {"HOME": str(home), "TMPDIR": str(tmpdir), "PATH": f"{venv_bin}:{load_setup().HOST_PATH}",
           "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8", **(extra or {})}
    if pass_proxies:
        env.update({key: os.environ[key] for key in PROXY_ENV if os.environ.get(key)})
    env.update(server.get("env", {}))
    return env


@dataclass
class Smoke:
    """What one server's smoke adds to the shared run."""
    server: str                                   # manifest id
    run_l1: Callable[["Session"], None]           # the server's L1 checks, on session.call
    packages: tuple[str, ...] = ()                # versions recorded in the report
    call_timeout: float | None = None             # tools/call timeout for slow tools (seconds)
    extra_env: dict[str, str] = field(default_factory=dict)
    pass_proxies: bool = False                    # network servers: pass the caller's proxy settings
    expected_cwd_files: tuple[str, ...] = ()      # artefacts the tools are expected to write into cwd
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None
    prepare: Callable[["Session"], None] | None = None   # before the server starts (extra L0, references)
    after: Callable[["Session"], None] | None = None     # after it stopped (e.g. short-lived probe servers)
    report_fields: Callable[["Session"], dict] | None = None


@dataclass
class Session:
    smoke: Smoke
    args: argparse.Namespace
    entry: dict                 # manifest entry
    server: dict                # --config entry
    checkout: Path
    report: Report
    tmp: Path
    home: Path
    cwd: Path
    scratch: Path               # TMPDIR
    env: dict
    client: StdioMCP | None = None
    call: Caller | None = None
    state: dict = field(default_factory=dict)   # per-server data shared between hooks

    @contextlib.contextmanager
    def spawn(self, name: str, *, env: dict | None = None,
              prepare_cwd: Callable[[Path], None] | None = None) -> Iterator[StdioMCP]:
        """A short-lived extra server with its own cwd ``<tmp>/<name>`` (e.g. a probe with a
        reduced environment); closed on exit."""
        cwd = self.tmp / name
        cwd.mkdir()
        if prepare_cwd:
            prepare_cwd(cwd)
        client = StdioMCP(self.server["command"], self.server["args"], cwd=cwd, env=env or self.env,
                          stderr_path=self.tmp / f"{name}.stderr.log")
        try:
            yield client
        finally:
            client.close()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def load_setup():
    """``setup.py`` as a module (manifest validation and config rendering)."""
    name = "mcp_e2e_setup"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, E2E_DIR / "setup.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def git_revision(checkout: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_problems(server: dict, expected: dict) -> list[str]:
    keys = sorted(set(server) | set(expected))
    return [f"{k}: {server.get(k)!r} (expected {expected.get(k)!r})" for k in keys if server.get(k) != expected.get(k)]


def parser_for(smoke: Smoke) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"smoke.py {smoke.server}")
    parser.add_argument("--config", required=True, help=f"generated <root>/{smoke.server}.mcp.json")
    parser.add_argument("--report", default=f"{smoke.server}-smoke-report.json")
    if smoke.add_arguments:
        smoke.add_arguments(parser)
    return parser


def _handshake(client: StdioMCP, report: Report, expected: list[str]) -> bool:
    try:
        info = client.initialize()
        report.add("L0", "initialize", "PASS", f"server={info.get('serverInfo')} protocol={info.get('protocolVersion')}")
        names = sorted(t["name"] for t in client.list_tools())
        report.add("L0", "tools/list", "PASS" if names == expected else "FAIL",
                   f"{len(names)} tools" + ("" if names == expected else f"; expected {expected}, got {names[:20]}"))
        again = sorted(t["name"] for t in client.list_tools())
        report.add("L0", "tools/list stable", "PASS" if again == names else "FAIL")
        return True
    except MCPError as exc:
        report.add("L0", "handshake", "FAIL", str(exc))
        return False


def main(smoke: Smoke, argv: list[str] | None = None) -> int:
    args = parser_for(smoke).parse_args(argv)
    setup = load_setup()
    entry = setup.load_manifest()[smoke.server]
    config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
    server = config["mcpServers"][smoke.server]
    checkout = Path(server["cwd"])
    revision = git_revision(checkout)

    report = Report()
    if revision != entry["revision"]:
        report.add("L0", "pinned revision", "FAIL", f"checkout at {revision}, manifest pins {entry['revision']}")
    problems = config_problems(server, setup.render_config(entry, checkout)["mcpServers"][smoke.server])
    report.add("L0", "config matches manifest", "FAIL" if problems else "PASS",
               "regenerate with setup.py; " + "; ".join(problems) if problems else "")

    stderr_tail = ""
    with tempfile.TemporaryDirectory(prefix=f"mcp-e2e-{smoke.server}-") as tmp:
        tmp_path = Path(tmp)
        home, cwd, scratch = tmp_path / "home", tmp_path / "cwd", tmp_path / "tmpdir"
        for d in (home, cwd, scratch):
            d.mkdir()
        env = server_env(server, home, scratch, extra=smoke.extra_env, pass_proxies=smoke.pass_proxies)
        session = Session(smoke, args, entry, server, checkout, report, tmp_path, home, cwd, scratch, env)
        if smoke.prepare:
            smoke.prepare(session)
        stderr_path = tmp_path / "server.stderr.log"
        client = session.client = StdioMCP(server["command"], server["args"], cwd=cwd, env=env,
                                           stderr_path=stderr_path)
        try:
            if _handshake(client, report, sorted(entry["expected_tools"])):
                session.call = Caller(client, report, timeout=smoke.call_timeout)
                smoke.run_l1(session)
                check_unknown_tool(client, report)
            alive = client.proc.poll() is None
            report.add("L1", "server alive after calls", "PASS" if alive else "FAIL",
                       "" if alive else f"exit code {client.proc.returncode}")
        finally:
            client.close()
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]

        polluted = client.non_json_stdout
        call = session.call
        per_tool = {k: v for k, v in sorted(call.stdout_by_tool.items()) if v} if call else {}
        report.add("L1", "stdout is pure JSON-RPC", "WARN" if polluted else "PASS",
                   f"{len(polluted)} non-JSON line(s), per tool {per_tool}, e.g. {polluted[:3]}" if polluted else "",
                   non_json_stdout=polluted[:50], non_json_stdout_by_tool=per_tool)
        leftovers = sorted(p.name for p in cwd.iterdir() if p.name not in smoke.expected_cwd_files)
        report.add("L1", "server leaves cwd untouched", "WARN" if leftovers else "PASS",
                   f"created {leftovers}" if leftovers else "")
        if smoke.after:
            smoke.after(session)

    document = {
        "server": smoke.server,
        "repository": entry["repository"],
        "revision": revision,
        **(smoke.report_fields(session) if smoke.report_fields else {}),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": package_versions(smoke.packages),
        "result": "FAIL" if report.failed else "PASS",
        "summary": dict(Counter(c["status"] for c in report.checks)),
        "checks": report.checks,
        "server_stderr_tail": stderr_tail,
    }
    Path(args.report).write_text(json.dumps(document, indent=2, ensure_ascii=False, default=str) + "\n",
                                 encoding="utf-8")
    print(f"\n{document['result']} {document['summary']}  report -> {Path(args.report).resolve()}")
    return 1 if report.failed else 0
