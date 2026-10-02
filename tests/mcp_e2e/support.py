"""Shared helpers for the MCP E2E tests in ``tests/mcp_e2e/``.

One copy of everything the test modules used to duplicate:

* the modules under test, loaded once (:data:`verify`, :data:`setup`,
  :data:`runner`, :data:`client`, :func:`smoke_module`, :meth:`Task.module`);
* agent logs: :class:`claude` (Claude Code stream-json) and :class:`codex`
  (``codex exec --json``) event builders, :func:`jsonl`;
* :func:`persist_like_run` and :meth:`Task.verify`, which lay out a run
  directory as ``asibench run`` does and call ``verify.verify_one``;
* scorer inputs: :func:`score_dirs`, :meth:`Task.eval_config`, :meth:`Task.total`;
* smoke stand-ins: :class:`StubClient` and the ``rpc_*`` tools/call responses.

``verify.verify_one`` is always looked up on the module at call time, so the
golden snapshot plugin (``tests/mcp_e2e/golden.py``) can wrap it.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
E2E_TASKS = ROOT / "examples/mcp-e2e-tasks"
BUNDLE = ROOT / "scripts/mcp/e2e"
SEED = 31415


def load(path: Path, name: str):
    """Import a file as module ``name`` (registered in ``sys.modules`` first, which
    dataclasses need on Python 3.14); a module already loaded under that name is reused."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

verify = load(BUNDLE / "verify_run.py", "mcp_e2e_verify_run")
setup = load(BUNDLE / "setup.py", "mcp_e2e_setup")      # the name e2e_smoke.runner.load_setup uses
runner = importlib.import_module("e2e_smoke.runner")
client = importlib.import_module("e2e_smoke.client")


def smoke_module(server: str):
    return importlib.import_module(f"e2e_smoke.servers.{server}")


def statuses(row: dict) -> dict[str, str]:
    """Check name -> status of a verify_one row."""
    return {name: check["status"] for name, check in row["checks"].items()}


def jsonl(events) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


# --------------------------------------------------------------------------
# Agent logs
# --------------------------------------------------------------------------

class claude:
    """Claude Code ``--output-format stream-json`` events."""

    @staticmethod
    def init(server: str, tools: list[str], status: str = "connected") -> dict:
        return {"type": "system", "subtype": "init", "mcp_servers": [{"name": server, "status": status}],
                "tools": tools}

    @staticmethod
    def tool_use(cid: str, name: str, arguments: dict) -> dict:
        return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": cid, "name": name,
                                                              "input": arguments}]}}

    @staticmethod
    def tool_result(cid: str, content, is_error: bool = False) -> dict:
        """``content``: a text result (str) or a list of content blocks."""
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": cid,
                                                         "is_error": is_error, "content": blocks}]}}

    @classmethod
    def call(cls, cid: str, name: str, arguments: dict, content, is_error: bool = False) -> list[dict]:
        return [cls.tool_use(cid, name, arguments), cls.tool_result(cid, content, is_error)]

    @staticmethod
    def result(num_turns: int) -> dict:
        return {"type": "result", "subtype": "success", "is_error": False, "num_turns": num_turns}


class codex:
    """``codex exec --json`` events (shapes as emitted by codex-cli 0.159.2)."""

    START = ({"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"})

    @staticmethod
    def mcp(item_id: str, server: str, tool: str, arguments, content=None, *, error: str | None = None,
            structured=None) -> list[dict]:
        """An ``mcp_tool_call`` item: started, then completed with ``content`` blocks (or a str
        for one text block) or failed with ``error``."""
        base = {"id": item_id, "type": "mcp_tool_call", "server": server, "tool": tool, "arguments": arguments}
        started = {"type": "item.started", "item": {**base, "result": None, "error": None, "status": "in_progress"}}
        if error is not None:
            done = {**base, "result": None, "error": {"message": error}, "status": "failed"}
        else:
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            done = {**base, "result": {"content": blocks, "structured_content": structured}, "error": None,
                    "status": "completed"}
        return [started, {"type": "item.completed", "item": done}]

    @staticmethod
    def shell(item_id: str, command: str, output: str = "", exit_code: int = 0) -> list[dict]:
        base = {"id": item_id, "type": "command_execution", "command": command}
        return [{"type": "item.started", "item": {**base, "aggregated_output": "", "exit_code": None,
                                                  "status": "in_progress"}},
                {"type": "item.completed", "item": {**base, "aggregated_output": output, "exit_code": exit_code,
                                                    "status": "completed"}}]

    @staticmethod
    def item(item: dict) -> dict:
        return {"type": "item.completed", "item": item}

    @staticmethod
    def done(usage: dict | None = None) -> dict:
        return {"type": "turn.completed", "usage": usage or {}}


# --------------------------------------------------------------------------
# Run directories as `asibench run` writes them
# --------------------------------------------------------------------------

def _orchestrator(tmp_path: Path | None):
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator
    orchestrator = object.__new__(BenchmarkOrchestrator)
    if tmp_path is not None:
        orchestrator.output_dir, orchestrator.repo_root = tmp_path / "out", ROOT
    return orchestrator


def persist_like_run(stream: str, harness: str = "claude", *, instance_id: str = "inst",
                     tmp_path: Path | None = None) -> tuple[str, list[dict]]:
    """(persisted raw stdout, trajectory steps) for an agent log, through the same code
    ``asibench run`` uses: the orchestrator's ``_sanitize_raw_artifact_text`` redacts user
    events (Claude tool results) and replaces absolute host paths with ``<abs_path>``;
    the trajectory is extracted from the unsanitized log."""
    if harness == "codex":
        from ai4sci_bench.trajectory.codex_extractor import extract_from_jsonl
    else:
        from ai4sci_bench.trajectory.claude_extractor import extract_from_jsonl
    base = tmp_path or Path(tempfile.gettempdir()) / "mcp-e2e"
    persisted = _orchestrator(base)._sanitize_raw_artifact_text(stream, raw_format="jsonl", workspace=base / "ws")
    return persisted, [step.to_dict() for step in extract_from_jsonl(stream, instance_id).steps]


@dataclass(frozen=True)
class Task:
    """One MCP E2E fake task in ``examples/mcp-e2e-tasks``."""
    task_id: str                       # e.g. "mcp_e2e.psi4_opt_freq"

    @property
    def dir(self) -> Path:
        return E2E_TASKS / Path(*self.task_id.split("."))

    @property
    def name(self) -> str:
        return self.task_id.split(".")[-1]

    @property
    def instance_id(self) -> str:
        return f"{self.task_id}__seed{SEED}"

    def module(self, stem: str):
        """``generate_gt`` / ``custom_scorer`` of the task, loaded once."""
        return load(self.dir / f"{stem}.py", f"mcp_e2e_{self.name}_{stem}")

    def eval_config(self) -> dict:
        import yaml
        return yaml.safe_load((self.dir / "task_eval.yaml").read_text())["evaluation"]

    def total(self, pred: Path, ref: Path) -> float:
        """Sum of all scoring items (not gates) of task_eval.yaml."""
        from ai4sci_bench.core.scorer import get_scorer
        return sum(get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]}).score
                   for item in self.eval_config()["scoring"])

    def verify(self, tmp_path: Path, stream: str, *, reference: dict, answer: Any, harness: str = "claude",
               persist: bool = True, raw_stdout: bool = True, trajectory: bool = True,
               files: dict[str, str] | None = None,
               edit_trajectory: Callable[[list[dict]], None] | None = None) -> dict:
        """Lay out one run result as ``asibench run`` does and return ``verify_one``'s row.

        ``persist``: the log goes through :func:`persist_like_run` and a trajectory is
        written; otherwise the raw log is written and there is no trajectory.
        ``raw_stdout=False`` keeps only the trajectory, ``trajectory=False`` only the persisted
        log. ``answer=None`` writes no result.json.
        """
        results, instances = tmp_path / "out", tmp_path / "instances"
        instance_id = self.instance_id
        task_out = results / self.task_id
        outputs = task_out / f"{instance_id}__b1.outputs"
        outputs.mkdir(parents=True)
        ref = instances / instance_id / "reference"
        ref.mkdir(parents=True)
        ref.joinpath("reference.json").write_text(json.dumps(reference))
        if answer is not None:
            outputs.joinpath("result.json").write_text(json.dumps(answer))
        for name, text in (files or {}).items():
            outputs.joinpath(name).write_text(text)
        agent: dict = {"persisted_outputs": {"dir": outputs.name}}
        stdout = f"{instance_id}__b1.agent_stdout.jsonl"
        if persist:
            persisted, steps = persist_like_run(stream, harness, instance_id=instance_id if harness == "claude"
                                                else "inst", tmp_path=tmp_path)
            if edit_trajectory:
                edit_trajectory(steps)
            if trajectory:
                traj = f"{instance_id}__b1.trajectory.json"
                task_out.joinpath(traj).write_text(json.dumps(steps))
                agent["trajectory_file"] = traj
            stream = persisted
        if raw_stdout:
            task_out.joinpath(stdout).write_text(stream)
            agent["raw_stdout_file"] = stdout
        result = {"task_id": self.task_id, "instance_id": instance_id, "prompt_level": "b1",
                  "status": "completed", "agent_output": agent}
        path = task_out / f"{instance_id}__b1.json"
        path.write_text(json.dumps(result))
        return verify.verify_one(path, result, instances, E2E_TASKS)


def all_tasks() -> list[Task]:
    return [Task(".".join(p.parent.relative_to(E2E_TASKS).parts))
            for p in sorted(E2E_TASKS.glob("*/*/task_meta.yaml"))]


def score_dirs(tmp_path: Path, prediction, reference) -> tuple[Path, Path]:
    """pred/ result.json and ref/ reference.json for a scorer; None leaves the file out,
    a str prediction is written verbatim (e.g. "not json")."""
    pred, ref = tmp_path / "pred", tmp_path / "ref"
    pred.mkdir(parents=True)
    ref.mkdir(parents=True)
    if prediction is not None:
        (pred / "result.json").write_text(prediction if isinstance(prediction, str) else json.dumps(prediction))
    if reference is not None:
        (ref / "reference.json").write_text(json.dumps(reference))
    return pred, ref


# --------------------------------------------------------------------------
# Smoke stand-ins
# --------------------------------------------------------------------------

class StubClient:
    """Stands in for StdioMCP: canned tools/call responses, optional stdout chatter."""

    def __init__(self, responses, chatter=0):
        self.responses = responses
        self.chatter = chatter
        self.non_json_stdout = []
        self.calls = []

    def call_tool(self, name, arguments, timeout=300.0):
        self.calls.append((name, arguments))
        self.non_json_stdout.extend(["noise"] * self.chatter)
        response = self.responses[name]
        return response(arguments) if callable(response) else response


def rpc_text(text, is_error=False):
    """A tools/call JSON-RPC response with one text block."""
    return {"result": {"content": [{"type": "text", "text": text}], "isError": is_error}}


def rpc_json(payload, is_error=False):
    return rpc_text(json.dumps(payload), is_error)


def rpc_image(data, mime="image/png"):
    return {"result": {"content": [{"type": "image", "data": data, "mimeType": mime}], "isError": False}}


def report_statuses(report) -> dict[str, str]:
    """Check name -> status of a smoke Report."""
    return {c["name"]: c["status"] for c in report.checks}


def png_bytes(width, height) -> bytes:
    """The PNG signature and IHDR chunk of a width x height image."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
            + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr)))
