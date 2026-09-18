import hashlib
import json
import os
import sys
from pathlib import Path


def send(x):
    print(json.dumps(x), flush=True)


session_cwd = None
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params", {})
    if method == "initialize":
        result = {
            "protocolVersion": 1,
            "agentInfo": {"name": "asi-contract-dummy", "version": "1"},
            "agentCapabilities": {},
        }
    elif method == "session/new":
        session_cwd = params.get("cwd")
        result = {"sessionId": "asi-dummy"}
    elif method == "session/prompt":
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "asi-dummy",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "write-probe",
                        "title": "Inspect workspace and write dummy artifacts",
                        "kind": "execute",
                        "status": "in_progress",
                    },
                },
            }
        )
        before = [
            str(p.relative_to("/workspace"))
            for p in Path("/workspace").rglob("*")
            if p.is_file()
        ]
        forbidden = [
            "/verifier",
            "/reference",
            "/task_manifest.json",
            "/workspace/verifier",
            "/workspace/reference",
            "/workspace/task_manifest.json",
            "/tests",
        ]
        report = {
            "process_cwd": os.getcwd(),
            "session_cwd": session_cwd,
            "uid": os.getuid(),
            "workspace_before": sorted(before),
            "root_entries": sorted(p.name for p in Path("/").iterdir()),
            "forbidden_exists": {p: Path(p).exists() for p in forbidden},
            "prompt_blocks": params.get("prompt"),
        }
        for name, body in [
            ("analysis.py", "# dummy contract probe; not a scientific solution\n"),
            ("certificate.json", '{"dummy":true}\n'),
        ]:
            Path(name).write_text(body)
        report["net_devices"] = Path("/proc/net/dev").read_text()
        report["routes"] = Path("/proc/net/route").read_text()
        report["outputs"] = {
            name: {
                "absolute": str(Path(name).resolve()),
                "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest(),
            }
            for name in ("analysis.py", "certificate.json")
        }
        try:
            report["instruction_equals_prompt_bytes"] = (
                Path("/instruction.md").read_bytes()
                == Path("/workspace/prompt.md").read_bytes()
            )
        except OSError as e:
            report["instruction_read_error"] = str(e)
        report["output_dir_exists"] = Path("/workspace/output").exists()
        report["artifacts_mount_exists"] = Path("/logs/artifacts").is_dir()
        log = Path("/logs/agent/asi-probe.json")
        try:
            log.write_text(json.dumps(report, indent=2))
            report["log_write"] = "ok"
        except Exception as e:
            report["log_write"] = str(e)
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "asi-dummy",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": json.dumps(report)},
                    },
                },
            }
        )
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "asi-dummy",
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "write-probe",
                        "status": "completed",
                    },
                },
            }
        )
        result = {"stopReason": "end_turn"}
    elif method in ("session/set_model", "session/set_config_option"):
        result = {}
    elif rid is None:
        continue
    else:
        send(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32601, "message": "unsupported"},
            }
        )
        continue
    send({"jsonrpc": "2.0", "id": rid, "result": result})
