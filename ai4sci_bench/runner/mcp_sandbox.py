"""Per-run MCP configuration staging. No host binaries or directories are mapped."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
import tomllib
from urllib.parse import urlsplit

from ai4sci_bench.mcp_config import codex_mcp_toml

CONTAINER_ROOT = "/opt/asibench-mcp"
PYTHON = "/opt/venv/bin/python"


def validate_os_mcp(servers: dict, metadata: dict | None = None) -> None:
    """Require explicit container deployment; host configs never become mounts."""
    for name, entry in servers.items():
        spec = entry.get("sandbox")
        if not isinstance(spec, dict) or spec.get("deployment") not in {"instance", "service"}:
            raise ValueError(f"MCP {name!r}: --sandbox os requires explicit sandbox.deployment (instance or service)")
        if set(spec) - {"deployment", "service", "startup_timeout_seconds", "env_from"}:
            raise ValueError(f"MCP {name!r}: unsupported sandbox fields")
        refs = spec.get("env_from", {})
        if not isinstance(refs, dict) or any(
            not isinstance(k, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k)
            or not isinstance(v, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", v)
            or k in {"HOME", "TMPDIR", "PATH", "PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH"}
            for k, v in refs.items()
        ):
            raise ValueError(f"MCP {name!r}: env_from must map safe environment names to host environment names")
        timeout = spec.get("startup_timeout_seconds", 30)
        if type(timeout) is not int or not 1 <= timeout <= 300:
            raise ValueError(f"MCP {name!r}: startup_timeout_seconds must be 1..300")
        if spec["deployment"] == "instance":
            if "command" not in entry or "service" in spec:
                raise ValueError(f"MCP {name!r}: instance deployment requires stdio command")
            # All paths are container paths; no host path rewriting or mounts.
            command = entry["command"]
            if not command.startswith(("/opt/", "/usr/", "/bin/")) or ".." in Path(command).parts:
                raise ValueError(f"MCP {name!r}: command must be an absolute image path under /opt, /usr or /bin")
            cwd = entry.get("cwd", "/workspace")
            if not (cwd == "/workspace" or cwd.startswith(("/workspace/", "/opt/", "/usr/"))) or ".." in Path(cwd).parts:
                raise ValueError(f"MCP {name!r}: cwd must be a container workspace/image path")
        else:
            service = spec.get("service")
            url = urlsplit(entry.get("url", ""))
            if url.port is None:
                raise ValueError(f"MCP {name!r}: service URL requires an explicit port")
            if not isinstance(service, str) or not service or url.scheme != "http" or url.hostname != service or url.username or url.password or url.fragment:
                raise ValueError(f"MCP {name!r}: service requires http://<service>:<port>/ endpoint")
            if entry.get("headers") or entry.get("env") or refs:
                raise ValueError(f"MCP {name!r}: service MCP authentication/environment overrides are unsupported; use an internal unauthenticated endpoint")
            if metadata is not None:
                from ai4sci_bench.runner.os_sandbox import OSSandbox
                declared = {s["name"]: s for s in OSSandbox._task_services(metadata)}
                if service not in declared or url.port != declared[service]["port"]:
                    raise ValueError(f"MCP {name!r}: endpoint must match declared runtime.services name and port")


def mcp_provenance(servers: dict) -> dict:
    """Record identity and field names, never config values or secrets."""
    return {
        name: {
            "deployment": entry.get("sandbox", {}).get("deployment", "host"),
            "transport": "stdio" if "command" in entry else "http",
            "config_sha256": hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
            "environment_names": sorted(set(entry.get("env", {})) | set(entry.get("sandbox", {}).get("env_from", {}))),
            "network": "internal-service" if entry.get("sandbox", {}).get("deployment") == "service" else "inherits-instance-policy",
        }
        for name, entry in servers.items()
    }


@contextmanager
def stage_os_mcp(servers: dict, metadata: dict, agent_type: str, cmd: list[str], extra_env: dict | None):
    """Yield command/env/mounts; clean staging even on timeout or exceptions."""
    if not servers:
        yield cmd, extra_env, []
        return
    validate_os_mcp(servers, metadata)
    with tempfile.TemporaryDirectory(prefix="asibench-mcp-") as directory:
        root = Path(directory)
        shutil.copyfile(Path(__file__).with_name("mcp_container.py"), root / "runtime.py")
        resolved = json.loads(json.dumps(servers))
        for name, entry in resolved.items():
            for target, source in entry["sandbox"].get("env_from", {}).items():
                if source not in os.environ:
                    raise ValueError(f"MCP {name!r}: credential environment reference {source!r} is unset")
                entry.setdefault("env", {})[target] = os.environ[source]
        (root / "source.json").write_text(json.dumps(resolved))
        (root / "source.json").chmod(0o600)
        rendered = {}
        for name, entry in servers.items():
            if "command" in entry:
                rendered[name] = {"command": PYTHON, "args": [f"{CONTAINER_ROOT}/runtime.py", "server", name]}
            else:
                rendered[name] = {k: v for k, v in entry.items() if k != "sandbox"}
        (root / "mcp.json").write_text(json.dumps({"mcpServers": rendered}))
        env = dict(extra_env or {})
        command = list(cmd)
        if agent_type == "claude_code":
            index = command.index("--mcp-config")
            command[index + 1] = f"{CONTAINER_ROOT}/mcp.json"
        elif agent_type == "codex":
            # Preserve only generated provider routing, never ambient MCP config.
            # Native-provider mode supplies a freshly generated CODEX_HOME.
            config = ""
            source_home = env.pop("CODEX_HOME", None)
            if source_home:
                parsed = tomllib.loads((Path(source_home) / "config.toml").read_text())
                if set(parsed) - {"model_provider", "model_providers"}:
                    raise ValueError("OS MCP accepts only generated Codex provider configuration; ambient CODEX_HOME is unsupported")
                config = (Path(source_home) / "config.toml").read_text()
            (root / "config.toml").write_text(config)
            (root / "config.toml").chmod(0o600)
            # CLI -c overrides only our exact server tables; standard auth stays
            # in the sandbox's isolated ~/.codex, not a host config directory.
            tables = tomllib.loads(codex_mcp_toml(rendered))
            for name, entry in tables.get("mcp_servers", {}).items():
                entry["required"] = True
                for key, value in entry.items():
                    # JSON literals cover the values used here (no null).
                    if isinstance(value, dict):
                        literal = "{ " + ", ".join(json.dumps(k) + " = " + json.dumps(v) for k, v in value.items()) + " }"
                    else:
                        literal = json.dumps(value)
                    command[2:2] = ["-c", f"mcp_servers.{name}.{key}={literal}"]
        else:
            raise ValueError("OS MCP supports only Claude/Codex")
        # Staging is private and not persisted as task output.
        mounts = ["-v", f"{root}:{CONTAINER_ROOT}:ro"]
        if agent_type == "codex" and config:
            # The existing auth-file mount remains separate from this config.
            mounts += ["-v", f"{root / 'config.toml'}:/home/agent/.codex/config.toml:ro"]
        command = [PYTHON, f"{CONTAINER_ROOT}/runtime.py", "preflight", "--", *command]
        yield command, env, mounts
