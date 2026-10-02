#!/usr/bin/env python3
"""Install one pinned MCP server for end-to-end (E2E) testing.

Stdlib only. For the server ``<id>`` listed in ``manifest.json`` this:

1. clones the upstream repository into ``<root>/<id>`` (or reuses an existing
   clean checkout) and detaches at the pinned revision;
2. builds an isolated virtualenv with ``uv`` using the manifest's Python
   version (``UV_PYTHON`` from the caller's shell is ignored on purpose):
   ``uv sync --frozen`` against the upstream lockfile, or, for upstreams without
   one, ``uv pip install`` of exact manifest pins with ``--exclude-newer``; or,
   for servers that need conda-only packages (psi4), a ``micromamba`` prefix
   created from a committed per-platform ``@EXPLICIT`` lock (exact package
   URLs + SHA-256, no solver at install time);
3. writes a portable ``<root>/<id>.mcp.json`` for ``asibench run --mcp-config``
   (``{checkout}`` in launch args / env values becomes the absolute checkout).

Servers that vendor prebuilt native code declare ``host_requirements``
(machine, CPU flags, loadable system libraries); these are checked before
anything is cloned and are never installed by this script.

It never installs anything into the ASI-Bench environment, never touches
operator credentials and never runs business tool calls; use the matching
``smoke_<id>.py`` afterwards.

Usage::

    python3 scripts/mcp/e2e/setup.py pyscf [--root ~/mcp]
    python3 scripts/mcp/e2e/setup.py arxiv [--root ~/mcp]
    python3 scripts/mcp/e2e/setup.py jsbsim [--root ~/mcp]
    python3 scripts/mcp/e2e/setup.py s4 [--root ~/mcp]
    python3 scripts/mcp/e2e/setup.py psi4 [--root ~/mcp]

Conda locks are regenerated (maintainers only, needs network) with::

    python3 scripts/mcp/e2e/setup.py psi4 --lock [--root ~/mcp]
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
# Placeholder in launch args / env values for the absolute checkout path, e.g. "{checkout}/main.py".
CHECKOUT = "{checkout}"
# uv-sync-frozen: upstream ships pyproject.toml + uv.lock -> `uv sync --frozen`.
# uv-pip-pinned:  upstream has no lockfile -> fresh venv + `uv pip install` of the
#                 manifest's exact `name==version` pins, resolved with --exclude-newer.
# conda-explicit: conda-only dependencies -> `micromamba create --file <lock>` from a
#                 committed @EXPLICIT lock for the host's conda platform.
INSTALL_MODES = ("uv-sync-frozen", "uv-pip-pinned", "conda-explicit")
PIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9][A-Za-z0-9.+!_-]*")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
# Optional per-server host checks for upstreams that vendor prebuilt native code.
HOST_REQUIREMENT_KEYS = ("machine", "cpu_flags", "shared_libraries")
# conda-explicit: exact `name=version` specs, solved once per platform into a lock.
CONDA_SPEC_RE = re.compile(r"[a-z0-9][a-z0-9._-]*=[A-Za-z0-9][A-Za-z0-9._+]*")
CONDA_KEYS = ("channel", "specs", "locks")
CONDA_PLATFORMS = {("Linux", "x86_64"): "linux-64", ("Linux", "aarch64"): "linux-aarch64",
                   ("Darwin", "arm64"): "osx-arm64", ("Darwin", "x86_64"): "osx-64"}
CONDA_URL_RE = re.compile(r"https://conda\.anaconda\.org/(?P<channel>[a-z0-9-]+)/(?P<subdir>[a-z0-9-]+)/"
                          r"[A-Za-z0-9._+-]+\.(?:conda|tar\.bz2)#sha256:[0-9a-f]{64}")
# Solver view of the target hosts when locking (micromamba cannot detect glibc of
# a foreign platform). 2.28 is conda-forge's Linux baseline.
LOCK_GLIBC = "2.28"
# Environment variables that would redirect or reconfigure the installers.
_INSTALLER_ENV_DROP = ("UV_PYTHON", "VIRTUAL_ENV", "PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
                       "CONDA_PKGS_DIRS", "CONDA_ENVS_PATH", "CONDARC", "MAMBARC", "MAMBA_ROOT_PREFIX")


class SetupError(RuntimeError):
    pass


def _check_install_fields(sid: str, entry: dict) -> None:
    sync_args = entry.get("uv_sync_args", [])
    if not isinstance(sync_args, list) or not all(isinstance(a, str) and a.startswith("--") for a in sync_args) \
            or any(a.split("=", 1)[0] in {"--python", "--frozen"} for a in sync_args):
        raise SetupError(f"{sid}: uv_sync_args must be extra --flags (not --python/--frozen)")
    conda = entry["install"] == "conda-explicit"
    if not conda and "conda" in entry:
        raise SetupError(f"{sid}: conda only applies to install conda-explicit")
    if conda:
        if sync_args:
            raise SetupError(f"{sid}: uv_sync_args do not apply to install conda-explicit")
        _check_conda_fields(sid, entry)
    pinned = entry["install"] == "uv-pip-pinned"
    if not pinned:
        if "requirements" in entry or "exclude_newer" in entry:
            raise SetupError(f"{sid}: requirements/exclude_newer only apply to install uv-pip-pinned")
        return
    if sync_args:
        raise SetupError(f"{sid}: uv_sync_args do not apply to install uv-pip-pinned")
    requirements = entry.get("requirements")
    if not isinstance(requirements, list) or not requirements \
            or not all(isinstance(r, str) and PIN_RE.fullmatch(r) for r in requirements):
        raise SetupError(f"{sid}: uv-pip-pinned needs requirements as exact 'name==version' pins")
    if not isinstance(entry.get("exclude_newer"), str) or not TIMESTAMP_RE.fullmatch(entry["exclude_newer"]):
        raise SetupError(f"{sid}: uv-pip-pinned needs exclude_newer as YYYY-MM-DDTHH:MM:SSZ "
                         "(fixes the transitive resolution)")


def _check_conda_fields(sid: str, entry: dict) -> None:
    conda = entry.get("conda")
    if not isinstance(conda, dict) or set(conda) != set(CONDA_KEYS):
        raise SetupError(f"{sid}: conda-explicit needs conda with exactly {', '.join(CONDA_KEYS)}")
    if not isinstance(conda["channel"], str) or not re.fullmatch(r"[a-z0-9-]+", conda["channel"]):
        raise SetupError(f"{sid}: conda.channel must be a channel name such as conda-forge")
    specs = conda["specs"]
    if not isinstance(specs, list) or not specs or not all(isinstance(x, str) and CONDA_SPEC_RE.fullmatch(x)
                                                           for x in specs):
        raise SetupError(f"{sid}: conda.specs must be exact 'name=version' specs")
    if f"python={entry['python']}" not in specs and not any(x.startswith(f"python={entry['python']}.")
                                                             for x in specs):
        raise SetupError(f"{sid}: conda.specs must pin python to the manifest version {entry['python']}")
    locks = conda["locks"]
    if not isinstance(locks, dict) or not locks or set(locks) - set(CONDA_PLATFORMS.values()) \
            or not all(isinstance(v, str) and not Path(v).is_absolute() and ".." not in Path(v).parts
                       for v in locks.values()):
        raise SetupError(f"{sid}: conda.locks must map conda platforms "
                         f"({', '.join(sorted(CONDA_PLATFORMS.values()))}) to relative lock paths")


def conda_platform(system: str | None = None, machine: str | None = None) -> str:
    key = (system or platform.system(), machine or platform.machine())
    if key not in CONDA_PLATFORMS:
        raise SetupError(f"no conda platform for {key[0]}/{key[1]}")
    return CONDA_PLATFORMS[key]


def lock_header(entry: dict, conda_plat: str) -> list[str]:
    conda = entry["conda"]
    return [f"# server: {entry['id']}", f"# platform: {conda_plat}", f"# channel: {conda['channel']}",
            f"# specs: {' '.join(conda['specs'])}"]


def read_conda_lock(entry: dict, conda_plat: str, bundle: Path | None = None) -> Path:
    """Return the lock for ``conda_plat`` after checking it matches the manifest exactly."""
    bundle = bundle or HERE
    locks = entry["conda"]["locks"]
    if conda_plat not in locks:
        raise SetupError(f"{entry['id']}: no conda lock for platform {conda_plat} "
                         f"(available: {', '.join(sorted(locks))})")
    path = bundle / locks[conda_plat]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SetupError(f"{entry['id']}: cannot read conda lock {path}: {exc}") from None
    comments = [line for line in lines if line.startswith("#")]
    missing = [h for h in lock_header(entry, conda_plat) if h not in comments]
    if missing:
        raise SetupError(f"{entry['id']}: {path.name} is stale or foreign (header lacks {missing}); "
                         f"regenerate with: setup.py {entry['id']} --lock")
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    if not body or body[0] != "@EXPLICIT" or len(body) < 2:
        raise SetupError(f"{entry['id']}: {path.name} is not a conda @EXPLICIT lock")
    for line in body[1:]:
        match = CONDA_URL_RE.fullmatch(line)
        if not match or match["channel"] != entry["conda"]["channel"] \
                or match["subdir"] not in {conda_plat, "noarch"}:
            raise SetupError(f"{entry['id']}: {path.name}: not a {entry['conda']['channel']} "
                             f"{conda_plat}/noarch URL with #sha256: {line[:160]!r}")
    return path


def _check_host_requirements(sid: str, entry: dict) -> None:
    if "host_requirements" not in entry:
        return
    req = entry["host_requirements"]
    if not isinstance(req, dict) or not req or set(req) - set(HOST_REQUIREMENT_KEYS) - {"reason"}:
        raise SetupError(f"{sid}: host_requirements may only contain {', '.join(HOST_REQUIREMENT_KEYS)} and reason")
    for key in HOST_REQUIREMENT_KEYS:
        values = req.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) and v and v.strip() == v for v in values):
            raise SetupError(f"{sid}: host_requirements.{key} must be a list of non-empty strings")
    if not isinstance(req.get("reason", ""), str):
        raise SetupError(f"{sid}: host_requirements.reason must be a string")


def check_host(entry: dict, *, machine: str | None = None, cpuinfo: Path = Path("/proc/cpuinfo"),
               loader=ctypes.CDLL) -> None:
    """Fail fast when the host cannot run a server's prebuilt native code.

    Only checks; never installs system packages (that needs an administrator).
    """
    req = entry.get("host_requirements")
    if not req:
        return
    problems = []
    machine = machine or platform.machine()
    if req.get("machine") and machine not in req["machine"]:
        problems.append(f"machine {machine!r} is not one of {req['machine']}")
    if req.get("cpu_flags"):
        try:
            text = cpuinfo.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        flags = {f for line in text.splitlines() if line.split(":", 1)[0].strip() == "flags"
                 for f in line.split(":", 1)[1].split()}
        missing = [f for f in req["cpu_flags"] if f not in flags]
        if missing:
            problems.append(f"CPU lacks {missing} (read from {cpuinfo})")
    for lib in req.get("shared_libraries", []):
        try:
            loader(lib)
        except OSError as exc:
            problems.append(f"cannot load {lib}: {exc}")
    if problems:
        reason = f" ({req['reason']})" if req.get("reason") else ""
        raise SetupError(f"{entry['id']}: host requirements not met{reason}:\n  - " + "\n  - ".join(problems))


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
        if entry["install"] not in INSTALL_MODES:
            raise SetupError(f"{sid}: unsupported install mode {entry['install']!r}")
        launch = entry["launch"]
        if Path(launch["command"]).is_absolute() or any(Path(a).is_absolute() for a in launch["args"]):
            raise SetupError(f"{sid}: launch paths must be relative to the checkout")
        if any(CHECKOUT in a and not a.startswith(CHECKOUT + "/") for a in launch["args"]):
            raise SetupError(f"{sid}: {CHECKOUT} may only prefix a path argument")
        env = launch.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise SetupError(f"{sid}: launch.env must map strings to strings")
        if any(Path(v).is_absolute() for v in env.values()):
            raise SetupError(f"{sid}: launch.env paths must use {CHECKOUT}/..., not absolute paths")
        if any(CHECKOUT in v and v != CHECKOUT and not v.startswith(CHECKOUT + "/") for v in env.values()):
            raise SetupError(f"{sid}: {CHECKOUT} may only be or prefix a path value in launch.env")
        _check_install_fields(sid, entry)
        _check_host_requirements(sid, entry)
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


def _installer_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _INSTALLER_ENV_DROP}


def _micromamba() -> str:
    path = shutil.which("micromamba")
    if path is None:
        raise SetupError("micromamba not found on PATH; install the standalone binary first "
                         "(https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)")
    return path


def _mamba_env(dest: Path) -> dict:
    env = _installer_env()
    # Keep the package cache next to the checkouts, away from ~/micromamba and any base env.
    env["MAMBA_ROOT_PREFIX"] = str(dest.parent / ".micromamba")
    return env


def build_conda_env(entry: dict, dest: Path, *, conda_plat: str | None = None) -> Path:
    """Fresh micromamba prefix at <checkout>/.venv from the committed explicit lock."""
    lock = read_conda_lock(entry, conda_plat or conda_platform())
    micromamba = _micromamba()
    prefix = dest / ".venv"
    if prefix.exists() or prefix.is_symlink():
        if prefix.is_symlink() or not (prefix / "conda-meta").is_dir():
            raise SetupError(f"{prefix} exists but is not a conda prefix; remove it first")
        print(f"+ rm -rf {prefix}", flush=True)
        shutil.rmtree(prefix)       # never mix a previous environment into the locked one
    run([micromamba, "create", "--yes", "--no-rc", "--prefix", str(prefix), "--file", str(lock)],
        cwd=dest, env=_mamba_env(dest))
    return prefix / "bin" / "python"


def write_conda_locks(entry: dict, root: Path, bundle: Path | None = None) -> list[Path]:
    """Solve the manifest specs once per locked platform and write @EXPLICIT locks (needs network)."""
    micromamba = _micromamba()
    conda = entry["conda"]
    env = _mamba_env(root / entry["id"])
    env["CONDA_OVERRIDE_GLIBC"] = LOCK_GLIBC
    version = run([micromamba, "--version"], env=env)
    written = []
    for conda_plat, rel in sorted(conda["locks"].items()):
        out = run([micromamba, "create", "--no-rc", "--dry-run", "--json", "--yes",
                   "--prefix", str(root / f".lock-{entry['id']}-{conda_plat}"), "--platform", conda_plat,
                   "--override-channels", "-c", conda["channel"], *conda["specs"]], env=env)
        try:
            packages = json.loads(out)["actions"]["LINK"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SetupError(f"unexpected micromamba --json output for {conda_plat}: {exc}") from None
        urls = sorted(f"{p['url']}#sha256:{p['sha256']}" for p in packages)
        path = (bundle or HERE) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        header = [f"# Generated by scripts/mcp/e2e/setup.py {entry['id']} --lock with micromamba {version}",
                  f"# on {dt.datetime.now(dt.timezone.utc).date().isoformat()}; CONDA_OVERRIDE_GLIBC={LOCK_GLIBC}. "
                  "Do not edit by hand.", *lock_header(entry, conda_plat)]
        path.write_text("\n".join([*header, "@EXPLICIT", *urls]) + "\n", encoding="utf-8")
        read_conda_lock(entry, conda_plat, bundle)
        print(f"wrote {path} ({len(urls)} packages)")
        written.append(path)
    return written


def build_env(entry: dict, dest: Path) -> Path:
    if entry["install"] == "conda-explicit":
        python = build_conda_env(entry, dest)
    else:
        if shutil.which("uv") is None:
            raise SetupError("uv not found on PATH; install it first (https://docs.astral.sh/uv/)")
        env = _installer_env()
        python = dest / ".venv" / "bin" / "python"
        if entry["install"] == "uv-pip-pinned":
            # --clear: never mix a previous resolution into the pinned one.
            run(["uv", "venv", "--clear", "--python", entry["python"], str(dest / ".venv")], cwd=dest, env=env)
            run(["uv", "pip", "install", "--python", str(python), "--exclude-newer", entry["exclude_newer"],
                 *entry["requirements"]], cwd=dest, env=env)
        else:
            run(["uv", "sync", "--frozen", *entry.get("uv_sync_args", []), "--python", entry["python"]],
                cwd=dest, env=env)
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
        server["env"] = {k: v.replace(CHECKOUT, str(dest)) for k, v in launch["env"].items()}
    return {"mcpServers": {entry["id"]: server}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server", help="manifest id, e.g. pyscf")
    parser.add_argument("--root", default=os.environ.get("MCP_E2E_ROOT", "~/mcp"),
                        help="install root (default: $MCP_E2E_ROOT or ~/mcp)")
    parser.add_argument("--lock", action="store_true",
                        help="conda-explicit only: re-solve conda.specs and rewrite the committed locks")
    args = parser.parse_args(argv)

    try:
        servers = load_manifest()
        if args.server not in servers:
            raise SetupError(f"Unknown server {args.server!r}; known: {', '.join(sorted(servers))}")
        entry = servers[args.server]
        root = Path(args.root).expanduser().resolve()
        dest = root / entry["id"]
        if args.lock:
            if entry["install"] != "conda-explicit":
                raise SetupError(f"--lock only applies to install conda-explicit, not {entry['install']}")
            write_conda_locks(entry, root)
            return 0
        check_host(entry)
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
