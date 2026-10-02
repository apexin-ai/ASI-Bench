#!/usr/bin/env python3
"""Install one pinned MCP server for end-to-end (E2E) testing.

Stdlib only. For the server ``<id>`` listed in ``manifest.json`` this:

1. checks the host against the server's ``host_requirements`` (machine, CPU
   flags, loadable system libraries) before anything is cloned; nothing is
   ever installed on the host;
2. clones the upstream repository into ``<root>/<id>`` (or reuses an existing
   clean checkout) and detaches at the pinned revision;
3. builds an isolated environment at ``<root>/<id>/.venv`` with the server's
   install mode (``UV_PYTHON`` and other installer variables from the caller's
   shell are ignored on purpose):

   ``uv-sync-frozen``  ``uv sync --frozen`` against the upstream lockfile
                       (optional extra ``uv_sync_args``);
   ``uv-pip-pinned``   no upstream lockfile: fresh venv and ``uv pip install``
                       of exact ``requirements`` pins with ``exclude_newer``;
   ``conda-explicit``  conda-only packages (psi4): ``micromamba create`` from a
                       committed per-platform ``@EXPLICIT`` lock (exact URLs +
                       SHA-256, no solver at install time) for the ``conda``
                       specs; ``--lock`` re-solves and rewrites the locks;

4. writes a portable ``<root>/<id>.mcp.json`` for ``asibench run --mcp-config``
   (``{checkout}`` in launch args / env values becomes the absolute checkout).

The manifest is validated strictly: an entry may only contain the common keys
(:data:`COMMON_KEYS`) plus the fields of its install mode, so a field of
another mode or a misspelt key is an error, not silently ignored. A new install
mode is one :class:`Installer` subclass registered in :data:`INSTALLERS`; a new
host check is one probe in :data:`HOST_PROBES`.

It never installs anything into the ASI-Bench environment, never touches
operator credentials and never runs business tool calls; use the matching
``smoke_<id>.py`` afterwards.

Usage::

    python3 scripts/mcp/e2e/setup.py <id> [--root ~/mcp]      # pyscf, arxiv, jsbsim, s4, psi4

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
from typing import Callable

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
# Placeholder in launch args / env values for the absolute checkout path, e.g. "{checkout}/main.py".
CHECKOUT = "{checkout}"
REVISION_RE = re.compile(r"[0-9a-f]{40}")
PYTHON_RE = re.compile(r"3\.\d+")
PIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9][A-Za-z0-9.+!_-]*")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
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


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SetupError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}{proc.stderr}"
        )
    return proc.stdout.strip()


def _installer_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _INSTALLER_ENV_DROP}


def _string_list(value, *, nonempty: bool = False) -> bool:
    return isinstance(value, list) and (bool(value) or not nonempty) and \
        all(isinstance(v, str) and v and v.strip() == v for v in value)


# --------------------------------------------------------------------------
# Install modes
# --------------------------------------------------------------------------

Validator = Callable[[str, object, dict], None]   # (server id, value or None if absent, entry)


class Installer:
    """One install mode: the manifest fields it owns and how it builds ``<dest>/.venv``."""
    mode: str = ""
    # field -> validator; a validator is also called with None when the field is absent,
    # so it decides itself whether the field is required.
    fields: dict[str, Validator] = {}

    def validate(self, sid: str, entry: dict) -> None:
        for key, check in self.fields.items():
            check(sid, entry.get(key), entry)

    def install(self, entry: dict, dest: Path) -> Path:
        """Build the environment; return its Python interpreter."""
        raise NotImplementedError

    def lock(self, entry: dict, root: Path) -> list[Path]:
        raise SetupError(f"--lock only applies to install conda-explicit, not {self.mode}")


def _uv() -> None:
    if shutil.which("uv") is None:
        raise SetupError("uv not found on PATH; install it first (https://docs.astral.sh/uv/)")


def _check_uv_sync_args(sid: str, value, entry: dict) -> None:
    if value is None:
        return
    if not _string_list(value) or not all(a.startswith("--") for a in value) \
            or any(a.split("=", 1)[0] in {"--python", "--frozen"} for a in value):
        raise SetupError(f"{sid}: uv_sync_args must be extra --flags (not --python/--frozen)")


class UvSyncFrozen(Installer):
    """Upstream ships pyproject.toml + uv.lock -> ``uv sync --frozen``."""
    mode = "uv-sync-frozen"
    fields = {"uv_sync_args": _check_uv_sync_args}

    def install(self, entry: dict, dest: Path) -> Path:
        _uv()
        run(["uv", "sync", "--frozen", *entry.get("uv_sync_args", []), "--python", entry["python"]],
            cwd=dest, env=_installer_env())
        return dest / ".venv" / "bin" / "python"


def _check_requirements(sid: str, value, entry: dict) -> None:
    if not _string_list(value, nonempty=True) or not all(PIN_RE.fullmatch(r) for r in value):
        raise SetupError(f"{sid}: uv-pip-pinned needs requirements as exact 'name==version' pins")


def _check_exclude_newer(sid: str, value, entry: dict) -> None:
    if not isinstance(value, str) or not TIMESTAMP_RE.fullmatch(value):
        raise SetupError(f"{sid}: uv-pip-pinned needs exclude_newer as YYYY-MM-DDTHH:MM:SSZ "
                         "(fixes the transitive resolution)")


class UvPipPinned(Installer):
    """No upstream lockfile -> fresh venv + ``uv pip install`` of exact pins with --exclude-newer."""
    mode = "uv-pip-pinned"
    fields = {"requirements": _check_requirements, "exclude_newer": _check_exclude_newer}

    def install(self, entry: dict, dest: Path) -> Path:
        _uv()
        env = _installer_env()
        python = dest / ".venv" / "bin" / "python"
        # --clear: never mix a previous resolution into the pinned one.
        run(["uv", "venv", "--clear", "--python", entry["python"], str(dest / ".venv")], cwd=dest, env=env)
        run(["uv", "pip", "install", "--python", str(python), "--exclude-newer", entry["exclude_newer"],
             *entry["requirements"]], cwd=dest, env=env)
        return python


def _check_conda(sid: str, conda, entry: dict) -> None:
    if not isinstance(conda, dict) or set(conda) != set(CONDA_KEYS):
        raise SetupError(f"{sid}: conda-explicit needs conda with exactly {', '.join(CONDA_KEYS)}")
    if not isinstance(conda["channel"], str) or not re.fullmatch(r"[a-z0-9-]+", conda["channel"]):
        raise SetupError(f"{sid}: conda.channel must be a channel name such as conda-forge")
    specs = conda["specs"]
    if not _string_list(specs, nonempty=True) or not all(CONDA_SPEC_RE.fullmatch(x) for x in specs):
        raise SetupError(f"{sid}: conda.specs must be exact 'name=version' specs")
    python = entry.get("python")
    if f"python={python}" not in specs and not any(x.startswith(f"python={python}.") for x in specs):
        raise SetupError(f"{sid}: conda.specs must pin python to the manifest version {python}")
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


class CondaExplicit(Installer):
    """Conda-only dependencies -> ``micromamba create --file <lock>`` from a committed
    @EXPLICIT lock for the host's conda platform; ``lock`` re-solves the specs."""
    mode = "conda-explicit"
    fields = {"conda": _check_conda}

    def install(self, entry: dict, dest: Path) -> Path:
        lock = read_conda_lock(entry, conda_platform())
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

    def lock(self, entry: dict, root: Path, bundle: Path | None = None) -> list[Path]:
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
                      f"# on {dt.datetime.now(dt.timezone.utc).date().isoformat()}; "
                      f"CONDA_OVERRIDE_GLIBC={LOCK_GLIBC}. Do not edit by hand.", *lock_header(entry, conda_plat)]
            path.write_text("\n".join([*header, "@EXPLICIT", *urls]) + "\n", encoding="utf-8")
            read_conda_lock(entry, conda_plat, bundle)
            print(f"wrote {path} ({len(urls)} packages)")
            written.append(path)
        return written


INSTALLERS: dict[str, Installer] = {i.mode: i for i in (UvSyncFrozen(), UvPipPinned(), CondaExplicit())}
INSTALL_MODES = tuple(INSTALLERS)


def write_conda_locks(entry: dict, root: Path, bundle: Path | None = None) -> list[Path]:
    return INSTALLERS["conda-explicit"].lock(entry, root, bundle)


def build_env(entry: dict, dest: Path) -> Path:
    python = INSTALLERS[entry["install"]].install(entry, dest)
    version = run([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
    if version != entry["python"]:
        raise SetupError(f"{entry['id']}: venv Python {version} != manifest {entry['python']}")
    return python


# --------------------------------------------------------------------------
# Host requirements
# --------------------------------------------------------------------------

def _probe_machine(wanted: list[str], host: dict) -> list[str]:
    return [] if host["machine"] in wanted else [f"machine {host['machine']!r} is not one of {wanted}"]


def _probe_cpu_flags(wanted: list[str], host: dict) -> list[str]:
    try:
        text = host["cpuinfo"].read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    flags = {f for line in text.splitlines() if line.split(":", 1)[0].strip() == "flags"
             for f in line.split(":", 1)[1].split()}
    missing = [f for f in wanted if f not in flags]
    return [f"CPU lacks {missing} (read from {host['cpuinfo']})"] if missing else []


def _probe_shared_libraries(wanted: list[str], host: dict) -> list[str]:
    problems = []
    for lib in wanted:
        try:
            host["loader"](lib)
        except OSError as exc:
            problems.append(f"cannot load {lib}: {exc}")
    return problems


# host_requirements key -> probe(required values, host facts) -> problems
HOST_PROBES: dict[str, Callable[[list[str], dict], list[str]]] = {
    "machine": _probe_machine,
    "cpu_flags": _probe_cpu_flags,
    "shared_libraries": _probe_shared_libraries,
}
HOST_REQUIREMENT_KEYS = tuple(HOST_PROBES)


def _check_host_requirements(sid: str, req, entry: dict) -> None:
    if req is None:
        return
    if not isinstance(req, dict) or not set(req) & set(HOST_PROBES) or set(req) - set(HOST_PROBES) - {"reason"}:
        raise SetupError(f"{sid}: host_requirements may only contain {', '.join(HOST_PROBES)} and reason")
    for key in HOST_PROBES:
        if key in req and not _string_list(req[key], nonempty=True):
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
    host = {"machine": machine or platform.machine(), "cpuinfo": cpuinfo, "loader": loader}
    problems = [p for key, probe in HOST_PROBES.items() if req.get(key) for p in probe(req[key], host)]
    if problems:
        reason = f" ({req['reason']})" if req.get("reason") else ""
        raise SetupError(f"{entry['id']}: host requirements not met{reason}:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def _check_string(pattern: re.Pattern | None, what: str) -> Validator:
    def check(sid: str, value, entry: dict) -> None:
        if not isinstance(value, str) or not value or (pattern and not pattern.fullmatch(value)):
            raise SetupError(f"{sid}: {what}")
    return check


def _check_launch(sid: str, launch, entry: dict) -> None:
    if not isinstance(launch, dict) or "command" not in launch or set(launch) - {"command", "args", "env"}:
        raise SetupError(f"{sid}: launch must have command and may have args, env")
    args, env = launch.get("args", []), launch.get("env", {})
    if not isinstance(launch["command"], str) or not launch["command"] \
            or not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise SetupError(f"{sid}: launch.command must be a string and launch.args a list of strings")
    if Path(launch["command"]).is_absolute() or any(Path(a).is_absolute() for a in args):
        raise SetupError(f"{sid}: launch paths must be relative to the checkout")
    if any(CHECKOUT in a and not a.startswith(CHECKOUT + "/") for a in args):
        raise SetupError(f"{sid}: {CHECKOUT} may only prefix a path argument")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise SetupError(f"{sid}: launch.env must map strings to strings")
    if any(Path(v).is_absolute() for v in env.values()):
        raise SetupError(f"{sid}: launch.env paths must use {CHECKOUT}/..., not absolute paths")
    if any(CHECKOUT in v and v != CHECKOUT and not v.startswith(CHECKOUT + "/") for v in env.values()):
        raise SetupError(f"{sid}: {CHECKOUT} may only be or prefix a path value in launch.env")


def _check_expected_tools(sid: str, value, entry: dict) -> None:
    if not _string_list(value, nonempty=True) or value != sorted(set(value)):
        raise SetupError(f"{sid}: expected_tools must be a sorted list of distinct tool names")


def _check_install(sid: str, value, entry: dict) -> None:
    if value not in INSTALLERS:
        raise SetupError(f"{sid}: unsupported install mode {value!r}; known: {', '.join(INSTALLERS)}")


# Keys every entry may have (validators decide which are required); install modes add their own.
COMMON_KEYS: dict[str, Validator] = {
    "id": _check_string(re.compile(r"[a-z0-9][a-z0-9_-]*"), "id must be a lowercase name"),
    "catalog_id": _check_string(None, "catalog_id must be a non-empty string"),
    "repository": _check_string(re.compile(r"https://\S+"), "repository must be an https URL"),
    "revision": _check_string(REVISION_RE, "revision must be a full 40-char commit SHA"),
    "python": _check_string(PYTHON_RE, "python must be a 3.x version such as 3.12"),
    "install": _check_install,
    "launch": _check_launch,
    "smoke": _check_string(re.compile(r"smoke_[a-z0-9_]+\.py"), "smoke must name a smoke_<id>.py script"),
    "expected_tools": _check_expected_tools,
    "host_requirements": _check_host_requirements,
}


def validate_entry(entry) -> None:
    sid = entry.get("id", "?") if isinstance(entry, dict) else "?"
    if not isinstance(entry, dict):
        raise SetupError(f"{sid}: a server entry must be an object")
    for key, check in COMMON_KEYS.items():
        check(sid, entry.get(key), entry)
    installer = INSTALLERS[entry["install"]]
    extra = sorted(set(entry) - set(COMMON_KEYS) - set(installer.fields))
    owned = {key: mode for mode, other in INSTALLERS.items() for key in other.fields}
    foreign = [key for key in extra if key in owned]
    if foreign:
        modes = sorted({owned[key] for key in foreign})
        raise SetupError(f"{sid}: fields {foreign} only apply to install {' / '.join(modes)}, "
                         f"not {installer.mode}")
    if extra:
        raise SetupError(f"{sid}: unknown manifest key(s) {extra}; allowed: "
                         f"{sorted([*COMMON_KEYS, *installer.fields])}")
    installer.validate(sid, entry)


def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise SetupError(f"Unsupported manifest schema: {document.get('schema_version')!r}")
    if set(document) != {"schema_version", "servers"} or not isinstance(document["servers"], list):
        raise SetupError("manifest must contain exactly schema_version and a servers list")
    servers: dict[str, dict] = {}
    for entry in document["servers"]:
        validate_entry(entry)
        if entry["id"] in servers:
            raise SetupError(f"Duplicate manifest id: {entry['id']}")
        servers[entry["id"]] = entry
    return servers


# --------------------------------------------------------------------------
# Checkout and config
# --------------------------------------------------------------------------

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


def render_config(entry: dict, dest: Path) -> dict:
    launch = entry["launch"]
    server = {
        "command": str(dest / launch["command"]),
        "args": [arg.replace(CHECKOUT, str(dest)) for arg in launch.get("args", [])],
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
            INSTALLERS[entry["install"]].lock(entry, root)
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
