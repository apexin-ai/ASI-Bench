"""Prepare per-run Wolfram state; never print license material."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent


def prepare_environment() -> dict[str, str]:
    env = dict(os.environ)
    state_root = env.get('WOLFRAM_STATE_ROOT')
    if state_root:
        root = Path(state_root)
        root.mkdir(parents=True, exist_ok=True)
        home = Path(tempfile.mkdtemp(prefix='run-', dir=root))
        env.update(HOME=str(home), TMPDIR=str(home))
    else:
        home = Path(env.get('HOME', '/home/agent'))
        home.mkdir(parents=True, exist_ok=True)
    # Seed read-only build-time packages/data using file symlinks and writable
    # directories, avoiding a 1.2 GiB copy for each preflight/agent session.
    template = Path('/opt/wolfram-state')
    if template.is_dir():
        for relative in ['.WolframEngine/Paclets', '.Wolfram/Objects']:
            source = template / relative
            target = home / relative
            if source.is_dir() and not target.exists():
                shutil.copytree(source, target, copy_function=lambda src, dst: os.symlink(src, dst))
    # ASI-Bench gives stdio servers a fresh HOME. Use preinstalled paclets,
    # not the package/credential state from the agent or a developer's host.
    env.setdefault('MCP_SERVER_NAME', 'WolframLanguage')
    env.setdefault('WOLFRAMINIT', '-pacletreadonly')
    env.setdefault('XDG_CACHE_HOME', str(home / '.cache'))
    Path(env['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
    license_b64 = env.pop('WOLFRAM_LICENSE_B64', None)
    if license_b64:
        try:
            content = base64.b64decode(license_b64, validate=True)
            if not content or len(content) > 1024 * 1024 or b'\0' in content:
                raise ValueError()
        except (ValueError, TypeError):
            raise RuntimeError('invalid WOLFRAM_LICENSE_B64 (expected base64 mathpass content)') from None
        for relative in ['.Wolfram/Licensing', '.WolframEngine/Licensing', '.Mathematica/Licensing']:
            directory = home / relative
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory / 'mathpass'
            # Fresh private HOME is required; do not follow existing links.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as out:
                out.write(content)
    return env


def official_command() -> list[str]:
    # Explicit JSON argv override is useful for pinned private installations and
    # protocol fixtures. No shell evaluation or ambient client configuration.
    raw = os.environ.get('WOLFRAM_MCP_COMMAND_JSON')
    if raw:
        command = json.loads(raw)
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or '\0' in x for x in command):
            raise ValueError('WOLFRAM_MCP_COMMAND_JSON must be a nonempty JSON argv list')
        return command
    binary = os.environ.get('WOLFRAMSCRIPT', '/usr/bin/wolframscript')
    if not Path(binary).is_file() and not shutil.which(binary):
        raise RuntimeError('wolframscript is missing; build the licensed Wolfram runtime image')
    return [binary, '-script', str(HERE / 'start.wls')]
