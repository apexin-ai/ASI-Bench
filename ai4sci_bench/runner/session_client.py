"""Container-side launcher. Mounted read-only next to the upstream Unix bridge.

Only a per-cell HOME is persistent; fresh placeholder transport credentials and
XDG paths are recreated. No host user config or benchmark reference is mounted.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import threading
import time


def main():
    deadline = float(sys.argv[1])
    spec = importlib.util.spec_from_file_location('upstream_relay_client', '/opt/asibench-network/client.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    config = json.loads(Path('/opt/asibench-network/socket/client.json').read_text())
    if config['agent_type'] != 'claude_code':
        raise ValueError('Session checkpoint launcher supports only Claude Code')
    with client.Server(('127.0.0.1', 0), client.Bridge) as server, tempfile.TemporaryDirectory() as directory:
        for key in list(os.environ):
            if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'}:
                os.environ.pop(key)
        command = client.configure_client(config, f'http://127.0.0.1:{server.server_address[1]}',
                                          sys.argv[2:], Path(directory))
        home = Path('/opt/asibench-resume/home')
        os.environ['HOME'] = str(home)
        os.environ['CLAUDE_CONFIG_DIR'] = str(home / '.claude')
        # Required for deterministic isolated runs; version is inherited from image.
        os.environ['DISABLE_AUTOUPDATER'] = '1'
        threading.Thread(target=server.serve_forever, daemon=True).start()
        # Host and local Docker share CLOCK_MONOTONIC. Startup/preflight time
        # consumes the same remaining budget; upstream's extra 30s is cleanup only.
        with subprocess.Popen(command, start_new_session=True) as process:
            try:
                return process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process exited at the deadline between wait and kill.
                process.wait()
                return 124


if __name__ == '__main__':
    sys.exit(main())
