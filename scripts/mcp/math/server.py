"""Repository-owned SageMath/SymPy MCP: native programs and task-local artifacts."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import StrictInt

from runtime import HERE, Jobs, safe_path


def probe(backend, env):
    process = subprocess.run([env.get('MATH_PYTHON', sys.executable), str(HERE / 'worker.py'),
                              '--probe', backend], env=env, capture_output=True, text=True, timeout=90,
                             stdin=subprocess.DEVNULL)
    if process.returncode:
        raise RuntimeError(f'{backend} import failed: {process.stderr[-2000:]}')
    return json.loads(process.stdout)


def build_server(jobs, info):
    backend = jobs.backend
    port = int(os.environ.get('MATH_MCP_PORT', '8765'))
    mcp = FastMCP('ASI-Bench ' + backend, host=os.environ.get('MATH_MCP_HOST', '0.0.0.0'),
                  port=port, json_response=True,
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                      allowed_hosts=[f'localhost:{port}', f'127.0.0.1:{port}', f'{backend}:{port}'],
                      allowed_origins=[f'http://localhost:{port}', f'http://{backend}:{port}']))

    @mcp.tool()
    def get_info() -> dict:
        """Return the actual imported backend version, syntax and execution limits."""
        return {**info, 'syntaxes': ['python', 'sage'] if backend == 'sagemath' else ['python'],
                'concurrent_job_limit': 2, 'timeout_max_seconds': 3600,
                'session': 'fresh process per job; final expression or result variable returned',
                'trust_boundary': 'native code has container privileges; use OS sandbox',
                'notes': 'Sage uses Python syntax unless syntax=sage; SymPy uses exact Rational for fractions.'}

    @mcp.tool()
    def upload_file(name: str, data: str, encoding: str = 'utf-8') -> dict:
        """Upload a relative input file, at most 16 MiB; no overwrite. Binary uses base64."""
        return jobs.put(name, data, encoding)

    def submit(code, script_name, input_files, timeout_seconds, syntax):
        if bool(code.strip()) == bool(script_name):
            raise ValueError('provide exactly one of code or script_name')
        if script_name:
            path = safe_path(jobs.workspace, script_name, exists=True)
            if path.stat().st_size > 1024 * 1024:
                raise ValueError('script exceeds 1 MiB')
            code = path.read_text(encoding='utf-8')
        return jobs.submit(code, input_files, timeout_seconds, syntax)

    @mcp.tool()
    def submit_script(code: str = '', script_name: str = '', input_files: list[str] | None = None,
                      timeout_seconds: StrictInt = 120, syntax: str = 'python') -> dict:
        """Run native Python/Sage code asynchronously. Return job_id, then wait/get_result.

        Full installed backend API is available. Code has container privileges.
        Stage workspace inputs explicitly. Jobs use fresh processes, not persistent sessions.
        """
        return submit(code, script_name, input_files, timeout_seconds, syntax)

    @mcp.tool()
    async def evaluate(code: str, timeout_seconds: StrictInt = 30, syntax: str = 'python') -> dict:
        """Evaluate a short program and return its final expression or error with job_id.

        Waits at most 60 seconds; long computations should use submit_script and polling.
        Use exact Rational/QQ for fractions; ordinary Python 1/3 is a float.
        """
        job = submit(code, '', None, timeout_seconds, syntax)
        await asyncio.to_thread(jobs.wait, job['job_id'], min(timeout_seconds + 1, 60))
        return result(job['job_id'])

    @mcp.tool()
    def list_jobs() -> dict:
        """List this server lifetime's job records; no persisted PID reuse."""
        with jobs.lock:
            return {'jobs': [dict(record) for record in jobs.records.values()]}

    @mcp.tool()
    def get_status(job_id: str) -> dict:
        """Get running, completed, failed, timeout or cancelled state."""
        return jobs.status(job_id)

    @mcp.tool()
    async def wait_job(job_id: str, wait_seconds: StrictInt = 30) -> dict:
        """Wait up to 60 seconds without blocking other HTTP clients or cancellation."""
        return await asyncio.to_thread(jobs.wait, job_id, wait_seconds)

    @mcp.tool()
    def cancel_job(job_id: str) -> dict:
        """Kill a running job process group; preserve terminal job states."""
        return jobs.cancel(job_id)

    def result(job_id):
        status = jobs.status(job_id)
        payload = None
        path = safe_path(jobs.directory(job_id), 'result.json')
        if status['state'] in {'completed', 'failed'} and path.is_file():
            if path.stat().st_size > 65536:
                raise ValueError('oversized result metadata')
            payload = json.loads(path.read_text())
        return {'status': status, 'result': payload, 'artifacts': jobs.files(job_id)}

    @mcp.tool()
    def get_result(job_id: str) -> dict:
        """Return status, bounded exact-expression preview and file inventory.

        Large exact expressions and LaTeX are files, not a forced float conversion.
        Failed/timeout jobs are never represented as successful mathematics.
        """
        return result(job_id)

    @mcp.tool()
    def list_files(job_id: str | None = None, offset: StrictInt = 0, limit: StrictInt = 100) -> dict:
        """List workspace or job artifacts with bounded pagination; omit symlinks."""
        return jobs.files(job_id, offset, limit)

    @mcp.tool()
    def read_file(name: str, job_id: str | None = None, offset: StrictInt = 0,
                  limit: StrictInt = 65536, encoding: str = 'utf-8') -> dict:
        """Download text or base64 bytes; offsets are bytes, with chunk SHA-256."""
        return jobs.read(name, job_id, offset, limit, encoding)

    return mcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['sagemath', 'sympy'], required=True)
    parser.add_argument('--transport', choices=['stdio', 'streamable-http'], default='stdio')
    args = parser.parse_args()
    env = dict(os.environ)
    # The framework's temporary HOME is on noexec /tmp. Sage needs executable caches.
    state_root = Path(env.get('MATH_STATE_ROOT', '/home/agent/.cache/math'))
    state_root.mkdir(parents=True, exist_ok=True)
    home = tempfile.mkdtemp(prefix=args.backend + '-', dir=state_root)
    env.update(HOME=home, TMPDIR=home, DOT_SAGE=str(Path(home) / '.sage'),
               XDG_CACHE_HOME=str(Path(home) / '.cache'), MPLCONFIGDIR=str(Path(home) / '.matplotlib'))
    jobs = None
    def terminate(signum, frame):
        if jobs:
            jobs.close()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    try:
        # Backend import errors are fatal startup errors; never silently switch engines.
        info = probe(args.backend, env)
        jobs = Jobs(os.environ.get('MATH_MCP_WORKSPACE', '/workspace'), env, args.backend)
        build_server(jobs, info).run(transport=args.transport)
    finally:
        if jobs:
            jobs.close()
        shutil.rmtree(home, ignore_errors=True)


if __name__ == '__main__':
    main()
