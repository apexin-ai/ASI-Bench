"""LAMMPS process lifecycle and bounded artifact access (not a script sandbox)."""
from __future__ import annotations
import atexit
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid

MAX_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
TERMINAL = {'completed', 'failed', 'cancelled', 'timeout', 'output_limit', 'interrupted'}
RESERVED = {'in.lammps', 'log.lammps', 'screen.log', 'stderr.log', 'status.json'}


def bounded(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in [{low}, {high}]')
    return value


def finite(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be finite in [{low}, {high}]')
    return float(value)


def safe_path(root: Path, name: str, *, exists=False) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts or name == '.':
        raise ValueError('expected a relative path without parent traversal')
    path = root / relative
    # Reject even internal symlinks: output APIs never follow job-created links.
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError('symlinks are not supported')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('path escapes root')
    if exists and not path.is_file():
        raise ValueError('file does not exist')
    return path


def read_bytes(path: Path) -> bytes:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('file exceeds 16 MiB; use read_output with offset/limit')
    return path.read_bytes()


def parse_thermo(text: str, max_rows: int = 2000) -> dict:
    """Parse all numeric thermo tables without treating minimizer footers as rows."""
    blocks, headers, rows, total = [], None, [], 0
    def flush():
        nonlocal headers, rows
        if headers is not None:
            blocks.append({'columns': headers, 'rows': rows})
        headers, rows = None, []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == 'Step':
            flush(); headers = parts; continue
        if headers is None:
            continue
        try:
            values = [float(v) for v in parts]
            numeric = len(values) == len(headers) and all(math.isfinite(v) for v in values)
        except ValueError:
            numeric = False
        if numeric:
            total += 1
            if total <= max_rows:
                rows.append(values)
        elif parts and not line.lstrip().startswith('WARNING:'):
            flush()
    flush()
    stop = re.findall(r'Stopping criterion\s*=\s*([^\n]+)', text)
    # These are LAMMPS's actual stop reasons, not inference from the exit code.
    converged = (stop[-1].strip() in {'force tolerance', 'energy tolerance'}) if stop else None
    return {'blocks': blocks, 'row_count': total, 'truncated': total > max_rows,
            'minimization_stop_reason': stop[-1].strip() if stop else None,
            'converged': converged}


class Jobs:
    def __init__(self, workspace=None):
        self.workspace = Path(workspace or os.environ.get('LAMMPS_MCP_WORKSPACE', '/workspace')).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.root = safe_path(self.workspace, 'lammps-runs')
        self.root.mkdir(exist_ok=True)
        self.binary = os.environ.get('LAMMPS_BINARY', '/opt/venv/bin/lmp')
        self.lock = threading.RLock()
        self.live = {}
        self.records = {}
        self.closed = False
        # Never trust persisted PIDs from an earlier server, or kill unrelated processes.
        for path in self.root.glob('*/status.json'):
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                if path.stat().st_size > 65536: continue
                record = json.loads(path.read_text())
                job_id = path.parent.name
                if not re.fullmatch(r'[0-9a-f]{32}', job_id): continue
                if record.get('state') not in TERMINAL:
                    record['state'] = 'interrupted'
                    record['detail'] = 'server restarted; completion is not certified'
                self.records[job_id] = record
            except (OSError, ValueError):
                continue
        atexit.register(self.close)

    def directory(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id) or job_id not in self.records:
            raise ValueError('unknown job_id')
        return safe_path(self.root, job_id)

    def _save(self, job_id):
        # Opening without following links is important even inside the script trust boundary.
        path = safe_path(self.root / job_id, 'status.json')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as out:
            json.dump(self.records[job_id], out, allow_nan=False)

    def submit(self, script, input_files=None, timeout_seconds=120, mpi_ranks=1, prepared=None):
        if not isinstance(script, str) or not script.strip() or len(script.encode()) > 1024 * 1024:
            raise ValueError('script must contain 1 byte..1 MiB of LAMMPS input')
        bounded(timeout_seconds, 1, 600, 'timeout_seconds')
        bounded(mpi_ranks, 1, 4, 'mpi_ranks')
        input_files = input_files or []
        if len(input_files) > 64: raise ValueError('at most 64 input files')
        with self.lock:
            if self.closed: raise RuntimeError('server is shutting down')
            if len(self.live) >= 2: raise ValueError('at most two concurrent jobs')
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir(mode=0o700)
            try:
                total = 0
                for name in input_files:
                    if Path(name).parts[0] in RESERVED: raise ValueError('reserved input filename')
                    src = safe_path(self.workspace, name, exists=True)
                    data = read_bytes(src); total += len(data)
                    if total > MAX_BYTES: raise ValueError('staged inputs exceed 16 MiB')
                    dst = safe_path(directory, name); dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
                if prepared: prepared(directory)
                (directory / 'in.lammps').write_text(script)
                env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
                # Explicit MPI count; never derive CPU count from the host.
                command = [self.binary, '-in', 'in.lammps', '-log', 'log.lammps', '-screen', 'none']
                if mpi_ranks > 1: command = [str(Path(self.binary).with_name('mpiexec')), '-n', str(mpi_ranks), *command]
                # File limits apply to the complete launcher process and descendants.
                # prlimit avoids unsafe preexec_fn calls from a threaded MCP server.
                command = ['/usr/bin/prlimit', f'--fsize={MAX_OUTPUT_BYTES}', '--nofile=256:256', '--', *command]
                with (directory / 'screen.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
                    process = subprocess.Popen(command, cwd=directory, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                record = {'job_id': job_id, 'state': 'running', 'started_at_unix': time.time(),
                          'timeout_seconds': timeout_seconds, 'mpi_ranks': mpi_ranks,
                          'script_sha256': hashlib.sha256(script.encode()).hexdigest(),
                          'directory': str(directory), 'exit_code': None}
                self.records[job_id] = record
                self.live[job_id] = process
                self._save(job_id)
            except Exception:
                if job_id in self.live:
                    self._kill(self.live.pop(job_id))
                shutil.rmtree(directory)
                self.records.pop(job_id, None)
                raise
            threading.Thread(target=self._watch, args=(job_id, process), daemon=True).start()
            return dict(record)

    @staticmethod
    def _kill(process):
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=10)

    def _watch(self, job_id, process):
        reason = None
        while process.poll() is None:
            with self.lock:
                record = self.records[job_id]
                if record['state'] != 'running': reason = record['state']; break
                if time.time() - record['started_at_unix'] > record['timeout_seconds']:
                    reason = 'timeout'; break
            if self._over_budget(self.root / job_id):
                reason = 'output_limit'; break
            time.sleep(.1)
        if reason is None and self._over_budget(self.root / job_id): reason = 'output_limit'
        # Also reap subprocesses left in the job's process group after lmp exits.
        self._kill(process)
        with self.lock:
            record = self.records[job_id]
            if record['state'] == 'running':
                record['state'] = reason or ('completed' if process.returncode == 0 else 'failed')
            record.update(exit_code=process.returncode, finished_at_unix=time.time())
            self.live.pop(job_id, None)
            try: self._save(job_id)
            except (OSError, ValueError): record['persistence_error'] = True

    @staticmethod
    def _over_budget(directory):
        size, entries = 0, 0
        for base, dirs, files in os.walk(directory, followlinks=False):
            entries += len(dirs) + len(files)
            if entries > 2048: return True
            for name in files:
                try: size += (Path(base)/name).lstat().st_size
                except FileNotFoundError: pass
                if size > MAX_OUTPUT_BYTES: return True
        return False

    def status(self, job_id):
        with self.lock:
            self.directory(job_id)
            return dict(self.records[job_id])

    def wait(self, job_id, seconds=30):
        bounded(seconds, 0, 60, 'wait_seconds')
        deadline = time.monotonic() + seconds
        while True:
            status = self.status(job_id)
            if status['state'] in TERMINAL or time.monotonic() >= deadline: return status
            time.sleep(.1)

    def cancel(self, job_id):
        with self.lock:
            self.directory(job_id)
            if job_id in self.live:
                self.records[job_id]['state'] = 'cancelled'
                self._kill(self.live[job_id])
        # Wait for monitor to persist terminal state and return exit code.
        deadline = time.monotonic() + 10
        while job_id in self.live and time.monotonic() < deadline: time.sleep(.02)
        return self.status(job_id)

    def close(self):
        with self.lock:
            self.closed = True
            for job_id, process in list(self.live.items()):
                self.records[job_id]['state'] = 'interrupted'
                self._kill(process)

    def read_output(self, job_id, name, offset=0, limit=65536, encoding='utf-8'):
        bounded(offset, 0, 2**40, 'offset')
        bounded(limit, 1, 1024 * 1024, 'limit')
        if encoding not in {'utf-8', 'base64'}: raise ValueError('encoding must be utf-8 or base64')
        path = safe_path(self.directory(job_id), name, exists=True)
        with path.open('rb') as handle:
            handle.seek(offset); data = handle.read(limit)
        return {'path': str(path), 'size_bytes': path.stat().st_size, 'offset': offset,
                'next_offset': offset + len(data), 'eof': offset + len(data) >= path.stat().st_size,
                'encoding': encoding, 'data': base64.b64encode(data).decode() if encoding == 'base64' else data.decode('utf-8', errors='replace')}

    def files(self, job_id):
        directory = self.directory(job_id)
        result = []
        for base, dirs, names in os.walk(directory, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(base)/d).is_symlink()]
            for name in names:
                path = Path(base)/name
                if path.is_symlink() or not path.is_file(): continue
                result.append({'name': str(path.relative_to(directory)), 'size_bytes': path.stat().st_size})
                if len(result) >= 2048: return {'files': result, 'truncated': True}
        return {'files': result, 'truncated': False}
