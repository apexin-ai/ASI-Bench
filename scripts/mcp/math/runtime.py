"""Bounded SageMath/SymPy processes and file transfer; scripts have container privileges."""
from __future__ import annotations
import atexit
from contextlib import suppress
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
MAX_BYTES = 16 * 1024 * 1024
MAX_OUTPUT = 128 * 1024 * 1024
TERMINAL = {'completed', 'failed', 'cancelled', 'timeout', 'aborted', 'interrupted'}
RESERVED = {'input.py', 'request.json', 'result.json', 'result.repr.txt', 'result.exact.txt', 'result.latex.txt', '.result.json.tmp', 'stdout.log', 'stderr.log'}


def bounded(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def safe_path(root: Path, name: str, *, exists=False) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts or name == '.':
        raise ValueError('expected a relative file path without parent traversal')
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError('symlinks are not supported')
    if not cursor.resolve().is_relative_to(root.resolve()):
        raise ValueError('path escapes workspace')
    if exists and not cursor.is_file():
        raise ValueError('file does not exist')
    return cursor


class Jobs:
    def __init__(self, workspace: str | Path, env: dict[str, str], backend: str):
        if backend not in {"sagemath", "sympy"}:
            raise ValueError("backend must be sagemath or sympy")
        self.backend = backend
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.root = safe_path(self.workspace, 'math-jobs')
        self.root.mkdir(exist_ok=True)
        self.env = env
        self.lock = threading.RLock()
        self.live: dict[str, subprocess.Popen] = {}
        self.records: dict[str, dict] = {}
        self.closed = False
        # Never reuse a PID or trust a previous server's completion records.
        atexit.register(self.close)

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch('[0-9a-f]{32}', job_id) or job_id not in self.records:
            raise ValueError('unknown job_id (jobs belong to this server lifetime)')
        return safe_path(self.root, job_id)

    def submit(self, code: str, input_files: list[str] | None, timeout_seconds: int, syntax: str = "python") -> dict:
        if syntax not in ({"python", "sage"} if self.backend == "sagemath" else {"python"}):
            raise ValueError("sage syntax requires sagemath; otherwise use python")
        if not code.strip() or len(code.encode()) > 1024 * 1024:
            raise ValueError('code must contain 1 byte..1 MiB of code')
        bounded(timeout_seconds, 1, 3600, 'timeout_seconds')
        inputs = input_files or []
        if len(inputs) > 64:
            raise ValueError('at most 64 input files')
        with self.lock:
            if self.closed:
                raise RuntimeError('server is shutting down')
            if len(self.live) >= 2:
                raise ValueError('at most two concurrent native jobs')
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir(mode=0o700)
            try:
                total = 0
                for name in inputs:
                    if Path(name).parts and Path(name).parts[0] in RESERVED | {'math-jobs'}:
                        raise ValueError('reserved input filename')
                    src = safe_path(self.workspace, name, exists=True)
                    total += src.stat().st_size
                    if total > MAX_BYTES:
                        raise ValueError('staged inputs exceed 16 MiB')
                    dst = safe_path(directory, name)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                (directory / 'input.py').write_text(code, encoding='utf-8')
                (directory / 'request.json').write_text(json.dumps({'backend': self.backend, 'syntax': syntax}))
                env = {**self.env, 'MPLBACKEND': 'Agg'}
                binary = env.get('MATH_PYTHON', os.sys.executable)
                command = [binary, str(HERE / 'worker.py'), str(directory)]
                # A separate process group and an external deadline handle uninterruptible code.
                if Path('/usr/bin/prlimit').exists():
                    command = ['/usr/bin/prlimit', f'--fsize={MAX_OUTPUT}', '--nofile=256:256', '--', *command]
                with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
                    proc = subprocess.Popen(command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                            stdout=out, stderr=err, start_new_session=True)
                record = {'job_id': job_id, 'state': 'running', 'started_at_unix': time.time(),
                          'timeout_seconds': timeout_seconds, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                          'exit_code': None, 'backend': self.backend, 'syntax': syntax, 'session': 'fresh process; no implicit persistent variables'}
                self.records[job_id] = record
                self.live[job_id] = proc
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            threading.Thread(target=self._watch, args=(job_id, proc), daemon=True).start()
            return dict(record)

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _watch(self, job_id, proc):
        record = self.records[job_id]
        try:
            # Deadline includes backend startup, computation and export.
            proc.wait(timeout=record['timeout_seconds'])
            with self.lock:
                if record['state'] == 'running':
                    record['state'] = 'failed'
                    record['exit_code'] = proc.returncode
                    try:
                        path = safe_path(self.directory(job_id), 'result.json', exists=True)
                        if path.stat().st_size > 65536:
                            raise ValueError('oversized result metadata')
                        result = json.loads(path.read_text())
                        if not isinstance(result, dict):
                            raise ValueError('result metadata must be an object')
                        state = result.get('state')
                        if result.get('backend') == self.backend and state in TERMINAL and (state != 'completed' or proc.returncode == 0):
                            record['state'] = state
                    except (OSError, ValueError):
                        record['detail'] = 'no valid result metadata; inspect stderr.log/stdout.log'
        except subprocess.TimeoutExpired:
            self._kill(proc)
            proc.wait(timeout=10)
            with self.lock:
                if record['state'] == 'running':
                    record.update(state='timeout', detail='external deadline; worker process group terminated')
        finally:
            # Kill residual children even if the launcher exited normally.
            self._kill(proc)
            with self.lock:
                record.update(exit_code=proc.returncode, finished_at_unix=time.time())
                self.live.pop(job_id, None)

    def status(self, job_id):
        with self.lock:
            self.directory(job_id)
            return dict(self.records[job_id])

    def wait(self, job_id, wait_seconds):
        bounded(wait_seconds, 0, 60, 'wait_seconds')
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if self.status(job_id)['state'] != 'running':
                break
            time.sleep(.05)
        return self.status(job_id)

    def cancel(self, job_id):
        with self.lock:
            self.directory(job_id)
            proc = self.live.get(job_id)
            if proc and self.records[job_id]['state'] == 'running':
                self.records[job_id]['state'] = 'cancelled'
                self._kill(proc)
            return dict(self.records[job_id])

    def close(self):
        with self.lock:
            self.closed = True
            processes = list(self.live.values())
            for job_id in list(self.live):
                self.cancel(job_id)
        for process in processes:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)

    def put(self, name, data, encoding='utf-8'):
        if Path(name).parts and Path(name).parts[0] == 'math-jobs':
            raise ValueError('upload to input workspace, not managed job outputs')
        if encoding not in {'utf-8', 'base64'} or len(data) > MAX_BYTES * 2:
            raise ValueError('invalid encoding or oversized payload')
        payload = data.encode() if encoding == 'utf-8' else base64.b64decode(data, validate=True)
        if len(payload) > MAX_BYTES:
            raise ValueError('upload exceeds 16 MiB')
        path = safe_path(self.workspace, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as out:
            out.write(payload)
        return {'name': name, 'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

    def files(self, job_id=None, offset=0, limit=100):
        bounded(offset, 0, 1000000, 'offset'); bounded(limit, 1, 500, 'limit')
        root = self.directory(job_id) if job_id else self.workspace
        files = []
        # No recursive symlink traversal; stop instead of unbounded file enumeration.
        for parent, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not (Path(parent) / d).is_symlink())
            for name in sorted(names):
                path = Path(parent) / name
                if path.is_symlink() or not path.is_file():
                    continue
                files.append({'name': str(path.relative_to(root)), 'size_bytes': path.stat().st_size})
                if len(files) > 10000:
                    raise ValueError('file enumeration exceeds 10000 entries')
        return {'files': files[offset:offset+limit], 'count': len(files),
                'next_offset': min(offset+limit, len(files)), 'eof': offset+limit >= len(files)}

    def read(self, name, job_id=None, offset=0, limit=65536, encoding='utf-8'):
        bounded(offset, 0, MAX_OUTPUT, 'offset'); bounded(limit, 1, 1048576, 'limit')
        if encoding not in {'utf-8', 'base64'}:
            raise ValueError('encoding must be utf-8 or base64')
        root = self.directory(job_id) if job_id else self.workspace
        path = safe_path(root, name, exists=True)
        with path.open('rb') as f:
            f.seek(offset); payload = f.read(limit)
        size = path.stat().st_size
        # Offsets are bytes; base64 is lossless even at arbitrary UTF-8 boundaries.
        return {'name': name, 'data': payload.decode('utf-8', errors='replace') if encoding == 'utf-8' else base64.b64encode(payload).decode(),
                'encoding': encoding, 'offset': offset, 'next_offset': offset+len(payload),
                'size_bytes': size, 'eof': offset+len(payload) >= size,
                'chunk_sha256': hashlib.sha256(payload).hexdigest()}
