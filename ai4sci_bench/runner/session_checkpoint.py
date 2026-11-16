"""Versioned, immutable Claude checkpoints and conservative recovery policy.

This module intentionally has no upstream imports, Docker calls or credentials.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import uuid

SCHEMA = 'asibench.claude-checkpoint.v1'
RECOVERABLE = {'external_api', 'external_infrastructure'}
USAGE_KEYS = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    """Do not dereference symlinks (agent outputs may point outside workspace)."""
    root = Path(root)
    found = {}
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            found[name] = {'symlink': str(path.readlink())}
        elif path.is_file():
            found[name] = {'sha256': digest(path), 'mode': path.stat().st_mode & 0o777}
        elif not path.is_dir():
            raise ValueError(f'Unsupported checkpoint filesystem entry: {name}')
    return found


def fingerprint(value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, 'value'):
            return item.value
        raise TypeError(f'Unsupported fingerprint value: {type(item).__name__}')
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=encode).encode()).hexdigest()


def cell_key(task, level):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', task) or level not in {'b1', 'b2', 'b3', 'b4'}:
        raise ValueError('Invalid checkpoint task/level')
    return f'{task}--{level}'


def classify(status, error='', requests=(), docker_state=None):
    """Terminal evidence only. Intermediate API retries do not make OOM resumable."""
    state = docker_state or {}
    if state.get('OOMKilled') or status == 'timeout' or state.get('ExitCode') == 124:
        return 'model_budget_or_process'
    if status == 'completed':
        return 'completed'
    messages = [r for r in requests if r.get('route') == '/v1/messages']
    last = messages[-1] if messages else {}
    text = str(error).lower()
    if state.get('ExitCode') in {137, 143}:
        return 'unknown'  # Cannot infer who killed the process.
    code = last.get('status')
    terminal_api = any(marker in text for marker in (
        'api error', 'api_error', 'authentication', 'unauthorized', 'quota',
        'rate limit', 'overloaded', 'upstream', 'connection error', 'connection_error',
        'stream error', 'stream_error', 'fetch failed', 'no available channel',
        '无可用渠道', 'credit balance'))
    if terminal_api and (code in {401, 402, 403, 408, 429, 500, 502, 503, 504, 520, 522, 524, 529}
            or last.get('transport_error') or last.get('stream_error')):
        # Only callers with an actual failed terminal generation use this path.
        return 'external_api'
    if any(marker in text for marker in ('docker daemon is not responsive',
                                         'cannot connect to the docker daemon',
                                         'could not create agent container',
                                         'infrastructure failure:')):
        return 'external_infrastructure'
    return 'unknown'


def usage_totals(requests):
    totals = dict.fromkeys(USAGE_KEYS, 0)
    complete = True
    seen = set()
    for request in requests:
        if request['id'] in seen:
            raise ValueError('Duplicate API accounting request ID')
        seen.add(request['id'])
        if request.get('route') != '/v1/messages':
            continue
        usage = request.get('usage') or {}
        # Failed requests can still be billable; unknown is never zero-cost proof.
        if not request.get('usage_complete'):
            complete = False
        for key in USAGE_KEYS:
            value = usage.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError('Invalid API token usage')
            totals[key] += value
    return {'observed_tokens': totals, 'usage_complete': complete,
            'actual_cost_usd': None, 'estimated_cost_usd': None}


def token_prices(value):
    """USD per million tokens, explicitly including both cache token classes."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(USAGE_KEYS):
        raise ValueError('Token prices must specify all four usage keys in USD per million tokens')
    if not all(isinstance(rate, (int, float)) and not isinstance(rate, bool)
               and math.isfinite(rate) and rate >= 0 for rate in value.values()):
        raise ValueError('Token prices must be finite nonnegative numbers')
    return value


def estimate_cost(accounting, prices):
    if prices is not None:
        accounting['estimated_cost_usd'] = sum(
            accounting['observed_tokens'][key] * prices[key] / 1_000_000 for key in USAGE_KEYS)
    accounting['estimate_basis'] = 'observed-usage-only' if prices else 'no-price-schedule'
    return accounting


def accounting_summary(root, cells):
    """Authoritative accounting; legacy upstream summaries may omit unknown costs."""
    selected = {}
    for cell in cells:
        path = Path(root) / cell / 'checkpoint.json'
        if path.is_file():
            record = json.loads(path.read_text())
            if record.get('state') == 'finished':
                selected[cell] = {'budget': record['budget'], 'accounting': record['accounting'],
                                  'failure_class': record['failure_class'], 'complete': record['complete']}
    accounts = [cell['accounting'] for cell in selected.values()]
    return {'schema': 'asibench.session-accounting.v1', 'cells': selected,
            'expected_cells': len(cells), 'recorded_cells': len(selected),
            'observed_tokens': {key: sum(a['observed_tokens'][key] for a in accounts) for key in USAGE_KEYS},
            'usage_complete': len(selected) == len(cells) and all(a['usage_complete'] for a in accounts),
            'actual_cost_usd': None,
            'estimated_cost_usd': (sum(a['estimated_cost_usd'] for a in accounts)
                                   if accounts and all(a['estimated_cost_usd'] is not None for a in accounts) else None)}


def load_checkpoint(path, verify=True):
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'checkpoint.json'
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError('Unsupported Claude checkpoint schema')
    if data.get('state') == 'finished' and data.get('failure_class') not in RECOVERABLE:
        raise ValueError('Checkpoint is not an external API/infrastructure failure')
    if not data.get('complete'):
        raise ValueError('Missing or incomplete Claude checkpoint; legacy logs alone cannot resume')
    if data.get('failure_class') not in RECOVERABLE:
        raise ValueError('Checkpoint is not an external API/infrastructure failure')
    budget = data['budget']
    used, limit = budget['charged_seconds'], budget['limit_seconds']
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (used, limit)):
        raise ValueError('Invalid checkpoint budget')
    if used >= limit or limit - used < 1:
        raise ValueError('Original execution budget is exhausted')
    if not math.isclose(budget.get('remaining_seconds', -1), limit - used, abs_tol=1e-6):
        raise ValueError('Checkpoint remaining budget is inconsistent')
    segments = data.get('segments', [])
    if segments:
        if len({s['id'] for s in segments}) != len(segments):
            raise ValueError('Duplicate continuation segment IDs')
        if not math.isclose(sum(s['charged_seconds'] for s in segments), used, abs_tol=1e-6):
            raise ValueError('Checkpoint cumulative budget is inconsistent')
    try:
        uuid.UUID(data['session_id'])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('Invalid Claude session ID') from exc
    root = path.parent
    if verify:
        if fingerprint(data['config']) != data.get('config_sha256'):
            raise ValueError('Checkpoint configuration integrity mismatch')
        for name in ('workspace', 'home', 'history'):
            if not (root / name).is_dir() or inventory(root / name) != data['files'][name]:
                raise ValueError(f'Checkpoint {name} integrity mismatch')
        archive = root / 'rootfs-image.tar'
        if not archive.is_file() or digest(archive) != data['rootfs']['sha256']:
            raise ValueError('Checkpoint container image integrity mismatch')
        sessions = list((root / 'home' / '.claude' / 'projects').rglob(data['session_id'] + '.jsonl'))
        if len(sessions) != 1 or not sessions[0].stat().st_size:
            raise ValueError('Missing or ambiguous native Claude session')
    return path, data


class Claim:
    """One committed continuation per checkpoint; a crashed claim needs an audit.

    flock guards concurrent submissions; persistent claim prevents silent branching
    or replay after a child has already consumed model budget.
    """
    def __init__(self, source, destination):
        self.path = Path(source).parent / 'continuation.json'
        self.destination = str(Path(destination).resolve())
        self.stream = None
        self.started = False

    def __enter__(self):
        import fcntl
        self.stream = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                raise ValueError('Checkpoint already claimed; inspect its continuation instead of branching')
            atomic_json(self.path, {'destination': self.destination, 'state': 'claimed'})
        except BaseException:
            self.stream.close()
            raise
        return self

    def begin_execution(self):
        atomic_json(self.path, {'destination': self.destination, 'state': 'executing'})
        self.started = True

    def __exit__(self, *_):
        # Preparation/validation failures before Docker dispatch spend no model
        # budget and must not permanently consume the source checkpoint.
        try:
            if not self.started and self.path.exists():
                current = json.loads(self.path.read_text())
                if current.get('destination') == self.destination:
                    self.path.unlink()
        finally:
            self.stream.close()


class ResponseUsage:
    """Bounded streaming parser: keep usage/error metadata, never message content."""
    def __init__(self, route):
        self.record = {'id': uuid.uuid4().hex, 'route': route, 'usage': {}, 'usage_complete': False}
        self.buffer = b''
        self.sse = False
        self.saw_start = False
        self.saw_output = False
        self.saw_stop = False

    def event(self, value):
        kind = value.get('type')
        if kind == 'error':
            self.record['stream_error'] = True
        if kind == 'message_start':
            self.saw_start = True
            value = value.get('message') or {}
        if kind == 'message_stop':
            self.saw_stop = True
        usage = value.get('usage') or {}
        if 'output_tokens' in usage and kind in {'message_delta', 'message'}:
            self.saw_output = True
        for key in USAGE_KEYS:
            if isinstance(usage.get(key), int) and usage[key] >= 0:
                self.record['usage'][key] = usage[key]

    def feed(self, chunk):
        self.buffer += chunk
        if self.sse:
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                if line.startswith(b'data:'):
                    try:
                        value = json.loads(line[5:])
                        if isinstance(value, dict):
                            self.event(value)
                    except ValueError:
                        self.record['parser_error'] = True
        if len(self.buffer) > 1024 * 1024:
            self.buffer = b''
            self.record['parser_overflow'] = True

    def finish(self):
        if not self.sse:
            try:
                value = json.loads(self.buffer)
                if isinstance(value, dict):
                    self.event(value)
            except ValueError:
                pass
        self.record['usage_complete'] = bool(
            self.record.get('status') == 200 and not self.record.get('transport_error')
            and not self.record.get('stream_error') and not self.record.get('parser_overflow')
            and not self.record.get('parser_error')
            and 'input_tokens' in self.record['usage'] and self.saw_output
            and (not self.sse or (self.saw_start and self.saw_stop)))
        if self.sse and self.record.get('status') == 200 and not self.saw_stop:
            self.record['stream_error'] = True
        return self.record
