"""Native Claude checkpoint lifecycle, called explicitly by adapter and sandbox.

Each execution owns its state; no monkey patches, global environment switches,
production credentials or experiment-specific lock files are required.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

from ai4sci_bench.runner.session_checkpoint import (SCHEMA, Claim, atomic_json, cell_key, classify,
                                digest, fingerprint, inventory, load_checkpoint,
                                usage_totals, token_prices, estimate_cost)

HERE = Path(__file__).resolve().parent


def docker(*args, timeout=120):
    return subprocess.run(['docker', *map(str, args)], check=True, capture_output=True,
                          text=True, timeout=timeout).stdout.strip()


def implementation_identity():
    paths = ('runner/session_checkpoint.py', 'runner/claude_checkpoint.py',
             'runner/session_client.py', 'runner/session_resume.py',
             'runner/os_sandbox.py', 'runner/model_relay.py',
             'runner/orchestrator.py', 'adapters/claude_code_cli.py')
    return {name: digest(HERE.parent / name) for name in paths}


def frozen_config(adapter, instance, prices=None):
    from ai4sci_bench import __version__
    # Bind restoration to the installed implementation, not a developer checkout.
    implementation = implementation_identity()
    return {'framework_version': __version__, 'implementation': implementation,
            'agent': 'claude_code_cli', 'model': adapter.model,
            'timeout_seconds': instance.effective_timeout_seconds,
            'effort': adapter.effort, 'sandbox': adapter.sandbox,
            'tool_mode': adapter.tool_mode.value, 'network_policy': adapter.effective_network_policy,
            'api_protocol': adapter.api_protocol, 'supports_image_input': adapter.supports_image_input,
            'permission_mode': adapter.permission_mode,
            'task_id': instance.task_id, 'instance_id': instance.instance_id,
            'level': instance.prompt_level.value, 'parameters': instance.parameters,
            'task_definition': fingerprint({k: v for k, v in instance.metadata.items() if not k.startswith('_')}),
            'input_files': inventory(instance.workspace_dir),
            'token_prices_usd_per_million': token_prices(prices),
            'cpu_limit': adapter._os_sandbox.cpu_limit, 'memory_limit': adapter._os_sandbox.memory_limit}


def copy_tree(source, target):
    # Symlinks are data: never read arbitrary host paths through agent symlinks.
    shutil.copytree(source, target, symlinks=True)


def instance_key(instance):
    return checkpoint_key(instance.task_id, instance.instance_id, instance.prompt_level.value)


def checkpoint_key(task, instance_id, level):
    # A run can contain multiple instances of the same task/level.
    return cell_key(task, level) + '--' + fingerprint(instance_id)[:24]


class CheckpointExecution:
    def __init__(self, adapter, instance, root, source=None, prices=None):
        self.root = root
        self.instance = instance
        self.adapter = adapter
        self.config = frozen_config(adapter, instance, prices)
        self.parent = None
        self.parent_path = None
        self.requests = []
        self.request_lock = threading.Lock()
        self.rootfs = None
        self.docker_state = {}
        self.original_image = None
        self.execution_image = None
        self.capture_error = None
        self.container = None
        self.started = time.monotonic()
        self.execution_seconds = 0.0
        self.session_id = str(uuid.uuid4())
        self.limit = instance.effective_timeout_seconds
        self.previous_charge = 0.0
        self.previous_segments = []
        if source:
            self.parent_path, self.parent = load_checkpoint(source)
            if self.parent['config'] != self.config:
                raise ValueError('Task inputs, image configuration or frozen experiment parameters changed')
            self.session_id = self.parent['session_id']
            self.limit = self.parent['budget']['limit_seconds']
            self.previous_charge = self.parent['budget']['charged_seconds']
            self.previous_segments = self.parent['segments']
            if instance.effective_timeout_seconds != self.limit:
                raise ValueError('Resume cannot reset or change the original time budget')
        root.mkdir(parents=True, exist_ok=False)
        root.chmod(0o700)
        home = root / 'home'
        if self.parent:
            copy_tree(self.parent_path.parent / 'home', home)
            if (self.parent_path.parent / 'history').is_dir():
                copy_tree(self.parent_path.parent / 'history', root / 'history')
            workspace = instance.workspace_dir
            # Prepared input fingerprint was checked before replacing fresh workspace.
            for entry in workspace.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            shutil.copytree(self.parent_path.parent / 'workspace', workspace, symlinks=True, dirs_exist_ok=True)
        else:
            home.mkdir(mode=0o700)
        # Docker resolves paths on the host daemon, not inside the outer worker.
        launcher = root / 'launcher.py'
        shutil.copyfile(HERE / 'session_client.py', launcher)
        launcher.chmod(0o644)
        if os.geteuid() == 0:
            for entry in [root, home, *home.rglob('*')]:
                os.chown(entry, 1000, 1000, follow_symlinks=False)
        self.write_running()

    def write_running(self):
        atomic_json(self.root / 'checkpoint.json', {
            'schema': SCHEMA, 'complete': False, 'state': 'running',
            'session_id': self.session_id, 'config': self.config,
            'parent': str(self.parent_path) if self.parent_path else None})

    @property
    def remaining(self):
        # Round down: never add time by rounding fractions upward.
        return math.floor(self.limit - self.previous_charge)

    def observe(self, request):
        with self.request_lock:
            self.requests.append(request)
            with (self.root / 'requests.jsonl').open('a') as stream:
                stream.write(json.dumps(request, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

    def capture(self, container):
        """Freeze before snapshot; full image is portable across pool nodes.

        Bind-mounted home/workspace are copied separately. tmpfs (/tmp and /dev/shm)
        and processes are intentionally not checkpointed.
        """
        self.docker_state = json.loads(docker('inspect', '--format', '{{json .State}}', container))
        if self.docker_state.get('Running'):
            docker('stop', '--time', '2', container, timeout=20)
            self.docker_state = json.loads(docker('inspect', '--format', '{{json .State}}', container))

    def save_image(self):
        image_id = docker('commit', self.container, timeout=300)
        if not image_id.startswith('sha256:'):
            raise ValueError('Docker commit did not return an immutable image ID')
        temporary = self.root / 'rootfs-image.tar.tmp'
        try:
            docker('save', '--output', temporary, image_id, timeout=1800)
            temporary.replace(self.root / 'rootfs-image.tar')
            self.rootfs = {'image_id': image_id, 'base_image_id': self.original_image,
                           'sha256': digest(self.root / 'rootfs-image.tar')}
        finally:
            # The archive owns the snapshot; do not leak untagged images on workers.
            try:
                docker('image', 'rm', image_id)
            except (OSError, subprocess.SubprocessError):
                pass

    def finish(self, output, error=None):
        status = output.status.value if output is not None else 'failed'
        failure = classify(status, output.error_message if output else '', self.requests, self.docker_state)
        self.capture_error = self.capture_error or (type(error).__name__ if error else None)
        if failure in {'external_api', 'external_infrastructure'} and self.container and not self.capture_error:
            try:
                self.save_image()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.capture_error = type(exc).__name__
        try:
            history = self.root / 'history' / uuid.uuid4().hex
            history.mkdir(parents=True)
            (history / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.requests))
            if output is not None:
                for name, content in [('stdout.jsonl', output.raw_stdout), ('stderr.log', output.raw_stderr)]:
                    text = content or ''
                    if isinstance(text, bytes):
                        text = text.decode(errors='replace')
                    for name_env, secret in os.environ.items():
                        if secret and len(secret) >= 8 and any(k in name_env.upper() for k in ('API_KEY', 'BEARER_TOKEN')):
                            text = text.replace(secret, '<redacted>')
                    if self.adapter.api_key:
                        text = text.replace(self.adapter.api_key, '<redacted>')
                    (history / name).write_text(text)
            copy_tree(self.instance.workspace_dir, self.root / 'workspace')
            files = {name: inventory(self.root / name) for name in ('home', 'workspace', 'history')}
        except (OSError, ValueError, shutil.Error) as exc:
            files = {}
            self.capture_error = type(exc).__name__
        session_files = list((self.root / 'home' / '.claude' / 'projects').rglob(self.session_id + '.jsonl'))
        session_present = len(session_files) == 1 and session_files[0].is_file() and session_files[0].stat().st_size > 0
        charge = self.execution_seconds
        segment = {'id': self.root.name + '-' + uuid.uuid4().hex,
                   'checkpoint': str(self.root / 'checkpoint.json'),
                   'history': str(history.relative_to(self.root)),
                   'status': status, 'failure_class': failure,
                   'charged_seconds': charge,
                   'api_request_seconds': sum(r.get('elapsed_seconds', 0) for r in self.requests),
                   'confirmed_failed_api_seconds': sum(r.get('elapsed_seconds', 0) for r in self.requests
                       if r.get('status', 0) >= 400 or r.get('transport_error') or r.get('stream_error')),
                   'api_wait_refunded_seconds': 0,
                   'accounting': estimate_cost(usage_totals(self.requests), self.config['token_prices_usd_per_million']),
                   'request_count': len(self.requests)}
        segments = self.previous_segments + [segment]
        totals = {key: sum(s['accounting']['observed_tokens'][key] for s in segments)
                  for key in segment['accounting']['observed_tokens']}
        record = {'schema': SCHEMA, 'complete': bool(session_present and self.rootfs and files and not self.capture_error),
                  'state': 'finished', 'session_id': self.session_id,
                  'instance_bundle': str(self.instance.reference_dir.parent.resolve()),
                  'connection': {'api_key_env': self.adapter.api_key_env or 'ANTHROPIC_API_KEY',
                                 'api_base_env': self.adapter.api_base_env, 'api_base': self.adapter.api_base},
                  'config': self.config, 'config_sha256': fingerprint(self.config),
                  'parent': str(self.parent_path) if self.parent_path else None,
                  'failure_class': failure, 'capture_error': self.capture_error,
                  'termination': {'exit_code': self.docker_state.get('ExitCode'),
                                  'oom_killed': self.docker_state.get('OOMKilled')},
                  'files': files, 'rootfs': self.rootfs,
                  'budget': {'limit_seconds': self.limit, 'charged_seconds': self.previous_charge + charge,
                             'remaining_seconds': max(0, self.limit - self.previous_charge - charge),
                             'policy': 'conservative-walltime-no-api-refund-v1'},
                  'segments': segments,
                  'accounting': {'observed_tokens': totals,
                                 'usage_complete': all(s['accounting']['usage_complete'] for s in segments),
                                 'actual_cost_usd': None,
                                 'estimated_cost_usd': (sum(s['accounting']['estimated_cost_usd'] for s in segments)
                                                       if self.config['token_prices_usd_per_million'] is not None else None),
                                 'estimate_basis': 'observed-usage-only'},
                  'limitations': ['Processes, tmpfs and /dev/shm are not restored',
                                  'Provider billing is unknown; observed tokens include context replay',
                                  'No automatic retry or recovery of incomplete checkpoints']}
        atomic_json(self.root / 'checkpoint.json', record)
        if self.parent_path:
            atomic_json(self.parent_path.parent / 'continuation.json',
                        {'destination': str(self.root / 'checkpoint.json'), 'state': 'finished'})
        return record

    def prepare_container(self, container, image_identity, exec_args, mounts):
        self.container = container
        self.original_image = image_identity
        self.started = time.monotonic()
        self.deadline = self.started + self.remaining
        image = self.execution_image if self.parent else None
        if self.parent and image_identity != self.parent['rootfs']['base_image_id']:
            raise ValueError('Task base image changed; refusing session recovery')
        command = list(exec_args)
        index = command.index('/opt/asibench-network/client.py')
        command[index:index + 1] = ['/opt/asibench-resume/launcher.py', str(self.deadline)]
        mounts = [*mounts, '-v', f'{self.root / "home"}:/opt/asibench-resume/home:rw',
                  '-v', f'{self.root / "launcher.py"}:/opt/asibench-resume/launcher.py:ro']
        if self.parent:
            self.claim.begin_execution()
        return image, command, mounts

    def capture_safely(self, container):
        self.execution_seconds += time.monotonic() - self.started
        try:
            self.capture(container)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.capture_error = type(exc).__name__


def run_checkpointed(adapter, instance, solve):
    """Continue one logical execution, keeping each physical segment immutable."""
    from ai4sci_bench.core.types import CostInfo, RunStatus
    options = adapter.checkpoint_options
    key = instance_key(instance)
    root = Path(options['root']).resolve() / key
    source = options.get('resume', {}).get(key)
    if options.get('resume') and source is None:
        raise ValueError('Instance is absent from the explicit resume plan')
    claim = Claim(source, root / 'checkpoint.json') if source else nullcontext()
    with claim:
        context = CheckpointExecution(adapter, instance, root, source, options.get('token_prices'))
        context.claim = claim
        output, error = None, None
        try:
            if source:
                docker('load', '--input', context.parent_path.parent / 'rootfs-image.tar', timeout=1800)
                context.execution_image = context.parent['rootfs']['image_id']
                if docker('image', 'inspect', '--format', '{{.Id}}', context.execution_image) != context.execution_image:
                    raise ValueError('Restored container image identity mismatch')
            output = solve(instance, checkpoint=context)
            output.process_exit_code = context.docker_state.get('ExitCode')
            if output.process_exit_code == 124:
                output.status = RunStatus.TIMEOUT
                output.error_message = 'Original cumulative execution time budget exhausted'
            return output
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                record = context.finish(output, error)
                if output is not None:
                    output.execution_time_seconds = record['budget']['charged_seconds']
                    account = record['accounting']
                    if account['estimated_cost_usd'] is None:
                        output.cost = None
                    else:
                        usage = account['observed_tokens']
                        inputs = usage['input_tokens'] + usage['cache_read_input_tokens'] + usage['cache_creation_input_tokens']
                        output.cost = CostInfo(input_tokens=inputs, output_tokens=usage['output_tokens'],
                                              total_tokens=inputs + usage['output_tokens'],
                                              estimated_cost_usd=account['estimated_cost_usd'])
                    output.network_runtime = output.network_runtime or {}
                    output.network_runtime['session_checkpoint'] = {
                        'schema': SCHEMA, 'path': str(root / 'checkpoint.json'),
                        'session_id': context.session_id, 'segment_count': len(record['segments']),
                        'failure_class': record['failure_class'], 'budget': record['budget'],
                        'accounting': account}
            finally:
                if context.container:
                    adapter._os_sandbox._remove_container(context.container)
                if context.execution_image:
                    try:
                        docker('image', 'rm', context.execution_image)
                    except (OSError, subprocess.SubprocessError):
                        pass
