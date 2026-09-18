"""Trusted evaluator materialization and scoring input snapshots; no scoring CLI yet.

Never imports ASI or executes task/prediction code on the host. The manifest
is trusted adapter configuration, not an arbitrary user-supplied code manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

ADAPTER_ROOT = Path(__file__).resolve().parent
EVALUATOR_CACHE = Path.home() / '.cache/benchflow/asi_bench/evaluator_sources'
DEFAULT_SPEC = ADAPTER_ROOT / 'evaluator_files.json'
INITIALIZERS = {
    'ai4sci_bench/core/__init__.py': (
        'from .types import PromptLevel, ScoreDetail\n'
        'from .scorer import Scorer, get_scorer, register_scorer\n'
    ),
    'ai4sci_bench/scorers/__init__.py': '',
    'ai4sci_bench/runner/__init__.py': '',
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def _relative(value: str) -> Path:
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or any(x in ('', '.', '..') for x in value.split('/'))):
        raise ValueError(f'unsafe evaluator path: {value!r}')
    return Path(value)


def _regular_bytes(root: Path, relative: str) -> bytes:
    path = root
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'unsafe evaluator root: {root}')
    parts = _relative(relative).parts
    for index, component in enumerate(parts):
        path = path / component
        mode = path.lstat().st_mode
        wanted = stat.S_ISREG(mode) if index == len(parts)-1 else stat.S_ISDIR(mode)
        if not wanted:
            raise ValueError(f'unsafe evaluator file: {path}')
    with path.open('rb') as stream:
        data = stream.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError(f'evaluator file exceeds size limit: {path}')
    return data


def _validate_spec(spec: dict) -> None:
    """Validate copy paths and derive local code identities, not manual hash pins."""
    for entry in spec['files']:
        _relative(entry['path'])
        kind = entry['kind']
        if kind == 'copy':
            _relative(entry['source'])
            if not re.fullmatch(r'[0-9a-f]{64}', entry['source_sha256']):
                raise ValueError('invalid upstream source digest')
            entry['sha256'] = entry['source_sha256']
        elif kind == 'initializer' and entry['path'] in INITIALIZERS:
            entry['sha256'] = _sha(INITIALIZERS[entry['path']].encode())
        elif kind == 'runtime_adapter' and entry['path'] == 'ai4sci_bench/runner/task_env.py':
            entry['sha256'] = _sha(_regular_bytes(ADAPTER_ROOT, entry['template']))
        else:
            raise ValueError('unsupported evaluator file kind/path')
    for key in ('bridge', 'launcher', 'scoring'):
        entry = spec[key]
        entry['sha256'] = _sha(_regular_bytes(ADAPTER_ROOT, entry['path']))


def load_evaluator_spec(spec_path: Path = DEFAULT_SPEC) -> dict:
    spec_path = Path(spec_path)
    spec = json.loads(_regular_bytes(spec_path.parent, spec_path.name))
    try:
        _validate_spec(spec)
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid evaluator spec schema') from exc
    return spec


def _manifest(spec: dict) -> dict:
    payload = {'schema_version': 1, 'spec_sha256': _sha(_canonical(spec)),
               'upstream': spec['upstream'], 'bridge': spec['bridge'], 'launcher': spec['launcher'], 'scoring': spec['scoring'], 'files': spec['files']}
    return {**payload, 'bundle_digest': _sha(_canonical(payload))}


def scoring_profile_identity(spec: dict | None = None) -> dict:
    """Static P2.1 identity only; no ASI imports, cache writes or Docker calls."""
    if spec is None:
        spec = load_evaluator_spec()
    else:
        _validate_spec(spec)
    manifest = _manifest(spec)
    return {
        'profile_id': 'asi-seed31415-v1',
        'launcher': 'verifier/test.sh',
        'entrypoint': 'verifier/score_entry.py',
        'evaluator_spec_sha256': manifest['spec_sha256'],
        'expected_evaluator_bundle_digest': manifest['bundle_digest'],
    }


def download_evaluator_sources(spec: dict) -> Path:
    """Cache only hash-pinned upstream files; never execute them on the host."""
    upstream = spec['upstream']
    root = EVALUATOR_CACHE / upstream['revision']
    for entry in spec['files']:
        if entry['kind'] != 'copy':
            continue
        relative = entry['source']
        if root.exists():
            try:
                if _sha(_regular_bytes(root, relative)) == entry['source_sha256']:
                    continue
            except FileNotFoundError:
                pass
        url = f"https://raw.githubusercontent.com/{upstream['repo']}/{upstream['revision']}/{relative}"
        with urlopen(url, timeout=30) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError(f'evaluator file exceeds size limit: {relative}')
        if _sha(data) != entry['source_sha256']:
            raise ValueError(f'downloaded evaluator file mismatch: {relative}')
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return root


def materialize_evaluator(source_root: Path | None, destination: Path,
                          spec: dict | None = None) -> Path:
    """Assemble pinned sources and local replacements into a fresh runtime."""
    spec = load_evaluator_spec() if spec is None else spec
    _validate_spec(spec)
    destination = Path(destination)
    if source_root is None:
        source_root = download_evaluator_sources(spec)
    contents = {}
    for entry in spec['files']:
        if entry['kind'] == 'copy':
            data = _regular_bytes(source_root, entry['source'])
            if _sha(data) != entry['source_sha256']:
                raise ValueError(f"upstream working file mismatch: {entry['source']}")
        elif entry['kind'] == 'runtime_adapter':
            data = _regular_bytes(ADAPTER_ROOT, entry['template'])
        else:
            data = INITIALIZERS[entry['path']].encode()
        contents[entry['path']] = data
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for relative, data in contents.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o644)
        (destination / 'evaluator_manifest.json').write_bytes(_canonical(_manifest(spec)) + b'\n')
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination


# P2.2: host input validation only. Never import task or prediction code.
MAX_INPUT_FILE_BYTES = 64 * 1024 * 1024
MAX_INPUT_TOTAL_BYTES = 512 * 1024 * 1024
MAX_INPUT_FILES = 4096


def _input_path(value: str) -> Path:
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(ord(c) < 32 for c in value)
            or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('unsafe scoring input path')
    return Path(value)


def _input_root(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for parent in reversed((path, *path.parents)):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError('unsafe scoring input directory')
    return path


def _input_bytes(root: Path, name: str, limit: int | None = None) -> bytes:
    """Descriptor-relative traversal: do not follow replaced parents, links or FIFOs."""
    limit = MAX_INPUT_FILE_BYTES if limit is None else limit
    parts = _input_path(name).parts
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        child = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(child, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError('unsafe or oversized scoring input')
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            if (len(data) > limit or len(data) != before.st_size
                    or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ValueError('scoring input changed while reading')
            return data
    finally:
        os.close(fd)


def _input_digest(data: bytes) -> dict:
    return {'bytes': len(data), 'sha256': _sha(data)}


def _input_tree(root: Path, declared=()) -> dict:
    from benchmarks.asi_bench.prepare import is_inventory_noise

    _input_root(root)
    files, total, directory_count = {}, 0, 0
    def raise_walk_error(error):
        raise error

    for parent, dirs, names in os.walk(root, followlinks=False, onerror=raise_walk_error):
        directory_count += len(dirs)
        if directory_count > MAX_INPUT_FILES:
            raise ValueError('scoring input directory count exceeds limit')
        for name in dirs:
            if not stat.S_ISDIR((Path(parent) / name).lstat().st_mode):
                raise ValueError('unsafe scoring input directory')
        for name in sorted(names):
            relative = (Path(parent) / name).relative_to(root).as_posix()
            if relative not in declared and is_inventory_noise(Path(parent) / name):
                continue
            if len(files) >= MAX_INPUT_FILES:
                raise ValueError('scoring input file count exceeds limit')
            data = _input_bytes(root, relative)
            total += len(data)
            if total > MAX_INPUT_TOTAL_BYTES:
                raise ValueError('scoring input total bytes exceed limit')
            files[relative] = _input_digest(data)
    return files


def _decode_input_json(data: bytes) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate scoring JSON field')
            value[key] = item
        return value
    value = json.loads(data, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError('scoring input JSON must be an object')
    return value


def _input_json(root: Path, name: str) -> dict:
    return _decode_input_json(_input_bytes(root, name, 8 * 1024 * 1024))


def _model_provenance(route: dict) -> dict:
    from urllib.parse import urlsplit

    if not isinstance(route, dict) or set(route) != {'endpoint', 'api_key_env', 'routing'}:
        raise ValueError('invalid model route provenance')
    endpoint = route['endpoint']
    if endpoint is not None:
        if not isinstance(endpoint, str):
            raise ValueError('invalid model endpoint')
        url = urlsplit(endpoint)
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError('unsafe model endpoint')
    if route['api_key_env'] not in (None, 'ASI_MODEL_API_KEY') or route['routing'] not in ('explicit', 'native_environment'):
        raise ValueError('invalid model route selector')
    return dict(route)


def inspect_scoring_inputs(prepared_dir: Path, result_file: Path) -> dict:
    """Cross-check local execution evidence, not a cryptographic authenticity claim."""
    from benchmarks.asi_bench.solve import inspect_prepared_task, result_layout

    prepared = _input_root(prepared_dir)
    result_file = Path(os.path.abspath(result_file))
    run = _input_root(result_file.parent.parent)
    # Bound all reads before invoking existing prepared/raw integrity checks.
    manifest = _input_json(prepared, 'task_manifest.json')
    prepared_files = _input_tree(prepared, {entry['target'] for entry in manifest['files']})
    raw = _input_root(Path(manifest['source_assets']['raw_dir']))
    if run.is_relative_to(raw) or run.is_relative_to(prepared):
        raise ValueError('run overlaps immutable inputs')
    source_manifest = _input_json(raw, 'sources.json')
    raw_files = _input_tree(raw, {entry['path'] for entry in source_manifest['files']})
    contract = inspect_prepared_task(prepared)
    spec = load_evaluator_spec()
    profile = scoring_profile_identity(spec)
    prepared_spec = spec
    if manifest.get('scoring') != profile:
        # Exact pre-workspace bridge only. It is validated but never executed;
        # reports bind the current evaluator, while prepared stays immutable.
        prepared_spec = {**spec, 'bridge': {**spec['bridge'], 'adaptation_version': 6,
            'sha256': '9b6f90917cdfedf8c8ea35b01b2b1730830cd31127c28e862e3a6355f1f793a2'}}
        legacy = _manifest(prepared_spec)
        legacy_profile = {**profile, 'evaluator_spec_sha256': legacy['spec_sha256'],
                          'expected_evaluator_bundle_digest': legacy['bundle_digest']}
        if manifest.get('scoring') != legacy_profile:
            raise ValueError('unsupported prepared scoring identity')
    for field, name in (('launcher', 'test.sh'), ('bridge', 'score_entry.py'), ('scoring', 'scoring.py')):
        if prepared_files['verifier/' + name]['sha256'] != prepared_spec[field]['sha256']:
            raise ValueError('prepared scoring asset identity mismatch')
    meta_bytes = _input_bytes(run, 'run_metadata.json', 8 * 1024 * 1024)
    meta = _decode_input_json(meta_bytes)
    result_relative = result_file.relative_to(run).as_posix()
    result_bytes = _input_bytes(run, result_relative, 8 * 1024 * 1024)
    result = _decode_input_json(result_bytes)
    if meta.get('result_file') != result_relative or not isinstance(meta.get('finished_at'), str) or not meta['finished_at']:
        raise ValueError('unfinished or misbound run metadata')
    task_id = manifest['task']['id']
    instance_id, level = manifest['instance']['id'], manifest['task']['level']
    for key, expected in {'adapter_schema_version': 1, 'official': False, 'mode': 'produce-only',
                          'task_id': task_id, 'instance_id': instance_id, 'prompt_level': level,
                          'status': 'completed', 'attempt_status': 'completed',
                          'failure_reason': None, 'evaluation_status': 'pending'}.items():
        if key not in result or key not in meta or result[key] != expected or meta[key] != expected:
            raise ValueError('ineligible or inconsistent run identity/status: ' + key)
    for key in ('attempt', 'agent', 'model', 'requested_effort', 'model_route', 'sources', 'prepared', 'started_at', 'budgets'):
        if key not in result or key not in meta or result[key] != meta[key]:
            raise ValueError('result metadata binding mismatch: ' + key)
    for key in ('agent', 'model'):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError('missing agent/model provenance')
    if not isinstance(result['requested_effort'], (str, type(None))):
        raise ValueError('invalid effort provenance')
    route = _model_provenance(result['model_route'])
    expected_prepared = {'converter_version': 'prepare-v5', 'manifest_sha256': prepared_files['task_manifest.json']['sha256']}
    if result['prepared'] != expected_prepared or result['sources'] != manifest['sources']:
        raise ValueError('prepared/source result binding mismatch')
    layout = result_layout(run, task_id, instance_id, level, result['attempt'])
    if result_file != layout.result_file:
        raise ValueError('unexpected result filename')
    for key, expected in (('collection_status', 'collected'), ('prediction_status', 'present'), ('cleanup_status', 'completed')):
        if result.get(key) != expected:
            raise ValueError('ineligible result status: ' + key)
    if (result.get('diagnostics') != [] or result.get('native_error_present') is not False
            or result.get('native_error_category') is not None):
        raise ValueError('result contains execution diagnostics')
    native_path = _input_path(result['native_result'])
    if native_path.parts[0] != 'benchflow' or native_path.name != 'result.json':
        raise ValueError('native result is outside rollout evidence')
    native_bytes = _input_bytes(run, native_path.as_posix(), 8 * 1024 * 1024)
    native = _decode_input_json(native_bytes)
    for key in ('error', 'error_category', 'verifier_error', 'export_error', 'rewards'):
        if key not in native or native[key] is not None:
            raise ValueError('native execution or verifier failure: ' + key)
    # Native serialization omits scoring when no verifier was run.
    if native.get('scoring') is not None:
        raise ValueError('native execution or verifier failure: scoring')
    if (native.get('task_name') != prepared.name or native.get('rollout_name') != native_path.parent.name
            or native.get('agent') != result['agent'] or native.get('model') != result['model']
            or not native.get('finished_at')):
        raise ValueError('native result identity mismatch')
    collection = result['collection']
    if (collection.get('missing') != [] or collection.get('invalid') != []
            or collection.get('prediction_status') != 'present'):
        raise ValueError('incomplete collection')
    freeze = collection.get('freeze', {})
    if (freeze.get('method') != 'docker-stop' or freeze.get('running') is not False
            or type(freeze.get('pid')) is not int or freeze['pid'] != 0
            or freeze.get('shared_workspace') is not False
            or not isinstance(freeze.get('container_id'), str)
            or not re.fullmatch('[0-9a-f]{64}', freeze['container_id'])
            or result.get('network', {}).get('container_id') != freeze['container_id']):
        raise ValueError('missing or inconsistent stopped-writer evidence')
    output = result['agent_output']
    persisted = output['persisted_outputs']
    if (persisted.get('dir') != layout.outputs_dir.name
            or output.get('code_files') != [s.name for s in contract.outputs if s.type == 'code']
            or output.get('data_files') != [s.name for s in contract.outputs if s.type == 'data']):
        raise ValueError('persisted output path/declarations mismatch')
    output_types = {s.name: s.type for s in contract.outputs}
    prediction_files = _input_tree(layout.outputs_dir, output_types)
    if set(prediction_files) != set(output_types):
        raise ValueError('unexpected prediction inventory')
    entries = persisted.get('files', [])
    artifacts = collection.get('artifacts', [])
    if len(entries) != len(output_types) or len(artifacts) != len(output_types):
        raise ValueError('duplicate or missing prediction entries')
    expected_entries = [{'path': name, **info} for name, info in prediction_files.items()]
    expected_artifacts = [{'name': name, 'type': output_types[name],
                           'size': info['bytes'], 'sha256': info['sha256']} for name, info in prediction_files.items()]
    if (sorted(entries, key=lambda x: x['path']) != sorted(expected_entries, key=lambda x: x['path'])
            or sorted(artifacts, key=lambda x: x['name']) != sorted(expected_artifacts, key=lambda x: x['name'])):
        raise ValueError('prediction/collection digest mismatch')
    inventories = {prepared: prepared_files, raw: raw_files, layout.outputs_dir: prediction_files}
    inputs = {}
    for name in raw_files:
        task_prefix = 'task_bundle/' + manifest['sources']['github']['path'] + '/'
        instance_prefix = 'instance/' + instance_id + '/'
        if name.startswith(task_prefix):
            inputs['task_bundle/' + name[len(task_prefix):]] = (raw, name)
        elif name.startswith(instance_prefix):
            relative = name[len(instance_prefix):]
            if relative == 'instance_meta.json' or relative.startswith(('data/', 'reference/')):
                inputs['instance/' + relative] = (raw, name)
    for name in prediction_files:
        inputs['prediction/' + name] = (layout.outputs_dir, name)
    inputs['instance_parameters.json'] = (prepared, 'verifier/instance_parameters.json')
    required = {'task_bundle/task_meta.yaml', 'task_bundle/task_eval.yaml', 'instance/instance_meta.json'}
    if not required <= inputs.keys():
        raise ValueError('missing required scoring/reference inputs')
    instance_meta = _input_json(raw, f'instance/{instance_id}/instance_meta.json')
    if (instance_meta.get('task_id') != task_id
            or _input_json(prepared, 'verifier/instance_parameters.json') !=
            (instance_meta.get('params_used') or {})):
        raise ValueError('instance parameter binding mismatch')
    inventory = {name: inventories[root][rel] for name, (root, rel) in inputs.items()}
    if len(inventory) > MAX_INPUT_FILES or sum(e['bytes'] for e in inventory.values()) > MAX_INPUT_TOTAL_BYTES:
        raise ValueError('combined scoring inputs exceed limits')
    provenance = {'result_sha256': _sha(result_bytes),
                  'native_result_sha256': _sha(native_bytes),
                  'prepared_manifest_sha256': expected_prepared['manifest_sha256'],
                  'source_manifest_sha256': raw_files['sources.json']['sha256'],
                  **{k: result[k] for k in ('agent', 'model', 'requested_effort')},
                  'sources': {k: {field: value[field] for field in ('repo', 'path', 'resolved_revision')}
                              for k, value in manifest['sources'].items()}}
    image_id = result.get('network', {}).get('image_id')
    if not isinstance(image_id, str) or not re.fullmatch('sha256:[0-9a-f]{64}', image_id):
        raise ValueError('missing immutable solve image ID; rerun solve')
    request = {'schema_version': 2, 'official': False, 'seed': 31415, 'task_id': task_id,
               'instance_id': instance_id, 'prompt_level': level,
               'attempt': {'number': result['attempt'], **{k: result[k] for k in
                           ('attempt_status', 'collection_status', 'prediction_status')},
                           'persisted_outputs': prediction_files},
               'provenance': provenance, 'files': inventory}
    evidence = {
        (run, 'run_metadata.json'): _input_digest(meta_bytes),
        (run, result_relative): _input_digest(result_bytes),
        (run, native_path.as_posix()): _input_digest(native_bytes),
        (prepared, 'task_manifest.json'): prepared_files['task_manifest.json'],
        (raw, 'sources.json'): raw_files['sources.json'],
    }
    return {'inputs': inputs, 'evidence': evidence,
            'image_id': image_id, 'prepared_dir': prepared,
            'environment_files': {name: digest for name, digest in prepared_files.items()
                                  if name.startswith('environment/')},
            'request_fields': request,
            'host_provenance': {'solve_image_id': image_id, 'model_route': route, 'run_metadata_sha256': _sha(meta_bytes)},
            'scoring': profile}


def _remove_snapshot(path: Path) -> None:
    if path.exists():
        for parent, _, _ in os.walk(path):
            os.chmod(parent, 0o700)
        shutil.rmtree(path)


@contextmanager
def scoring_input_snapshot(checked: dict):
    """Yield temporary read-only input plus request fields; runtime is bound later."""
    # Resolve only our OS-selected temporary parent (macOS /var -> /private/var).
    temporary = Path(tempfile.mkdtemp(prefix='asi-scoring-')).resolve()
    root = temporary / 'input'
    try:
        root.mkdir()
        for name, (source, relative) in checked['inputs'].items():
            data = _input_bytes(source, relative)
            if _input_digest(data) != checked['request_fields']['files'][name]:
                raise ValueError('input changed during snapshot copy')
            target = root / _input_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
        for (source, relative), expected in checked['evidence'].items():
            if _input_digest(_input_bytes(source, relative)) != expected:
                raise ValueError('source evidence changed during snapshot copy')
        for path in root.rglob('*'):
            path.chmod(0o500 if path.is_dir() else 0o400)
        root.chmod(0o500)
        yield {'input_dir': root, 'request_fields': checked['request_fields'],
               'host_provenance': checked['host_provenance'], 'scoring': checked['scoring']}
    finally:
        _remove_snapshot(temporary)


def inspect_scoring_image(image_id: str) -> dict:
    """Inspect an existing content-addressed image; never pull, build or commit."""
    if not isinstance(image_id, str) or not re.fullmatch('sha256:[0-9a-f]{64}', image_id):
        raise ValueError('expected immutable scoring image ID')
    result = subprocess.run(['docker', 'image', 'inspect', image_id], check=True,
                            capture_output=True, text=True, timeout=30)
    image = json.loads(result.stdout)[0]
    if image['Id'] != image_id or image['Os'] != 'linux':
        raise ValueError('scoring image identity/platform mismatch')
    return image


def materialize_scoring_workspace(inputs: Path, destination: Path) -> None:
    """Recreate ASI's prediction workspace from verified outputs and public data."""
    shutil.copytree(inputs / 'prediction', destination)
    data = inputs / 'instance/data'
    if data.is_dir():
        # Do not merge Agent output into trusted instance data.
        shutil.copytree(data, destination / 'data')
    for path in [destination, *destination.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)


async def _score_snapshot(snapshot: dict, image_id: str, source_root: Path | None,
                          output_dir: Path, *, timeout: int = 600) -> dict:
    """Run trusted snapshot in a fresh, offline container of the scoring image.

    Internal boundary also used by frozen-input integration tests. Production
    callers must enter through score_result, which validates the result chain.
    """
    import asyncio
    import uuid

    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    if timeout <= 0:
        raise ValueError('scoring timeout must be positive')
    inspect_scoring_image(image_id)
    spec = load_evaluator_spec()
    evaluator = _manifest(spec)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='asi-score-') as temporary:
        root = Path(temporary).resolve()
        root.chmod(0o755)
        bundle = materialize_evaluator(source_root, root / 'evaluator', spec)
        inputs, bridge, exported = (root / name for name in ('input', 'bridge', 'output'))
        shutil.copytree(snapshot['input_dir'], inputs)
        for path in [inputs, *inputs.rglob('*')]:
            path.chmod(0o755 if path.is_dir() else 0o644)
        runtime = {'image_id': image_id, 'evaluator_bundle_digest': evaluator['bundle_digest'],
                   'bridge_sha256': spec['bridge']['sha256'],
                   'launcher_sha256': spec['launcher']['sha256']}
        (inputs / 'evaluation_manifest.json').write_bytes(
            _canonical({**snapshot['request_fields'], 'runtime': runtime}) + b'\n')
        workspace = root / 'scoring-workspace'
        materialize_scoring_workspace(inputs, workspace)
        bridge.mkdir()
        for key, name in [('bridge', 'score_entry.py'), ('launcher', 'test.sh'), ('scoring', 'scoring.py')]:
            (bridge / name).write_bytes(_regular_bytes(ADAPTER_ROOT, spec[key]['path']))
        exported.mkdir(mode=0o777)
        exported.chmod(0o777)
        environment = root / 'environment'
        environment.mkdir()
        # Bind only this invocation's frozen inputs/code/output, never Agent
        # workspace or container writable layers. No build or mutable image tag.
        volumes = [{'type': 'bind', 'source': str(source), 'target': target, 'read_only': readonly}
                   for source, target, readonly in [(inputs, '/input', True),
                       (bridge, '/opt/bridge', True), (bundle, '/opt/asi-evaluator', True),
                       (exported, '/output', False), (workspace, '/prediction', True)]]
        compose = {'services': {'main': {
            'image': image_id, 'pull_policy': 'never', 'user': '10001:10001',
            'read_only': True, 'network_mode': 'none', 'cap_drop': ['ALL'],
            'security_opt': ['no-new-privileges:true'],
            'deploy': {'resources': {'limits': {'pids': 128}}},
            'tmpfs': ['/tmp:rw,nosuid,nodev,size=536870912,mode=1777'],
            'volumes': volumes, 'working_dir': '/output',
        }}}
        # JSON is a YAML subset; avoids shell interpolation of host paths.
        (environment / 'docker-compose.yaml').write_text(json.dumps(compose))
        paths = RolloutPaths(root / 'rollout')
        paths.mkdir()
        sandbox = DockerSandbox(environment_dir=environment, environment_name='asi-score',
            session_id=uuid.uuid4().hex, rollout_paths=paths,
            task_env_config=SandboxConfig(docker_image=image_id, network_mode='no-network',
                                          cpus=2, memory_mb=4096))
        try:
            await asyncio.wait_for(sandbox.start(force_build=False), 180)
            executed = await asyncio.wait_for(sandbox.exec(
                '/bin/sh /opt/bridge/test.sh', user='10001:10001', cwd='/output',
                env={'PYTHONDONTWRITEBYTECODE': '1', 'HOME': '/tmp',
                     'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                timeout_sec=timeout), timeout + 10)
            (output_dir / 'execution.json').write_bytes(_canonical({
                'image_id': image_id, 'solve_image_id': snapshot['host_provenance']['solve_image_id'],
                'return_code': executed.return_code,
                'stdout': executed.stdout, 'stderr': executed.stderr}) + b'\n')
        except BaseException as exc:
            (output_dir / 'execution.json').write_bytes(_canonical({
                'image_id': image_id, 'evaluation_status': 'failed',
                'error': type(exc).__name__, 'message': str(exc)}) + b'\n')
            raise
        finally:
            # Kill scorer descendants before reading output, including timeout
            # and partial-start paths. Failure never publishes a reward.
            await asyncio.wait_for(sandbox.stop(delete=False), 180)
        report = _input_json(exported, 'asibench_score.json')
        (output_dir / 'asibench_score.json').write_bytes(_canonical(report) + b'\n')
        if executed.return_code != 0 or report.get('evaluation_status') != 'completed':
            raise RuntimeError('scoring failed; see execution.json and asibench_score.json')
        import math

        reward = report.get('reward')
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward) or not 0 <= reward <= 1
                or float(_input_bytes(exported, 'reward.txt')) != reward):
            raise ValueError('invalid scoring reward')
        (output_dir / 'reward.txt').write_text(str(reward) + '\n')
        return report


def build_scoring_image(checked: dict, *, timeout: int = 600) -> str:
    """Build a fresh scoring image from a verified copy of prepared/environment."""
    with tempfile.TemporaryDirectory(prefix='asi-score-build-') as temporary:
        root = Path(temporary)
        context = root / 'environment'
        context.mkdir()
        for name, expected in checked['environment_files'].items():
            data = _input_bytes(checked['prepared_dir'], name)
            if _input_digest(data) != expected:
                raise ValueError('prepared environment changed before scoring build: ' + name)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        iidfile = root / 'image-id'
        # Inherit stdout/stderr so Docker's original build errors remain visible.
        subprocess.run(['docker', 'build', '--iidfile', str(iidfile), str(context)],
                       check=True, timeout=timeout)
        image_id = iidfile.read_text().strip()
        inspect_scoring_image(image_id)
        return image_id


async def score_result(prepared_dir: Path, result_file: Path, output_dir: Path, *,
                       source_root: Path | None = None, timeout: int = 600) -> dict:
    """Score in an image rebuilt from the same prepared environment definition."""
    import asyncio

    if timeout <= 0:
        raise ValueError('scoring timeout must be positive')
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    checked = inspect_scoring_inputs(prepared_dir, result_file)
    with scoring_input_snapshot(checked) as snapshot:
        if source_root is None:
            source_root = await asyncio.to_thread(download_evaluator_sources, load_evaluator_spec())
        image_id = await asyncio.to_thread(build_scoring_image, checked, timeout=timeout)
        return await _score_snapshot(snapshot, image_id, source_root, output_dir,
                                     timeout=timeout)
