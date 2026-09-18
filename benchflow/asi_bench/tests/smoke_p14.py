"""Historical frozen-prediction fixtures; execution uses the production scorer."""
import hashlib
import json
from pathlib import Path

from benchmarks.asi_bench.tests.smoke_score import main


def inventory(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'symlink: {p}')
        if p.is_file():
            result[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('unsafe artifact path')
    p = Path(value)
    if p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('unsafe artifact path')
    return p


def freeze_prediction(result_path, target):
    result = json.loads(result_path.read_text())
    if result['attempt_status'] != 'completed' or result['prediction_status'] != 'present':
        raise ValueError('requires completed attempt with prediction')
    outputs = result['agent_output']['persisted_outputs']
    rel = relative(outputs['dir'])
    source = result_path.parent / rel
    for p in (source, *source.parents):
        if p.is_symlink():
            raise ValueError('unsafe prediction path')
    entries = outputs['files']
    if sorted(e['path'] for e in entries) != ['analysis.py', 'certificate.json']:
        raise ValueError('unexpected prediction inventory')
    if set(inventory(source)) != {'analysis.py', 'certificate.json'}:
        raise ValueError('unexpected prediction files')
    data = {}
    for e in entries:
        name = str(relative(e['path']))
        value = (source / name).read_bytes()
        if len(value) != e['bytes'] or hashlib.sha256(value).hexdigest() != e['sha256']:
            raise ValueError(f'prediction integrity mismatch: {name}')
        data[name] = value
    target.mkdir()
    for name, value in data.items():
        (target / name).write_bytes(value)
    return result


# Test-only observability, loaded through PYTHONPATH at interpreter startup.
# No scoring logic is implemented here; the unmodified launcher invokes the bridge.
AUDIT = r'''
import atexit, json, sys
from pathlib import Path
launches, helper_returns, native_calls = [], [], []
def save_audit():
    Path('/output/runtime-audit.json').write_text(json.dumps({
        'helper_returns': helper_returns, 'native_calls': native_calls, 'subprocesses': launches}, indent=2))
atexit.register(save_audit)
def audit(event, args):
    if event == 'subprocess.Popen':
        command = args[1]
        launches.append(list(command))
        worker = (len(command) == 2 and command[0] == '/usr/local/bin/python3.11'
            and command[1].startswith('/tmp/mpsc-submission-') and command[1].endswith('/submission_worker.py'))
        if not worker:
            raise RuntimeError('unexpected subprocess: ' + repr(command))
    elif event in ('os.system', 'os.exec', 'os.posix_spawn'):
        raise RuntimeError('unexpected process launch: ' + event)
def profile(frame, event, arg):
    if frame.f_globals.get('__name__') == 'ai4sci_bench.runner.task_env':
        name = frame.f_code.co_name
        if event == 'call':
            native_calls.append(name)
            if name in ('_build_env', '_run', '_acquire_lock'):
                raise RuntimeError('native cold-build path reached: ' + name)
        if event == 'return' and name == 'ensure_env':
            helper_returns.append({'python': str(arg.python_executable)})
sys.addaudithook(audit)
sys.setprofile(profile)
'''


if __name__ == '__main__':
    main()
