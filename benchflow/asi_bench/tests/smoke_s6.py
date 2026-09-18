"""Opt-in real Docker acceptance through solve_task, no model calls or scoring.

Unlike the historical S5 smoke, this never silently repairs a model endpoint.
ASI_S6_MODEL_ENV must explicitly supply a usable native proxy route.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from benchflow.agents.registry import register_agent
from benchmarks.asi_bench.solve import SolveConfig, solve_task


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docker', action='store_true', required=True)
    parser.add_argument('--mode', choices=['success', 'missing', 'symlink', 'background', 'agent_error'], default='success')
    args = parser.parse_args()
    from dotenv import dotenv_values

    prepared = Path(os.environ['ASI_S5_PREPARED'])
    raw = Path(os.environ['ASI_S5_RAW'])
    env_file = Path(os.environ['ASI_S6_MODEL_ENV'])
    settings = dotenv_values(env_file)
    root = Path(tempfile.mkdtemp(prefix='asi-s6-'))
    run = root / 'run'
    repo = Path(__file__).resolve().parents[3]
    source_files = sorted({
        *repo.joinpath('src/benchflow').rglob('*.py'),
        *repo.joinpath('benchmarks/asi_bench').rglob('*.py'),
        repo / 'pyproject.toml', repo / 'uv.lock',
    })
    identity = {
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in source_files},
        'mode': args.mode,
        'command': f'uv run python -m benchmarks.asi_bench.tests.smoke_s6 --docker --mode {args.mode}',
        'prepared': str(prepared), 'raw': str(raw),
    }
    (root / 'source-identity.json').write_text(json.dumps(identity, indent=2))
    before, raw_before = snapshot(prepared), snapshot(raw)
    body = (Path(__file__).parent / 'fixtures/s6_dummy.py').read_bytes()
    payload = base64.b64encode(body).decode()
    register_agent('asi-s6-dummy',
                   f'echo {payload} | base64 -d > /usr/local/bin/asi-s6-dummy.py',
                   'python3 /usr/local/bin/asi-s6-dummy.py ' + shlex.quote(args.mode))
    print('EVIDENCE', root, flush=True)
    result = await solve_task(SolveConfig(
        prepared, run, 'asi-s6-dummy', 'vllm/' + settings['ASI_MODEL_ID'],
        model_env_file=env_file,
    ))
    assert snapshot(prepared) == before and snapshot(raw) == raw_before
    (root / 'integrity.json').write_text(json.dumps({'raw_unchanged': True, 'prepared_unchanged': True}))
    key = settings['ASI_MODEL_API_KEY'].encode()
    assert not any(key in p.read_bytes() for p in root.rglob('*') if p.is_file())
    metadata = json.loads((run / 'run_metadata.json').read_text())
    persisted_result = json.loads((run / metadata['result_file']).read_text())
    assert persisted_result == result
    for name in ('status', 'attempt_status', 'failure_reason', 'evaluation_status'):
        assert metadata[name] == result[name]
    assert result['evaluation_status'] == 'pending'
    assert result['cleanup_status'] == 'completed', result['diagnostics']
    assert result['collection_status'] == 'collected', result['diagnostics']
    assert result['network'] is not None, result['diagnostics']
    assert result['network']['agent_uid'] != 0
    assert result['network']['agent_cap_eff'] == 0
    assert result['network']['privileged'] is False
    native = json.loads((run / result['native_result']).read_text())
    assert native.get('scoring') is None and native.get('verifier_error') is None
    from benchflow.sandbox.docker import _sanitize_docker_compose_project_name

    project = _sanitize_docker_compose_project_name(native['rollout_name'])
    audit = {'compose_project': project}
    for kind, command in (
        ('containers', ['ps', '-a', '-q']),
        ('networks', ['network', 'ls', '-q']),
        ('volumes', ['volume', 'ls', '-q']),
    ):
        completed = subprocess.run(
            ['docker', *command, '--filter', f'label=com.docker.compose.project={project}'],
            capture_output=True, text=True, check=True, timeout=30,
        )
        audit[kind] = completed.stdout.split()
    (root / 'cleanup-audit.json').write_text(json.dumps(audit, indent=2))
    assert not any(audit[kind] for kind in ('containers', 'networks', 'volumes')), audit
    assert result['collection']['freeze']['running'] is False
    native_dir = (run / result['native_result']).parent
    probe = json.loads((native_dir / 'agent/asi-probe.json').read_text())
    assert probe['process_cwd'] == probe['session_cwd'] == '/workspace'
    assert probe['workspace_before'] == ['data/public_cases.json', 'data/system.json', 'prompt.md', 'task_info.json']
    assert not any(probe['forbidden_exists'].values())
    assert ''.join(b['text'] for b in probe['prompt_blocks'] if b['type'] == 'text') == (prepared / 'environment/inputs/prompt.md').read_text()
    if args.mode == 'background':
        assert probe['background_pid'] > 0

    output = result['agent_output']['persisted_outputs']
    task_dir = run / result['task_id']
    for item in output['files']:
        if not item.get('missing'):
            data = (task_dir / output['dir'] / item['path']).read_bytes()
            assert len(data) == item['bytes']
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            if args.mode != 'background':
                assert item['sha256'] == probe['outputs'][item['path']]['sha256']
    if args.mode == 'background':
        assert json.loads((task_dir / output['dir'] / 'certificate.json').read_text()) == {'background': True}

    if args.mode != 'agent_error':
        assert result['attempt_status'] == 'completed'
        assert result['failure_reason'] is None
    if args.mode in ('success', 'background'):
        assert result['status'] == 'completed', result['diagnostics']
    elif args.mode == 'agent_error':
        assert result['attempt_status'] == 'execution_failed'
        assert result['failure_reason'] == 'execution_error'
        assert result['prediction_status'] == 'present'
        assert result['status'] == 'failed'
    else:
        assert result['status'] == 'failed'
        assert result['prediction_status'] == ('invalid' if args.mode == 'symlink' else 'missing')
        assert output['files'][1]['missing'] is True
    assert all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
               for name, digest in identity['sha256'].items()), 'sources changed during smoke'
    (root / 'acceptance.json').write_text(json.dumps({'mode': args.mode, 'passed': True}))
    print(json.dumps({k: result[k] for k in ('status', 'attempt_status', 'collection_status',
                                          'prediction_status', 'cleanup_status')}, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
