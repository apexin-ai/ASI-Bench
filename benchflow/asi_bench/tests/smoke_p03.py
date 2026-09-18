"""Opt-in P0.3 native Docker lifecycle probe; no Agent or scientific scoring.

Run with --docker. Uses a local Python image, no pull/build or model secrets.
Evidence is retained in a fresh temp directory (or --output, which must not exist).
"""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

BRIDGE = r'''
import hashlib, json, os, socket, sys
from pathlib import Path
sys.path.insert(0, '/opt/asi-evaluator')
from probe_evaluator import echo
assert os.getuid() == 10001 and os.getgid() == 10001
assert os.getcwd() == '/output'
assert not Path('/var/run/docker.sock').exists()
assert not Path('/run/docker.sock').exists()
assert 'ASI_P03_HOST_ONLY' not in os.environ
assert not any(k in os.environ for k in ('HF_TOKEN', 'GITHUB_TOKEN', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY'))
protected = {}
for name in ('/input/value.txt', '/opt/asi-evaluator/probe_evaluator.py', '/opt/bridge/probe.py'):
    p = Path(name)
    before = p.read_bytes()
    try:
        p.write_text('tamper')
    except PermissionError:
        protected[name] = True
    else:
        raise AssertionError('writable trusted input: ' + name)
    assert p.read_bytes() == before
for name in ('/input/new', '/opt/asi-evaluator/new', '/opt/bridge/new'):
    try:
        Path(name).touch()
    except PermissionError:
        pass
    else:
        raise AssertionError('writable trusted directory')
network = []
for host in ('1.1.1.1', '8.8.8.8'):
    try:
        with socket.create_connection((host, 443), timeout=1):
            raise AssertionError('external network reachable')
    except OSError as exc:
        network.append({'host': host, 'errno': exc.errno, 'error': str(exc)})
interfaces = sorted(name for _, name in socket.if_nameindex())
routes = Path('/proc/net/route').read_text()
assert not routes.splitlines()[1:], routes
assert all(item['errno'] == 101 for item in network), network
Path('/tmp/probe-write').write_text('temporary')
value = Path('/input/value.txt').read_bytes()
report = {'uid': os.getuid(), 'gid': os.getgid(), 'cwd': os.getcwd(),
          'protected': protected, 'interfaces': interfaces, 'routes': routes, 'network': network,
          'input_sha256': hashlib.sha256(value).hexdigest(), 'echo': echo(value.decode()),
          'socket_absent': True, 'host_secret_env_absent': True}
Path('probe.json').write_text(json.dumps(report, indent=2))
print('P03_STDOUT', flush=True)
print('P03_STDERR', file=sys.stderr, flush=True)
'''


def docker(*args, check=True):
    return subprocess.run(['docker', *args], capture_output=True, text=True,
                          timeout=30, check=check)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


async def run_case(root, image_id, mode):
    case = root / mode
    case.mkdir()
    environment = case / 'environment'
    environment.mkdir()
    # Root is reserved for trusted upload/permission setup; probe runs as 10001.
    (environment / 'docker-compose.yaml').write_text('''services:
  main:
    deploy:
      resources:
        limits:
          pids: 64
    security_opt:
      - no-new-privileges:true
    tmpfs:
      - /tmp:size=16777216,mode=1777
''')
    paths = RolloutPaths(case / 'logs')
    paths.mkdir()
    project = 'asi-p03-' + uuid.uuid4().hex[:16]
    sandbox = DockerSandbox(
        environment_dir=environment, environment_name=project, session_id=project,
        rollout_paths=paths, keep_containers=False,
        task_env_config=SandboxConfig(docker_image=image_id, network_mode='no-network',
                                      cpus=1, memory_mb=256),
    )
    report = {'mode': mode, 'project': project, 'status': 'failed'}
    start = time.monotonic()
    try:
        await asyncio.wait_for(sandbox.start(force_build=False), 90)
        cid = docker('ps', '-aq', '--filter', f'label=com.docker.compose.project={project}').stdout.strip()
        assert cid and '\n' not in cid
        report['container_id'] = cid
        inspected = json.loads(docker('inspect', cid).stdout)[0]
        # Retain only relevant config, never an unrestricted host environment dump.
        host = inspected['HostConfig']
        report['docker'] = {k: host.get(k) for k in (
            'NetworkMode', 'Memory', 'NanoCpus', 'PidsLimit', 'Tmpfs',
            'Privileged', 'CapAdd', 'SecurityOpt')}
        report['mounts'] = inspected['Mounts']
        assert host['NetworkMode'] == 'none'
        assert host['Memory'] == 256 * 1024**2 and host['NanoCpus'] == 10**9
        assert host['PidsLimit'] == 64 and not host['Privileged']
        assert not host.get('CapAdd')
        assert any('no-new-privileges' in x for x in host['SecurityOpt'])
        assert 'size=16777216' in host['Tmpfs']['/tmp']
        binds = {m['Destination']: Path(m['Source']).resolve() for m in inspected['Mounts'] if m['Type'] == 'bind'}
        assert binds == {f'/logs/{name}': (case / 'logs' / name).resolve()
                         for name in ('agent', 'verifier', 'artifacts')}, binds
        assert not any(m['Type'] == 'volume' for m in inspected['Mounts'])
        assert not any('docker.sock' in str(m) for m in inspected['Mounts'])
        for dirname, filename, text, destination in (
            ('input', 'value.txt', 'p03 ordinary input\n', '/input'),
            ('evaluator', 'probe_evaluator.py', 'def echo(value):\n    return value\n', '/opt/asi-evaluator'),
            ('bridge', 'probe.py', BRIDGE, '/opt/bridge'),
        ):
            source = case / dirname
            source.mkdir()
            (source / filename).write_text(text)
            await sandbox.upload_dir(source, destination)
        setup = await sandbox.exec(
            'chown -R 0:0 /input /opt/asi-evaluator /opt/bridge && '
            'chmod -R a-w /input /opt/asi-evaluator /opt/bridge && '
            'mkdir /output && chown 10001:10001 /output && chmod 700 /output',
            user='root', timeout_sec=10)
        assert setup.return_code == 0, setup.stdout
        result = await sandbox.exec('python /opt/bridge/probe.py', user='10001:10001',
                                    cwd='/output', env={'PYTHONDONTWRITEBYTECODE': '1'}, timeout_sec=15)
        report['exec'] = {'return_code': result.return_code, 'stdout': result.stdout, 'stderr': result.stderr}
        assert result.return_code == 0, result.stdout
        assert 'P03_STDOUT' in result.stdout and 'P03_STDERR' in result.stdout
        if mode == 'failure':
            result = await sandbox.exec("printf 'EXPECTED_FAILURE\\n' >&2; exit 7",
                                        user='10001:10001', cwd='/output', timeout_sec=10)
            report['expected_failure'] = {'return_code': result.return_code, 'stdout': result.stdout}
            assert result.return_code == 7 and 'EXPECTED_FAILURE' in result.stdout
        if mode == 'timeout':
            command = '''python -c 'import os,subprocess,time; from pathlib import Path; p=subprocess.Popen(["sleep","120"]); Path("worker.pid").write_text(str(p.pid)); time.sleep(120)' '''
            t = time.monotonic()
            try:
                await sandbox.exec(command, user='10001:10001', cwd='/output', timeout_sec=2)
            except RuntimeError as exc:
                assert 'timed out' in str(exc), str(exc)
                report['timeout'] = {'error': str(exc), 'elapsed_seconds': time.monotonic()-t}
            else:
                raise AssertionError('native exec did not time out')
            assert report['timeout']['elapsed_seconds'] < 15
            probe = await sandbox.exec('test -f /output/worker.pid && kill -0 $(cat /output/worker.pid)',
                                       user='root', timeout_sec=10)
            report['worker_alive_before_teardown'] = probe.return_code == 0
        exported = case / 'exported'
        exported.mkdir()
        await sandbox.download_dir('/output', exported)
        probe = json.loads((exported / 'probe.json').read_text())
        assert probe['input_sha256'] == hashlib.sha256((case/'input/value.txt').read_bytes()).hexdigest()
        assert probe['echo'] == 'p03 ordinary input\n'
        report['exported'] = probe
        report['status'] = 'passed'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            # delete=True removes images too. No volumes are declared; down suffices.
            await asyncio.wait_for(sandbox.stop(delete=False), 180)
            remaining = docker('ps', '-aq', '--filter', f'label=com.docker.compose.project={project}').stdout.strip()
            networks = docker('network', 'ls', '-q', '--filter', f'label=com.docker.compose.project={project}').stdout.strip()
            volumes = docker('volume', 'ls', '-q', '--filter', f'label=com.docker.compose.project={project}').stdout.strip()
            image_after = json.loads(docker('image', 'inspect', image_id).stdout)[0]['Id']
            assert not remaining and not networks and not volumes
            assert image_after == image_id
            report['cleanup'] = {'status': 'completed', 'containers': remaining,
                                 'networks': networks, 'volumes': volumes, 'image_retained': image_after}
        except BaseException as exc:
            report['status'] = 'failed'
            report['cleanup'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            raise
        finally:
            report['elapsed_seconds'] = time.monotonic()-start
            save(case/'evidence.json', report)
    return report


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker', action='store_true', required=True)
    parser.add_argument('--image', default='python:3.11-slim')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.environ['ASI_P03_HOST_ONLY'] = 'public-test-sentinel'
    if args.output:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix='asi-p03-')).resolve()
    print(f'EVIDENCE={root}', flush=True)
    image = json.loads(docker('image', 'inspect', args.image).stdout)[0]
    report = {'status': 'failed', 'image_requested': args.image, 'image_id': image['Id'],
              'docker_version': docker('version', '--format', '{{.Server.Version}}').stdout.strip(),
              'cases': []}
    try:
        for mode in ('success', 'failure', 'timeout'):
            report['cases'].append(await run_case(root, image['Id'], mode))
        report['status'] = 'passed'
    finally:
        save(root/'summary.json', report)
    print(json.dumps({'status': report['status'], 'evidence': str(root)}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
