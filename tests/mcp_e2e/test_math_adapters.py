"""SageMath/SymPy contracts, real local CAS checks and opt-in real OS Docker acceptance."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog
from scripts.mcp.e2e.e2e_smoke.client import StdioMCP

ROOT = Path(__file__).parents[2]
ADAPTER = ROOT / 'scripts/mcp/math'
TOOLS = {'get_info', 'evaluate', 'submit_script', 'list_jobs', 'get_status', 'wait_job',
         'cancel_job', 'get_result', 'upload_file', 'list_files', 'read_file'}


def load_runtime():
    spec = importlib.util.spec_from_file_location('math_job_runtime', ADAPTER / 'runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(client, tool, **arguments):
    response = client.call_tool(tool, arguments, timeout=120)
    assert 'error' not in response, response
    result = response['result']
    assert not result.get('isError'), result
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])


@pytest.mark.parametrize('backend', ['sagemath', 'sympy'])
def test_catalog_and_deployment_contracts(backend):
    entry = load_science_mcp_catalog()[backend]
    config = load_mcp_config(ADAPTER / f'{backend}.os.json')
    assert config[backend] == entry['server']
    assert entry['availability'] == 'repository_adapter'
    assert config[backend]['sandbox']['deployment'] == 'instance'
    service = yaml.safe_load((ADAPTER / f'{backend}.service.yaml').read_text())['runtime']['services'][0]
    http = load_mcp_config(ADAPTER / f'{backend}.service.json')[backend]
    assert http['url'] == f'http://{service["name"]}:{service["port"]}/mcp'
    assert service['image'] == f'asibench-{backend}:test'
    assert 'env_from' not in service
    from ai4sci_bench.runner.mcp_sandbox import validate_os_mcp
    validate_os_mcp(config, {})
    validate_os_mcp({backend: http}, {'runtime': {'services': [service]}})


def test_file_transfer_paths_bounds_and_no_overwrite(tmp_path):
    jobs = load_runtime().Jobs(tmp_path, dict(os.environ), 'sympy')
    try:
        for name in ['../outside', '/absolute', '.', 'math-jobs/forged']:
            with pytest.raises(ValueError):
                jobs.put(name, 'bad')
        (tmp_path / 'symlink').symlink_to(tmp_path / 'target')
        with pytest.raises(ValueError):
            jobs.put('symlink', 'bad')
        payload = bytes(range(256))
        jobs.put('binary.dat', base64.b64encode(payload).decode(), 'base64')
        with pytest.raises(FileExistsError):
            jobs.put('binary.dat', 'overwrite')
        result = jobs.read('binary.dat', offset=7, limit=13, encoding='base64')
        assert base64.b64decode(result['data']) == payload[7:20]
        assert result['next_offset'] == 20 and not result['eof']
        with pytest.raises(ValueError):
            jobs.submit('1', [], 0)
        with pytest.raises(ValueError):
            jobs.submit('1', [], 1, 'sage')
        with pytest.raises(ValueError):
            jobs.submit('1', ['request.json'], 1)
        with pytest.raises(ValueError):
            jobs.status('not-a-job')
    finally:
        jobs.close()


@pytest.fixture
def sympy_client(tmp_path):
    pytest.importorskip('mcp')
    pytest.importorskip('sympy')
    env = {**os.environ, 'MATH_MCP_WORKSPACE': str(tmp_path), 'MATH_STATE_ROOT': str(tmp_path / 'state'),
           'MATH_PYTHON': sys.executable, 'OPENBLAS_NUM_THREADS': '1'}
    client = StdioMCP(sys.executable, [str(ADAPTER / 'server.py'), '--backend', 'sympy'],
                      cwd=tmp_path, env=env, stderr_path=tmp_path / 'server.stderr')
    try:
        client.initialize()
        yield client
    finally:
        client.close()
        assert not client.non_json_stdout


def test_real_sympy_exact_algebra_and_files(sympy_client):
    client = sympy_client
    assert {tool['name'] for tool in client.list_tools()} == TOOLS
    assert call(client, 'get_info')['backend'] == 'sympy'
    # Independent references: integral of x^4 on [0,1], a 2x2 determinant, roots sum.
    result = call(client, 'evaluate', code="x = symbols('x')\n(integrate(x**4, (x,0,1)), Matrix([[1,2],[3,4]]).det(), sum(solve(x**2-5*x+6,x)))")
    assert result['status']['state'] == 'completed'
    assert result['result']['representation'] == '(1/5, -2, 5)'
    assert 'Rational(1, 5)' in result['result']['exact']
    jid = result['status']['job_id']
    assert 'result.exact.txt' in {f['name'] for f in call(client, 'list_files', job_id=jid)['files']}
    assert call(client, 'read_file', job_id=jid, name='result.repr.txt')['data'] == '(1/5, -2, 5)'
    call(client, 'upload_file', name='input.txt', data='7')
    call(client, 'upload_file', name='program.py', data="from pathlib import Path\nn=int(Path('input.txt').read_text())\nPath('answer.bin').write_bytes(bytes(range(n)))\nRational(n,11)")
    job = call(client, 'submit_script', script_name='program.py', input_files=['input.txt'])
    assert call(client, 'wait_job', job_id=job['job_id'], wait_seconds=10)['state'] == 'completed'
    assert call(client, 'get_status', job_id=job['job_id'])['exit_code'] == 0
    binary = call(client, 'read_file', job_id=job['job_id'], name='answer.bin', encoding='base64')
    assert base64.b64decode(binary['data']) == bytes(range(7))
    assert len(call(client, 'list_jobs')['jobs']) == 2
    assert call(client, 'get_result', job_id=job['job_id'])['result']['representation'] == '7/11'


def test_errors_isolation_timeout_and_cancellation(sympy_client):
    client = sympy_client
    error = call(client, 'evaluate', code="raise ValueError('deliberate failure')")
    assert error['status']['state'] == 'failed'
    assert error['result']['error_type'] == 'ValueError'
    assert call(client, 'evaluate', code='x')['status']['state'] == 'failed'
    assert call(client, 'evaluate', code='import time; time.sleep(20)', timeout_seconds=1)['status']['state'] == 'timeout'
    job = call(client, 'submit_script', code='import time; time.sleep(30)')
    assert call(client, 'cancel_job', job_id=job['job_id'])['state'] == 'cancelled'
    assert call(client, 'evaluate', code='Rational(1,3)')['result']['representation'] == '1/3'
    assert client.call_tool('submit_script', {'code': '1', 'script_name': 'other.py'})['result']['isError']


def test_large_unicode_preview_preserves_full_result(sympy_client):
    result = call(sympy_client, 'evaluate', code="'🧮' * 9000")
    assert result['status']['state'] == 'completed'
    assert result['result']['truncated'] and result['result']['exact_truncated']
    jid = result['status']['job_id']
    data = call(sympy_client, 'read_file', job_id=jid, name='result.repr.txt', encoding='base64')
    assert base64.b64decode(data['data']).decode() == repr('🧮' * 9000)


def test_missing_backend_fails_startup_without_fallback(tmp_path):
    pytest.importorskip('mcp')
    env = {**os.environ, 'MATH_STATE_ROOT': str(tmp_path / 'state'), 'MATH_PYTHON': '/bin/false'}
    result = subprocess.run([sys.executable, str(ADAPTER / 'server.py'), '--backend', 'sagemath'],
                             env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and not result.stdout
    assert 'sagemath import failed' in result.stderr
    assert not list((tmp_path / 'state').iterdir())


def test_concurrency_limit_and_shutdown_cleanup(tmp_path):
    worker = tmp_path / 'lifecycle-worker'
    worker.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(30)\n')
    worker.chmod(0o755)
    jobs = load_runtime().Jobs(tmp_path, {**os.environ, 'MATH_PYTHON': str(worker)}, 'sympy')
    try:
        first = jobs.submit('unused lifecycle fixture', [], 30)
        second = jobs.submit('unused lifecycle fixture', [], 30)
        processes = list(jobs.live.values())
        with pytest.raises(ValueError, match='two concurrent'):
            jobs.submit('must not start', [], 30)
        jobs.close()
        assert all(process.poll() is not None for process in processes)
        assert jobs.status(first['job_id'])['state'] == 'cancelled'
        assert jobs.status(second['job_id'])['state'] == 'cancelled'
        with pytest.raises(RuntimeError, match='shutting down'):
            jobs.submit('must not start', [], 30)
    finally:
        jobs.close()


def test_external_timeout_and_descendant_cleanup(tmp_path):
    runtime = load_runtime()
    # Lifecycle-only worker: no claim of mathematical computation.
    worker = tmp_path / 'hanging-worker'
    worker.write_text('#!' + sys.executable + '\nimport subprocess,time\nfrom pathlib import Path\np=subprocess.Popen(["sleep","30"])\nPath("child.pid").write_text(str(p.pid))\ntime.sleep(30)\n')
    worker.chmod(0o755)
    jobs = runtime.Jobs(tmp_path, {**os.environ, 'MATH_PYTHON': str(worker)}, 'sympy')
    try:
        job = jobs.submit('unused lifecycle fixture', [], 1)
        directory = jobs.directory(job['job_id'])
        assert jobs.wait(job['job_id'], 5)['state'] == 'timeout'
        pid = int((directory / 'child.pid').read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            path = Path(f'/proc/{pid}/stat')
            if not path.exists() or path.read_text().split()[2] == 'Z':
                break
            time.sleep(.05)
        else:
            pytest.fail('descendant still running after timeout')
    finally:
        jobs.close()


DOCKER = os.environ.get('ASIBENCH_MATH_DOCKER') == '1'


@pytest.mark.skipif(not DOCKER, reason='build real SageMath and SymPy images; set ASIBENCH_MATH_DOCKER=1')
@pytest.mark.parametrize('backend', ['sagemath', 'sympy'])
def test_real_docker_stdio_science_and_artifacts(tmp_path, backend):
    tmp_path.chmod(0o777)
    # Actual Sage upstream once lacked system roots: model HTTPS then fails despite CAS success.
    certificate = subprocess.run(['docker', 'run', '--rm', '--network', 'none',
        '--entrypoint', '/opt/venv/bin/python', f'asibench-{backend}:test', '-c',
        'import ssl; assert ssl.create_default_context().cert_store_stats()["x509_ca"] > 0'],
        capture_output=True, text=True, timeout=30)
    assert certificate.returncode == 0, certificate.stderr
    client = StdioMCP('docker', ['run', '--rm', '-i', '--network', 'none', '--memory', '4g',
                     '--cpus', '2', '--pids-limit', '512', '--cap-drop', 'ALL',
                     '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,noexec',
                     '-v', f'{tmp_path}:/workspace', f'asibench-{backend}:test'],
                     cwd=tmp_path, env=dict(os.environ), stderr_path=tmp_path / 'docker.stderr')
    try:
        client.initialize()
        assert call(client, 'get_info')['backend'] == backend
        assert {tool['name'] for tool in client.list_tools()} == TOOLS
        if backend == 'sagemath':
            code = 'R.<x> = PolynomialRing(QQ)\n(FiniteField(7)(3)^6, (x^2-1).factor(), matrix(QQ,[[1,2],[3,4]]).det(), SymmetricGroup(4).order(), 1/3)'
            result = call(client, 'evaluate', code=code, syntax='sage')
            assert result['result']['representation'] == '(1, (x - 1) * (x + 1), -2, 24, 1/3)'
        else:
            result = call(client, 'evaluate', code="x=symbols('x'); (integrate(x**4,(x,0,1)), Matrix([[1,2],[3,4]]).det(), factor(x**2-1))")
            assert result['result']['representation'] == '(1/5, -2, (x - 1)*(x + 1))'
        assert result['status']['state'] == 'completed'
        # Transfer input, use original backend's Python syntax, and export a real PNG.
        call(client, 'upload_file', name='data.txt', data='7')
        code = "from pathlib import Path\nimport matplotlib.pyplot as plt\nn=int(Path('data.txt').read_text())\nplt.plot([0,1],[0,n]); plt.savefig('plot.png'); plt.close()\nPath('binary.dat').write_bytes(bytes(range(n)))\n" + ('QQ(n)/11' if backend == 'sagemath' else 'Rational(n,11)')
        job = call(client, 'submit_script', code=code, input_files=['data.txt'])
        assert call(client, 'wait_job', job_id=job['job_id'], wait_seconds=60)['state'] == 'completed'
        assert call(client, 'get_result', job_id=job['job_id'])['result']['representation'] == '7/11'
        png = call(client, 'read_file', job_id=job['job_id'], name='plot.png', encoding='base64', limit=1048576)
        data = base64.b64decode(png['data'])
        assert data[:8] == b'\x89PNG\r\n\x1a\n' and png['eof']
        from PIL import Image
        import io
        Image.open(io.BytesIO(data)).verify()
        call(client, 'upload_file', name='program.py', data='print("scientific stdout stays out of MCP")\n' + ('QQ(2)/7' if backend == 'sagemath' else 'Rational(2,7)'))
        script = call(client, 'submit_script', script_name='program.py')
        assert call(client, 'wait_job', job_id=script['job_id'], wait_seconds=60)['state'] == 'completed'
        assert 'scientific stdout' in call(client, 'read_file', job_id=script['job_id'], name='stdout.log')['data']
        assert call(client, 'evaluate', code='1/0')['status']['state'] == 'failed'
        assert call(client, 'evaluate', code='import time; time.sleep(30)', timeout_seconds=1)['status']['state'] == 'timeout'
        hanging = call(client, 'submit_script', code='import time; time.sleep(30)')
        assert call(client, 'cancel_job', job_id=hanging['job_id'])['state'] == 'cancelled'
        assert call(client, 'get_status', job_id=hanging['job_id'])['state'] == 'cancelled'
        assert len(call(client, 'list_jobs')['jobs']) >= 6
        assert call(client, 'list_files')['count'] > 0
    finally:
        client.close()
        assert not client.non_json_stdout


@pytest.mark.skipif(not DOCKER, reason='requires built real mathematics images and Docker')
@pytest.mark.parametrize('backend', ['sagemath', 'sympy'])
@pytest.mark.parametrize('deployment', ['instance', 'service'])
def test_framework_preflight_and_actual_scientific_call(tmp_path, monkeypatch, backend, deployment):
    """Real OSSandbox agent-config injection with a protocol client, no model substitution claim."""
    from ai4sci_bench.runner.os_sandbox import OSSandbox
    from tests.test_mcp_os_docker import ImageBuilder
    tmp_path.chmod(0o777)
    script = ROOT / 'tests/mcp_e2e/math_container_check.py'
    (tmp_path / 'check.py').write_text(script.read_text())
    config = load_mcp_config(ADAPTER / f'{backend}.{"os" if deployment == "instance" else "service"}.json')
    metadata = yaml.safe_load((ADAPTER / f'{backend}.service.yaml').read_text()) if deployment == 'service' else {}
    sandbox = OSSandbox(ROOT, image_builder=ImageBuilder(f'asibench-{backend}:test'))
    monkeypatch.setattr(sandbox, '_prepare_auth_mounts', lambda _: [])
    monkeypatch.delenv('AI4SCI_DOCKER_NETWORK', raising=False)
    # Other validations may run concurrently. Inspect only containers we created.
    created = []
    original_run = subprocess.run
    def track_run(command, *args, **kwargs):
        if isinstance(command, list) and command[:1] == ['docker'] and '--name' in command:
            created.append(command[command.index('--name') + 1])
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', track_run)
    runtime = {}
    success, log, stdout, stderr, identity = sandbox.run_agent(metadata, workspace=tmp_path,
        agent_cmd=['/opt/venv/bin/python', '/workspace/check.py'], timeout=180,
        agent_type='codex', mcp_servers=config, mcp_runtime=runtime,
        extra_env={'CHECK_BACKEND': backend, 'CHECK_DEPLOYMENT': deployment})
    assert success, log + stdout + stderr
    assert 'MATH_FRAMEWORK_SCIENCE_PASS' in stdout
    assert json.loads((tmp_path / 'verified.json').read_text())['answer'] == '1/3'
    assert identity.startswith('sha256:')
    assert created
    for name in created:
        assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0


@pytest.mark.skipif(not DOCKER, reason='requires both built images and Docker')
def test_both_http_backends_in_one_offline_task(tmp_path, monkeypatch):
    from ai4sci_bench.runner.os_sandbox import OSSandbox
    tmp_path.chmod(0o777)
    metadata = yaml.safe_load((ADAPTER / 'both.service.yaml').read_text())
    metadata['runtime']['image'] = 'asibench-sympy:test'
    monkeypatch.delenv('AI4SCI_DOCKER_NETWORK', raising=False)
    code = '''import asyncio,json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
async def main():
    for backend, expression in [('sagemath','QQ(1)/3'), ('sympy','Rational(1,3)')]:
        async with streamable_http_client('http://'+backend+':8765/mcp') as (read,write,_):
            async with ClientSession(read,write) as client:
                await client.initialize()
                response=await client.call_tool('evaluate',{'code':expression})
                assert not response.isError,response
                result=response.structuredContent or json.loads(response.content[0].text)
                assert result['status']['state']=='completed',result
                assert result['result']['backend']==backend
                assert result['result']['representation']=='1/3'
    print('BOTH_MATH_SERVICES_PASS')
asyncio.run(main())
'''
    (tmp_path / 'check.py').write_text(code)
    success, log, stdout, *_ = OSSandbox(ROOT).execute_python(metadata, workspace=tmp_path,
                                                  code_file='check.py', timeout=180)
    assert success, log
    assert 'BOTH_MATH_SERVICES_PASS' in stdout
