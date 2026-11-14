"""Offline contracts and opt-in real native-engine Docker acceptance."""
import base64
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import pytest

from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog
from .support import ROOT, client
from .engineering_cases import CASES, INVALID

HERE = ROOT / 'scripts/mcp/engineering'
spec = importlib.util.spec_from_file_location('engineering_job_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
BACKENDS = list(CASES)
LIVE = os.environ.get('ASIBENCH_ENGINEERING_DOCKER') == '1'


def image(backend):
    return 'asibench-engineering-' + ('eda' if backend in BACKENDS[:4] else 'cae') + ':20261114'


@pytest.mark.parametrize('backend', BACKENDS)
def test_catalog_config_and_os_codex_staging(backend):
    from ai4sci_bench.runner.mcp_sandbox import stage_os_mcp
    name = backend + '_native'
    entry = load_science_mcp_catalog()[name]
    servers = load_mcp_config(HERE / f'{backend}.os.json')
    assert entry['server']['args'] == servers[name]['args']
    with stage_os_mcp(servers, {}, 'codex', ['codex', 'exec'], None) as (cmd, _, _):
        assert f'mcp_servers.{name}.required=true' in cmd
    services = json.loads((HERE / f'{backend}.service.yaml').read_text())['runtime']['services']
    service = load_mcp_config(HERE / f'{backend}.service.json')[name]
    assert services[0]['name'] == service['sandbox']['service']
    from ai4sci_bench.runner.mcp_sandbox import validate_os_mcp
    validate_os_mcp({name: service}, {'runtime': {'services': services}})


def test_transfer_rejects_traversal_symlink_and_managed_outputs(tmp_path):
    jobs = runtime.Jobs(tmp_path / 'ws', dict(os.environ), 'ngspice')
    try:
        (tmp_path / 'ws/link').symlink_to(tmp_path)
        for name in ['../outside', '/tmp/outside', 'link/outside', 'engineering-jobs/fake']:
            with pytest.raises(ValueError): jobs.put(name, 'bad')
        info = jobs.put('inputs.bin', base64.b64encode(b'\0abc').decode(), 'base64')
        assert info['size_bytes'] == 4
        assert base64.b64decode(jobs.read('inputs.bin', encoding='base64')['data']) == b'\0abc'
        with pytest.raises(FileExistsError): jobs.put('inputs.bin', 'again')
        with pytest.raises(ValueError): jobs.submit('x', [], True)
        with pytest.raises(ValueError): jobs.submit('x', [], 10, '--bad')
    finally: jobs.close()


def test_timeout_cancel_and_failed_stage(tmp_path):
    bindir = tmp_path / 'bin'; bindir.mkdir()
    binary = bindir / 'ngspice'
    binary.write_text('#!/bin/sh\n/bin/sleep 20\n'); binary.chmod(0o755)
    jobs = runtime.Jobs(tmp_path/'ws', {**os.environ, 'PATH':str(bindir)}, 'ngspice')
    try:
        job = jobs.submit('test', [], 1)['job_id']
        assert jobs.wait(job, 10)['state'] == 'timeout'
        job = jobs.submit('test', [], 30)['job_id']
        assert jobs.cancel(job)['state'] == 'cancelled'
        jobs.wait(job, 2)
        deadline = time.monotonic()+5
        while jobs.live and time.monotonic()<deadline: time.sleep(.05)
        binary.write_text('#!/bin/sh\necho "Error: synthetic"\nexit 0\n');binary.chmod(0o755)
        job = jobs.submit('test', [], 10)['job_id']
        assert jobs.wait(job, 10)['state'] == 'failed'
        assert jobs.status(job)['exit_code'] != 0
    finally: jobs.close()


def call(mcp, tool, **arguments):
    response = mcp.call_tool(tool, arguments)
    assert 'error' not in response, response
    result = response['result']
    assert not result.get('isError'), result
    value = result.get('structuredContent')
    if value is None: value = json.loads(client.text_of(result))
    if isinstance(value, dict) and set(value) == {'result'}: value = value['result']
    return value


def execute(mcp, backend, invalid=False):
    code, files = CASES[backend] if not invalid else (INVALID[backend], {})
    for name, data in files.items(): call(mcp, 'upload_file', name=name, data=data)
    job = call(mcp, 'submit_script', code=code, input_files=list(files), timeout_seconds=120)['job_id']
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        status = call(mcp, 'wait_job', job_id=job, wait_seconds=10)
        if status['state'] != 'running': break
    assert status['state'] == ('failed' if invalid else 'completed'), call(mcp, 'read_file', job_id=job, name='stdout.log')
    return job


@pytest.mark.skipif(not LIVE, reason='build engineering images and set ASIBENCH_ENGINEERING_DOCKER=1')
@pytest.mark.parametrize('backend', BACKENDS)
def test_real_offline_stdio_native_result_and_errors(backend, tmp_path):
    mcp = client.StdioMCP('docker', ['run','--rm','-i','--network','none', '--cpus','2','--memory','2g',
        '-e','HOME=/tmp/fresh-home','--entrypoint','/opt/venv/bin/python',image(backend),
        '/opt/engineering-mcp/server.py','--backend',backend],
        cwd=tmp_path,env=dict(os.environ),stderr_path=tmp_path/'stderr.log')
    try:
        mcp.initialize()
        assert {t['name'] for t in mcp.list_tools()} == {'get_info','upload_file','submit_script','list_jobs',
            'get_status','wait_job','cancel_job','get_result','list_files','read_file'}
        assert call(mcp,'get_info')['backend'] == backend
        job = execute(mcp, backend)
        def read(name):return call(mcp,'read_file',name=name,job_id=job)['data']
        output = read('stdout.log')
        if backend == 'ngspice':
            values = [float(x) for x in read('divider.txt').split()]
            assert values[-1] == pytest.approx(5., abs=1e-8)
        elif backend in {'iverilog','verilator'}:
            rows = [tuple(map(int,x)) for x in re.findall(r'SUM (\d+) (\d+) (\d+)', output)]
            assert rows == [(a,b,a+b) for a in range(4) for b in range(4)]
        elif backend == 'opensta':
            assert 'slack (MET)' in output
            assert re.search(r'0\.100000\s+1\.100000',output), output
        elif backend == 'calculix':
            text = read('input.dat')
            rows = [tuple(map(float,x)) for x in re.findall(r'^\s*([2367])\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s*$',text,re.M)]
            assert len(rows)==4,text
            assert [x[1] for x in rows] == pytest.approx([.004]*4, abs=1e-8)
        else:
            import csv,io
            rows=list(csv.DictReader(io.StringIO(read('Decay_res.csv'))))
            assert float(rows[-1]['time'])==pytest.approx(1.)
            assert float(rows[-1]['x'])==pytest.approx(math.exp(-1),abs=2e-6)
        # Independently inspect expected result files and fail bad native input.
        (tmp_path/'validated.json').write_text(json.dumps({'backend':backend,'job_id':job,'state':'completed','output':output}))
        execute(mcp,backend,invalid=True)
        assert call(mcp,'get_info')['backend']==backend
    finally:mcp.close()
    assert not mcp.non_json_stdout,mcp.non_json_stdout


@pytest.mark.skipif(not LIVE, reason='requires engineering images and real Docker')
@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('deployment', ['instance', 'service'])
def test_framework_preflight_and_native_job(tmp_path, monkeypatch, backend, deployment):
    from ai4sci_bench.runner.os_sandbox import OSSandbox
    from tests.test_mcp_os_docker import ImageBuilder
    tmp_path.chmod(0o777)
    (tmp_path/'check.py').write_text((ROOT/'tests/mcp_e2e/engineering_container_check.py').read_text())
    (tmp_path/'case.json').write_text(json.dumps(CASES[backend]))
    servers=load_mcp_config(HERE/f'{backend}.{"os" if deployment=="instance" else "service"}.json')
    metadata=json.loads((HERE/f'{backend}.service.yaml').read_text()) if deployment=='service' else {}
    sandbox=OSSandbox(ROOT,image_builder=ImageBuilder(image(backend)))
    monkeypatch.setattr(sandbox,'_prepare_auth_mounts',lambda _:[])
    monkeypatch.delenv('AI4SCI_DOCKER_NETWORK',raising=False)
    names=[]
    original=subprocess.run
    def track(command,*args,**kwargs):
        if isinstance(command,list) and command[:1]==['docker'] and '--name' in command:
            names.append(command[command.index('--name')+1])
        return original(command,*args,**kwargs)
    monkeypatch.setattr(subprocess,'run',track)
    provenance={}
    success,log,stdout,stderr,identity=sandbox.run_agent(metadata,workspace=tmp_path,
        agent_cmd=['/opt/venv/bin/python','/workspace/check.py'],timeout=180,agent_type='codex',
        mcp_servers=servers,mcp_runtime=provenance,
        extra_env={'CHECK_BACKEND':backend,'CHECK_DEPLOYMENT':deployment})
    assert success,log+stdout+stderr
    assert 'ENGINEERING_FRAMEWORK_PASS' in stdout
    assert json.loads((tmp_path/'verified.json').read_text())['backend']==backend
    assert identity.startswith('sha256:')
    assert provenance['servers'][backend+'_native']['deployment']==deployment
    assert names
    for name in names:
        assert original(['docker','inspect',name],capture_output=True).returncode!=0
