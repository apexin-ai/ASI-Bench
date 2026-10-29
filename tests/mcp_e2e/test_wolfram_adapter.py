"""Gateway/lifecycle tests and separate opt-in licensed scientific acceptance."""
import base64
import importlib.util
import json
import os
import re
import io
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest
from scripts.mcp.e2e.e2e_smoke.client import StdioMCP

ROOT = Path(__file__).parents[2]
ADAPTER = ROOT/'scripts/mcp/wolfram'
FIXTURE = ROOT/'tests/fixtures/wolfram'


def load_runtime():
    spec = importlib.util.spec_from_file_location('wolfram_runtime', ADAPTER/'runtime.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def value(response):
    assert 'error' not in response, response
    result = response['result']; assert not result.get('isError'), result
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])


def call(client, tool, **kwargs):
    return value(client.call_tool(tool, kwargs, timeout=45))


@pytest.fixture
def protocol_client(tmp_path):
    pytest.importorskip('mcp', reason='install scripts/mcp/wolfram/requirements.lock for protocol tests')
    env = {**os.environ, 'WOLFRAM_MCP_COMMAND_JSON': json.dumps([sys.executable, str(FIXTURE/'official.py')]),
           'WOLFRAM_MCP_WORKSPACE': str(tmp_path), 'WOLFRAMSCRIPT': str(FIXTURE/'native.py'),
           'WOLFRAM_MCP_TIMEOUT': '3', 'HOME': str(tmp_path/'home')}
    env.pop('WOLFRAM_LICENSE_B64', None); env.pop('WOLFRAMSCRIPT_ENTITLEMENTID', None)
    client = StdioMCP(sys.executable, [str(ADAPTER/'server.py')], cwd=tmp_path, env=env,
                      stderr_path=tmp_path/'gateway.stderr')
    try:
        client.initialize(); yield client
    finally:
        client.close()
        assert not client.non_json_stdout


def test_official_pagination_content_errors_and_primitives(protocol_client):
    client = protocol_client
    names = {x['name'] for x in client.list_tools()}
    assert {'WolframLanguageEvaluator', 'FixtureSecond', 'wolfram_submit', 'wolfram_restart'} <= names
    response = client.call_tool('WolframLanguageEvaluator', {'code': 'exact-symbolic-fixture'})['result']
    assert response['structuredContent'] == {'echo': 'exact-symbolic-fixture'}
    assert [b['type'] for b in response['content']] == ['text', 'image', 'resource']
    assert response['_meta'] == {'fixture': True}
    assert client.call_tool('WolframLanguageEvaluator', {'code': 'error'})['result']['isError']
    assert client.request('prompts/list')['result']['prompts'][0]['name'] == 'fixture-prompt'
    assert client.request('prompts/get', {'name': 'fixture-prompt'})['result']['messages'][0]['content']['text'] == 'fixture prompt'
    assert client.request('resources/list')['result']['resources'][0]['name'] == 'fixture'
    assert client.request('resources/read', {'uri': 'file:///workspace/fixture.txt'})['result']['contents'][0]['text'] == 'fixture resource'
    assert client.request('resources/templates/list')['result']['resourceTemplates'] == []
    info = call(client, 'wolfram_info')
    assert info['official_server']['name'] == 'protocol-fixture-not-Wolfram'


def test_jobs_files_and_cancellation_via_mcp(protocol_client):
    client = protocol_client
    payload = bytes(range(256))
    call(client, 'wolfram_upload', name='input.dat', data=base64.b64encode(payload).decode(), encoding='base64')
    job = call(client, 'wolfram_submit', code='lifecycle fixture', input_files=['input.dat'])
    jid = job['job_id']
    assert call(client, 'wolfram_wait', job_id=jid, wait_seconds=10)['state'] == 'completed'
    assert call(client, 'wolfram_status', job_id=jid)['exit_code'] == 0
    assert call(client, 'wolfram_result', job_id=jid)['result']['fixture'] is True
    assert len(call(client, 'wolfram_jobs')['jobs']) == 1
    assert any(f['name'] == 'binary.wxf' for f in call(client, 'wolfram_files', job_id=jid)['files'])
    out = call(client, 'wolfram_read', job_id=jid, name='binary.wxf', offset=7, limit=13, encoding='base64')
    assert base64.b64decode(out['data']) == payload[7:20] and out['next_offset'] == 20
    job = call(client, 'wolfram_submit', code='fail')
    assert call(client, 'wolfram_wait', job_id=job['job_id'], wait_seconds=10)['state'] == 'failed'
    assert 'fixture failure' in call(client, 'wolfram_read', job_id=job['job_id'], name='stderr.log')['data']
    job = call(client, 'wolfram_submit', code='hang')
    assert call(client, 'wolfram_cancel', job_id=job['job_id'])['state'] == 'cancelled'
    call(client, 'wolfram_upload', name='program.wl', data='script fixture')
    job = call(client, 'wolfram_submit', script_name='program.wl')
    assert call(client, 'wolfram_wait', job_id=job['job_id'], wait_seconds=10)['state'] == 'completed'
    assert call(client, 'wolfram_restart')['state'] == 'restarted'
    assert not client.call_tool('WolframLanguageEvaluator', {'code': 'after restart'})['result'].get('isError')


def test_paths_and_process_group_cleanup(tmp_path):
    runtime = load_runtime()
    env = {**os.environ, 'WOLFRAMSCRIPT': str(FIXTURE/'native.py')}
    jobs = runtime.Jobs(tmp_path, env)
    try:
        for name in ['../outside', '/tmp/escape', '.', 'wolfram-jobs/forged']:
            with pytest.raises(ValueError): jobs.put(name, 'x')
        (tmp_path/'link').symlink_to(tmp_path/'outside')
        with pytest.raises(ValueError): jobs.put('link', 'x')
        jobs.put('input', 'first')
        with pytest.raises(FileExistsError): jobs.put('input', 'overwrite')
        job = jobs.submit('hang', [], 1); jid = job['job_id']
        path = jobs.directory(jid)/'child.pid'
        deadline = time.monotonic()+5
        while not path.exists() and time.monotonic()<deadline: time.sleep(.02)
        assert path.exists()
        pid = int(path.read_text())
        jobs.cancel(jid)
        deadline=time.monotonic()+5
        while jid in jobs.live and time.monotonic()<deadline: time.sleep(.02)
        assert jid not in jobs.live
        # A zombie has exited; a running descendant is a cleanup failure.
        stat = Path(f'/proc/{pid}/stat')
        assert not stat.exists() or stat.read_text().split()[2] == 'Z'
    finally: jobs.close()


def test_bootstrap_license_not_forwarded(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('wolfram_bootstrap',ADAPTER/'bootstrap.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setenv('HOME',str(tmp_path));monkeypatch.setenv('WOLFRAM_LICENSE_B64',base64.b64encode(b'fixture-license-not-valid').decode())
    env=module.prepare_environment();assert 'WOLFRAM_LICENSE_B64' not in env
    path=tmp_path/'.WolframEngine/Licensing/mathpass'
    assert path.read_bytes()==b'fixture-license-not-valid' and path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv('WOLFRAM_LICENSE_B64','invalid?')
    with pytest.raises(RuntimeError,match='invalid WOLFRAM_LICENSE_B64'):module.prepare_environment()


def test_gateway_timeout_requires_explicit_restart(protocol_client):
    response=protocol_client.call_tool('WolframLanguageEvaluator',{'code':'hang'},timeout=20)
    assert 'error' in response or response['result'].get('isError')
    response=protocol_client.call_tool('WolframLanguageEvaluator',{'code':'no silent replay'},timeout=20)
    assert 'error' in response or response['result'].get('isError')
    assert call(protocol_client,'wolfram_restart')['state']=='restarted'
    assert not protocol_client.call_tool('WolframLanguageEvaluator',{'code':'recovered'})['result'].get('isError')


LIVE = os.environ.get('ASIBENCH_WOLFRAM_DOCKER') == '1'


@pytest.mark.skipif(not LIVE, reason='requires built Wolfram image and a valid runtime license')
def test_licensed_real_kernel_and_native_artifacts(tmp_path):
    # Explicit opt-in. A missing or invalid license is a FAILURE, never a fake pass.
    env=dict(os.environ)
    args=['run','--rm','-i','--cpus','2','--memory','8g','--pids-limit','512']
    if env.get('WOLFRAM_LICENSE_B64'):
        args+=['--network','none','-e','WOLFRAM_LICENSE_B64']
    elif env.get('WOLFRAMSCRIPT_ENTITLEMENTID'):
        args+=['-e','WOLFRAMSCRIPT_ENTITLEMENTID']
    else: pytest.fail('set WOLFRAM_LICENSE_B64 or WOLFRAMSCRIPT_ENTITLEMENTID')
    args+=[os.environ.get('ASIBENCH_WOLFRAM_IMAGE','asibench-wolfram:test')]
    client=StdioMCP('docker',args,cwd=tmp_path,env=env,stderr_path=tmp_path/'wolfram.stderr')
    try:
        client.initialize(timeout=300)
        result=client.call_tool('WolframLanguageEvaluator',{'code':'Integrate[x^2, {x,0,1}]','timeConstraint':30},timeout=120)
        assert 'error' not in result and not result['result'].get('isError'),result
        text=''.join(x.get('text','') for x in result['result']['content'])
        assert '1/3' in text or '1 / 3' in text, result
        assigned=client.call_tool('WolframLanguageEvaluator',{'code':'asiBenchProbe = 41; asiBenchProbe','timeConstraint':30},timeout=120)
        assert 'error' not in assigned and not assigned['result'].get('isError'),assigned
        session_text=''.join(x.get('text','') for x in assigned['result']['content'])
        session=re.search(r'Pass session="([^"]+)"',session_text)
        assert session, assigned
        continued=client.call_tool('WolframLanguageEvaluator',{'code':'asiBenchProbe + 1','session':session.group(1),'timeConstraint':30},timeout=120)
        assert 'error' not in continued and not continued['result'].get('isError'),continued
        assert re.search(r'\b42\b',''.join(x.get('text','') for x in continued['result']['content'])),continued
        job=call(client,'wolfram_submit',code='Export["data.csv", {{1,2},{3,4}}, "CSV"]; Export["plot.png", Plot[x^2,{x,0,1}], "PNG"]; 1/3')
        status=call(client,'wolfram_wait',job_id=job['job_id'],wait_seconds=60)
        assert status['state']=='completed',call(client,'wolfram_result',job_id=job['job_id'])
        expression=call(client,'wolfram_read',job_id=job['job_id'],name='result.inputform.txt')['data'].strip()
        assert expression=='1/3'
        wxf=call(client,'wolfram_read',job_id=job['job_id'],name='result.wxf',encoding='base64')
        assert wxf['eof'] and base64.b64decode(wxf['data']).startswith((b'8:',b'8C:'))
        csv=call(client,'wolfram_read',job_id=job['job_id'],name='data.csv')['data']
        assert csv.replace('\r','').strip()=='1,2\n3,4'
        png=call(client,'wolfram_read',job_id=job['job_id'],name='plot.png',encoding='base64')
        assert base64.b64decode(png['data']).startswith(b'\x89PNG\r\n\x1a\n')
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(png['data']))) as image:
            image.load();assert image.width>0 and image.height>0
        scientific = '''sol = DSolveValue[{y'[x] == y[x], y[0] == 1}, y[x], x];
          mat = {{3,1},{1,2}}; rhs = {9,8};
          Export["checks.json", <|
            "identity" -> TrueQ[FullSimplify[Sin[x]^2 + Cos[x]^2 == 1]],
            "ode" -> TrueQ[FullSimplify[sol == Exp[x]]],
            "linear_residual" -> N[Norm[mat.LinearSolve[mat,rhs]-rhs]],
            "complex" -> ToString[(3+4 I)/(1-2 I), InputForm],
            "pi" -> ToString[N[Pi,60], InputForm]
          |>, "RawJSON"]; 1/3'''
        job=call(client,'wolfram_submit',code=scientific)
        assert call(client,'wolfram_wait',job_id=job['job_id'],wait_seconds=60)['state']=='completed'
        checks=json.loads(call(client,'wolfram_read',job_id=job['job_id'],name='checks.json')['data'])
        assert checks['identity'] and checks['ode'] and checks['linear_residual']<1e-12
        assert checks['complex'].replace(' ','')=='-1+2*I'
        assert checks['pi'].startswith('3.14159265358979323846264338327950288419716939937510')
        job=call(client,'wolfram_submit',code='$Failed')
        assert call(client,'wolfram_wait',job_id=job['job_id'],wait_seconds=60)['state']=='failed'
        job=call(client,'wolfram_submit',code='Pause[10]; 99',timeout_seconds=1)
        assert call(client,'wolfram_wait',job_id=job['job_id'],wait_seconds=60)['state']=='timeout'
    finally:
        client.close();assert not client.non_json_stdout


def test_http_transport_preserves_official_content(tmp_path):
    pytest.importorskip('mcp')
    import asyncio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'WOLFRAM_MCP_COMMAND_JSON':json.dumps([sys.executable,str(FIXTURE/'official.py')]),
         'WOLFRAM_MCP_WORKSPACE':str(tmp_path),'WOLFRAM_MCP_PORT':str(port),'WOLFRAM_MCP_HOST':'127.0.0.1',
         'HOME':str(tmp_path/'home')}
    env.pop('WOLFRAM_LICENSE_B64',None);env.pop('WOLFRAMSCRIPT_ENTITLEMENTID',None)
    with (tmp_path/'http.stderr').open('w') as log:
        process=subprocess.Popen([sys.executable,str(ADAPTER/'server.py'),'--transport','streamable-http'],env=env,stdout=subprocess.DEVNULL,stderr=log)
        try:
            deadline=time.monotonic()+15
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:
                    assert process.poll() is None, (tmp_path/'http.stderr').read_text()
                    if time.monotonic()>deadline:pytest.fail('gateway HTTP did not start')
                    time.sleep(.05)
            async def check():
                async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (read,write,_):
                    async with ClientSession(read,write) as session:
                        await session.initialize()
                        tools=await session.list_tools()
                        assert 'WolframLanguageEvaluator' in {t.name for t in tools.tools}
                        result=await session.call_tool('WolframLanguageEvaluator',{'code':'http fixture'})
                        assert result.structuredContent=={'echo':'http fixture'}
                        assert [x.type for x in result.content]==['text','image','resource']
                        result=await session.call_tool('wolfram_upload',{'name':'http-input','data':'via HTTP'})
                        assert not result.isError and (tmp_path/'http-input').read_text()=='via HTTP'
            asyncio.run(check())
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


def test_external_deadline_marks_timeout(tmp_path):
    runtime=load_runtime();jobs=runtime.Jobs(tmp_path,{**os.environ,'WOLFRAMSCRIPT':str(FIXTURE/'native.py')})
    try:
        job=jobs.submit('hang',[],1)
        status=jobs.wait(job['job_id'],40)
        assert status['state']=='timeout' and status['exit_code'] is not None
    finally:jobs.close()


@pytest.mark.skipif(os.environ.get('ASIBENCH_WOLFRAM_IMAGE_TEST')!='1',reason='build asibench-wolfram:test for real image/preflight checks (no license needed)')
def test_image_fresh_home_preflight_and_fail_closed(tmp_path):
    # Real Docker/launcher/gateway; the upstream is a labelled PROTOCOL fixture.
    # This check never claims actual Wolfram computation.
    preflight=ROOT/'ai4sci_bench/runner/mcp_container.py'
    fixture_env=json.dumps(['/opt/venv/bin/python','/opt/wolfram-fixtures/official.py'])
    entry={'command':'/opt/venv/bin/python','args':['/opt/wolfram-mcp/server.py'],
           'cwd':'/workspace','env':{'WOLFRAM_MCP_COMMAND_JSON':fixture_env},
           'sandbox':{'deployment':'instance','startup_timeout_seconds':30}}
    code='import json,sys;sys.path.insert(0,"/opt/checks");import mcp_container;mcp_container.probe_stdio(json.loads(sys.argv[1]));print("protocol preflight passed; scientific validation not performed")'
    command=['docker','run','--rm','--network','none','--memory','4g','--cpus','2',
             '-v',f'{FIXTURE}:/opt/wolfram-fixtures:ro','-v',f'{preflight}:/opt/checks/mcp_container.py:ro',
             'asibench-wolfram:test','/opt/venv/bin/python','-c',code,json.dumps(entry)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    assert 'protocol preflight passed' in result.stdout
    command=['docker','run','--rm','--network','none','--memory','4g','--cpus','2',
             '-e','WOLFRAM_MCP_TIMEOUT=15','asibench-wolfram:test']
    result=subprocess.run(command,capture_output=True,text=True,timeout=45)
    assert result.returncode!=0 and not result.stdout
    assert 'license' in result.stderr.lower() or 'activated' in result.stderr.lower()


@pytest.mark.skipif(not LIVE,reason='requires built image and valid OFFLINE runtime license')
def test_licensed_framework_sidecar(tmp_path):
    from ai4sci_bench.runner.os_sandbox import OSSandbox
    if not os.environ.get('WOLFRAM_LICENSE_B64'):
        pytest.fail('HTTP sidecar requires WOLFRAM_LICENSE_B64 valid for offline container execution')
    (tmp_path/'check.py').write_text((Path(__file__).with_name('wolfram_http_check.py')).read_text())
    tmp_path.chmod(0o777)
    metadata={'runtime':{'image':'asibench-wolfram:test','services':[{
        'name':'wolfram','image':'asibench-wolfram:test','port':8765,
        'command':['/opt/venv/bin/python','/opt/wolfram-mcp/server.py','--transport','streamable-http'],
        'startup_timeout_seconds':300,'memory_mb':8192,'cpus':2,
        'env_from':{'WOLFRAM_LICENSE_B64':'WOLFRAM_LICENSE_B64'}}]}}
    result=OSSandbox(ROOT).execute_python(metadata,workspace=tmp_path,code_file='check.py',timeout=180)
    assert result[0],result[1]
    assert '"kernel": "real Wolfram"' in result[2]


def test_hung_upstream_startup_fails_and_cleans_up(tmp_path):
    pytest.importorskip('mcp')
    env={**os.environ,'WOLFRAM_MCP_COMMAND_JSON':json.dumps([sys.executable,'-c','import time; time.sleep(60)']),
         'WOLFRAM_MCP_WORKSPACE':str(tmp_path),'WOLFRAM_MCP_TIMEOUT':'1','HOME':str(tmp_path/'home')}
    env.pop('WOLFRAM_LICENSE_B64',None);env.pop('WOLFRAMSCRIPT_ENTITLEMENTID',None)
    result=subprocess.run([sys.executable,str(ADAPTER/'server.py')],env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==70 and not result.stdout
    assert 'Future exception was never retrieved' not in result.stderr
