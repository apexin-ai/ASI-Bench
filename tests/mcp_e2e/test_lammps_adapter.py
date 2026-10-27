"""Protocol and real LAMMPS physics; no mock numerical success."""
import base64
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
import pytest
from scripts.mcp.e2e.e2e_smoke.client import StdioMCP

ROOT=Path(__file__).parents[2]
ADAPTER=ROOT/'scripts/mcp/lammps'
LIVE=os.environ.get('ASIBENCH_LAMMPS_DOCKER')=='1'
TOOLS={'get_info','get_engine_help','list_potentials','write_input','submit_script','list_jobs',
       'get_status','wait_job','cancel_job','get_result','list_files','read_output','read_log',
       'read_trajectory','restart','single_point','minimize','run_md'}
NATIVE='''units metal
atom_style atomic
boundary p p p
region box block 0 20 0 20 0 20
create_box 1 box
create_atoms 1 single 5 5 5
create_atoms 1 single 6.5 5 5
mass 1 39.948
pair_style lj/cut 3.0
pair_coeff * * 1.0 1.0
pair_modify shift yes
thermo_style custom step atoms pe ke etotal temp press vol fmax
thermo_modify norm no format float %.16g
run 0
write_dump all custom final.lammpstrj id type x y z fx fy fz modify sort id
write_restart final.restart
'''
DIMER='2\nLattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3 pbc="T T T"\nAr 5 5 5\nAr 6.5 5 5\n'


def value(response):
    assert 'error' not in response,response
    result=response['result']; assert not result.get('isError'),result
    data=result.get('structuredContent')
    return data if data is not None else json.loads(result['content'][0]['text'])


def call(client,tool,**kwargs):
    return value(client.call_tool(tool,kwargs,timeout=90))


def done(client,job):
    status=call(client,'wait_job',job_id=job['job_id'],wait_seconds=30)
    assert status['state']=='completed', (status,call(client,'read_output',job_id=job['job_id'],name='log.lammps'))
    return call(client,'get_result',job_id=job['job_id'])


@pytest.fixture
def client(tmp_path):
    if not LIVE: pytest.skip('build asibench-lammps:test and set ASIBENCH_LAMMPS_DOCKER=1')
    client=StdioMCP('docker',['run','--rm','-i','--network','none','--cpus','2','--memory','1g','--pids-limit','128',
        '-v',f'{tmp_path}:/workspace','asibench-lammps:test'],cwd=tmp_path,env=dict(os.environ),stderr_path=tmp_path/'server.stderr.log')
    exchanges=[]; original=client.request
    def record(method,params=None,timeout=120):
        response=original(method,params,timeout);exchanges.append({'method':method,'params':params,'response':response});return response
    client.request=record
    try:
        client.initialize();assert {t['name'] for t in client.list_tools()}==TOOLS
        assert {t['name'] for t in client.list_tools()}==TOOLS
        yield client
    finally:
        client.close()
        (tmp_path/'protocol.json').write_text(json.dumps(exchanges,indent=2)+'\n')
        assert not client.non_json_stdout


def test_native_and_complete_artifact_lifecycle(client,tmp_path):
    info=call(client,'get_info')
    assert '22 Jul 2025' in info['version'] and 'MANYBODY' in info['installed_packages']
    assert 'lj/cut' in call(client,'get_engine_help',limit=32000)['text']
    pots=call(client,'list_potentials')['potentials']
    expected_pot=json.loads((ADAPTER/'potentials.json').read_text())['high_level_potentials'][0]
    assert any(p['name']==expected_pot['name'] and p['sha256']==expected_pot['sha256'] for p in pots)
    call(client,'write_input',name='include.settings',data='variable answer equal 42\n')
    job=call(client,'submit_script',script='include include.settings\n'+NATIVE,input_files=['include.settings'])
    result=done(client,job);jid=job['job_id']
    assert call(client,'get_status',job_id=jid)['exit_code']==0
    assert any(x['job_id']==jid for x in call(client,'list_jobs')['jobs'])
    thermo=call(client,'read_log',job_id=jid)
    block=thermo['blocks'][0]
    expected=4*((1/1.5)**12-(1/1.5)**6)-4*((1/3)**12-(1/3)**6)
    assert block['rows'][-1][block['columns'].index('PotEng')]==pytest.approx(expected,abs=1e-12)
    frame=call(client,'read_trajectory',job_id=jid,name='final.lammpstrj')['frames'][0]
    fx_index=frame['columns'].index('fx')
    expected_fx=24*(1/1.5**7-2/1.5**13)
    assert frame['rows'][0][fx_index]==pytest.approx(expected_fx,rel=1e-5)
    files=call(client,'list_files',job_id=jid)['files'];assert any(x['name']=='final.restart' for x in files)
    binary=call(client,'read_output',job_id=jid,name='final.restart',encoding='base64',limit=1048576)
    assert binary['eof'] and base64.b64decode(binary['data'])==(tmp_path/f'lammps-runs/{jid}/final.restart').read_bytes()
    part=call(client,'read_output',job_id=jid,name='in.lammps',offset=3,limit=7)
    assert len(part['data'].encode())==7 and part['next_offset']==10 and not part['eof']
    call(client,'write_input',name='copied.restart',data=binary['data'],encoding='base64')
    for action in ['info','data','dump','continue']:
        restart=call(client,'restart',job_id=jid,action=action,continuation_script='run 0\n')
        out=done(client,restart)
        if action=='data': assert any(x['name']=='converted.data' for x in out['files'])
        if action=='dump': assert call(client,'read_trajectory',job_id=restart['job_id'],name='converted.lammpstrj')['frames'][0]['atom_count']==2
    assert call(client,'cancel_job',job_id=jid)['state']=='completed'


def test_high_level_physics_minimization_and_md(client,tmp_path):
    call(client,'write_input',name='dimer.extxyz',data=DIMER)
    single=done(client,call(client,'single_point',input_name='dimer.extxyz'))
    block=single['thermo']['blocks'][0]
    expected=4*((1/1.5)**12-(1/1.5)**6)-4*((1/3)**12-(1/3)**6)
    assert block['rows'][-1][2]==pytest.approx(expected,abs=1e-12)
    opt=done(client,call(client,'minimize',input_name='dimer.extxyz',fmax=.0001,steps=1000))
    assert opt['thermo']['converged'] is True
    assert opt['thermo']['blocks'][-1]['rows'][-1][2]==pytest.approx(-1-4*((1/3)**12-(1/3)**6),abs=1e-8)
    short=done(client,call(client,'minimize',input_name='dimer.extxyz',fmax=1e-8,steps=1))
    assert short['thermo']['converged'] is False
    # Independent ASE calculator/optimizer reference, in a separate interpreter.
    reference=Path(__file__).with_name('lammps_reference.py')
    p=subprocess.run(['docker','run','--rm','--network','none','-v',f'{reference}:/checks/ref.py:ro',
        'asibench-lammps:test','/opt/venv/bin/python','/checks/ref.py'],capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stderr
    ref=json.loads(p.stdout)
    assert single['thermo']['blocks'][0]['rows'][-1][2]==pytest.approx(ref['dimer_energy'],abs=1e-12)
    # Native dump output precision is configurable; default forces have 6 significant digits.
    frame=single['final_structure_and_forces']['frames'][0]
    assert frame['rows'][0][5:8]==pytest.approx(ref['dimer_forces'][0],rel=1e-5)
    assert opt['thermo']['blocks'][-1]['rows'][-1][2]==pytest.approx(ref['optimized_energy'],abs=1e-8)
    for name,model in [('copper.extxyz','cu_eam'),('silicon.extxyz','si_tersoff')]:
        call(client,'write_input',name=name,data=ref[name])
        answer=done(client,call(client,'single_point',input_name=name,model=model))
        # The reference directly calls engine API with independently prepared cell.
        assert answer['thermo']['blocks'][0]['rows'][-1][2]==pytest.approx(ref[model],abs=1e-8)
    for ensemble in ['nve','nvt','npt']:
        answer=done(client,call(client,'run_md',input_name='copper.extxyz',model='cu_eam',steps=20,ensemble=ensemble,temperature=50.,timestep=.0001))
        assert answer['status']['state']=='completed'
        frames=call(client,'read_trajectory',job_id=answer['status']['job_id'],max_frames=2,max_atoms=3)
        assert len(frames['frames'])==2 and frames['frames'][0]['atoms_truncated']
        if ensemble=='nve':
            block=answer['thermo']['blocks'][0];energy=block['columns'].index('TotEng')
            assert abs(block['rows'][-1][energy]-block['rows'][0][energy])<1e-4
    serial=done(client,call(client,'submit_script',script=NATIVE))
    parallel=done(client,call(client,'submit_script',script=NATIVE,mpi_ranks=2))
    assert parallel['thermo']['blocks'][0]['rows'][-1][2]==pytest.approx(serial['thermo']['blocks'][0]['rows'][-1][2],abs=1e-12)


def test_failure_timeout_cancellation_and_validation(client,tmp_path):
    bad=call(client,'submit_script',script='not_a_lammps_command\n')
    assert call(client,'wait_job',job_id=bad['job_id'],wait_seconds=20)['state']=='failed'
    sleeper='shell sleep 20\n'
    timeout=call(client,'submit_script',script=sleeper,timeout_seconds=1)
    assert call(client,'wait_job',job_id=timeout['job_id'],wait_seconds=10)['state']=='timeout'
    cancel=call(client,'submit_script',script=sleeper)
    assert call(client,'cancel_job',job_id=cancel['job_id'])['state']=='cancelled'
    # Two native jobs must have independent working directories and bounded concurrency.
    a=call(client,'submit_script',script=sleeper);b=call(client,'submit_script',script=sleeper)
    response=client.call_tool('submit_script',{'script':sleeper})
    assert response['result']['isError']
    for job in [a,b]: call(client,'cancel_job',job_id=job['job_id'])
    jid=bad['job_id']
    outside=tmp_path/'outside.txt';outside.write_text('unchanged')
    (tmp_path/f'lammps-runs/{jid}/linked').symlink_to(outside)
    for name,args in [('write_input',{'name':'../escape','data':'bad'}),
                      ('read_output',{'job_id':jid,'name':'linked'}),
                      ('read_output',{'job_id':jid,'name':'../../outside.txt'}),
                      ('get_status',{'job_id':'../escape'}),
                      ('submit_script',{'script':'run 0','mpi_ranks':True}),
                      ('submit_script',{'script':'run 0','timeout_seconds':601}),
                      ('restart',{'job_id':jid,'action':'unknown'})]:
        response=client.call_tool(name,args);assert response.get('error') or response['result'].get('isError'),(name,response)
    assert outside.read_text()=='unchanged'
    budget=call(client,'submit_script',script='shell dd if=/dev/zero of=huge.bin bs=1048576 count=129\nshell sleep 3\n')
    assert call(client,'wait_job',job_id=budget['job_id'],wait_seconds=10)['state']=='output_limit'
    assert call(client,'get_info')['native_scripts']


def runtime_module():
    spec=importlib.util.spec_from_file_location('lammps_runtime',ADAPTER/'runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_thermo_blocks_nonconvergence_and_truncation():
    module=runtime_module()
    text='Step PotEng\n0 -2\n1 -3\nLoop time of 1\nStep PotEng\n2 -4\nLoop time of 1\nStopping criterion = max iterations\n'
    result=module.parse_thermo(text,max_rows=2)
    assert result['row_count']==3 and result['truncated'] and len(result['blocks'])==2
    assert result['converged'] is False
    assert module.parse_thermo(text.replace('max iterations','force tolerance'))['converged'] is True
    assert module.parse_thermo('Step PotEng\n0 -1\nLoop time of 1')['converged'] is None


def test_job_recovery_does_not_certify_unfinished_process(tmp_path):
    module=runtime_module();jid='a'*32
    directory=tmp_path/'lammps-runs'/jid;directory.mkdir(parents=True)
    (directory/'status.json').write_text(json.dumps({'state':'running','job_id':jid,'pid':1}))
    jobs=module.Jobs(tmp_path)
    assert jobs.status(jid)['state']=='interrupted'
    jobs.close()


def test_catalog_instance_and_service_contract():
    from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog
    from ai4sci_bench.runner.mcp_sandbox import validate_os_mcp
    config=load_mcp_config(ADAPTER/'mcp.os.json');validate_os_mcp(config)
    assert load_science_mcp_catalog()['lammps']['server']==config['lammps']
    service=load_mcp_config(ADAPTER/'mcp.service.json')
    validate_os_mcp(service,{'runtime':{'services':[{'name':'lammps','image':'example/lammps:fixed','port':8765}]}})


def test_http_sidecar_upload_compute_and_binary_download(tmp_path):
    if not LIVE: pytest.skip('set ASIBENCH_LAMMPS_DOCKER=1')
    import uuid
    suffix=uuid.uuid4().hex[:12];network='lammps-check-'+suffix;container=network+'-server'
    def run(*args,timeout=60):
        return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    assert run('docker','network','create','--internal',network).returncode==0
    try:
        started=run('docker','run','-d','--rm','--name',container,'--network',network,'--network-alias','lammps',
                    '--memory','1g','--cpus','2','--pids-limit','128','asibench-lammps:test',
                    '/opt/venv/bin/python','/opt/lammps-mcp/server.py','--transport','streamable-http')
        assert started.returncode==0,started.stderr
        health=''
        for _ in range(40):
            health=run('docker','inspect','--format','{{.State.Health.Status}}',container).stdout.strip()
            if health=='healthy':break
            if health=='unhealthy':break
            time.sleep(1)
        assert health=='healthy',run('docker','logs',container).stderr
        check=Path(__file__).with_name('lammps_http_check.py')
        tested=run('docker','run','--rm','--network',network,'-v',f'{check}:/checks/client.py:ro',
                   'asibench-lammps:test','/opt/venv/bin/python','/checks/client.py',timeout=90)
        (tmp_path/'http-client.stdout').write_text(tested.stdout)
        (tmp_path/'http-client.stderr').write_text(tested.stderr)
        assert tested.returncode==0,tested.stderr
        assert json.loads(tested.stdout)['binary_transfer']
    finally:
        logs=run('docker','logs',container)
        (tmp_path/'http-server.log').write_text(logs.stdout+logs.stderr)
        run('docker','rm','-f',container)
        run('docker','network','rm',network)
