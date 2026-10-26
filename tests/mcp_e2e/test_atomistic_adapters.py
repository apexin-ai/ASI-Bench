"""Real, opt-in ASE/OVITO MCP stdio tests; no mocked scientific success."""
import json
import math
import os
from pathlib import Path
import subprocess
import pytest

from scripts.mcp.e2e.e2e_smoke.client import StdioMCP

LIVE=os.environ.get('ASIBENCH_ATOMISTIC_DOCKER')=='1'
ROOT=Path(__file__).parents[2]


@pytest.fixture
def pair(tmp_path):
    if not LIVE:
        pytest.skip('build asibench-ase-ovito:test and set ASIBENCH_ATOMISTIC_DOCKER=1')
    clients={}
    try:
        for name in ['ase','ovito']:
            client=StdioMCP('docker',['run','--rm','-i','--network','none','-v',f'{tmp_path}:/workspace',
                'asibench-ase-ovito:test','/opt/venv/bin/python',f'/opt/atomistic-mcp/{name}_server.py'],
                cwd=tmp_path,env=dict(os.environ),stderr_path=tmp_path/f'{name}.stderr.log')
            clients[name]=client
            exchanges=[]
            original=client.request
            def record(method, params=None, timeout=120, original=original, exchanges=exchanges):
                response=original(method,params,timeout)
                exchanges.append({'method':method,'params':params,'response':response})
                return response
            client.request=record
            client.exchanges=exchanges
            client.initialize()
            expected=({'get_info','build_bulk','read_structure','repeat_structure','single_point','optimize_structure'} if name=='ase'
                      else {'get_info','inspect_structure','identify_structure','compute_rdf','dislocation_analysis','render_snapshot'})
            assert {t['name'] for t in client.list_tools()}==expected
            assert {t['name'] for t in client.list_tools()}==expected
        yield clients,tmp_path
    finally:
        for name,client in clients.items():
            client.close()
            (tmp_path/f'{name}.protocol.json').write_text(json.dumps(client.exchanges,indent=2)+'\n')
        assert all(not c.non_json_stdout for c in clients.values()),'non-JSON native stdout'


def call(client,tool,**arguments):
    response=client.call_tool(tool,arguments,timeout=180)
    assert 'error' not in response,response
    result=response['result']
    assert not result.get('isError'),result
    value=result.get('structuredContent')
    if value is None: value=json.loads(result['content'][0]['text'])
    return value


def test_real_tools_and_independent_reference(pair):
    clients,workspace=pair
    ase,ovito=clients['ase'],clients['ovito']
    assert call(ase,'get_info')['ase_version']=='3.29.0'
    assert call(ovito,'get_info')['ovito_version'].startswith('3.17.1')
    built=call(ase,'build_bulk',symbol='Cu',crystalstructure='fcc',lattice_constant=3.61,repeat=[3,3,3],output_name='cu.extxyz')
    assert built['atom_count']==108
    read=call(ase,'read_structure',input_name='cu.extxyz')
    assert read['positions_A']==built['positions_A']
    repeat=call(ase,'repeat_structure',input_name='cu.extxyz',repeat=[2,1,1],output_name='repeat.cif')
    assert repeat['atom_count']==216
    assert call(ase,'read_structure',input_name='repeat.cif')['atom_count']==216
    prediction=call(ase,'single_point',input_name='cu.extxyz',calculator='emt')
    (workspace/'dimer.xyz').write_text('2\nAr dimer\nAr 0 0 0\nAr 1.5 0 0\n')
    lj=call(ase,'single_point',input_name='dimer.xyz',calculator='lennard_jones')
    expected=4*((1/1.5)**12-(1/1.5)**6)-4*((1/3)**12-(1/3)**6)
    assert math.isclose(lj['energy_eV'],expected,abs_tol=1e-12)
    optimized=call(ase,'optimize_structure',input_name='dimer.xyz',calculator='lennard_jones',fmax=.01,steps=100)
    assert optimized['converged'] and optimized['energy_eV']<lj['energy_eV']
    unfinished=call(ase,'optimize_structure',input_name='dimer.xyz',calculator='lennard_jones',fmax=.001,steps=1,output_name='unfinished.extxyz')
    assert unfinished['converged'] is False and unfinished['steps']==1
    inspected=call(ovito,'inspect_structure',input_name='cu.extxyz')
    assert inspected['particle_count']==108 and inspected['pbc']==[True]*3
    types=call(ovito,'identify_structure',input_name='cu.extxyz')
    assert types['counts']['FCC']==108 and max(types['rmsd'])<1e-6
    rdf=call(ovito,'compute_rdf',input_name='cu.extxyz',cutoff=4,bins=80)
    assert len(rdf['g_r'])==80 and set(rdf['coordination'])=={18}
    dxa=call(ovito,'dislocation_analysis',input_name='cu.extxyz',lattice='fcc')
    assert dxa['segment_count']==0 and dxa['total_line_length']==0
    rendered=call(ovito,'render_snapshot',input_name='cu.extxyz',width=320,height=240)
    from PIL import Image
    import hashlib
    image_path=workspace/'snapshot.png'
    with Image.open(image_path) as image:
        assert image.size==(320,240) and image.format=='PNG'
        assert any(low != high for low,high in image.convert('RGB').getextrema())
    assert rendered['sha256']==hashlib.sha256(image_path.read_bytes()).hexdigest()
    reference=Path(__file__).with_name('atomistic_reference.py')
    process=subprocess.run(['docker','run','--rm','--network','none','-v',f'{workspace}:/workspace:ro',
        '-v',f'{reference}:/checks/reference.py:ro','asibench-ase-ovito:test','/opt/venv/bin/python','/checks/reference.py'],
        capture_output=True,text=True,timeout=180)
    assert process.returncode==0,process.stderr
    ref=json.loads(process.stdout)
    assert math.isclose(prediction['energy_eV'],ref['energy_eV'],abs_tol=1e-10)
    assert prediction['forces_eV_per_A']==ref['forces_eV_per_A']
    assert math.isclose(optimized['energy_eV'],ref['optimized_energy_eV'],abs_tol=1e-10)
    assert rdf['r_A']==pytest.approx(ref['r_A'],abs=1e-12)
    assert rdf['g_r']==pytest.approx(ref['g_r'],abs=1e-8)
    for name,result in [('structure-types.json',types),('rdf.json',rdf),('dislocations.json',dxa)]:
        assert json.loads((workspace/name).read_text())=={k:v for k,v in result.items() if k!='analysis_path'}


def test_invalid_calls_and_no_side_effects(pair):
    clients,workspace=pair
    call(clients['ase'],'build_bulk',symbol='Cu',crystalstructure='fcc',lattice_constant=3.61,repeat=[3,3,3],output_name='cu.extxyz')
    (workspace/'dimer.xyz').write_text('2\n\nAr 0 0 0\nAr 1.5 0 0\n')
    outside=workspace.parent/'outside.json';outside.write_text('unchanged')
    (workspace/'linked.json').symlink_to(outside)
    (workspace/'too-many.xyz').write_text('500000\n\n')
    cases=[('ase','build_bulk',dict(symbol='Cu',crystalstructure='fcc',lattice_constant=3.61,repeat=[32]*3)),
           ('ase','build_bulk',dict(symbol='Cu',crystalstructure='fcc',lattice_constant=-1,repeat=[1]*3)),
           ('ase','single_point',dict(input_name='cu.extxyz',calculator='vasp')),
           ('ase','single_point',dict(input_name='dimer.xyz',calculator='emt')),
           ('ase','read_structure',dict(input_name='../outside.json')),
           ('ase','read_structure',dict(input_name='too-many.xyz')),
           ('ase','build_bulk',dict(symbol='Cu',crystalstructure='fcc',lattice_constant=3.61,repeat=[True,1,1])),
           ('ase','optimize_structure',dict(input_name='dimer.xyz',steps=0)),
           ('ovito','compute_rdf',dict(input_name='dimer.xyz',cutoff=4)),
           ('ovito','compute_rdf',dict(input_name='cu.extxyz',cutoff=20)),
           ('ovito','compute_rdf',dict(input_name='cu.extxyz',bins=513)),
           ('ovito','identify_structure',dict(input_name='cu.extxyz',output_name='linked.json')),
           ('ovito','identify_structure',dict(input_name='cu.extxyz',output_name='../bad.json')),
           ('ovito','dislocation_analysis',dict(input_name='cu.extxyz',lattice='unknown')),
           ('ovito','render_snapshot',dict(input_name='cu.extxyz',width=1025)),
           ('ovito','compute_rdf',dict(input_name='cu.extxyz',bins=True))]
    for server,tool,args in cases:
        result=clients[server].call_tool(tool,args)
        assert result.get('error') or result.get('result',{}).get('isError'),(tool,result)
    assert outside.read_text()=='unchanged'
    for client in clients.values():
        result=client.call_tool('__unknown__',{})
        assert result.get('error') or result.get('result',{}).get('isError')
        call(client,'get_info')


def test_local_catalog_and_os_contract():
    from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog
    from ai4sci_bench.runner.mcp_sandbox import validate_os_mcp
    config=load_mcp_config(ROOT/'scripts/mcp/atomistic/mcp.os.json')
    validate_os_mcp(config)
    catalog=load_science_mcp_catalog()
    assert catalog['ase_native']['server']==config['ase']
    assert catalog['ovito']['server']==config['ovito']
    # The pre-existing third-party ASE entry keeps its original contract.
    assert 'mcp-atomictoolkit' in catalog['ase']['source']


@pytest.fixture
def common_module(monkeypatch,tmp_path):
    import importlib.util
    monkeypatch.setenv('ATOMISTIC_MCP_WORKSPACE',str(tmp_path))
    spec=importlib.util.spec_from_file_location('atomistic_common',ROOT/'scripts/mcp/atomistic/common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name',['../escape.extxyz','/etc/passwd','missing.extxyz'])
def test_input_paths_reject_escape_and_missing(common_module,name):
    with pytest.raises(ValueError): common_module.input_path(name)


def test_output_symlink_and_numeric_bounds(common_module,tmp_path):
    source=tmp_path/'data.extxyz';source.write_text('unchanged')
    (tmp_path/'link.extxyz').symlink_to(source)
    with pytest.raises(ValueError):common_module.output_path('link.extxyz')
    for value in [True,0,float('nan'),float('inf')]:
        with pytest.raises(ValueError):common_module.positive(value,.1,20,'cutoff')
    for value in [True,1.0,0,33]:
        with pytest.raises(ValueError):common_module.bounded_int(value,1,32,'repeat')
