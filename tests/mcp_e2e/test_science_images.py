"""Opt-in business regressions against locally built science MCP Docker images.

ASIBENCH_SCIENCE_IMAGES=1 pytest -q tests/mcp_e2e/test_science_images.py
"""
import json
import os
import math
import subprocess
from pathlib import Path

import pytest

from scripts.mcp.e2e.e2e_smoke.client import StdioMCP

pytestmark = pytest.mark.skipif(os.environ.get('ASIBENCH_SCIENCE_IMAGES') != '1', reason='requires built science Docker images')


def payload(response):
    assert 'error' not in response, response
    result = response['result']
    assert not result.get('isError'), result
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])


@pytest.fixture(params=['gmsh', 'fenicsx'])
def server(request, tmp_path):
    name = request.param
    client = StdioMCP('docker', ['run', '--rm', '-i', '--network', 'none', '-v', f'{tmp_path}:/workspace', f'asibench-{name}:test'], cwd=tmp_path, env=dict(os.environ), stderr_path=tmp_path/'stderr.log')
    try:
        client.initialize()
        expected = {'get_info', 'mesh_box', 'inspect_mesh'} if name == 'gmsh' else {'get_info', 'solve_poisson_manufactured'}
        assert {tool['name'] for tool in client.list_tools()} == expected
        assert {tool['name'] for tool in client.list_tools()} == expected
        yield name, client, tmp_path
    finally:
        client.close()
    assert not client.non_json_stdout, client.non_json_stdout


def test_business_and_saved_outputs(server):
    name, client, workspace = server
    info = payload(client.call_tool('get_info', {}))
    if name == 'gmsh':
        assert info['gmsh_version']
        mesh = payload(client.call_tool('mesh_box', dict(length=1, width=1, height=1, element_size=.3)))
        saved = payload(client.call_tool('inspect_mesh', dict(mesh_name='box.msh')))
        assert mesh == saved
        reference = independent_reference(name, workspace)
        assert mesh['nodes'] == reference['nodes']
        assert mesh['elements'] == reference['elements']
        assert mesh['nodes'] > 0 and mesh['elements'] > 0
        assert (workspace/'box.msh').stat().st_size > 100
    else:
        assert info['dolfinx_version']
        values = {}
        for n, degree in [(8, 1), (16, 1), (8, 2)]:
            result = payload(client.call_tool('solve_poisson_manufactured', dict(nx=n, ny=n, degree=degree, output_name=f'p{n}d{degree}.xdmf')))
            assert result['finite'] and result['convergence_reason'] > 0
            assert result['mesh_cells'] == 2*n*n
            assert result['dofs'] == (n*degree+1)**2
            assert (workspace/f'p{n}d{degree}.h5').stat().st_size > 100
            values[n, degree] = result['l2_error']
        assert 3 < values[8, 1]/values[16, 1] < 5
        assert values[8, 2] < values[16, 1]
        reference = independent_reference(name, workspace)
        for (n, degree), error in values.items():
            assert math.isclose(error, reference[f'{n},{degree}']['l2_error'], rel_tol=1e-9, abs_tol=1e-12)
    payload(client.call_tool('get_info', {}))


def independent_reference(name, workspace):
    reference = Path(__file__).with_name('science_image_reference.py')
    result = subprocess.run(['docker', 'run', '--rm', '--network', 'none',
        '-v', f'{workspace}:/workspace:ro', '-v', f'{reference}:/checks/reference.py:ro',
        '--entrypoint', 'python3', f'asibench-{name}:test', '/checks/reference.py', name],
        capture_output=True, text=True, check=True, timeout=240)
    return json.loads(result.stdout)


def test_invalid_calls_are_rejected(server):
    name, client, workspace = server
    if name == 'gmsh':
        tool = 'mesh_box'
        valid = dict(length=1, width=1, height=1, element_size=.3)
        cases = [{**valid, 'length': -1}, {**valid, 'element_size': 1e-6},
                 {**valid, 'output_name': '../escape.msh'},
                 {**valid, 'output_name': 'bad.txt'}]
        outside = workspace.parent/'outside.msh'
        outside.write_text('untouched')
        (workspace/'link.msh').symlink_to(outside)
        cases.append({**valid, 'output_name': 'link.msh'})
    else:
        tool = 'solve_poisson_manufactured'
        valid = dict(nx=8, ny=8)
        cases = [{**valid, 'nx': 0}, {**valid, 'ny': 257}, {**valid, 'degree': 3},
                 {**valid, 'output_name': '../escape.xdmf'},
                 {**valid, 'output_name': 'bad.txt'}]
        outside = workspace.parent/'outside.h5'
        outside.write_text('untouched')
        (workspace/'link.h5').symlink_to(outside)
        cases.append({**valid, 'output_name': 'link.xdmf'})
    for arguments in cases:
        result = client.call_tool(tool, arguments)
        assert result.get('error') or result.get('result', {}).get('isError'), result
    result = client.call_tool('__missing__', {})
    assert result.get('error') or result.get('result', {}).get('isError'), result
    assert outside.read_text() == 'untouched'
    payload(client.call_tool('get_info', {}))
