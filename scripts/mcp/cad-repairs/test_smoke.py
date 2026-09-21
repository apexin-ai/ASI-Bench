"""Opt-in live stdio regression tests; never connect to real CAD hosts.

CAD_MCP_REPAIR_ROOT must contain sketchup/, fusion360/, cadquery/ with .venv.
Run with a Python environment containing pytest and mcp==1.30.0.
"""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(os.environ['CAD_MCP_REPAIR_ROOT']).resolve()


async def all_tools(session):
    tools, seen, cursor = [], set(), None
    while True:
        page = await session.list_tools(cursor=cursor)
        tools.extend(tool.model_dump(mode='json') for tool in page.tools)
        cursor = page.nextCursor
        if not cursor:
            break
        assert cursor not in seen, 'Repeated pagination cursor'
        seen.add(cursor)
    assert len({t['name'] for t in tools}) == len(tools)
    return tools


@pytest.mark.parametrize('name,expected', [('sketchup', 10), ('fusion360', 10), ('cadquery', 10)])
def test_stdio(name, expected, tmp_path):
    async def check():
        base = ROOT / name
        if name == 'fusion360':
            command = str(base / '.venv/bin/python')
            args = [str(base / 'src/main.py'), '--mcp']
        else:
            command = str(base / '.venv/bin' / ('sketchup-mcp' if name == 'sketchup' else 'mcp-cadquery'))
            args = [] if name == 'sketchup' else ['--mode', 'stdio']
        # Deliberately do not inherit credentials or PYTHONPATH; run outside source.
        env = {'HOME': str(tmp_path), 'PATH': str(base / '.venv/bin') + ':/usr/bin:/bin',
               'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1'}
        with (tmp_path / 'stderr.log').open('w') as err:
            try:
                async with asyncio.timeout(60):
                    async with stdio_client(StdioServerParameters(command=command, args=args, env=env, cwd=str(tmp_path)), errlog=err) as (rd, wr):
                        async with ClientSession(rd, wr) as session:
                            initialized = await session.initialize()
                            tools = await all_tools(session)
                            assert tools
                            if expected:
                                assert len(tools) == expected
                            assert tools == await all_tools(session)
                            await session.send_ping()
                            # Must be a failed tool result, not a successful empty response.
                            invalid = await session.call_tool('__nonexistent__', {})
                            assert invalid.isError
                            if name == 'fusion360':
                                bad = await session.call_tool('CreateSketch', {'plane': 123})
                                assert bad.isError
                                result = await session.call_tool('CreateSketch', {'plane': 'xy'})
                                assert not result.isError
                                text = '\n'.join(c.text for c in result.content if c.type == 'text')
                                assert 'adsk.core' in text and 'sketches.add' in text
                            elif name == 'cadquery':
                                bad = await session.call_tool('execute_cadquery_script', {})
                                assert bad.isError
                                backend_error = await session.call_tool('get_shape_properties', {'result_id': '__missing__'})
                                assert backend_error.isError
                                result = await session.call_tool('search_parts', {'query': ''})
                                assert not result.isError
                                payload = json.loads(result.content[0].text)
                                assert payload.get('success') is True
                            assert tools == await all_tools(session)
                            print(json.dumps({'server': name, 'protocol': initialized.protocolVersion,
                                              'tools': len(tools), 'status': 'PASS'}))
            except BaseException:
                err.flush()
                print((tmp_path / 'stderr.log').read_text())
                raise
    # SketchUp hardcodes localhost:9876 and connects at startup. Reserve the port
    # WITHOUT listening, so an installed real extension can never be reached.
    with socket.socket() as guard:
        if name == 'sketchup':
            guard.bind(('127.0.0.1', 9876))  # fail closed if already occupied
        asyncio.run(check())


def test_cadquery_installed_models(tmp_path):
    code = '''from mcp_cadquery_server.models import ExecuteCadqueryScriptArgs
from pydantic import ValidationError
assert ExecuteCadqueryScriptArgs(workspace_path='.', script='', parameters={'x': 1}).parameters == {'x': 1}
for field in ('parameters', 'parameter_sets'):
    try:
        ExecuteCadqueryScriptArgs(workspace_path='.', script='', **{field: 'invalid'})
    except ValidationError:
        pass
    else:
        raise AssertionError(field)
'''
    subprocess.run([str(ROOT / 'cadquery/.venv/bin/python'), '-I', '-c', code], cwd=tmp_path, check=True)
