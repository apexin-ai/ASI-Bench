"""Actual MCP client in OSSandbox; intentionally not a model-driven agent test."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
import sys
from pathlib import Path
import tomllib

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


@asynccontextmanager
async def connect(entry):
    if 'url' in entry:
        async with streamable_http_client(entry['url']) as (read, write, _):
            yield read, write
    else:
        async with stdio_client(StdioServerParameters(command=entry['command'], args=entry.get('args', []),
                                 env=entry.get('env', {}), cwd='/workspace')) as streams:
            yield streams


async def main():
    backend = os.environ['CHECK_BACKEND']
    overrides = [sys.argv[i + 1] for i, arg in enumerate(sys.argv[:-1]) if arg == '-c']
    entry = tomllib.loads('\n'.join(overrides))['mcp_servers'][backend]
    async with connect(entry) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            async def call(tool, args):
                result = await session.call_tool(tool, args)
                assert not result.isError, result
                return result.structuredContent or json.loads(result.content[0].text)
            assert (await call('get_info', {}))['backend'] == backend
            result = await call('evaluate', {'code': 'QQ(1)/3' if backend == 'sagemath' else 'Rational(1,3)'})
            assert result['status']['state'] == 'completed', result
            assert result['result']['representation'] == '1/3', result
            Path('/workspace/agent-marker').write_text('agent-only')
            files = await call('list_files', {})
            visible = any(item['name'] == 'agent-marker' for item in files['files'])
            assert visible == (os.environ['CHECK_DEPLOYMENT'] == 'instance')
            await call('upload_file', {'name': 'from-client.txt', 'data': '11'})
            code = "from pathlib import Path\nn=int(Path('from-client.txt').read_text())\nPath('export.bin').write_bytes(bytes(range(n)))\n" + ('QQ(n)/13' if backend == 'sagemath' else 'Rational(n,13)')
            job = await call('submit_script', {'code': code, 'input_files': ['from-client.txt']})
            assert (await call('wait_job', {'job_id': job['job_id'], 'wait_seconds': 60}))['state'] == 'completed'
            artifact = await call('read_file', {'job_id': job['job_id'], 'name': 'export.bin', 'encoding': 'base64'})
            import base64
            assert base64.b64decode(artifact['data']) == bytes(range(11))
            Path('/workspace/verified.json').write_text(json.dumps({'answer': '1/3', 'backend': backend, 'artifact': artifact}))
    print('MATH_FRAMEWORK_SCIENCE_PASS')


if __name__ == '__main__':
    asyncio.run(main())
