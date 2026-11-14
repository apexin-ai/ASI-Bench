"""Protocol client inside the real OSSandbox; not a model-driven agent."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


@asynccontextmanager
async def connect(entry):
    if 'url' in entry:
        async with streamable_http_client(entry['url']) as (read, write, _):
            yield read, write
    else:
        async with stdio_client(StdioServerParameters(command=entry['command'], args=entry['args'],
               env=entry.get('env', {}), cwd='/workspace')) as streams:
            yield streams


async def main():
    backend = os.environ['CHECK_BACKEND']
    overrides = [sys.argv[i+1] for i,arg in enumerate(sys.argv[:-1]) if arg == '-c']
    entry = tomllib.loads('\n'.join(overrides))['mcp_servers'][backend+'_native']
    code, files = json.loads(Path('/workspace/case.json').read_text())
    async with connect(entry) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            async def call(tool, args):
                response = await session.call_tool(tool, args)
                assert not response.isError, response
                value = response.structuredContent or json.loads(response.content[0].text)
                return value['result'] if isinstance(value,dict) and set(value)=={'result'} else value
            assert (await call('get_info',{}))['backend']==backend
            for name,data in files.items(): await call('upload_file',{'name':name,'data':data})
            job=await call('submit_script',{'code':code,'input_files':list(files),'timeout_seconds':120})
            for _ in range(15):
                status=await call('wait_job',{'job_id':job['job_id'],'wait_seconds':10})
                if status['state']!='running':break
            assert status['state']=='completed',status
            output=await call('read_file',{'job_id':job['job_id'],'name':'stdout.log'})
            # Persistent artifacts are transferred before the sidecar is removed.
            Path('/workspace/verified.json').write_text(json.dumps({'backend':backend,'status':status,'output':output}))
            Path('/workspace/agent-marker').write_text('instance only')
            inventory=await call('list_files',{})
            visible=any(x['name']=='agent-marker' for x in inventory['files'])
            assert visible==(os.environ['CHECK_DEPLOYMENT']=='instance')
    print('ENGINEERING_FRAMEWORK_PASS')


if __name__=='__main__':asyncio.run(main())
