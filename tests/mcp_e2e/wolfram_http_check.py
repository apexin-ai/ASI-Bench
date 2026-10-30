"""Licensed real-kernel check from a separate container with no shared files."""
import asyncio
import base64
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client('http://wolfram:8765/mcp') as (read,write,_):
        async with ClientSession(read,write) as session:
            await session.initialize()
            async def call(tool, **arguments):
                result=await session.call_tool(tool,arguments)
                assert not result.isError,(tool,result)
                return result.structuredContent or json.loads(result.content[0].text)
            await call('wolfram_upload',name='input.csv',data='1,2\n3,4\n')
            job=await call('wolfram_submit',code='Export["copy.csv", Import["input.csv", "CSV"], "CSV"]; 1/3',input_files=['input.csv'])
            state=await call('wolfram_wait',job_id=job['job_id'],wait_seconds=60)
            assert state['state']=='completed',state
            expression=await call('wolfram_read',job_id=job['job_id'],name='result.inputform.txt')
            assert expression['data'].strip()=='1/3'
            copied=await call('wolfram_read',job_id=job['job_id'],name='copy.csv',encoding='base64')
            assert base64.b64decode(copied['data']).decode().replace('\r','').strip()=='1,2\n3,4'
            print(json.dumps({'transport':'streamable-http','kernel':'real Wolfram','native_expression':'1/3','binary_artifact_transfer':True}))

if __name__=='__main__':asyncio.run(main())
