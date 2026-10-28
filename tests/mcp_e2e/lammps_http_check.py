"""A separate-container HTTP client: upload, compute, fetch bytes, no shared workspace."""
import asyncio
import json
import math
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

DIMER='2\nLattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3 pbc="T T T"\nAr 5 5 5\nAr 6.5 5 5\n'

async def main():
    async with streamable_http_client('http://lammps:8765/mcp') as (read,write,_):
        async with ClientSession(read,write) as session:
            await session.initialize()
            async def call(tool,**kwargs):
                result=await session.call_tool(tool,kwargs)
                assert not result.isError,(tool,result)
                return result.structuredContent or json.loads(result.content[0].text)
            await call('write_input',name='dimer.extxyz',data=DIMER)
            job=await call('single_point',input_name='dimer.extxyz')
            state=await call('wait_job',job_id=job['job_id'],wait_seconds=20)
            assert state['state']=='completed',state
            answer=await call('get_result',job_id=job['job_id'])
            energy=answer['thermo']['blocks'][0]['rows'][-1][2]
            expected=4*((1/1.5)**12-(1/1.5)**6)-4*((1/3)**12-(1/3)**6)
            assert math.isclose(energy,expected,abs_tol=1e-12)
            restart=await call('read_output',job_id=job['job_id'],name='final.restart',encoding='base64')
            assert restart['eof'] and restart['data']
            print(json.dumps({'transport':'streamable-http','energy_eV':energy,'status':state,'binary_transfer':True}))

if __name__=='__main__':asyncio.run(main())
