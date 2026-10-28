"""Real HTTP MCP initialization, not just an open TCP port."""
import asyncio
import os
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client(f'http://localhost:{os.environ.get("LAMMPS_MCP_PORT", "8765")}/mcp') as (read,write,_):
        async with ClientSession(read,write) as session:
            await session.initialize()
            tools=await session.list_tools()
            if not {'get_info','submit_script','get_result'} <= {t.name for t in tools.tools}:
                raise RuntimeError('missing LAMMPS tools')

if __name__=='__main__': asyncio.run(main())
