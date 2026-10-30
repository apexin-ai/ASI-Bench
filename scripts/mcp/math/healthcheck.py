"""Readiness requires the actual backend import and MCP tools, not just an open port."""
import asyncio
import os
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    port = os.environ.get('MATH_MCP_PORT', '8765')
    async with streamable_http_client(f'http://localhost:{port}/mcp') as (read, write, _):
        async with ClientSession(read, write) as client:
            await client.initialize()
            info = await client.call_tool('get_info', {})
            if info.isError:
                raise RuntimeError('backend unavailable')
            tools = await client.list_tools()
            if not {'evaluate', 'submit_script', 'read_file'} <= {t.name for t in tools.tools}:
                raise RuntimeError('missing mathematics tools')


if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(main(), 8))
