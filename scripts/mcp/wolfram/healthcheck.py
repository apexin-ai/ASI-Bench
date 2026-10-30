"""HTTP readiness checks the gateway AND its already-running official upstream."""
import asyncio
import os
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client(f'http://localhost:{os.environ.get("WOLFRAM_MCP_PORT", "8765")}/mcp') as (read, write, _):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tools = await client.list_tools()
            if not {'WolframLanguageEvaluator', 'wolfram_submit', 'wolfram_read'} <= {t.name for t in tools.tools}:
                raise RuntimeError('missing Wolfram tools')

if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(main(), 8))
