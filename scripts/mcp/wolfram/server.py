"""Official Wolfram MCP gateway plus native jobs and lossless file transfer."""
from __future__ import annotations
import argparse
import asyncio
from contextlib import suppress
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import signal
import shutil
import sys

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
import uvicorn
from pydantic import StrictInt

from bootstrap import HERE, official_command, prepare_environment
from runtime import Jobs, MAX_BYTES, safe_path

log = logging.getLogger('wolfram-mcp')


class Official:
    """One upstream per task, owned by a single task for safe SDK cleanup."""
    def __init__(self, command, env, workspace, timeout=180):
        self.command, self.env, self.workspace, self.timeout = command, env, workspace, timeout
        self.task = None
        self.session = None
        self.ready = None
        self.stop = None
        self.failed = False
        self.lock = asyncio.Lock()
        self.calls = asyncio.Lock()
        self.info = None

    async def _owner(self):
        try:
            params = StdioServerParameters(command=self.command[0], args=self.command[1:], env=self.env, cwd=str(self.workspace))
            async def roots(context):
                return types.ListRootsResult(roots=[types.Root(uri=self.workspace.as_uri(), name='task-workspace')])
            async with stdio_client(params, errlog=sys.stderr) as (read, write):
                async with ClientSession(read, write, list_roots_callback=roots,
                                         read_timeout_seconds=timedelta(seconds=self.timeout)) as session:
                    self.info = await session.initialize()
                    self.session = session
                    if not self.ready.done():
                        self.ready.set_result(None)
                    await self.stop.wait()
        except BaseException as exc:
            self.failed = True
            if not self.ready.done():
                self.ready.set_exception(RuntimeError('official Wolfram MCP failed to start; check runtime installation and license'))
            if not isinstance(exc, asyncio.CancelledError):
                log.error('Official Wolfram MCP disconnected (%s); no fallback backend', type(exc).__name__)
        finally:
            self.session = None

    async def _start(self):
        self.ready = asyncio.get_running_loop().create_future()
        self.stop = asyncio.Event()
        self.failed = False
        self.task = asyncio.create_task(self._owner())
        try:
            await asyncio.wait_for(asyncio.shield(self.ready), self.timeout)
        except BaseException:
            self.ready.cancel()
            self.task.cancel()
            await self.task
            raise

    async def start(self):
        async with self.lock:
            await self._start()

    async def _close(self):
        if self.task:
            self.stop.set()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 5)
            except asyncio.TimeoutError:
                self.task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.task
            self.task = None

    async def close(self):
        async with self.lock:
            await self._close()

    async def invoke(self, method, *args, **kwargs):
        async with self.calls:
            if self.failed or self.session is None:
                raise RuntimeError('official MCP unavailable; use wolfram_restart to create a fresh upstream (sessions may be lost)')
            try:
                return await asyncio.wait_for(getattr(self.session, method)(*args, **kwargs), self.timeout)
            except (TimeoutError, asyncio.CancelledError):
                # Do not silently replay code after an uncertain execution outcome.
                self.failed = True
                raise
            except McpError as exc:
                if exc.error.code == 408:
                    self.failed = True
                raise

    async def restart(self):
        async with self.lock:
            await self._close()
            await self._start()
        return {'state': 'restarted', 'note': 'Do not assume previous evaluator sessions, files or unsaved state were restored.'}

    async def all_tools(self):
        tools, cursor, seen = [], None, set()
        while True:
            response = await self.invoke('list_tools', cursor=cursor)
            tools.extend(response.tools)
            cursor = response.nextCursor
            if not cursor:
                return tools
            if cursor in seen:
                raise RuntimeError('official tools/list repeated pagination cursor')
            seen.add(cursor)


def build_server(jobs: Jobs, official: Official):
    host = os.environ.get('WOLFRAM_MCP_HOST', '0.0.0.0')
    port = int(os.environ.get('WOLFRAM_MCP_PORT', '8765'))
    mcp = FastMCP('ASI-Bench Wolfram', host=host, port=port, json_response=True,
                 transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                    allowed_hosts=[f'localhost:{port}', f'127.0.0.1:{port}', f'wolfram:{port}'],
                    allowed_origins=[f'http://localhost:{port}', f'http://wolfram:{port}']))

    @mcp.tool()
    async def wolfram_info() -> dict:
        """Report upstream server identity and installed image versions, without credentials.

        Availability proves the official kernel started, not every optional frontend/cloud feature.
        """
        lock = HERE / 'upstream.lock.json'
        return {'official_server': official.info.serverInfo.model_dump(mode='json') if official.info else None,
                'upstream_lock': json.loads(lock.read_text()) if lock.exists() else None,
                'workspace': str(jobs.workspace), 'native_job_limit': 2,
                'native_job_timeout_max_seconds': 3600,
                'frontend': 'optional installation; verify UsingFrontEnd before relying on it',
                'trust_boundary': 'Wolfram code has container privileges; file API checks are not a code sandbox'}

    @mcp.tool()
    async def wolfram_restart() -> dict:
        """Stop/restart official MCP and its kernels after failure; do not assume sessions survive.

        Independent native jobs are unaffected. No computation is silently retried.
        """
        return await official.restart()

    @mcp.tool()
    def wolfram_submit(code: str = '', script_name: str | None = None,
                       input_files: list[str] | None = None, timeout_seconds: StrictInt = 120) -> dict:
        """Run arbitrary native WL asynchronously in a FRESH kernel, independent of Evaluator sessions.

        Give code OR an uploaded .wl/.wls script_name. Stage relative input_files into
        the job directory. Save exports there, poll/wait, then download artifacts.
        Return value is preserved as InputForm and WXF, not a lossy JSON float.
        For persistent interactive definitions use official WolframLanguageEvaluator(session=...).
        """
        if bool(code) == bool(script_name):
            raise ValueError('provide exactly one of code or script_name')
        if script_name:
            path = safe_path(jobs.workspace, script_name, exists=True)
            if path.stat().st_size > 1024 * 1024:
                raise ValueError('script exceeds 1 MiB')
            code = path.read_text()
        return jobs.submit(code, input_files, timeout_seconds)

    @mcp.tool()
    def wolfram_capabilities(timeout_seconds: StrictInt = 60) -> dict:
        """Submit a real version/paclet/frontend probe; read capabilities.json after completion.

        Reports installed frontend binaries, not successful GUI/online-service authorization.
        """
        code = '''Export["capabilities.json", <|
          "version" -> $Version, "system_id" -> $SystemID,
          "processor_count" -> $ProcessorCount,
          "paclets" -> (KeyTake[Association[#], {"Name", "Version"}] & /@ PacletFind[]),
          "frontend_available_in_session" -> ($FrontEnd =!= Null)
        |>, "RawJSON"]; $Version'''
        return jobs.submit(code, [], timeout_seconds)

    @mcp.tool()
    def wolfram_notebook(name: str, action: str = 'evaluate', timeout_seconds: StrictInt = 120) -> dict:
        """Submit real front-end Notebook evaluation or rendering (requires usable licensed frontend).

        Upload the .nb first. action=evaluate inserts results and saves evaluated.nb;
        action=render creates notebook.png. Does not fake a frontend with a text parser.
        Dynamic/Manipulate UI interaction is not provided by this headless file workflow.
        """
        if action not in {'evaluate', 'render'}:
            raise ValueError('action must be evaluate or render')
        path = safe_path(jobs.workspace, name, exists=True)
        if path.suffix.lower() != '.nb' or path.stat().st_size > MAX_BYTES:
            raise ValueError('expected an uploaded .nb file <=16 MiB')
        # A WL string literal (Unicode retained; no shell interpolation).
        wl_name = json.dumps(name, ensure_ascii=False)
        operation = ('NotebookEvaluate[nb, InsertResults -> True]; '
                     'NotebookSave[nb, FileNameJoin[{Directory[], "evaluated.nb"}]]'
                     if action == 'evaluate' else
                     'Export["notebook.png", Rasterize[nb, "Image"], "PNG"]')
        code = ('UsingFrontEnd[Module[{nb, result}, nb = NotebookOpen[' + wl_name + ', Visible -> False]; '
                'If[Head[nb] =!= NotebookObject, Return[$Failed]]; '
                'result = Check[' + operation + ', $Failed]; NotebookClose[nb]; result]]')
        return jobs.submit(code, [name], timeout_seconds)

    @mcp.tool()
    def wolfram_jobs() -> dict:
        """List native jobs from this server lifetime (official Evaluator sessions are separate)."""
        with jobs.lock:
            return {'jobs': [dict(r) for r in jobs.records.values()]}

    @mcp.tool()
    def wolfram_status(job_id: str) -> dict:
        """Read actual running/completed/failed/timeout/cancelled native job status."""
        return jobs.status(job_id)

    @mcp.tool()
    async def wolfram_wait(job_id: str, wait_seconds: StrictInt = 30) -> dict:
        """Wait at most 60 seconds; running does not mean completed."""
        return await asyncio.to_thread(jobs.wait, job_id, wait_seconds)

    @mcp.tool()
    def wolfram_cancel(job_id: str) -> dict:
        """Kill the native job's complete process group; does not affect official sessions."""
        return jobs.cancel(job_id)

    @mcp.tool()
    def wolfram_result(job_id: str) -> dict:
        """Get native job status, bounded exact-expression preview and artifact list.

        Messages/prints are logs; an unevaluated expression is not proof of a solved problem.
        Download result.inputform.txt/result.wxf for complete precision and structure.
        """
        state = jobs.status(job_id)
        result = None
        if state['state'] != 'running':
            path = safe_path(jobs.directory(job_id), 'result.json')
            if path.is_file() and path.stat().st_size <= 65536:
                result = json.loads(path.read_text())
        return {'status': state, 'result': result, 'artifacts': jobs.files(job_id)}

    @mcp.tool()
    def wolfram_upload(name: str, data: str, encoding: str = 'utf-8') -> dict:
        """Upload a relative task input once (<=16 MiB); use base64 for binary data/.nb/WXF."""
        return jobs.put(name, data, encoding)

    @mcp.tool()
    def wolfram_files(job_id: str | None = None, offset: StrictInt = 0, limit: StrictInt = 100) -> dict:
        """List bounded task-workspace or native-job artifacts. No symlinks followed."""
        return jobs.files(job_id, offset, limit)

    @mcp.tool()
    def wolfram_read(name: str, job_id: str | None = None, offset: StrictInt = 0,
                     limit: StrictInt = 65536, encoding: str = 'utf-8') -> dict:
        """Download paginated artifacts/logs; byte offsets and per-chunk SHA-256.

        Use base64 for lossless binary/UTF-8 chunk boundaries; sidecars have no shared workspace.
        """
        return jobs.read(name, job_id, offset, limit, encoding)

    server = mcp._mcp_server
    local_list = server.request_handlers[types.ListToolsRequest]
    local_call = server.request_handlers[types.CallToolRequest]
    local_names = {t.name for t in mcp._tool_manager.list_tools()}
    remote_names = set()

    async def list_tools(request):
        # Collapse upstream pagination into the single gateway tools/list response.
        remote = [] if official.failed else await official.all_tools()
        if local_names & {t.name for t in remote}:
            raise RuntimeError('official/local tool name collision')
        remote_names.clear(); remote_names.update(t.name for t in remote)
        local = await local_list(types.ListToolsRequest(method='tools/list'))
        return types.ServerResult(types.ListToolsResult(tools=[*local.root.tools, *remote]))

    async def call_tool(request):
        name = request.params.name
        if name in local_names:
            # Prime the SDK's local schema cache without querying a failed upstream.
            # Recovery and file retrieval must remain usable after a timeout.
            await local_list(types.ListToolsRequest(method='tools/list'))
            return await local_call(request)
        if name not in remote_names:
            remote_names.update(t.name for t in await official.all_tools())
        if name not in remote_names:
            raise ValueError('unknown Wolfram tool')
        meta = request.params.meta.model_dump(exclude_none=True) if request.params.meta else None
        # Preserve errors, structuredContent, images, embedded resources and metadata.
        result = await official.invoke('call_tool', name, request.params.arguments or {}, meta=meta)
        return types.ServerResult(result)

    server.request_handlers[types.ListToolsRequest] = list_tools
    server.request_handlers[types.CallToolRequest] = call_tool

    @server.list_prompts()
    async def list_prompts():
        if not official.info.capabilities.prompts:
            return []
        items, cursor = [], None
        seen = set()
        while True:
            result = await official.invoke('list_prompts', cursor=cursor)
            items.extend(result.prompts); cursor = result.nextCursor
            if not cursor:
                return items
            if cursor in seen:
                raise RuntimeError('repeated prompt cursor')
            seen.add(cursor)

    @server.get_prompt()
    async def get_prompt(name, arguments):
        return await official.invoke('get_prompt', name, arguments)

    # The built-in WolframLanguage preset does not need MCP Apps. Resource and
    # prompt primitives are forwarded without enabling an unsupported UI client.
    async def resources(request):
        if not official.info.capabilities.resources:
            result = types.ListResourcesResult(resources=[])
        else:
            result = await official.invoke('list_resources', cursor=request.params.cursor if request.params else None)
        return types.ServerResult(result)

    async def templates(request):
        if not official.info.capabilities.resources:
            result = types.ListResourceTemplatesResult(resourceTemplates=[])
        else:
            result = await official.invoke('list_resource_templates', cursor=request.params.cursor if request.params else None)
        return types.ServerResult(result)

    async def read_resource(request):
        return types.ServerResult(await official.invoke('read_resource', request.params.uri))

    server.request_handlers[types.ListResourcesRequest] = resources
    server.request_handlers[types.ListResourceTemplatesRequest] = templates
    server.request_handlers[types.ReadResourceRequest] = read_resource
    return mcp


async def run(transport):
    env = prepare_environment()
    jobs = official = None
    try:
        workspace = Path(env.get('WOLFRAM_MCP_WORKSPACE', '/workspace')).resolve()
        timeout = int(env.get('WOLFRAM_MCP_TIMEOUT', '180'))
        if not 1 <= timeout <= 3600:
            raise ValueError('WOLFRAM_MCP_TIMEOUT must be 1..3600')
        jobs = Jobs(workspace, env)
        official = Official(official_command(), env, workspace, timeout)
        current = asyncio.current_task()
        if transport == 'stdio':
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, current.cancel)
        await official.start()
        names = {tool.name for tool in await official.all_tools()}
        if 'WolframLanguageEvaluator' not in names:
            raise RuntimeError('official server does not expose WolframLanguageEvaluator; select WolframLanguage preset')
        mcp = build_server(jobs, official)
        if transport == 'stdio':
            await mcp.run_stdio_async()
        else:
            config = uvicorn.Config(mcp.streamable_http_app(), host=mcp.settings.host, port=mcp.settings.port,
                                    log_level='warning', access_log=False)
            await uvicorn.Server(config).serve()
    finally:
        if jobs:
            jobs.close()
        if official:
            await official.close()
        if env.get('WOLFRAM_STATE_ROOT'):
            shutil.rmtree(env['HOME'], ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', choices=['stdio', 'streamable-http'], default='stdio')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        asyncio.run(run(args.transport))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as exc:
        # License values are never included in our diagnostics.
        print(f'Wolfram MCP startup/runtime failure ({type(exc).__name__}); check installation and license', file=sys.stderr)
        raise SystemExit(70)


if __name__ == '__main__':
    main()
