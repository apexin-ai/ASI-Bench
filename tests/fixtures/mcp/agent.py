"""Protocol client stand-in; never calls a model API or reads login tokens."""
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import urllib.request

if "--hang-agent" in sys.argv:
    import time
    time.sleep(60)

if "--mcp-config" in sys.argv:
    servers = json.loads(Path(sys.argv[sys.argv.index("--mcp-config") + 1]).read_text())["mcpServers"]
else:
    overrides = [sys.argv[i + 1] for i, arg in enumerate(sys.argv[:-1]) if arg == "-c"]
    servers = tomllib.loads("\n".join(overrides))["mcp_servers"]
for name, server in servers.items():
    request = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 19, "b": 23}}}
    if "command" in server:
        result = subprocess.run([server["command"], *server["args"]], input=json.dumps(request) + "\n", capture_output=True, text=True, check=True)
        reply = json.loads(result.stdout)
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(server["url"], data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}), timeout=10) as response:
            reply = json.load(response)
    assert reply["result"]["content"][0]["text"] == "42"
Path("/workspace/result.json").write_text(json.dumps({"answer": 42, "servers": sorted(servers)}))
print("MCP_PROTOCOL_CLIENT_PASS")
