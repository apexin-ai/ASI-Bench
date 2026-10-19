"""Tiny deterministic MCP server for sandbox protocol integration tests."""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def respond(request):
    method = request.get("method")
    if "id" not in request:
        return None
    result = {}
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "sandbox-fixture", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "add", "description": "Add integers", "inputSchema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}]}
    elif method == "tools/call":
        params = request["params"]["arguments"]
        result = {"content": [{"type": "text", "text": str(params["a"] + params["b"])}]}
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


if __name__ == "__main__":
    if "--http" in sys.argv:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
            def do_POST(self):
                reply = respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200 if reply else 202)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if reply:
                    self.wfile.write(json.dumps(reply).encode())
            def log_message(self, *args):
                pass
        HTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
    else:
        assert "ASIBENCH_TEST_AGENT_SECRET" not in os.environ
        if "--hang" in sys.argv:
            import time
            time.sleep(60)
        if "--bad" in sys.argv:
            print("not json", flush=True)
        for line in sys.stdin:
            reply = respond(json.loads(line))
            if reply:
                print(json.dumps(reply), flush=True)
