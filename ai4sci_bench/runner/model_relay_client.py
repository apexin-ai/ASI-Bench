"""Container-local TCP/Unix bridge. The host relay enforces all authorization.

This stdlib-only file is mounted read-only; it runs before the CLI or MCP
preflight and provides the CLI's expected HTTP transport on loopback.
"""
import os
import select
import socket
import socketserver
import subprocess
import sys
import threading


class Bridge(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            self.relay()
        except OSError:
            pass  # Disconnected CLI requests are routine; no traceback noise.

    def relay(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect("/opt/asibench-network/socket/model.sock")
            peers = {self.request: upstream, upstream: self.request}
            while True:
                readable, _, _ = select.select(list(peers), [], [], 300)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(64 * 1024)
                    if not data:
                        return
                    peers[source].sendall(data)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True


if __name__ == "__main__":
    with Server(("127.0.0.1", 0), Bridge) as server:
        os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}"
        os.environ["ANTHROPIC_API_KEY"] = "asibench-model-relay"
        for key in list(os.environ):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
                os.environ.pop(key)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        sys.exit(subprocess.call(sys.argv[1:]))
