# Official published image: AgentTools 2.2.0 + Engine 14.3. See upstream.lock.json.
FROM --platform=linux/amd64 ghcr.io/wolframresearch/mcpserver@sha256:22a0894758f5653737ef3355558fab0524e6801c36c95e7cbb0583afc3f3960f
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv util-linux ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/wolfram-mcp/requirements.lock
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r /opt/wolfram-mcp/requirements.lock
# Preserve only build-time paclets/cache, never the build user's license.
# Paclet discovery under a fresh HOME is configured in start.wls.
RUN mkdir -p /home/agent/.codex /workspace /opt/wolfram-state/.Wolfram /opt/wolfram-state/.WolframEngine \
    && cp -a /home/wolframengine/.Wolfram/Objects /opt/wolfram-state/.Wolfram/ \
    && cp -a /home/wolframengine/.WolframEngine/Paclets /opt/wolfram-state/.WolframEngine/ \
    && chmod -R a+rX /opt/wolfram-state \
    && chown -R 1000:1000 /home/agent /workspace
COPY bootstrap.py runtime.py server.py healthcheck.py start.wls runner.wls upstream.lock.json /opt/wolfram-mcp/
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/agent MCP_SERVER_NAME=WolframLanguage WOLFRAM_MCP_WORKSPACE=/workspace WOLFRAM_STATE_ROOT=/home/agent/.cache/wolfram
USER 1000:1000
WORKDIR /workspace
ENTRYPOINT []
HEALTHCHECK --interval=15s --timeout=10s --start-period=300s --retries=3 CMD ["/opt/venv/bin/python", "/opt/wolfram-mcp/healthcheck.py"]
CMD ["/opt/venv/bin/python", "/opt/wolfram-mcp/server.py"]
