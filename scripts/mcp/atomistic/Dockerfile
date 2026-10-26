FROM --platform=linux/amd64 python:3.12-slim@sha256:e8a38829209aea82774882e5157e1f7c3e8c8393ff53a697b3580896e57f345c
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libopengl0 libglib2.0-0 libxkbcommon0 libdbus-1-3 libgomp1 mesa-vulkan-drivers libvulkan1 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/atomistic-mcp/requirements.lock
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r /opt/atomistic-mcp/requirements.lock
COPY common.py ase_server.py ovito_server.py /opt/atomistic-mcp/
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin ATOMISTIC_MCP_WORKSPACE=/workspace QT_QPA_PLATFORM=ovitoheadless OVITO_GUI_MODE=0 OMP_NUM_THREADS=1
RUN /opt/venv/bin/python -c "import ase, ovito; from ovito.modifiers import PolyhedralTemplateMatchingModifier, CoordinationAnalysisModifier, DislocationAnalysisModifier; print(ase.__version__, ovito.version_string)" \
    && mkdir -p /home/agent/.codex /workspace && chown -R 1000:1000 /home/agent /workspace
ENV HOME=/home/agent
USER 1000:1000
WORKDIR /workspace
# Full agent image is built by ASI-Bench's normal CLI overlay; do not override its command.
CMD ["/opt/venv/bin/python", "/opt/atomistic-mcp/ase_server.py"]
