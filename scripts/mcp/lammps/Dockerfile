FROM --platform=linux/amd64 python:3.12-slim@sha256:e8a38829209aea82774882e5157e1f7c3e8c8393ff53a697b3580896e57f345c
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 util-linux \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/lammps-mcp/requirements.lock
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r /opt/lammps-mcp/requirements.lock
COPY runtime.py science.py server.py /opt/lammps-mcp/
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin LAMMPS_MCP_WORKSPACE=/workspace LAMMPS_BINARY=/opt/venv/bin/lmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HOME=/home/agent LD_LIBRARY_PATH=/opt/venv/lib
RUN /opt/venv/bin/lmp -h > /opt/lammps-mcp/engine-help.txt \
    && /opt/venv/bin/python -c "from lammps import lammps; l=lammps(cmdargs=['-screen','none','-log','none']); print(l.version()); l.close()" \
    && mkdir -p /home/agent/.codex /workspace && chown -R 1000:1000 /home/agent /workspace
USER 1000:1000
WORKDIR /workspace
# HTTP sidecars override command to --transport streamable-http. Healthcheck performs MCP initialize.
HEALTHCHECK --interval=5s --timeout=5s --start-period=10s --retries=3 CMD ["/opt/venv/bin/python", "/opt/lammps-mcp/healthcheck.py"]
COPY healthcheck.py /opt/lammps-mcp/
CMD ["/opt/venv/bin/python", "/opt/lammps-mcp/server.py"]
