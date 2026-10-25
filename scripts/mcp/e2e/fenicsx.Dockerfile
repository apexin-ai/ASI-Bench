# FEniCSx adapter image. The digest is the linux/amd64 v0.11.0 image listed by
# Docker Hub on 2026-10-22; update it together with the DOLFINx revision.
FROM --platform=linux/amd64 dolfinx/dolfinx:v0.11.0@sha256:58b27e84a2f26b98ce2d9ccc537b0ee6a59e2fcfdf386626d5ed9ddf43425ece

RUN python3 -m pip install --no-cache-dir "mcp==1.30.0" "pydantic==2.13.5"
COPY fenicsx_server.py /opt/fenicsx-mcp/fenicsx_server.py
ENV FENICSX_MCP_TRANSPORT=stdio FENICSX_MCP_WORKSPACE=/workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/fenicsx-mcp/fenicsx_server.py"]
