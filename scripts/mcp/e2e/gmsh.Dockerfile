FROM --platform=linux/amd64 python:3.12-slim@sha256:e8a38829209aea82774882e5157e1f7c3e8c8393ff53a697b3580896e57f345c

# The gmsh Python wheel imports the native library and needs GLU even for
# headless meshing. No GUI is enabled by the ASI-Bench adapter.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglu1-mesa \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "gmsh==4.15.2" "mcp==1.30.0" "pydantic==2.13.5" "numpy==2.5.3"
COPY gmsh_server.py /opt/gmsh-mcp/gmsh_server.py
ENV GMSH_MCP_TRANSPORT=stdio GMSH_MCP_WORKSPACE=/workspace
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/gmsh-mcp/gmsh_server.py"]
