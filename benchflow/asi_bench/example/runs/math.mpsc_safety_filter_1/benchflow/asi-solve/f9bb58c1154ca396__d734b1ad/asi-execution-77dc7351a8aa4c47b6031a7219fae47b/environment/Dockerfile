FROM python:3.11-slim
COPY runtime-constraints.txt /opt/runtime-constraints.txt
RUN ["python", "-m", "pip", "install", "--no-cache-dir", "-c", "/opt/runtime-constraints.txt", "numpy", "PyYAML", "packaging", "numpy>=1.26", "scipy>=1.11", "cvxpy>=1.4", "clarabel>=0.7", "scs>=3.2"]
WORKDIR /workspace
COPY inputs/ /workspace/
CMD ["sleep", "infinity"]
