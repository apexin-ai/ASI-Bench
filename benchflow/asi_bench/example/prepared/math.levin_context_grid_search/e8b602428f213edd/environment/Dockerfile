FROM python:3.11-slim
COPY runtime-constraints.txt /opt/runtime-constraints.txt
RUN ["python", "-m", "pip", "install", "--no-cache-dir", "-c", "/opt/runtime-constraints.txt", "numpy", "PyYAML", "packaging"]
WORKDIR /workspace
COPY inputs/ /workspace/
CMD ["sleep", "infinity"]
