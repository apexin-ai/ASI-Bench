#!/bin/sh
set -eu
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HOME=/home/evaluator
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=/opt/asi-evaluator:/opt/bridge
cd /output
exec /usr/local/bin/python /opt/bridge/score_entry.py \
    --manifest /input/evaluation_manifest.json --output-dir /output
