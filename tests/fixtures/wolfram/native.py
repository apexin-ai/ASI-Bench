#!/usr/bin/env python3
"""Lifecycle fixture; no mathematics and not a fake scientific success."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(os.environ['ASIBENCH_WOLFRAM_OUTPUT'])
code = Path(os.environ['ASIBENCH_WOLFRAM_INPUT']).read_text()
if code == 'hang':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (root/'child.pid').write_text(str(child.pid))
    time.sleep(60)
if code == 'fail':
    print('fixture failure', file=sys.stderr)
    sys.exit(2)
(root/'result.json').write_text(json.dumps({'state': 'completed', 'fixture': True}))
(root/'binary.wxf').write_bytes(bytes(range(256)))
