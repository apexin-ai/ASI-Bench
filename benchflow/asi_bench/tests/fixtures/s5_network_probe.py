import json
import os
from pathlib import Path

print(
    json.dumps(
        {
            "uid": os.getuid(),
            "cwd": os.getcwd(),
            "files": sorted(
                str(p.relative_to("/workspace"))
                for p in Path("/workspace").rglob("*")
                if p.is_file()
            ),
            "interfaces": sorted(p.name for p in Path("/sys/class/net").iterdir()),
            "routes": Path("/proc/net/route").read_text(),
            "capabilities": [
                s
                for s in Path("/proc/self/status").read_text().splitlines()
                if s.startswith("Cap")
            ],
            "hidden_readable": {
                p: os.access(p, os.R_OK)
                for p in [
                    "/verifier",
                    "/reference",
                    "/workspace/reference",
                    "/task_manifest.json",
                ]
            },
        }
    )
)
