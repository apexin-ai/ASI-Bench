"""Value helpers shared by several server smokes (stdlib only)."""
from __future__ import annotations

import contextlib
import math
import os
import struct
import sys

Atoms = list[tuple[str, tuple[float, float, float]]]


@contextlib.contextmanager
def quiet_fds():
    """Silence fd 1/2 while a reference library runs (PySCF, geomeTRIC and psi4 write to
    the process streams directly, bypassing ``sys.stdout``)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def sorted_distances(atoms: Atoms) -> list[float]:
    """All interatomic distances, sorted: invariant to rotation, translation and permutation."""
    return sorted(math.dist(atoms[i][1], atoms[j][1]) for i in range(len(atoms)) for j in range(i + 1, len(atoms)))


def max_abs_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG IHDR chunk."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def image_blocks(result: dict) -> list[dict]:
    """Image content blocks of a tools/call result."""
    return [b for b in result.get("content", []) if isinstance(b, dict) and b.get("type") == "image"]
