"""Shared test fixtures."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    PromptLevel,
    RunStatus,
    ScoreDetail,
    TaskInstance,
)


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_pred_dir(tmp_dir):
    """Create a prediction directory with sample .npy files."""
    pred_dir = tmp_dir / "pred"
    pred_dir.mkdir()

    # Vorticity frames: 5 frames of 10x10
    vort = np.random.randn(5, 10, 10).astype(np.float32)
    np.save(pred_dir / "vorticity_frames.npy", vort)

    # Kinetic energy: 201 time points
    ke = np.abs(np.random.randn(201).astype(np.float32)) + 1.0
    np.save(pred_dir / "kinetic_energy.npy", ke)

    return pred_dir


@pytest.fixture
def sample_ref_dir(tmp_dir):
    """Create a reference directory with matching .npy files."""
    ref_dir = tmp_dir / "ref"
    ref_dir.mkdir()

    vort = np.random.randn(5, 10, 10).astype(np.float32)
    np.save(ref_dir / "vorticity_frames_ref.npy", vort)

    ke = np.abs(np.random.randn(201).astype(np.float32)) + 1.0
    np.save(ref_dir / "kinetic_energy_ref.npy", ke)

    return ref_dir


@pytest.fixture
def matching_pred_ref(tmp_dir):
    """Create pred and ref dirs with identical data (should score perfectly)."""
    pred_dir = tmp_dir / "pred"
    ref_dir = tmp_dir / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()

    vort = np.random.randn(5, 10, 10).astype(np.float32)
    np.save(pred_dir / "vorticity_frames.npy", vort)
    np.save(ref_dir / "vorticity_frames_ref.npy", vort.copy())

    ke = np.abs(np.random.randn(201).astype(np.float32)) + 1.0
    np.save(pred_dir / "kinetic_energy.npy", ke)
    np.save(ref_dir / "kinetic_energy_ref.npy", ke.copy())

    return pred_dir, ref_dir


@pytest.fixture
def sample_code_file(tmp_dir):
    """Create a sample simulation.py file."""
    pred_dir = tmp_dir / "pred"
    pred_dir.mkdir(exist_ok=True)

    code = '''import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

n = 255
vorticity = ti.field(dtype=ti.f32, shape=(n, n))

@ti.kernel
def compute_vorticity():
    for i, j in vorticity:
        vorticity[i, j] = 0.0

@ti.kernel
def advect():
    for i, j in vorticity:
        vorticity[i, j] = vorticity[i, j] * 0.99

compute_vorticity()
advect()
'''
    (pred_dir / "simulation.py").write_text(code)
    return pred_dir


@pytest.fixture
def sample_task_dir(tmp_dir):
    """Create a minimal task directory structure."""
    task_dir = tmp_dir / "tasks" / "physics" / "test_task"
    task_dir.mkdir(parents=True)

    metadata = {
        "id": "physics.test_task",
        "name": "Test Task",
        "version": "1.0",
        "status": "test",
        "domain": "physics",
        "subdomain": "test",
        "tags": ["test"],
        "difficulty": {
            "estimated_lines": "10-50",
            "estimated_time_minutes": "1-5",
            "requires_gpu": False,
            "requires_network": False,
        },
        "prompts": {
            "b1": "prompt_b1.md",
            "b2": "prompt_b2.md",
            "b3": "prompt_b3.md",
            "b4": "prompt_b4.md",
        },
        "input": {"files": []},
        "output": {
            "files": [
                {"name": "output.npy", "type": "data", "dtype": "float32"},
            ]
        },
        "evaluation": {
            "gates": [],
            "scoring": [
                {
                    "scorer": "numerical",
                    "weight": 100,
                    "config": {
                        "metric": "relative_l2",
                        "pred_file": "output.npy",
                        "ref_file": "output_ref.npy",
                        "threshold": 0.1,
                    },
                }
            ],
        },
        "generation": {
            "script": "generate_gt.py",
            "parameters": {
                "size": {"type": "int", "range": [10, 100], "default": 50},
            },
            "timeout_seconds": 60,
        },
    }

    import yaml
    (task_dir / "task.yaml").write_text(yaml.dump(metadata))
    (task_dir / "prompt_b1.md").write_text("# Test Task\nFull description for B1.")
    (task_dir / "prompt_b2.md").write_text("# Test Task\nMethod hint for B2.")
    (task_dir / "prompt_b3.md").write_text("# Test Task\nGoal only for B3.")
    (task_dir / "prompt_b4.md").write_text("# Test Task\nGoal plus irrelevant context for B4.")

    # Create generate_gt.py
    (task_dir / "generate_gt.py").write_text('''
import json
import time
from pathlib import Path
import numpy as np

INPUT_SPEC = []
OUTPUT_SPEC = [
    {"name": "output.npy", "shape": lambda p: [p.get("size", 50)], "dtype": "float32"},
]
DEFAULT_PARAMS = {"size": 50, "seed": 0}

def generate(output_dir, params):
    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(output_dir) / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(p.get("seed", 0))
    arr = rng.random(p["size"]).astype(np.float32)
    np.save(ref_dir / "output_ref.npy", arr)
    for level in ["b1", "b2", "b3", "b4"]:
        Path(output_dir, f"prompt_{level}.md").write_text(f"# Test\\nSize={p['size']}")
    meta = {
        "params_used": p,
        "input_files": [],
        "reference_files": ["output_ref.npy"],
        "generation_time_seconds": round(time.time() - t0, 2),
    }
    Path(output_dir, "instance_meta.json").write_text(json.dumps(meta, indent=2))
    return meta

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    result = generate(args.output_dir, json.loads(args.params))
    print(json.dumps(result, indent=2))
''')

    return task_dir


@pytest.fixture
def sample_task_instance(tmp_dir, sample_task_dir):
    """Create a sample TaskInstance."""
    workspace = tmp_dir / "workspace"
    workspace.mkdir()
    ref_dir = tmp_dir / "reference"
    ref_dir.mkdir()

    # Create reference data
    arr = np.random.randn(50).astype(np.float32)
    np.save(ref_dir / "output_ref.npy", arr)

    # Create prompt
    (workspace / "prompt.md").write_text("# Test Task\nMethod hint for B2.")
    (workspace / "data").mkdir(exist_ok=True)

    import yaml
    metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
    metadata["_task_dir"] = sample_task_dir
    metadata["_task_yaml"] = sample_task_dir / "task.yaml"

    return TaskInstance(
        task_id="physics.test_task",
        instance_id="physics.test_task__size50_seed0",
        task_dir=sample_task_dir,
        workspace_dir=workspace,
        reference_dir=ref_dir,
        prompt_level=PromptLevel.B2,
        parameters={"size": 50, "seed": 0},
        metadata=metadata,
    )
