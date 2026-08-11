"""Parametric ground-truth generator template.

Standard interface — all task generate_gt.py files must implement:
  - INPUT_SPEC:     list of input file specifications
  - OUTPUT_SPEC:    list of output file specifications
  - DEFAULT_PARAMS: dict of default parameter values
  - generate():     function to generate one task instance

Framework calls: python generate_gt.py --output-dir <dir> --params '<json>'

Output directory layout produced by this script:
  <output_dir>/
  ├── data/                   input files for the agent
  ├── reference/              ground-truth files for evaluation
  ├── prompt_b1.md            rendered B1 prompt (with parameter values)
  ├── prompt_b2.md            rendered B2 prompt
  ├── prompt_b3.md            rendered B3 prompt
  ├── prompt_b4.md            rendered B4 prompt (B3 + redundant info)
  └── instance_meta.json      generation metadata
"""

import argparse
import json
import time
from pathlib import Path

# ── I/O specification (names/types match task_meta.yaml; shape/dtype match task_eval.yaml) ──

# Input files the agent will receive (written to data/)
INPUT_SPEC = [
    # Example:
    # {"name": "input_data.npy",
    #  "shape": lambda p: [p["size"]],
    #  "dtype": "float32",
    #  "description": "Input data"},
]

# Output files the agent must produce (evaluated against reference/)
OUTPUT_SPEC = [
    # Example:
    # {"name": "result.npy",
    #  "shape": lambda p: [p["size"]],
    #  "dtype": "float32",
    #  "description": "Computation result"},
]

# Default parameter values (must match task_eval.yaml generation.parameters)
DEFAULT_PARAMS = {
    # "size": 100,
    "seed": 0,  # GT reproducibility seed (injected by framework as seed+idx)
}


def generate(output_dir: Path, params: dict) -> dict:
    """Generate one task instance.

    Args:
        output_dir: target directory; framework creates it before calling this.
        params:     parameter dict; may be partial — use DEFAULT_PARAMS as fallback.

    Returns:
        instance_meta dict written to instance_meta.json.
    """
    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = output_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    # TODO: 1. Generate and save input data to data_dir
    # TODO: 2. Run reference implementation and save ground truth to ref_dir

    # 3. Render parametrized prompts
    task_dir = Path(__file__).parent
    for level in ["b1", "b2", "b3", "b4"]:
        template_path = task_dir / f"prompt_{level}.md"
        if template_path.exists():
            prompt_text = template_path.read_text(encoding="utf-8")
            # TODO: Substitute parameter values if needed
            # prompt_text = prompt_text.replace("{{grid_size}}", str(p["grid_size"]))
        else:
            prompt_text = f"# Task\n\nTODO: Write prompt for level {level}."
        (output_dir / f"prompt_{level}.md").write_text(prompt_text, encoding="utf-8")

    # 4. Write instance metadata
    meta = {
        "params_used": p,
        "input_files": [s["name"] for s in INPUT_SPEC],
        "reference_files": [s["name"] for s in OUTPUT_SPEC],
        "generation_time_seconds": round(time.time() - t0, 2),
    }
    (output_dir / "instance_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    result = generate(args.output_dir, json.loads(args.params))
    print(json.dumps(result, indent=2))
