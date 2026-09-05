# BenchFlow Integration

BenchFlow owns scheduling, agent execution, retries, and artifact retention.
ASI-Bench provides a stable command boundary for deterministic scoring of one
already-materialized attempt:

```bash
asibench benchflow-score \
  --manifest benchflow_manifest.json \
  --output benchflow_score.json
```

Only the public seed31415 dataset can be scored this way. The result is local
and non-official; the command rejects seed42 and never generates instances or
references.

## Install and pull an instance

Install the scientific scoring dependencies, then pull the public instance
bundle. This example uses one task and one prompt level so the run-result and
score-result relationship stays explicit:

```bash
git clone https://github.com/apexin-ai/ASI-Bench.git
cd ASI-Bench
python -m venv .venv
. .venv/bin/activate
pip install -e '.[full]'

export TASK_ID=materials.relaxation_mode_recovery
export INSTANCE_ID="${TASK_ID}__seed31415"
export PROMPT_LEVEL=b1
export INSTANCES_DIR="$PWD/hf_instances_seed31415"
export TASKS_DIR="$PWD/tasks"
export RUN_DIR="$PWD/benchflow-run"
export BENCHFLOW_RUN_ID=example-run-001

asibench task pull \
  --repo seed31415 \
  --tasks "$TASK_ID" \
  --output-dir "$INSTANCES_DIR"
```

## Run the agent

BenchFlow should invoke `asibench run` with `--fail-on-agent-error`. Replace
the agent and model with the adapter configured in your deployment:

```bash
asibench run \
  --instances-dir "$INSTANCES_DIR" \
  --tasks "$TASK_ID" \
  --prompt-levels "$PROMPT_LEVEL" \
  --tasks-dir "$TASKS_DIR" \
  --agent codex_cli \
  --agent-config '{"model":"gpt-5.6-sol"}' \
  --sandbox linux_ns \
  --output-dir "$RUN_DIR" \
  --fail-on-agent-error
```

Retain the output directory even when the command exits non-zero. Failed and
timed-out attempts still contain evidence required by the scoring stage. The
process exit status signals execution failure; it is not the score.

## Create the schema-v2 manifest

Create one manifest for each saved attempt. Read the prediction directory from
the persisted run result instead of reconstructing or guessing its path:

```bash
export RUN_RESULT="$RUN_DIR/$TASK_ID/${INSTANCE_ID}__${PROMPT_LEVEL}.json"

python - <<'PY'
import json
import os
from pathlib import Path

run_result = Path(os.environ["RUN_RESULT"]).resolve()
result = json.loads(run_result.read_text(encoding="utf-8"))
persisted = result["agent_output"]["persisted_outputs"]["dir"]
if not persisted:
    raise SystemExit("run result has no persisted prediction directory")

manifest = {
    "schema_version": 2,
    "benchmark": "ASI-Bench",
    "seed": 31415,
    "task_id": os.environ["TASK_ID"],
    "instance_id": os.environ["INSTANCE_ID"],
    "prompt_level": os.environ["PROMPT_LEVEL"],
    "run_result": str(run_result),
    "prediction_dir": str((run_result.parent / persisted).resolve()),
    "instance_dir": str(
        (Path(os.environ["INSTANCES_DIR"]) / os.environ["INSTANCE_ID"]).resolve()
    ),
    "tasks_dir": str(Path(os.environ["TASKS_DIR"]).resolve()),
    "benchflow_run_id": os.environ["BENCHFLOW_RUN_ID"],
    "attempt_id": str(result.get("attempt", 1)),
}
Path("benchflow_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY
```

The required fields are `task_id`, `instance_id`, `run_result`,
`prediction_dir`, `instance_dir`, and `tasks_dir`. Use absolute paths when the
execution and scoring stages have different working directories.

`instance_dir` must contain the public `reference/` directory, and `tasks_dir`
must point to this checkout's public task bundle. The adapter rejects
non-`__seed31415` instance IDs, mismatched run-result and prediction paths, and
attempts to generate references.

## Score the attempt

```bash
asibench benchflow-score \
  --manifest benchflow_manifest.json \
  --output benchflow_score.json
```

If the task uses an LLM or VLM Judge, configure its runtime transport with the
same safe options accepted by `asibench score` and `asibench run-score`:

- `--judge-api-base`
- `--judge-api-key-env`
- `--judge-api-protocol`

`--judge-api-key-env` accepts the name of an environment variable, never the
secret itself. Do not persist Judge secrets in the BenchFlow manifest.

## Interpret the result

BenchFlow should parse `benchflow_score.json`, not human-readable stdout.

| Field | Meaning |
|---|---|
| `status` | Overall status: `completed`, `attempt_failed`, or `evaluation_invalid` |
| `attempt_status` | Agent execution outcome, independent of scoring |
| `evaluation_status` | Whether the public scorer completed without an internal error |
| `retryable` | Whether BenchFlow should consider another agent attempt |
| `score` / `max_score` | Non-official local score for this attempt |
| `gate_results` / `score_details` | Stable per-check `ScoreDetail` records |

For example, `status: attempt_failed`, `attempt_status: execution_failed`, and
`evaluation_status: completed` means that the scorer ran successfully but the
agent attempt failed. This differs from a completed attempt that earned a low
score or failed a hard gate.

The result also records prediction and run-result SHA-256 values, scorer and
task revisions, framework version, and any harness, model, effort, or sandbox
provenance supplied in the manifest.

## Multiple tasks and retries

Repeat the manifest and scoring steps for every saved run-result JSON. With
`--retries`, attempt files after the first include `__attemptN` in the
filename. Each manifest must use the exact result file and its own
`persisted_outputs.dir`.

BenchFlow may aggregate the resulting score JSON records, but must preserve
each attempt's `attempt_status` and `evaluation_status` instead of replacing
them with the runner process exit code.
