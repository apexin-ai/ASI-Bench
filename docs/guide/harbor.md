# Harbor Integration

Harbor owns the task environment, agent execution, concurrency, and artifact
collection. ASI-Bench supplies the task bundle and scoring implementation. The
integration is for already materialized public `seed31415` instances; scores
are local and non-official.

The normal flow is:

```text
pull instance → export Harbor task → run agent → verify artifacts → inspect score
```

## Prerequisites

Install ASI-Bench from the checkout and install Harbor according to the Harbor
deployment you use. The Docker provider is required for the task images in this
guide. Run the commands from the ASI-Bench checkout:

```bash
cd /path/to/ASI-Bench
python -m venv .venv
. .venv/bin/activate
pip install -e '.[full]'
```

Build a wheel from the checkout. The command below discovers the wheel name,
so it does not assume a particular in-development version:

```bash
export WHEEL_DIR=/tmp/asibench-wheel
uv build --wheel --out-dir "$WHEEL_DIR"
export ASIBENCH_WHEEL="$(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' -print -quit)"
test -n "$ASIBENCH_WHEEL"
```

## Prepare an instance

Pull one public task instance. Keep the task ID in an environment variable so
the same values can be reused by the following commands:

```bash
export TASK_ID=robotics.particle_filter
export INSTANCE_ID="${TASK_ID}__seed31415"
export PROMPT_LEVEL=b1
export TASKS_DIR="$PWD/tasks"
export INSTANCES_DIR=/tmp/asi-seed31415

asibench task pull \
  --repo seed31415 \
  --tasks "$TASK_ID" \
  --output-dir "$INSTANCES_DIR"
```

This downloads the public task definition and instance data. It does not run
an agent or create a Harbor task yet.

## Export a Harbor task

Export one prompt level into a separate dataset directory:

```bash
export HARBOR_DATASET=/tmp/asi-harbor-b1

asibench harbor export \
  --task-dir "$TASKS_DIR/${TASK_ID//./\/}" \
  --instance-dir "$INSTANCES_DIR/$INSTANCE_ID" \
  --level "$PROMPT_LEVEL" \
  --wheel "$ASIBENCH_WHEEL" \
  --output-dir "$HARBOR_DATASET"
```

The output contains a Harbor task directory with `task.toml`, the prompt,
agent environment files, and verifier files. Public prompts and inputs are
available to the agent. The public reference and scorer remain in the verifier
side of the task package.

## Run an agent

Use `nop` first when checking only the task wiring:

```bash
harbor run \
  -p "$HARBOR_DATASET/tasks/robotics-particle-filter-b1" \
  -a nop
```

Run a real agent by selecting the Harbor agent, model, and effort explicitly:

```bash
export HARBOR_JOBS=/tmp/harbor-jobs

harbor run \
  -p "$HARBOR_DATASET/tasks/robotics-particle-filter-b1" \
  -a codex \
  -m gpt-6.1-sol \
  --effort low \
  --jobs-dir "$HARBOR_JOBS" \
  --n-concurrent 1 \
  --job-name codex-particle-filter-low
```

The API endpoint and key belong in the environment where Harbor runs. Keep
them in an ignored local environment file or a secret manager; do not put a
secret in `task.toml`, the task directory, or a command-line argument.

Harbor starts an agent container, mounts the task workspace, and collects the
files declared by the task as artifacts. The saved job directory contains the
trial log, agent trajectory, artifacts, and verifier result.

## Export and run a dataset

Audit a materialized seed31415 tree before exporting it:

```bash
asibench harbor audit \
  --tasks-dir "$TASKS_DIR" \
  --instances-dir "$INSTANCES_DIR" \
  --out /tmp/asi-harbor-audit.json
```

Export all compatible task/instance/level cells:

```bash
asibench harbor export-batch \
  --tasks-dir "$TASKS_DIR" \
  --instances-dir "$INSTANCES_DIR" \
  --wheel "$ASIBENCH_WHEEL" \
  --output-dir /tmp/asi-harbor-seed31415 \
  --skip-unsupported
```

`--skip-unsupported` keeps compatible cells and records omitted cells in the
export manifest. Without it, any blocked cell stops the batch export. An audit
of the current seed31415 set reports export compatibility; it does not claim
that an agent can solve every exported task.

Run the exported dataset with the same `harbor run` command, pointing `-p` at
the dataset root. Use `--n-concurrent` to control parallel trials and retain
the jobs directory for later inspection.

## How scoring is connected

The task's verifier test invokes the ASI-Bench entry point inside a separate
verifier container:

```bash
asibench harbor-verify \
  --task-dir /tests/task_bundle \
  --instance-dir /tests/instance/<task>__seed31415 \
  --outputs-dir /workspace \
  --prompt-level b1 \
  --out /logs/verifier
```

The verifier restores the agent artifacts, loads the task evaluation contract,
and calls the existing ASI-Bench scorer. It writes:

```text
/logs/verifier/reward.json       # Harbor reward, final_score / 100
/logs/verifier/score_detail.json # gates, score details, and diagnostics
```

Harbor reads `reward.json` to finish the trial. A missing submission can be a
valid zero score; an evaluator failure is recorded as invalid and returns a
non-zero verifier status.

## Special task environments

TSP, Max3SAT, and CMOS/ngspice scoring use a constrained Compose executor
sidecar on an internal network. The verifier sends only the candidate output
and required simulation inputs to the sidecar. Reference data and Judge
credentials stay in the verifier, and the integration does not mount the host
Docker socket.

The Harbor deployment must support the task's `tests/docker-compose.yaml` and
allow the verifier to reach any configured Judge endpoint.

## Judge configuration

For tasks using an LLM or VLM Judge, provide runtime settings through the
environment or the corresponding CLI options:

```text
ASIBENCH_JUDGE_API_BASE       # optional endpoint
ASIBENCH_JUDGE_API_KEY_ENV    # name of the variable holding the key
ASIBENCH_JUDGE_API_PROTOCOL   # native or openai
```

`ASIBENCH_JUDGE_API_KEY_ENV` is a variable name, never the key itself. The
verifier container needs network access to the Judge endpoint.

## Aggregate results

After Harbor has produced trial directories, create a local summary:

```bash
asibench harbor aggregate \
  --details-dir "$HARBOR_JOBS" \
  --out /tmp/asi-harbor-scores.json
```

The aggregator reads `score_detail.json` files, includes completed zero scores,
and excludes evaluator-invalid trials. The summary is useful for local
inspection but is not an official benchmark result.

## Troubleshooting checklist

- If Docker cannot build the task image, check the Harbor Docker provider and
  the network available during image builds.
- If an agent finishes without a score, inspect the trial log and the verifier
  container logs under the Harbor jobs directory.
- If a task is absent after batch export, inspect its
  `compatibility_report.json`; `--skip-unsupported` intentionally omits blocked
  cells.
- If a Judge task is invalid, verify the named key variable, endpoint, protocol,
  and verifier network access. Never copy the key into a task file.

This integration has been smoke-tested with real Codex runs on UKF B1 and TSP
B1. Those checks validate the execution and scoring path; they are not a
full-agent evaluation of all seed31415 tasks.
