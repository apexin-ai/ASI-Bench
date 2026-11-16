# Native Claude session recovery

ASI-Bench can opt in to persistent Claude Code checkpoints and explicitly resume
selected task instances after an external API fault. This is a framework feature;
experiment repositories only configure and record its use. It does not migrate
or restart existing campaigns automatically.

## Capture checkpoints

Start a new produce-only run with original materialized instance bundles:

```bash
asibench run --agent claude_code_cli --sandbox os \
  --agent-config '{"model":"claude-opus-5-5","effort":"max","api_protocol":"anthropic","api_key_env":"MODEL_API_KEY","api_base_env":"MODEL_API_BASE","network_policy":"model-only"}' \
  --instances-dir /private/bundles --tasks my.task --prompt-levels b1,b2,b3,b4 \
  --checkpoint-sessions --output-dir /private/run-1 --fail-on-agent-error
```

Use your task's actual ID and model endpoint. API credentials stay in the host
relay; only environment variable names and endpoint provenance are recorded.
Checkpoint v1 supports Claude Code, native Anthropic, OS sandbox, model-only
networking, a local Unix-socket Docker daemon, and one attempt. MCP and task
services are unsupported. Other runs retain their existing behavior. The CLI
version is inherited from the task's immutable agent image.

Private `checkpoints/` holds the native Claude HOME/session, full workspace,
per-segment logs/request accounting, and, for eligible external failures, a
portable Docker image archive. It can be large. Keep the whole directory private:
it contains full conversation and task input, unlike sanitized public results.
Standard submission bundles exclude this directory. Store/transfer checkpoints
only within the trusted execution environment, with their original permissions.

## Preview and resume selected levels

```bash
asibench resume --from /private/run-1 \
  --tasks my.task --levels b2,b4 --instances-dir /private/bundles \
  --output-dir /private/resume-1 --dry-run

asibench resume --from /private/run-1 \
  --tasks my.task --levels b2,b4 --instances-dir /private/bundles \
  --output-dir /private/resume-1 --parallel 10
```

`--from` accepts a run directory, checkpoint directory, or `checkpoint.json`.
`--instance-ids` selects particular instances. With no selectors, all eligible
checkpoints are selected and rejected checkpoints are listed. Explicit selections
that include ineligible or missing cells fail before model dispatch. Provide
`--tasks-dir` when the public task definitions are elsewhere. Original bundles
are inferred if still available, or supplied using `--instances-dir`; they remain
read-only. `--dry-run` verifies files and inputs without credentials, Docker, model
requests, claims, or creating an output directory.

Restoration checks file/image hashes, implementation, task/input metadata,
model, effort, isolation, resources and original budget. Inputs and profiles must
match. `--api-key-env` and `--api-base-env` can select replacement credentials or
a recovered native endpoint; these options take environment variable names.
The output must be new. Each source can have only one continuation, protected by
a lock and persistent claim. If recovery fails again, resume the child checkpoint.
A crash after dispatch leaves a claim requiring manual audit; do not delete claims
and replay executions whose token consumption is unknown.

`asibench resume` produces results only. Use `asibench score` separately as for
any other run. Historical `run --resume` still means skipping saved results; it
does not restore Claude sessions.

## Eligibility, budgets and limits

A recoverable checkpoint requires a terminal external API error corroborated by
its last Messages response/transport failure, or a recognized infrastructure
failure, plus an intact session, workspace and image. CountTokens failures alone
do not qualify. OOM, timeout, exhausted original budget, safeguards/400 errors,
and unknown process kills are ineligible. Missing/incomplete checkpoints never
fall back to a fresh attempt. Abrupt host loss may leave an incomplete checkpoint.
Old logs without native sessions and filesystem snapshots cannot be imported.

Recovery preserves the session UUID and conversation, workspace, persistent HOME
and container root filesystem. Live processes, `/tmp`, `/dev/shm` and other tmpfs
state are not restored. The continuation adds a neutral instruction to continue
the interrupted task. Remaining time is the original limit minus all prior
execution segments, enforced by a container-side deadline. API waiting consumes
the same budget; it is not refunded. Offline time between segments and snapshot
storage work are not agent execution time.

`session-accounting.json` and checkpoint ledgers accumulate observed input,
output and cache tokens across all ancestors, including context replay and
internal retries visible to the relay. Resuming saves previous work, but providers
may bill context again. Unreported usage is explicitly incomplete. Actual billed
cost remains unknown; it is never inferred to be zero. Optional
`run --token-prices prices.json` accepts four USD-per-million rates:

```json
{"input_tokens": 5, "output_tokens": 25, "cache_creation_input_tokens": 6.25, "cache_read_input_tokens": 0.5}
```

These example rates are not a provider quote. They produce estimates for observed
usage only, which may be lower than actual billing. The price schedule is frozen
across the continuation chain.
