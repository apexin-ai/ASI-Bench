# Troubleshooting Results Below Expectations

A low score does not always mean that the model is weak. First distinguish a
completed but low-quality attempt from an execution or evaluation failure, then
check whether the selected harness matches the task.

## Start with the run status

- **Execution failed:** inspect the saved run result, stdout, and stderr for
  missing credentials, unavailable CLI binaries, dependency errors, timeouts,
  or missing output files. Fix the run before interpreting its score.
- **Evaluation failed:** verify that seed31415 instances include `reference/`,
  that `--tasks-dir` points to the matching framework checkout, and that any
  required LLM/VLM Judge endpoint is configured. An evaluator error is not a
  model score.
- **Execution and evaluation completed, but the score is low:** inspect the
  per-gate score details and generated artifacts. Continue with the checks
  below.

Produce-only runs are unscored. Their numeric zero values are placeholders and
must not be interpreted as a real `0.0` score. Use public seed31415 local
scoring or the seed42 submission workflow before comparing results.

## Check whether `direct_llm` is the right baseline

`direct_llm` makes one model API call, extracts code from that single response,
and executes it once. It is a **single-turn** baseline with **no agentic tools**:
the model cannot inspect the workspace with tools, run its code, read an error,
edit files, or retry within the attempt.

That behavior is useful when measuring a model's no-tool capability. It can be
a poor fit for tasks that require exploration or iterative debugging. If the
generated program is almost correct but fails because it did not inspect input
files, test assumptions, or repair a runtime error, try an agentic CLI harness:

- `codex_cli`
- `claude_code_cli`
- `kimi_code_cli`
- Other compatible choices include `pi_cli` and `opencode_cli`.

These harnesses can perform multiple model/tool turns within one benchmark
attempt. They may read files, execute commands, observe results, and revise the
solution, subject to the adapter's tool mode and sandbox. This can improve
performance on iterative tasks, but it is not a guaranteed score increase and
usually costs more time and tokens.

For example:

```bash
asibench run \
  --instances-dir hf_instances_seed31415/ \
  --agent codex_cli \
  --agent-config '{"model":"gpt-5.6-sol"}' \
  --sandbox linux_ns \
  --output-dir out_codex/
```

Use a separate output directory for each harness or configuration. For a fair
comparison, keep the instance set, prompt levels, model where supported,
timeout, sandbox policy, and scoring revision fixed.

## Other common causes of unexpectedly low scores

### The model or reasoning setting is too weak

Confirm the effective model and reasoning effort recorded in provenance. A
provider alias may resolve differently than expected, and low reasoning effort
may be insufficient for long scientific workflows. Change one setting at a
time so the comparison remains interpretable.

### The agent did not produce the required artifacts

Read the task's output contract and the scorer's gate details. Correct code is
not enough if it writes the wrong filename, schema, units, shape, image, or data
format. Check the persisted output directory rather than relying only on the
agent's final message.

### The task environment or sandbox differs

Missing packages, blocked network access, unavailable system tools, or a
different sandbox can change the result. Review stdout/stderr and recorded
provenance. Official seed42 submissions require `--sandbox os`; local
seed31415 experiments should still use consistent environments when compared.

### The run timed out or an API call failed

Look for timeout, rate-limit, authentication, context-length, or provider
errors. Increase `--timeout` only when the task genuinely needs more time.
Treat transient API failures as failed attempts, not evidence of model quality.

### The score varies between attempts

Model sampling and agent decisions are stochastic. Run repeated attempts before
drawing conclusions:

```bash
asibench run-score \
  --instances-dir hf_instances_seed31415/ \
  --tasks-dir /path/to/ASI-Bench/tasks/ \
  --agent codex_cli \
  --agent-config '{"model":"gpt-5.6-sol"}' \
  --sandbox linux_ns \
  --repetitions 3 \
  --output-dir repeated_results/
```

Compare individual attempts as well as their aggregate. Do not hide execution
failures inside an average.

## A practical diagnosis order

1. Confirm that the attempt completed and was actually scored.
2. Inspect stdout, stderr, persisted outputs, and per-gate score details.
3. Confirm model, effort, timeout, sandbox, framework version, and scorer/task
   revision in provenance.
4. If `direct_llm` failed for an iterative reason, rerun with an agentic CLI
   harness such as `codex_cli`, `claude_code_cli`, or `kimi_code_cli`.
5. Repeat the experiment with controlled settings before concluding that a
   model or harness underperforms.

See [Getting started](getting-started.md) for adapter and sandbox setup and
[How scoring works](how-scoring-works.md) for the seed31415/seed42 scoring
boundary.
