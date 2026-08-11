# Getting Started

ASI-Bench runs LLM agents on real scientific tasks. This repository is
produce-only: it collects outputs and submits them to
`https://asibench.apexin.ai/` for official scoring.

## Install

```bash
pip install asibench
asibench --help
```

Python 3.11+. `asibench` is the canonical command in all user documentation.
The legacy `ai4sci-bench` entry point remains available only as a
backwards-compatibility alias.

## Run the benchmark with your agent

```bash
# 1. Pull each fixed-seed instance set (prompts + input data)
asibench task pull --repo seed42 --output-dir hf_instances_seed42/
asibench task pull --repo seed31415 --output-dir hf_instances_seed31415/

# 2. Run your agent and collect its outputs — no local scoring
asibench run \
  --instances-dir hf_instances_seed42/ \
  --agent-cmd 'python my_agent.py --workspace {workspace}' \
  --sandbox linux_ns \
  --output-dir out_seed42/

# 3. Sign in so the website can identify the submitter
asibench login

# 4. Upload a draft to the website for official scoring
asibench submit --results-dir out_seed42/
```

Repeat steps 2–3 with `hf_instances_seed31415/` and a distinct output directory
to run the seed31415 set.

`asibench run` evaluates `b1,b2,b3,b4` by default. Use `--prompt-levels` only
when you intentionally want a subset.


### Plugging in your agent

- **Any CLI / script:** `--agent-cmd '<command>'` (file-exchange mode). The
  framework prepares a workspace with the inputs and prompt, runs your command,
  and collects the output files your agent writes.
- **Built-in adapters:** `--agent direct_llm|claude_code_cli|codex_cli|…` with
  `--agent-config '{...}'` (model, API key/base, reasoning effort). Compatible
  adapters accept a third-party `api_base` + `api_protocol` (`openai` or
  `anthropic`).

The extension boundary is model versus harness:

- **New model, existing harness.** Pass the new model identifier in
  `--agent-config`. For a non-native endpoint, compatible adapters also accept
  `api_base`, `api_key`, and `api_protocol`. This normally requires no framework
  change as long as the existing harness accepts the model and its endpoint is
  OpenAI- or Anthropic-protocol compatible. Provider-specific behavior can still
  require an adapter update.
- **New agent harness.** Any new CLI or script that follows the workspace/file
  exchange contract can start with `--agent-cmd` and use `none` or `linux_ns`.
  First-class support—registration under `--agent`, harness-specific auth and
  configuration, proxy integration, and Docker-based `os` isolation—requires a
  new adapter to be implemented and tested in ASI-Bench.

### Sandboxing

Choose an isolation level with `--sandbox`:

- `none` — run on the host (fastest, least isolated)
- `task` — a per-task Python environment honoring the task's declared packages
- `os` — a Docker container
- `linux_ns` — lightweight Linux-namespace isolation

File-exchange agents configured with `--agent-cmd` support `none` and
`linux_ns`. Docker-based `os` isolation requires a compatible built-in adapter.

Every run records provenance (agent, model, effort, sandbox, framework version) so
results are reproducible and comparable.

When installed from PyPI, task environments install the same released
`asibench` version instead of treating `site-packages` as an editable source
checkout. Their caches live under `~/.asibench/`. Before a run starts,
`linux_ns` verifies user-namespace support and `os` verifies the Docker daemon;
missing host capabilities produce a non-zero command exit.

### Timeout

`asibench run --timeout` is the only source of the agent execution timeout and
defaults to 10800 seconds. A task cannot override it through `task.yaml` or
`task_meta.yaml`; timeout fields added to task metadata are ignored.

## Submit for scoring

`asibench submit` builds a bundle of your declared outputs with checksums and
provenance. By default it uploads the bundle to the ASI-Bench website as a draft and
prints a **confirm link**:

1. Open the link in your browser.
2. Review the completeness / integrity summary the portal parsed from your bundle.
3. Click **Confirm** to enter the scoring queue.

Uploading alone is not enough — your run is a **draft** until you confirm. The
website then scores it against private reference answers and can publish it to the
leaderboard. Self-reported scores are never used.

Use `--no-upload` when you intentionally want to build a local bundle without
submitting it to the website.

## Next steps

- [How scoring works](how-scoring-works.md) — what is public, what is private, and why.
- [Authoring a task](authoring-a-task.md) — contribute a new task to the benchmark.
