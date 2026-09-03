# Testing

Install development dependencies and run the default offline-safe suite. Tests
marked `integration` or `e2e` are excluded so a configured API key or installed
agent CLI cannot trigger paid/external execution unexpectedly:

```bash
uv run pytest -q
```

The `CI` GitHub Actions workflow runs this suite automatically on every push
and pull request with Python 3.11 and 3.13. It uses `uv sync --locked` and
`uv run --frozen`, so CI fails instead of silently rewriting a stale lockfile.
The stable `CI required` job aggregates the test matrix and package build for
use as a required branch-protection check.

Publishing is tied to a GitHub Release by `.github/workflows/publish.yml`. The
workflow checks that a tag such as `v0.1.2` matches the package version, reruns
the locked offline-safe suite, builds and validates both distributions, and
publishes with the repository's `PYPI_API_TOKEN` Actions Secret. Validate the
workflow contract without contacting PyPI:

```bash
uv run pytest -q tests/test_ci_workflow.py
```

README regression coverage also keeps the agent extension contract explicit:
new models for a compatible existing harness are configuration changes, while a
new harness starts through `--agent-cmd` or needs a new adapter for first-class
integration. The same test module locks the published paper title and arXiv URL:

```bash
uv run pytest -q \
  tests/test_readme_examples.py::test_agent_docs_distinguish_new_models_from_new_harnesses
```

Run external tests only when intentionally providing credentials and accepting
their network/runtime cost:

```bash
uv run pytest -q -m integration
uv run pytest -q -m e2e
```

## pi / opencode native adapters

The `pi_cli` and `opencode_cli` adapters provide first-class support for the
pi and opencode coding agents with native JSONL trajectories, token-cost
extraction, and fake-success (terminal API error) detection.

Offline unit tests (no agent CLI or API key needed):

```bash
uv run pytest -q \
  tests/test_pi_cli_adapter.py \
  tests/test_opencode_cli_adapter.py \
  tests/test_native_agent_extractors.py
```

Coverage includes: command construction (prompt via stdin, never argv),
auth mode resolution (local login / provider API key / explicit endpoint with
`api_protocol`), tool-mode isolation, JSONL log parsing, `CostInfo` summing,
terminal-error detection, `--sandbox os` integration (auth mounts, Docker
install commands, network whitelist, agent_type plumbing), trajectory
extractors, and JSONL schema detection dispatch.

The extractors are pinned to verified CLI event schemas (pi v0.84.3,
opencode v1.17.15). If either CLI changes its event format, update the
fixtures in `tests/test_native_agent_extractors.py` together with
`ai4sci_bench/trajectory/pi_extractor.py` /
`ai4sci_bench/trajectory/opencode_extractor.py`.

Manual smoke test (requires a real provider key; results are non-official):

```bash
asibench run --agent pi_cli \
  --agent-config '{"model": "<provider/model>"}' \
  --instances-dir hf_instances_seed31415/ \
  --prompt-levels b3 --output-dir out_smoke/
```

Check points: result JSON has `raw_stdout_format=jsonl`, non-generic
`trajectory_summary` (steps > 1), non-null `cost`; a deliberately invalid
API key must yield `FAILED` + `error_message` (pi exits 0 on API errors).

OS smoke tests must use `--sandbox os` and the exact seed31415 instance ID.
The image must report Node 22 plus pi `0.84.3` or opencode `1.17.15`; result
provenance must contain a Docker image SHA. The 2026-08-31 acceptance run used
`materials.relaxation_mode_recovery__seed31415` B1 with an OpenAI-compatible
endpoint. Both adapters completed, produced `analysis.py` and
`results/modes.csv`, persisted native token usage, and scored `93.02/100` via:

```bash
asibench score --repo seed31415 \
  --results-dir <agent-results> \
  --instances-dir <seed31415-instances> \
  --tasks-dir tasks
```

The Docker regression suite also covers stdin forwarding, pinned CLI install
commands, Node/base-schema cache invalidation, and root-only image build steps
followed by a non-root runtime:

```bash
uv run pytest -q tests/test_os_sandbox.py tests/test_os_sandbox_adapters.py \
  tests/test_pi_cli_adapter.py tests/test_opencode_cli_adapter.py \
  tests/test_native_agent_extractors.py tests/test_integration.py
```

## Per-run harness home isolation (claude_code / kimi_code / codex)

CLI harnesses keep session transcripts, history, and auto-memory under their
home directories (e.g. `~/.claude/projects/`, `~/.codex/sessions/`,
`KIMI_CODE_HOME/sessions/`). To prevent that state from leaking between
sequentially executed instances:

- **OS sandbox**: each instance runs in a one-shot `--rm` container with a
  fresh `HOME=/home/agent`, so harness session state is destroyed with the
  container.
- **Host-side runs** (`--sandbox none|task|linux_ns`):
  - `claude_code_cli` builds an isolated per-run `HOME` under
    `.ai4sci-bench/claude_home/<run_key>/` (tool_mode ≠ unrestricted), copying
    only `.credentials.json` and `settings.json` and setting
    `HOME`/`USERPROFILE`/`CLAUDE_CONFIG_DIR`. Sessions, `~/.claude.json`, and
    ambient user config are excluded.
  - `codex_cli` builds an isolated per-run `CODEX_HOME` under
    `.ai4sci-bench/codex_home/<run_key>/` with `--ignore-user-config`.
  - `kimi_code_cli` generates a per-instance `KIMI_CODE_HOME` subdirectory
    (keyed by `run_key`) under one adapter-level temp root; an explicitly
    user-provided `kimi_home` remains shared by choice.

```bash
uv run pytest -q tests/test_adapters.py tests/test_kimi_adapter.py \
  tests/test_subprocess_base.py
```

Coverage includes: isolated HOME construction and auth mirroring, exclusion
of session/memory surfaces (`.claude/projects/`, `~/.claude.json`), distinct
homes per instance + prompt level, `CLAUDE_CONFIG_DIR` override semantics,
unrestricted-mode opt-out, per-instance kimi homes for both host env and
`--sandbox os` rw mounts, explicit `kimi_home` passthrough, and teardown
cleanup of the whole home root.

## CLI Task Draft upload and browser confirmation

Task submission tests cover manual PAT validation and storage, endpoint
normalization, safe Task-relative file collection, exact Draft synchronization,
snapshot reconciliation, browser opening, headless/CI behavior, and the rule
that the CLI never performs the final Proposal submit. They also verify that
every local-test evidence entry records `sandbox: os` and that invalid evidence
is rejected before credential resolution or network access:

```bash
uv run pytest -q tests/test_cli_task_submit.py tests/test_token_login.py \
  tests/test_device_login_cli.py tests/test_endpoints.py
```

## Task difficulty gate

Task-author difficulty checks run and record B1–B4, but only B3/B4 control the
verdict. Their mean scores must be strictly below the default ceiling of 40;
B1/B2 are serialized as ungated `RECORDED` rows. The command defaults to and
requires the Docker `os` sandbox. The CLI rejects a threshold above 40, and
catalog flagging likewise ignores B1/B2. Cover the terminal,
JSON, Markdown, CSV, persistence, and catalog contracts with:

```bash
uv run pytest -q tests/test_difficulty_check.py tests/test_static_validator.py
```

## Pull filtering and custom pre-submit paths

Regression coverage verifies that the shared seed42 and seed31415
`tasks/<instance-id>/` Hugging Face layout is imported while root-level instance
directories are ignored, grouped `task pull` /
`task create` CLI commands work, retired top-level aliases stay absent, and a
task-filtered pull ignores unrelated
directories retained in a reused Hugging Face snapshot. It also locks the
reference split: seed31415 preserves public `reference/` directories, while
seed42 filters them both from downloads and reused caches. It verifies that
`validate --pre-submit <task-dir>` infers a non-default tasks root when
`--tasks-dir` is omitted. The CLI contract requires an explicit `--repo seed42`
or `--repo seed31415`; omitting the option and using retired, demo, or arbitrary
repository values must fail before any download. The public-task policy test
additionally locks the 15 restored seed42 task definitions to public metadata
while asserting that no official B1–B4 prompt files are tracked in GitHub. It
also prevents the n-body catalog entry from regressing to the retired scattering
contract and rejects evaluation/generation fields in final-task metadata:

```bash
uv run pytest -q \
  tests/test_hf_pull.py::TestFlattenFlatLayout \
  tests/test_hf_pull.py::TestRootFlatLayout \
  tests/test_cli.py::TestCLITaskPull \
  tests/test_static_validator.py::TestCLIPreSubmitMode \
  tests/test_public_task_policy.py
```

## Public local scoring and retired internal surfaces

The public distribution intentionally excludes the former internal orchestration
commands (`batch-run`, `eval`, `quickeval`, `fullrun`, `rerun-flagged`, and
`pipeline`), scoring flags on `run`, private submission scoring modules, and the
old GitHub task-PR automation. Benchmark `run` remains produce-only. The separate
`score --repo seed31415` command uses public HF references and GitHub scorers,
writes a non-official report without rewriting run results, and rejects seed42
before path access. Keep both boundaries covered with:

```bash
uv run pytest -q \
  tests/test_retired_features.py \
  tests/test_local_scoring.py \
  tests/test_no_local_scoring.py \
  tests/test_review.py \
  tests/test_hf_pull.py::TestUnscoredSubmissionReporting
```

Produce-only numeric zeros are serialization placeholders, not scores.
Per-task reports must omit an all-unscored score table and render unscored
levels as `N/A` when mixed with scored results. Pre-generated
`--instances-dir` trees are immutable inputs: framework metadata belongs under
the run output directory. Cover both contracts with:

```bash
uv run pytest -q \
  tests/test_reporting.py::TestAggregation \
  tests/test_runner.py::TestInstancesDirWorkspaceIsolation
```

The `math.nnls_modulus_deblur` scorer resolves observation, kernel, and
measurement metadata from either the agent output workspace or the instance's
`data/` directory, independent of the process working directory. Cover this
contract with:

```bash
uv run pytest -q tests/test_nnls_modulus_scorer.py
```

## Authenticated website submission

Result submission defaults to `https://asibench.apexin.ai/`, requires a submitter
identity, uploads a draft, and directs the user to the website confirmation page.
Before bundle creation or authentication, it requires every result instance to
carry the seed42 suffix and verified Docker OS provenance (`effective_mode=os`,
fail-closed enforcement, `docker_container` verification, and an image
identity). It rejects non-OS, seed31415, unknown, and mixed-seed runs. A provided
benchmark repository must be the seed42 alias or canonical repository ID.
Public fixed-seed submission bundles do not require
`framework_task_info.json`; bundle tests lock that this framework-only file
remains optional. The tests mock the upload and never create a real online
submission:

```bash
uv run pytest -q \
  tests/test_submission_bundle.py \
  tests/test_cli_task_submit.py \
  tests/test_difficulty_check.py \
  tests/test_device_login_cli.py \
  tests/test_endpoints.py
```

## Public task catalog policy

When updating formal-task scorers or published examples, validate both exact
allowlists and scan the public GitHub tree for GT generators, references, private
solver assets, undeclared artifacts, repository identifiers, and secret-like
content. `config/public_scorers.json` locks all 60 formal scoring contracts, 57
custom scorers, their exact per-task helper allowlists, and the private source revision.
Formal `task_eval.yaml` files may contain only scoring/output contracts and must
never contain `generation`. seed31415 references live on Hugging Face, not in
formal GitHub task directories. The separate Example policy locks five full public
sample tasks to their exact prompt, generator, scorer/configuration, and
reference-spec file sets. Lifecycle/CLI tests verify that sample tasks remain
accepted and opt-in runnable:

```bash
uv run pytest -q \
  tests/test_public_task_policy.py \
  tests/test_types.py::TestTaskLifecycle \
  tests/test_cli.py::TestCLIList \
  tests/test_issue33_task_loader_robustness.py::TestIssue33DiscoverTasksRobustness
```

The same policy test imports TSP, Max-3-SAT, Levin, MPSC, UCB, and CMOS scorers
in subprocesses containing only public files. It also scans evaluator runtimes
for generator/reference-builder symbols and forbids scorers from importing
`generate_gt`, so seed31415 remains locally scoreable without exposing seed42
GT through the public scorer API.

BenchFlow seed31415 adapter regression coverage validates the manifest seed and
instance boundary, required run-result binding, independent agent execution
status checks, deterministic artifact hashing, public reference usage, and
stable JSON score output. CLI coverage also verifies that
`run --fail-on-agent-error` saves evidence before returning non-zero and only
uses the final retry attempt:

```bash
uv run pytest -q tests/test_benchflow.py tests/test_cli.py
```

Persistence sanitization coverage verifies that HTTP(S) API endpoints remain
intact in reproducibility metadata while separate absolute host paths are
replaced with stable placeholders:

```bash
uv run pytest -q tests/test_runner.py -k sanitize
```

Native adapter tests pin the CLI versions whose command/config schemas were
verified (`pi` 0.84.3 and `opencode` 1.17.15); the normal run banner performs
the live `--version` probe when those binaries are installed. Full Docker
smoke tests remain environment-dependent.

## PyPI packaging

The packaging contract is covered by `tests/test_packaging.py`: the sole
distribution is `asibench`, the preferred and compatibility CLI entry points
both target `ai4sci_bench.cli:main`, and the import package remains
`ai4sci_bench`.

Before a release, build and inspect both artifacts:

```bash
uv build
uvx twine check dist/*
```

Install the wheel into a clean environment and smoke-test the metadata and both
commands:

```bash
python -m venv /tmp/asibench-release-check
/tmp/asibench-release-check/bin/pip install dist/asibench-*.whl
/tmp/asibench-release-check/bin/asibench --help
/tmp/asibench-release-check/bin/ai4sci-bench --help
/tmp/asibench-release-check/bin/asibench task pull \
  --repo seed42 \
  --tasks astronomy.nbody_close_encounters \
  --output-dir /tmp/asibench-release-check/pulled
```

The default wheel must install only the lightweight pull/runtime dependencies.
Use `pip install 'asibench[full]'` when testing the optional model and scientific
stack.

Task-sandbox regression coverage verifies both framework installation sources:
source checkouts use editable installation, while wheels pin the installed
distribution version and use a writable per-user runtime root. Run it together
with the host-backend fail-fast tests:

```bash
uv run pytest -q \
  tests/test_task_env.py \
  tests/test_cli.py::TestRunSandboxAvailability
```

Do not upload to PyPI or TestPyPI until the packaged framework source is ready
to be public: wheel and source-distribution contents are publicly downloadable.

## Dependency locking contract

`uv.lock` pins contributor and CI environments; it is not consumed by
`pip install asibench`. This is intentional for a Python library: wheel metadata
uses the compatible dependency ranges declared in `pyproject.toml`, so ASI-Bench
can coexist with the rest of a user's environment. CI separately installs the
built wheel in a clean environment, exercising the dependencies that `pip`
would resolve for an end user. Update `uv.lock` deliberately and commit it with
any dependency change.

### Independent three-run difficulty check

`asibench run-score` composes `run` and `score` for seed31415. It forwards
`--parallel` to the runner and can repeat the complete workflow with
`--repetitions`; each repetition has separate run and score output. Repeated
task jobs share one bounded queue, so later repetitions fill slots released by
the tail of an earlier repetition without multiplying the concurrency limit.
