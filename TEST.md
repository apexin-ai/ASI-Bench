# Testing

Install development dependencies and run the default offline-safe suite. Tests
marked `integration` or `e2e` are excluded so a configured API key or installed
agent CLI cannot trigger paid/external execution unexpectedly:

```bash
uv run pytest -q
```

Issue #8 Responses translation regressions run without paid model requests:

```bash
uv run pytest -q tests/test_api_proxy_responses.py \
  tests/test_api_proxy_anthropic_responses.py tests/test_third_party_api.py \
  tests/test_api_proxy_image_inputs.py tests/test_mimo_adapter.py
```

Coverage includes custom `input` vs function `arguments`, additional tool
declarations, explicit rejection of unsupported `include`/tool limits and
stateful history, request-local `drop_params=False`, original returned
reasoning IDs/encrypted content, concurrent proxy requests, monotonic SSE
sequence numbers, lifecycle ordering, and failed/incomplete terminal events.
The real-LiteLLM regression mocks model completions and prohibits HTTP: on
1.82.6 it verifies explicit rejection of unsupported custom-tool conversion;
on 1.97.0 it verifies a two-turn custom-tool round trip and executes a fixed,
asserted test script in a temporary directory to check `artifact.txt == "ok"`.
The exact additional-tools/include/max-tool-calls request reported in issue #8
must fail explicitly before upstream execution on both versions. No agent CLI
or live model is run. JSON non-object bodies and invalid custom names must
return client errors rather than crashing the handler.
Malformed upstream output and tool calls missing call IDs return HTTP 502 in
both JSON and SSE modes, before sending successful response headers.
The Anthropic-to-Responses suite verifies two-turn encrypted reasoning replay,
original reasoning IDs, visible and redacted thinking, signed replay-envelope
integrity, conversation/execution isolation, concurrent requests, terminal
states, and fail-closed malformed data. A loopback HTTP endpoint exercises the
locked LiteLLM release and confirms `include: ["reasoning.encrypted_content"]`
and `store: false` reach `/v1/responses`; it makes no external or paid request.
Live upstream streaming is not claimed because both translation directions are
deliberately buffered.

The `CI` GitHub Actions workflow runs this suite automatically on every push
and pull request with Python 3.11 and 3.13. It also runs the focused custom-task
Docker integration suite once on Ubuntu, including pi and Claude Code overlays.
It uses `uv sync --locked` and `uv run --frozen`, so CI fails instead of silently
rewriting a stale lockfile. The stable `CI required` job aggregates the test
matrix, Docker runtime probe, and package build for use as a required
branch-protection check.

Scientific MCP coverage is offline and never starts a third-party server. It
validates strict JSON parsing, catalog completeness, CLI generation/checking,
Claude allowlisting, Codex TOML conversion, per-run home isolation, and the
Docker fail-fast boundary:

```bash
uv run pytest -q tests/test_mcp_config.py tests/test_tool_isolation.py \
  tests/test_adapters.py tests/test_cli.py
```

Manual MCP smoke tests require the operator to install the selected upstream
server and simulator, edit all placeholders, and explicitly accept license or
network side effects. Real-server automation belongs under the opt-in
`integration` or `e2e` markers.

Integration helper coverage is offline: `uv run pytest -q
tests/test_integrations.py`. These tests never start OpenFOAM, FreeCAD, MATLAB,
COMSOL, or an upstream service.

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

## Runtime Judge API configuration

`score`, `run-score`, and `benchflow-score` share credential-safe runtime
overrides for text and image Judges. Offline coverage verifies CLI help and
forwarding, fail-fast validation, native-provider key selection,
OpenAI-compatible model routing, text/VLM parity, scoped BenchFlow/local-score
overrides, configured-temperature forwarding, agent-subprocess credential
isolation, and secret redaction:

```bash
uv run pytest -q \
  tests/test_cli.py \
  tests/test_judge_common.py \
  tests/test_llm_judge.py \
  tests/test_benchflow.py \
  tests/test_local_scoring.py
```

The key option takes an environment-variable name, never key material. A custom
endpoint must include all three settings: `--judge-api-base`,
`--judge-api-key-env`, and `--judge-api-protocol`. Tests must use dummy secrets,
mock all model calls, and assert that neither exceptions nor persisted score
details contain the secret. Native Gemini coverage uses a key-only override;
TokenRouter coverage expects the OpenAI-compatible
`openai/google/gemini-*` LiteLLM route. For an explicit OpenAI-compatible
endpoint, text, VLM, and CMOS Judges pass the configured temperature in the
request body so LiteLLM does not apply native-provider filtering; native
provider calls keep the top-level parameter and its normal validation.
Real-provider smoke tests belong under the opt-in `integration` or `e2e`
markers.

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

## Model-call observability

Run the call-boundary, transport-evidence, retry, persistence, denominator, and
privacy regression suite with:

```bash
uv run --frozen pytest -q tests/test_call_observability.py
```

The fixtures cover non-empty, empty, tool-only, partial, timed-out, malformed,
zero-token, empty-stream, and unavailable-stream outcomes; native and benchmark
retry linkage; provider IDs and finish reasons; stable call ordering; redacted
raw sidecars; and sidecar restoration by the result loader. A coverage gap or
unknown content state must produce `empty_response_rate: null` with
`empty_response_rate_status: unknown`. A numeric zero is valid only when all
attempted call boundaries and content states are observable. Codex `turn`
events are tested as agent-level coverage evidence rather than provider-call
boundaries because a tool-using turn may contain multiple completions.

The Docker regression suite also covers stdin forwarding, pinned CLI install
commands, Node/base-schema cache invalidation, and root-only image build steps
followed by a non-root runtime:

```bash
uv run pytest -q tests/test_os_sandbox.py tests/test_os_sandbox_adapters.py \
  tests/test_pi_cli_adapter.py tests/test_opencode_cli_adapter.py \
  tests/test_native_agent_extractors.py tests/test_integration.py
```

Custom task Dockerfile coverage verifies selected-agent overlays, per-agent
cache identities, single-agent installation, and direct reuse of the task base
when no agent is selected. A Docker integration probe builds the CMOS op-amp
base plus the pi overlay, runs it as UID/GID `12345:12345`, and checks writable
agent homes together with ngspice, the task Python environment, and the pi CLI:

```bash
uv run pytest -m integration -q \
  tests/test_os_sandbox_p1p2p3.py::TestTaskDockerfileAgentOverlayIntegration
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
  - `claude_code_cli` builds an isolated per-run `HOME` under a unique
    `.ai4sci-bench/claude_home/execution_*/<run_key-hash>/` root
    (tool_mode ≠ unrestricted), copying
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
homes per instance + prompt level and across repeated executions of the same
run key, collision-resistant run-key paths, `CLAUDE_CONFIG_DIR` override semantics,
unrestricted-mode opt-out, per-instance kimi homes for both host env and
`--sandbox os` rw mounts, concurrent Kimi root initialization, explicit
`kimi_home` passthrough, teardown cleanup of the whole home root, and exact
OS/Linux namespace timeout-marker classification. Timeout regressions must
cover genuine runner timeouts as well as successful and failed agent logs that
merely contain the words `timed out`.

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
requires the Docker `os` sandbox. It has no default agent: every explicit agent
requires a position-matched config with `model`, and at least one supported
multi-turn harness is mandatory. A direct-LLM-only check is rejected. Reports
record the effective model, effort, measured agent CLI/adapter version,
framework version, and sandbox. Task Draft validation independently rejects
missing agent names and direct-LLM-only evidence before authentication or
network access. The CLI rejects a threshold above 40, and catalog flagging
likewise ignores B1/B2. Terminal regressions also require a global progress bar
that identifies the current task, agent, prompt level, and instance before a
long-running agent call starts, then advances on completion or failure. Cover
the terminal, JSON, Markdown, CSV, persistence, submission evidence, runner
progress callback, and catalog contracts with:

```bash
uv run pytest -q tests/test_difficulty_check.py tests/test_static_validator.py \
  tests/test_cli_task_submit.py tests/test_parallel.py
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

## User-facing result troubleshooting

The documentation regression checks require README and Getting Started to link
the result troubleshooting guide. They also lock the explanation that
`direct_llm` is a single-turn, no-agentic-tools baseline and name the supported
agentic alternatives used for iterative tasks:

```bash
uv run pytest -q tests/test_readme_examples.py
```

When changing adapter behavior or names, update the troubleshooting guide and
this regression together so recommendations continue to match the code.

## Public local scoring and retired internal surfaces

The public distribution intentionally excludes the former internal orchestration
commands (`batch-run`, `eval`, `quickeval`, `fullrun`, `rerun-flagged`, and
`pipeline`), scoring flags on `run`, private submission scoring modules, and the
old GitHub task-PR automation. Benchmark `run` remains produce-only. The separate
`score --repo seed31415` command uses public HF references and GitHub scorers,
writes a non-official report without rewriting run results, and rejects seed42
before path access. Local score report schema v2 must preserve a legitimate
scored zero while representing an internal scorer failure with
`evaluation_status: evaluation_invalid`, `final_score: null`, and exclusion
from the aggregate denominator. MPSC setup failures must classify missing
trusted inputs/runtime as scorer errors and submitted controller failures as
ordinary scored-zero submission errors. Keep these boundaries covered with:

Judge/provider failures must carry `failure_kind: evaluator_unavailable`;
scorer, worker, and custom-loader crashes use `evaluator_runtime_error`; and
absent or unsafe immutable instance inputs use `missing_evaluator_input`.
Serial and parallel local scoring must both stage instance `data/` with the
persisted outputs in a fresh temporary workspace, reject symlinks and input
overwrites, leave the source `.outputs` tree unchanged, and exclude staging
failures from aggregate numerators and denominators.
The staging check copies the complete instance `data/` tree, so literal and
expanded input declarations (for example `pairs/<ii>/source.npy`) are both
covered. Missing contestant files remain ordinary submission failures rather
than evaluator failures.

Scorer-specific Python dependencies are opt-in: `evaluation.runtime: task`
prepares the task's declared runtime once per unique dependency specification
before workers start. Tests must prove that ordinary tasks never invoke the
runtime builder, opted-in workers can import packages from the isolated task
environment, repeated results reuse one environment, setup failures become
`evaluator_unavailable`, and a missing `uv` executable falls back to the
current compatible Python's `venv` and `pip`. A runtime resolved for a different
Python major/minor must fail closed before its site-packages are loaded.

`score --parallel N` performs full-batch preflight before evaluator work and
runs complete result evaluations in fresh spawned processes. The worker count
is bounded by `N`; gate/scorer/Judge loops inside one result stay sequential.
Coverage must preserve source ordering and serial score parity, prove isolation
from worker `cwd`/environment/import mutations, propagate credential-safe Judge
metadata, redact secrets, classify worker failures as invalid evaluations, and
preserve an existing report if atomic replacement fails. Docker image cache
creation remains protected by a tag-keyed cross-process lock.

```bash
uv run pytest -q \
  tests/test_retired_features.py \
  tests/test_local_scoring.py \
  tests/test_task_env.py \
  tests/test_mpsc_error_classification.py \
  tests/test_no_local_scoring.py \
  tests/test_review.py \
  tests/test_hf_pull.py::TestUnscoredSubmissionReporting \
  tests/test_os_sandbox.py::TestTaskImageBuilder
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
`config/public_task_runtimes.json` separately allowlists formal runtime files at
task granularity and must exactly match `runtime.dockerfile` declarations.
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

Formal-task scorer runtime regressions verify that the Levin scorer reads
training data from the reference bundle rather than agent outputs, its search
result type is constructible, and the CMOS scorer loads split task metadata
before building the ngspice image:

```bash
uv run --frozen pytest -q tests/test_task_scorer_runtime_regressions.py
```

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
`--parallel` as one workflow-wide limit and can repeat the complete workflow
with `--repetitions`; each repetition has separate run and score output. Agent
task jobs from every repetition first share one bounded queue. A phase barrier
then prevents agent jobs from overlapping scoring, and repetitions are scored
one at a time with up to the same number of isolated result workers. This
prevents nested \(N \times N\) worker fan-out while allowing later repetition
task jobs to fill slots released during the agent phase.
Runtime Judge endpoint, key-variable name, and protocol settings are forwarded
to every score subprocess without placing the secret on its command line.
The ASI-Bench agent-run child disables project `.env` reload during framework
startup and removes the Judge selectors plus a dedicated nonstandard Judge key.
It restores normal dotenv behavior before the evaluated agent starts;
conventional provider keys remain compatible.
# Scientific MCP catalog

The catalog regression test covers the built-in and extended scientific MCP
interface templates. Run:

```bash
.venv/bin/pytest -q tests/test_mcp_config.py
```

The extended catalog includes robotics, CAD/FEA, EDA, flight, traffic,
instrumentation, laboratory, and scientific-computing entries. Runtime
availability is machine-specific; use
`.claude-manager/artifacts/task-78/download_and_test.py` for best-effort local
package installation and executable probes.


## CAD MCP compatibility repair tests

Offline bundle validation (no CAD, network or extra MCP dependencies):

```bash
uv run --frozen pytest tests/test_cad_mcp_repairs.py tests/test_mcp_config.py -q
```

Checks patch hashes, pinned SDK snapshots, clean revision enforcement,
preflight-before-modification, checksum rejection and repeat-apply rejection.
For opt-in installed-server stdio tests, follow
`scripts/mcp/cad-repairs/README.md`. The live suite uses a temporary HOME/cwd,
port protection, and 60-second per-server timeouts; validates initialization,
all tools pages, repeated lists, errors and recovery; never calls real CAD.

## MCP E2E (setup, smoke, verifier, fake tasks)

All offline MCP E2E tests are in `tests/mcp_e2e/` (no network, no MCP server,
no upstream scientific libraries):

```bash
uv run --frozen pytest tests/mcp_e2e -q
```

| File | Covers |
|---|---|
| `support.py` | not a test: the one copy of loaders, Claude/Codex log builders, `persist_like_run`, `Task.verify` (run directory → `verify_one`), scorer dirs, smoke stubs |
| `test_setup.py` | `scripts/mcp/e2e/setup.py` |
| `test_smoke_runner.py` | `e2e_smoke/runner.py`, `client.py`, `smoke.py` |
| `test_smoke_<id>.py` | the library-free parts of `e2e_smoke/servers/<id>.py` |
| `test_verify_spec.py`, `test_verify_values.py` | `e2e_verify/spec.py`, `values.py` |
| `test_verify_evidence.py` | Codex extractor, persisted Codex JSONL, stdout parser per agent, Codex runs of the pyscf tasks |
| `test_task_contract.py` | conventions every fake task meets, parametrized over `examples/mcp-e2e-tasks` |
| `test_task_<task>.py` | one task: generator, scorers, verifier scenarios |

A new task needs only `test_task_<task>.py` with its data and scenarios (the
contract fails until it exists and covers generator determinism, submission
failures, evaluator failures and a genuine passing run); a new server needs
`test_smoke_<id>.py`. Build agent logs with `support.claude` / `support.codex`
and run the verifier with `Task.verify`; do not copy run-directory or stream
helpers into the test file.

Contract (`test_task_contract.py`): every task is `status: test`, discovered
only from its own directory and never from `tasks/`; prompts B1–B4 exist,
mention `result.json` and never `mcp__`, Claude or Codex; `e2e_check.json`
parses strictly, every call tool is one of the manifest's `expected_tools`,
`server_tools` (when given) equals them, and `mcp__` names in
`bypass_tools`/`suspicious_tools` are tools of the task's server.

Verifier golden snapshot: every `verify_run.verify_one` call made by a passing
test in `tests/mcp_e2e/` is recorded (verdict, failure, every check and
per-call status, tool sequence) and must equal `tests/mcp_e2e/golden.json`
(plugin `tests/mcp_e2e/golden.py`, loaded from `tests/conftest.py`). A
mismatch is reported as a test ERROR listing the changed fields; a new
verifier test without an entry, or a stale entry, also fails. After an
intended verifier change, regenerate and review the diff:

```bash
MCP_E2E_GOLDEN=update uv run --frozen pytest -q tests/mcp_e2e
```

The update refuses `-k` / `file::test` selections and red runs.

Persistence: `support.persist_like_run` runs the orchestrator's real
`_sanitize_raw_artifact_text` for Claude and Codex logs (user events redacted,
host paths → `<home>` / `<workspace>` / `<run_output_dir>` / `<repo_root>` /
`<abs_path>`) and extracts the trajectory from the unsanitized log, as
`asibench run` does; Codex scenarios are persisted too. `conftest.py` pins
`$HOME` to a sentinel for the whole package, so which placeholder a fixture
path becomes does not depend on who runs the suite.
`test_task_s4_grating_spectrum.py` checks that a `ctypes.CDLL('/…/libS4.so')`
command is scrubbed in the saved log yet still FAILs `no_bypass` (as-executed
text from the trajectory), and that a scrubbed command without a trajectory is
a `no_bypass` WARN; `test_verify_evidence.py` checks the same for a Codex
`command_execution` and that a tool *result* holding a host path (persisted as
`<abs_path>`, which the Codex JSONL keeps instead of redacting) is restored from
the trajectory by call id; `test_task_rdkit_conformer.py` checks that when
nothing unscrubbed is left the returned path is a `tool_correct` WARN (coverage
gap) rather than a FAIL, while the partially scrubbed pickles still chain;
`test_task_build123d_plate_measure.py` checks the same for scrubbed tool
*arguments*: a genuine run's `import_cad_file(path=…)` is restored from the
trajectory's `key_args` and PASSes `tool_chain`, importing a file the server
never wrote FAILs it, and exporting to another path is a `tool_correct` WARN;
`tests/test_trajectory.py` checks that the Claude extractor
skips events whose `message` is not an object instead of raising.

`test_setup.py`: manifest entries pinned to 40-char revisions and matching
catalog sources; unknown keys and fields of another install mode rejected
(each `Installer` owns its fields; owners are disjoint; a stub installer
registered in `INSTALLERS` validates, builds and refuses `--lock` without
other changes); rendered `*.mcp.json` valid with absolute paths, launch `env`,
`uv_sync_args` and `{checkout}` placeholders (the arxiv config keeps its CLI
flags literal and turns the ToolUniverse cache off; absolute `JBSIM_ROOT`);
absolute launch paths and foreign checkout remotes rejected; `uv-pip-pinned`
(fresh `uv venv --clear`, `uv pip install --exclude-newer` of exact pins);
`host_requirements` (machine, CPU flags, loadable libraries, every problem
listed, checked before cloning); `conda-explicit` (fresh `micromamba create
--file <lock>`, package cache under `<root>/.micromamba`, conda variables
dropped, non-conda `.venv` refused) and the committed locks (every
conda-explicit server and platform: header matches the manifest, each spec's
version really appears, conda-forge `<platform>`/`noarch` URLs with
`#sha256:`; stale, foreign, md5-only or wrong-channel locks rejected),
`setup.py --lock` from
`micromamba --dry-run --json`. The quantum_espresso config pins the local
runner (no Docker, no `mpirun`), resolves `pw.x` from the conda prefix via
`QE_PREFIX`, leaves `QE_WORKDIR` unset and keeps `mcp` below 2. Shared
checkouts: ids with the same `checkout`
install into one `<root>/<checkout>` (default is still `<root>/<id>`) while
keeping one `*.mcp.json` and one MCP server name per id, and a group that
differs on repository, revision, python or an install field, collides with
another manifest id, or names its checkout with a non-identifier is rejected.

`test_smoke_runner.py`: the stdio client paginates `tools/list`, surfaces
`isError` and records non-JSON stdout; the shared run end to end against a fake
stdio server from a pinned scratch checkout (config equals the rendered
manifest, handshake, unknown tool, stdout attribution per tool, cwd leftovers
minus declared artefacts) around the `prepare` / `run_l1` / `after` hooks; a
config that differs from the manifest FAILs; `check_rejected` and
`json_result` classification; a `Caller` passes a declared tools/call timeout
through to the client and leaves its default alone otherwise; every manifest
tool is called by its smoke and `smoke.py` resolves every manifest id.

`test_smoke_<id>.py`: pyscf — atom-string/XYZ/float-list parsing, invariant
distance comparison, PNG headers, in-band error split, plot and visualize
checks; arxiv — ID/URL splitting (incl. old-style IDs), Atom parsing, record
diffs, the snippet window/limit/cap reference, search/OR-precedence/snippet
checks with a stubbed reference, minimal server environment with proxies;
jsbsim — frame rounding, aircraft scan, telemetry comparison (printed
precision, dead fields), trim digests, missing-property/unknown-session
classification, a crash in the stock-script probe is a WARN; s4 — TMM against
closed forms, energy conservation, RCWA uniform limit and Li vs Laurent
convergence, shell → order mapping, grids clear of Rayleigh anomalies, stubbed
servers PASS while a TE/TM swap or Li-rule grating FAILs, defect probes WARN
only when a misuse is accepted; psi4 — geometry parsing, frequency
classification (imaginary sign dropped WARN), excited-state verdicts against a
stubbed reference, `ok:false` for a valid request FAILs; rdkit — structured
result unwrapping, nested-value deviation, three-state `classify`, file-path
confinement, `batch_map` item unpacking and the `fail_fast` / default image
name / `(*args, **kwargs)` / lost-property verdicts, tool tables covering the
manifest without overlap, hand anchors, and stubbed servers: descriptor checks
FAIL when an option is ignored, oxidation-number and invalid-input probes,
the coverage check; gpaw — the k-grid policy (hexagonal ×3 rule, vacuum by
atom span, wrap-around gap), the convergence recommendation on the recorded
MoS2 sweep (tolerance dependence, metallic fallback, sweep ceiling, wrong
arithmetic), gap analysis (direct/indirect/metallic, ties, k-point labels),
the `verify_run` state machine (three gap tolerances, missing results,
metastability, skipped reference), in-band errors and text/structuredContent
disagreement, and stubbed servers: the convergence gate FAILs when the
rejected run leaks a `summary.json`, a figure or a gap, the ignored `query`
and shared `gs.gpw` probes WARN, stdout pollution is attributed per tool;
quantum_espresso — banner parsing, the SG15 element index, the directory-order
pick re-run as `SG15Library` does it (D1) against the newest-version pick, the
k-grid rule with its odd snapping (ties go up), geometry helpers, our `pw.x` /
`bands.x` / `dos.x` inputs (documented settings, relax/vc-relax namelists,
`%.10f` band-path k-points, tetrahedra without smearing), the parsers on
pw.x 7.5's own layouts (every SCF step, total vs contribution force rows,
stress in kbar, final energy/cell/positions, ten-per-line `bands.dat`, `.gnu`,
`dos.dat`), the band-edge rule (direct/indirect/metal), nested comparisons,
and stubbed servers for every three-state check: a stale pick that follows the
scan WARNs and one that does not FAILs, `qe_get_kpath` (D10), a directory for
`output_dir` (D2), force rows (D13), a relaxation reply at the first step
(D11) vs the final one vs neither, a server output file from a different run,
a run outside `<cwd>/qe_calculations` or with another pseudopotential, the
`relax_and_scf` final SCF on the input geometry (D12), and the exact
`qe_get_job_status` / Materials Project errors;
alphafold_db — the `{status, data, metadata}` wrapper split,
the field-by-field comparison with an independently fetched entry (and that a
new model version alone never differs), pLDDT read from a four-residue PDB
fixture and binned at 50/70/90, AlphaMissense
per-position means, isoform accessions (`P04637-2`) accepted while a mismatched
`entryId` FAILs, summary selected by `model_identifier` (index 0 would be the
wrong model), the pLDDT mean tolerance and the one-residue bin-edge WARN, the
documented empty annotation state WARNing while an HTTP 500 or an
`isError=true` sentinel FAILs, the undeclared-`type` three-state (the MCP layer
rejecting it is a PASS, a silent `auto_query_params` override a WARN), and the
`return_schema` drift check read from a fake checkout; ncbi — `gene_table`
parsing for both strands (a plus-strand table carries no strand marker at all,
so requiring one made every plus-strand gene read as an unparsable reference),
the 0-based → 1-based locus shift (comparing the raw
values is the mistake the test pins), the reference-assembly entry chosen over
index 0, FASTA residue counting, the three in-band failure shapes plus zero hits
with no error field (a WARN, not a FAIL — only records for an impossible id
FAIL), a `retmax` dropped as falsy, an unusable `gene_table` reference blamed on
the smoke rather than on the tool, Datasets `v2alpha` disagreements kept to
WARN, and the identification check read from a fake checkout; atomictoolkit —
payload / in-band split, artifact lists (fresh ids, previews for structures
only, URL shape), the closed-form elastic fit against `np.polyfit`, the
textbook g(r) normalisation on an ideal gas, and three-state checks on stubbed
servers and references: the deprecated `FunctionTool` error, the plain-call
refusal of the task-required tools and their `tools/list` `taskSupport`, the
task-augmented call shape and `No active context found.`, BFGS lines on the task
path's stdout, ENAMETOOLONG for Si + EMT, the `auto` fallbacks, skin-0.3
coordination numbers, g(r) doubled, the cell-less molecule, `kim` listed as
available, and `create_download_artifact` accepting missing files.

`test_verify_spec.py` / `test_verify_values.py`: strict parsing (unknown,
inapplicable and invalid keys, dangling and optional-from-required
references, regexes, schema-1 normalisation, requirement grouping, `select` on
a call result) and value reading (`select` modes, structured-output
unwrapping, string/number chaining, geometry comparison, and the generic
`json_scalars` extractor: scalar leaves at any depth of a record keyed by the
requested uid, `None`/`False` never answerable, and a number canonicalised
together with its string spelling).

`test_verify_evidence.py`: event shapes from real `codex exec --json` runs
(codex-cli 0.159.2): the Codex extractor (call ids, server/tool, text results,
image media type without image data, failed and unfinished calls), arguments
and results surviving persistence, single-tool and scan → plot runs passing,
Codex's own resource listing not counting, direct backend use in a
`command_execution`, failed calls, uncopied answers, reformatted inputs (WARN),
a broken chain, a plot without an image, JSON-string arguments,
trajectory-only evidence, the stdout parser following `agent_name` (unknown
agents try each format; an unparseable log falls back to the trajectory) and a
lookalike server's tool not being the required tool.

`test_task_<task>.py`: seeded generator determinism and selection rules,
scorer credit curves and submission vs evaluator failures, and verifier
scenarios on Claude stream-json (and Codex JSONL where recorded): a genuine run
passes every check; the task's characteristic mistakes fail the right check
(pyscf_rhf_energy: direct PySCF, unavailable server vs uncalled tool, uncopied
answer; pyscf_bond_stretch: missing plot, retyped plot data, plot without
image, old trajectories WARN; arxiv_search_snippets: paper not from the search,
default cap, WebFetch/shell HTTP bypass, workspace trap; jsbsim_engine_run:
chunked steps WARN, another session id, no state read, trim WARN;
s4_grating_spectrum: default harmonics, uncopied spectrum values, RCWA code
WARN; psi4_opt_freq: reformatted geometry still chains, frequencies at the
start geometry, in-band `ok:false`, default method; gpaw_mos2_bandgap: a second
`run_id` breaking the chain, the one-call `run_verified_workflow` shortcut as a
bypass, an artefact listing that is not the run directory, a different
verification report, while `find`/`cp` in the server's own directory stay
PASS; ncbi_gene_protein_card: a gene record read for an id the search did not return
or fetched by symbol instead of uid breaks the chain, a value the record does
not carry fails `answer_from_tool` while the per-call check still passes (the
call result only pins the record's identity), all three in-band E-utilities
failure shapes plus an empty `uids` list count as failed calls, and
Biopython/entrez-direct/`datasets`/`curl` are bypasses while a plain WebSearch
is a WARN; alphafold_isoform_profile: every one of the three tools is required, the
`accession` alias of `qualifier` is a WARN, a score the annotation never carried
fails, the documented empty annotation state is a failed call, and fetching the
AlphaMissense CSV or UniProt is a bypass; its generator refuses equal isoform
lengths, disagreeing length fields, a tied top score, a score exactly at the
threshold, CSV disagreement and payloads above the 40 000-character cap).
With PySCF and geomeTRIC installed (`uv run --with pyscf==2.14.0 --with
geometric==1.1.1 ...`) the psi4 file also regenerates seed 31415 and compares
with the recorded reference.

`test_task_gpaw_mos2_bandgap.py` is the one task whose DFT reference cannot be
recomputed in a test, so it runs on a synthetic `SAMPLE_MEASURED` table
engineered to reach every branch (both convergence tolerances, both
`verify_run` verdicts, `params_verified` true and false) and checks the derived
policy against it; the recorded unit cell stands in for `ase.build.mx2`, so the
module imports without GPAW or ASE. Three further tests read the committed
`MEASURED` table instead: that it covers the whole `(ecut, kpts_density)` grid,
that the pure functions reproduce the gate *and the verification verdict the
server itself answered* at every grid point and gap tolerance, and that no
instance's gap sits within 3 meV of a `verify_run` branch boundary (they skip if
the table is ever emptied for a re-measurement). Only
`test_mx2_structure_matches_the_hardcoded_cell` and the generator test need ASE
(`uv run --with ase==3.29.0 ...`); the latter puts the real builder back.

`test_task_qe_si_bandstructure.py` reads the committed `MEASURED` table directly
(stdlib only): it covers exactly the 27-point (ecutwfc, grid, npoints_band) grid,
the k-spacing rule reproduces every `qe_suggest_kpoints` answer it recorded and
refuses a spacing whose ceiling hangs on rounding, every point generates, the
energies depend only on (cutoff, grid) and the band edges also on the path,
neighbouring instances sit at least twice the zero-credit tolerance apart, and
an inconsistent record (another pseudopotential copied, wrong band or point
count, a different file set, no gap) is refused. The measurement script's
literal round-trips to the committed table and `--check` reports every
difference. Verifier scenarios: a genuine Claude and Codex run pass; any
spelling of the suggested grid (`7 7 7`, `7x7x7`, `7`) chains, another grid,
another run directory or an unlisted band file breaks the chain; reading the
directory first (D2) and retrying is still a PASS; without the trajectory's raw
`output_dir` the path links are a WARN coverage gap; another Si pick (D1) fails
the index call; installs, imports, `pw.x` and the server's own interpreter are
bypasses while reading the band file with `cat` is a WARN and a pasted tool
payload is not.

Live L0/L1 smoke (network + upstream install, Linux, opt-in): follow
`scripts/mcp/e2e/README.md`, e.g. `python3 scripts/mcp/e2e/setup.py pyscf`
then `~/mcp/pyscf/.venv/bin/python scripts/mcp/e2e/smoke.py pyscf --config
~/mcp/pyscf.mcp.json`. It calls all seven tools and checks the results against
PySCF, RDKit and geomeTRIC run in the smoke process; expected outcome on the
pinned revision is `PASS` with 0 FAIL and 7–8 WARN (upstream defects listed in
`scripts/mcp/e2e/e2e_smoke/servers/pyscf.py`; the benzene symmetry probe is intermittent).
For arxiv: `python3 scripts/mcp/e2e/setup.py arxiv`, then
`~/mcp/tooluniverse/.venv/bin/python scripts/mcp/e2e/smoke.py arxiv --config
~/mcp/arxiv.mcp.json` (the checkout is shared per upstream repository, so it is
`~/mcp/tooluniverse`, not `~/mcp/arxiv`; needs access to export.arxiv.org and
arxiv.org, ~90 s).
Expected outcome on the pinned revision: `PASS` with 0 FAIL and 8 WARN (OR +
date precedence, three in-band errors plus a missing-paper one, `truncated`
flag, old-style ID, workspace-dependent tool filter).
For jsbsim: `python3 scripts/mcp/e2e/setup.py jsbsim`, then
`~/mcp/jsbsim/.venv/bin/python scripts/mcp/e2e/smoke.py jsbsim --config
~/mcp/jsbsim.mcp.json` (no network, ~2 s). Expected outcome on the pinned
revision: `PASS` with 0 FAIL and 17 WARN (dead telemetry fields, `cl` naming,
missing property, no-IC session, zero step, partial IC, unknown IC key, trim,
three in-band unknown-session errors, stdout lines of the unknown-aircraft
error, `execute_script` literal + temp file + stock-script segfault, and the two
launch-env probes).
For s4 (Linux x86-64 with AVX2/FMA/BMI2 and `libblas3 liblapack3`):
`python3 scripts/mcp/e2e/setup.py s4`, then `~/mcp/s4/.venv/bin/python
scripts/mcp/e2e/smoke.py s4 --config ~/mcp/s4.mcp.json` (no network, ~10 s).
Expected outcome on the pinned revision: `PASS` with 0 FAIL and 7 WARN
(swapped incidence/substrate, inner incidence layer, unknown layer material,
θ = 90° and 120°, negative thickness, duplicate layer names); a locally built
`libS4.so` adds an eighth WARN for the binary's SHA-256.
For psi4 (Linux x86-64 or aarch64, `micromamba` on `PATH`):
`python3 scripts/mcp/e2e/setup.py psi4`, then `~/mcp/psi4/.venv/bin/python
scripts/mcp/e2e/smoke.py psi4 --config ~/mcp/psi4.mcp.json` (no network,
~20 s). Expected outcome on the pinned revision: `PASS` with 0 FAIL and 11 WARN
(null `single_point` fields, in-band invalid multiplicity, `optimize`
`n_iterations` 0, zero IR intensities, ignored `temperature_K`, saddle-point
imaginary mode returned as real, `n_states` split between spins,
`optimize_excited_state` returning the ground-state minimum and the starting
geometry's excitation energy, stdout lines, `timer.dat` in the cwd).
For rdkit: `python3 scripts/mcp/e2e/setup.py rdkit`, then
`~/mcp/rdkit/.venv/bin/python scripts/mcp/e2e/smoke.py rdkit --config
~/mcp/rdkit.mcp.json` (no network, ~1 s, all 74 tools). Expected outcome on the
pinned revision: `PASS` with 0 FAIL and 28 WARN (`strict` description,
dropped invalid SMILES in `compute_descriptors`, four `(*args, **kwargs)`
tools, `CalcOxidationNumbers`, ignored unknown argument, explicit hydrogens
dropped, blind unpickling, one `EmbedParameters` default, SMILES-derived and
unconfined SDF file names, three `MolsMatrixToGridImage` crashes, default PNG
names, two `GetSubstructMatch` cases, six property tools plus the SDF
follow-up, unordered `batch_map` results and `fail_fast`). Upstream
mutations (MolWt → ExactMolWt, rounded TPSA, ignored `includeHs`, swapped
image size, dropped `useRandomCoords`, disabled pruning) each FAIL.
For gpaw (Linux x86-64 or aarch64, `micromamba` on `PATH`):
`python3 scripts/mcp/e2e/setup.py gpaw`, then `~/mcp/gpaw/.venv/bin/python
scripts/mcp/e2e/smoke.py gpaw --config ~/mcp/gpaw.mcp.json`. No credentials,
and only one check reaches the network (a Materials Project query without a
key, expected to fail with 401). It takes 24 min of single-threaded plane-wave
DFT (amd64, 2026-10-03): one relaxation, a convergence sweep, three band
calculations, two one-call workflows and one relaxation plus three SCFs as
references. Each step prints before it starts and the report's
`seconds_by_step` says where the time went; the smoke raises the tools/call
timeout to 2 h because the client default of 300 s failed a correct
`run_verified_workflow` call. Outcome on the pinned revision: `PASS` with
38 PASS, 0 FAIL
and 13 WARN (ignored `query`, MP query without credentials, three in-band
errors from `relax_structure`, fixed sweep range, path traversal, unknown run,
unpersisted `verification_note`, overwritten `gs.gpw`, the whole chain behind
one `run_verified_workflow` call, `calc_band_dos` stdout lines and the shared
stdout check). The two platforms agree on the same conda lock to ~1e-10 eV on
energies and the gap and ~1e-8 eV on the Fermi level (MoS2 monolayer: gap
1.6754359249486146 eV on aarch64, 1.6754359250622177 eV on amd64, direct K→K,
E = -22.0734417 eV, recommended ecut 300 eV / density 15), so an L2 tolerance
of 1e-6 eV is safe across platforms; it is GPAW *releases* that shift the
values by meV.
For atomictoolkit: `python3 scripts/mcp/e2e/setup.py atomictoolkit`, then
`~/mcp/atomictoolkit/.venv/bin/python scripts/mcp/e2e/smoke.py atomictoolkit
--config ~/mcp/atomictoolkit.mcp.json` (no network, ~5 s, all 18 tools; a second
short-lived server probes the task-augmented path). Expected outcome on the
pinned revision: `PASS` with 44 PASS, 0 FAIL and 31 WARN (four deprecated
wrappers, five task-required tools refused plainly and five unfetchable task
results, BFGS stdout on the task path, `kim` listed as available, `auto`
fallbacks, coordination skin, doubled g(r), cell-less molecule, ENAMETOOLONG
for Si, `create_download_artifact` relative URL and two silent accepts, six
in-band errors and the `tool_errors/` logs in the cwd). Mutating the server's
energy by 1e-7 relative, a translation by 0.1 %, or fixing the skin / g(r)
normalisation turns the matching checks into FAIL or PASS respectively.

`mcp_e2e.gpaw_mos2_bandgap` additionally pins that its answers survive upstream's
non-reproducible `structure_drift` check:
`test_the_verdict_may_not_depend_on_the_drift_check` forces that check both ways
and requires the verdict to be unchanged (and that the generator refuses a
tolerance where it is not), and
`test_a_drift_warning_does_not_break_the_verify_call` runs the verifier against a
simulated server that reports the check as `pass` and as `warn`.

Ground truth for `mcp_e2e.gpaw_mos2_bandgap` comes from the same installation:
`~/mcp/gpaw/.venv/bin/python scripts/mcp/e2e/measure_gpaw_table.py --config
~/mcp/gpaw.mcp.json --output ~/mcp/gpaw-table.json` drives one full chain per
`(ecut, kpts_density)` grid point, one of which also re-gates at the second
tolerance to confirm the generator may derive the recommendation rather than
measure it, and writes the `MEASURED` literal to paste into `generate_gt.py`.
Measured 2026-10-03 on AWS amd64: eight chains in 3799 s. Re-run it after
bumping the server revision or the conda lock; `--only 400:25` is an
eight-minute dry run. The run that produced the committed table also decided the
instance grid: ecut 350 eV was dropped because its relaxation stops on the force
criterion (2 BFGS steps, fmax 0.0495 against 0.05) and reports an indirect
Gamma->K gap, so ground truth there would be one GPAW release away from
changing.

For quantum_espresso: `python3 scripts/mcp/e2e/setup.py quantum_espresso`
(needs `micromamba`; the conda prefix is ~1.5 GB), then
`~/mcp/quantum_espresso/.venv/bin/python scripts/mcp/e2e/smoke.py
quantum_espresso --config ~/mcp/quantum_espresso.mcp.json`. No network and no
credentials; about a minute of single-threaded DFT on bulk Si (2 atoms,
30/120 Ry, 4×4×4, a 40-point band path), the server's calls and our own
`pw.x`/`bands.x`/`dos.x` reference runs taking 2–8 s each. Outcome on the
pinned revision, aarch64 VM and AWS amd64 (2026-10-07): `PASS` with 44 PASS,
18 WARN, 0 FAIL on both, every compared energy identical across the two
platforms; amd64 took about twice as long per DFT call (band structure 14.3 s,
DOS 14.6 s). The DFT tools agree with the references to the printed digit
(Si SCF `-15.75077338 Ry`, Fermi `6.5233 eV`; band gap 0.551 eV indirect;
2208 DOS points, dos.x EFermi 6.716 eV), and a reference with `degauss` 0.01
instead of the documented 0.02 turns all twelve DFT comparisons into FAILs. The
18 WARN: D1 (43 of 69 elements got an older file in the VM's directory
order, 33 on amd64; Si got the newest on both), D2 ×2, D10, D11 ×3 (relax, vc-relax, and
`relax_and_scf`'s relaxation energy), D12, D13 ×3, and seven documented in-band
errors (unparseable structure, three missing files, `qe_get_job_status` with
the local runner, two Materials Project tools without a key). The report
records `qe_versions`, `sg15_elements`, `sg15_pick` (the Si file and every
stale element), the workload and `seconds_by_step`.

Ground truth for `mcp_e2e.qe_si_bandstructure`:
`~/mcp/quantum_espresso/.venv/bin/python scripts/mcp/e2e/measure_qe_table.py
--config ~/mcp/quantum_espresso.mcp.json --output ~/mcp/qe-table.json` runs the
task's three-tool band-structure chain at all 27 grid points (180 s on the
aarch64 VM, 2026-10-07; a repeat run was identical) and writes the `MEASURED`
literal next to the JSON. `--ecut N` measures one cutoff (for a host with a
short command limit) and `--merge a.json b.json c.json` joins the parts; add
`--check` to compare with the table `generate_gt.py` carries (exit 1 on any
difference). Run `--check` on a new run host before an agent round.

The live agent run (generate → run → score → verify) is documented in
`examples/mcp-e2e-tasks/README.md`.

Local scoring of runs that produced no files:
`tests/test_local_scoring.py::test_run_that_recorded_no_outputs_is_a_scored_submission_failure`
checks that a result JSON which records no outputs is a completed zero while
other results in the batch still score, and
`test_vanished_output_directory_is_still_rejected` keeps a missing but recorded
`.outputs` directory a preflight error.

Produce-only retries:
`tests/test_retry.py::TestRetryStrategyAll::test_produce_only_run_keeps_every_attempt`
checks that `asibench run --retries N` (score=False) saves N result files with
attempts 1..N instead of overwriting attempt 1.
