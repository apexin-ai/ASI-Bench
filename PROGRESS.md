# Progress

## Task-scoped evaluator runtimes

- Problem: MPSC duplicated its dependency specification inside the scorer and
  attempted to create a task environment during evaluation. A scoring worker
  without the external `uv` executable therefore reported all levels as
  evaluator runtime failures even though the task metadata declared the needed
  SciPy/CVXPY solver packages.
- Resolution: add the opt-in `evaluation.runtime: task` contract. Local scoring
  now prepares and reuses the task's declared runtime before loading its scorer,
  injects the verified interpreter and site-packages into a fresh worker, and
  leaves non-opted-in tasks on the existing path. MPSC now uses `task_meta.yaml`
  as its only dependency source and reuses the framework runtime.
- Prevention: runtime setup is deduplicated by environment cache key, fails
  closed on Python minor-version mismatch, classifies setup failures as evaluator
  infrastructure errors, and falls back to standard-library `venv` when `uv` is
  unavailable but the current interpreter satisfies the task requirement.
- Verification: full default suite `2433 passed / 2 skipped / 24 deselected`;
  final MPSC runtime-reuse regression file `7 passed`; diff checks clean.
- Implementation commit: `35bd147`

## 2026-09: Scientific MCP catalog and integrations

- Added auditable MCP profiles and integration helpers for scientific
  simulators, CAD/FEA, EDA, robotics, flight, traffic, instrumentation, and
  laboratory tooling. Catalog entries remain operator-configured and do not
  bundle proprietary software or credentials.
- Verification: MCP catalog and integration tests pass; external runtimes are
  tested separately when locally available.
- Implementation commits: `74aeeea`, `72ccba8`, `98be95e`.

## Public formal-task scorers without GT disclosure

- Problem: formal task scoring logic was not auditable in the public repository,
  while publishing the private task tree directly would also expose GT
  generators, generation settings, references, and private solver assets.
- Resolution: import scoring-only contracts for all 60 formal tasks, 57 custom
  scorers, and one required scorer helper from source revision `f7d41c97`; strip
  every `generation` block and exclude every generator/reference asset.
- Prevention: `config/public_scorers.json` is the exact scorer allowlist and new
  fail-closed policy tests require scoring-only YAML, compile every custom
  scorer, and reject formal-task GT generators, reference specs, protected
  directories, undeclared files, secrets, and symlinks.
- Verification: source audit matched all 60 evaluation configs and normalized
  scorer blobs; default suite `2093 passed / 4 skipped / 22 deselected`; wheel
  and sdist passed `twine check --strict` and contained zero task files.
- Implementation commit: `af3c525`

## Latest-paper README alignment

- Problem: the README contributor list lagged the latest paper, and its compact
  B1–B4 description omitted the matched-condition controls and precise
  information boundaries used to measure scientific autonomy.
- Resolution: synchronize the front-page contributor order and new contributor,
  document the four guidance conditions, add the paper's long-horizon execution
  scale, and align task-submission requirements with the 15-step Guided Flow.
- Prevention: compare future README revisions directly against the paper's title
  page, benchmark-design section, author appendix, and contribution appendix;
  retain the user's decision not to include a separate main-findings section.
- Verification: default suite `2090 passed / 4 skipped / 22 deselected`; wheel
  and sdist both passed `twine check --strict` with the updated README metadata.
- Implementation commit: `7f793b5`

## Model versus agent-harness extension contract

- Problem: user documentation listed configurable adapters but did not explain
  whether a future model or a wholly new agent harness requires framework work;
  Getting Started also overstated third-party routing as supported by every
  adapter.
- Resolution: document that compatible new models normally use the existing
  harness plus `--agent-config`, while new harnesses can start with generic
  `--agent-cmd` but require a new adapter for first-class integration and Docker
  `os` support.
- Prevention: README regression coverage requires both the model and harness
  extension paths to remain explicit in README and Getting Started.
- Verification: README tests pass 7 cases; the default suite passes `2086
  passed / 2 skipped / 22 deselected`.
- Implementation commit: `94cce96`

## CI public-tree compatibility

- Problem: the first online CI run failed because tracked `AGENTS.md` was a
  symlink, which the repository's public-tree secret-container policy rejects;
  the initial Actions versions also emitted Node.js 20 deprecation warnings.
- Resolution: store `AGENTS.md` as a regular copy of `CLAUDE.md`, test that the
  two remain identical, and move checkout, setup-uv, and artifact upload to
  current Node.js 24 action releases.
- Prevention: run the full suite only after new files have been staged so
  `git ls-files` policy checks see their final tracked modes, and verify the
  real GitHub Actions run after every workflow change.
- Verification: public-policy and CI regression tests pass; the full Python
  3.11 suite passes `2085 passed / 2 skipped / 22 deselected`.
- Implementation commit: `84c5308`

## Locked CI and package verification

- Problem: tests and packaging checks existed only as manual instructions, so
  pushes and pull requests could merge without exercising them; the latest
  README also omitted a still-required legacy CLI compatibility note.
- Resolution: add a read-only GitHub Actions workflow that tests the locked
  environment on Python 3.11/3.13, builds and validates wheel/sdist artifacts,
  installs the wheel cleanly, and exposes one stable `CI required` gate; restore
  the README compatibility note and document the library lockfile contract.
- Prevention: regression coverage locks the workflow triggers, permissions,
  locked test commands, package checks, aggregate gate, and identical regular
  copies of `AGENTS.md` and `CLAUDE.md`.
- Verification: full offline suite passes `2084 passed / 2 skipped / 22
  deselected` on both Python 3.11 and 3.13; `twine check --strict`, clean wheel
  installation, and both CLI smoke tests pass.
- Implementation commit: `624e8b0`

## README fenced-code rendering

- Problem: an extra `````bash`` opener before explanatory prose caused CommonMark
  to render the following paragraphs, heading, and task table as one code block.
- Resolution: remove the stray opening fence so only the runnable shell commands
  remain fenced.
- Prevention: a README regression test now detects headings swallowed by fenced
  code and unclosed fences.
- Verification: `tests/test_readme_examples.py` passes 6 tests; the default suite
  passes `2083 passed / 2 skipped / 22 deselected`.
- Implementation commit: `34c525f`

## Offline-safe current-contract tests

- Problem: nine assertions still encoded retired local scoring, pre-protocol API
  setup, one-level instance counts, and old Codex/scorer behavior. Two permanently
  skipped VIC tests referenced a removed task, live model/CLI tests ran under the
  default command, and dependency metadata tests emitted deprecation warnings.
- Resolution: remove retired local-score and VIC tests; update useful API,
  generator, parallel/resume, scorer, and Codex coverage to current contracts;
  exclude `integration` and `e2e` markers by default; record dependency versions
  through distribution metadata.
- Prevention: external tests require explicit `-m integration` or `-m e2e`, prompt
  fixture rewriting now fails loudly if its source block changes, and the missing
  dependency test contains a real assertion.
- Verification: default offline suite `2081 passed / 2 platform skips / 22 external
  tests deselected`; integration and e2e collections contain 20 and 2 tests.
- Implementation commit: `e9beb66`

## PyPI wheel sandbox runtime root

- Problem: `BenchmarkOrchestrator` inferred the repository root from
  `__file__`. In a wheel this resolves to `site-packages`, so `task` and
  `linux_ns` task environments attempted `uv pip install -e site-packages` and
  failed before the agent started; related `os` setup paths also treated the
  installed package directory as mutable runtime state.
- Resolution: resolve a real source checkout when present and otherwise use a
  writable `~/.asibench/runtime` root. Task environments install source
  checkouts editable, but wheel installations pin the active `asibench`
  distribution version. Wheel caches live under `~/.asibench/`. Run commands
  now fail before instance creation when Linux user namespaces or Docker are
  unavailable.
- Prevention: regression tests cover source and wheel install targets, runtime
  root fallback, and non-zero host-backend preflight failures.
- Verification: a clean 0.1.1 wheel created a nested task environment without
  treating `site-packages` as a project; the reported astronomy task started
  Codex under both `task` and Docker `os`. Docker sandbox self-test passed.
  The latest full suite passed 2079 tests with 18 skipped and the same 9 known
  unrelated failures.
- Implementation commit: `a851c48`

## PyPI 0.1.2 release preparation

- Problem: PyPI `0.1.1` predates the current public task-contract, documentation,
  CI, and CLI fixes on `main`.
- Resolution: synchronize every public framework version field at `0.1.2` and
  regenerate the development lock file before building release artifacts.
- Prevention: verify the wheel and sdist with strict metadata checks, scan both
  archives for private task files, and smoke-test both CLI entry points plus a
  public seed42 pull from a clean wheel installation before uploading.
- Verification: default suite `2086 passed / 2 skipped / 22 deselected`; wheel
  and sdist passed `twine check --strict`, contained no private-file hits, and
  the clean-install seed42 nbody pull succeeded.
- Implementation commit: `5a1d457`

## GitHub-aligned PyPI publishing

- Problem: PyPI releases required a manual local upload and could drift from the
  source and version published on GitHub.
- Resolution: publish only when a GitHub Release is published, require its tag to
  match the package version, rerun the locked suite, and build and validate both
  distributions before uploading with an Actions Secret.
- Prevention: a workflow regression test locks the trigger, read-only repository
  permission, tag/version gate, strict artifact validation, secret reference, and
  absence of token-shaped plaintext in the workflow.
- Verification: workflow tests `3 passed`; default suite `2087 passed / 2 skipped
  / 22 deselected`; `uv version --short` reports `0.1.2`.
- Implementation commit: `eee5c47`

## PyPI 0.1.3 release

- Problem: PyPI `0.1.2` did not include the current public documentation and the
  fixes that preserve pulled instance directories and distinguish unscored
  submissions from zero scores.
- Resolution: synchronize all public framework version fields at `0.1.3` and
  publish the matching `main` commit through the GitHub Release workflow.
- Prevention: require a clean locked test suite, strict wheel/sdist validation,
  an archive scan for benchmark-only files, and a clean-wheel smoke test of
  both CLI names plus a public Hugging Face pull before publishing.
- Verification: `2090 passed / 4 skipped / 22 deselected`; both distributions
  passed `twine check --strict`, contained no task prompts, generators, scorers,
  or references, and the clean Python 3.11 wheel pull from seed42 succeeded.
- Implementation commit: `d732b85`

## Clean public repository migration

- Problem: exporting the latest private-repository tree verbatim would have
  republished 37 prompt, generator, scorer, and reference files from five
  formal benchmark task directories, even without copying Git history.
- Resolution: build the new public repository from a fresh object database,
  retain only `task_meta.yaml` in formal task directories, and keep
  `tasks/_template/` as the sole framework-level author scaffold.
- Prevention: the public-task policy and documentation now enforce the
  metadata-only boundary, and migrations verify both the staged tree and the
  absence of old commit objects before pushing.
- Verification: public-policy tests passed `13 passed`; the complete default
  offline suite passed `2087 passed / 2 skipped / 22 deselected`.
- Implementation commit: `a877776`

## Restore allowlisted public task examples

- Problem: the clean repository migration also removed the five intentionally
  public, end-to-end sample tasks used to demonstrate task authoring and local
  scorer integration.
- Resolution: restore exactly the 37 prompt, GT generator, scorer/configuration,
  and reference files belonging to the five allowlisted samples.
- Prevention: `config/public_examples.json` and the public-task policy test keep
  every other formal task metadata-only and reject unlisted private-like files.
- Verification: focused policy/lifecycle tests passed `27 passed`; the complete
  default offline suite passed `2087 passed / 2 skipped / 22 deselected`.
- Implementation commit: `989808e`

## CLI Task Draft upload with browser confirmation

- Problem: `asibench task submit` only opened a blank Portal form, forcing
  authors to select the same files and re-enter evidence they already prepared
  locally.
- Resolution: authenticate with a manually created Portal PAT, exact-sync the
  complete Task-relative file snapshot into an owner-only Draft, and open that
  Draft for file/field review while keeping final submission browser-only.
  `task_submission.yaml` carries private author evidence without entering Task
  repository exports.
- Prevention: CLI tests cover safe file collection, hidden token validation,
  credential reuse, Draft-only behavior, exact snapshot reconciliation, and
  headless failure; Portal tests enforce strict manifest validation and the
  export exclusion.
- Verification: after rebasing onto the public-scoring release, the public suite
  passed `2102 passed / 2 skipped / 22 deselected`; Portal CLI-sync tests passed
  `76 passed`; Portal frontend passed `63` contract tests, `2` WebKit tests, and
  the production build. A live CLI-to-WebKit E2E reached 100% completeness,
  verified ten exact files, clicked the browser-only final submit, froze R1,
  and excluded the Portal-only manifest from export.
- Implementation commit: `9161998`

## Split public seed31415 and private seed42 scoring

- Problem: both fixed seeds were documented and enforced as website-only even
  though seed31415 intentionally publishes references, while the puller could
  also copy a cached `reference/` tree for private seed42.
- Resolution: add a standalone `asibench score --repo seed31415` path that uses
  GitHub scoring contracts and public HF references without rewriting run
  results; reject seed42 before path access and filter its references during
  both snapshot download and cached-tree copying.
- Prevention: regression tests lock the seed-specific CLI boundary, independent
  non-official report, source-result immutability, deferred scientific imports,
  and two-layer seed42 reference exclusion.
- Verification: a real GPT-5.5 medium seed31415 B1 result passed both hard gates
  and all four scorers; real HF pulls contained 0 seed42 reference files and 6
  seed31415 reference files; the full suite passed `2112 passed / 4 skipped /
  22 deselected`; wheel/sdist strict checks passed and contained no task files.
- Implementation commit: `573c760`

## Restrict official result submissions to seed42

- Problem: `asibench submit` documented the private-reference seed42 workflow
  but accepted seed31415, unknown-seed, and mixed-seed result directories; the
  optional `--benchmark-repo` value was unvalidated provenance rather than an
  enforcement boundary.
- Resolution: validate every result `instance_id` before creating the bundle,
  reading credentials, or contacting Portal; accept only the seed42 suffix and,
  when supplied, the seed42 alias or canonical Hugging Face repository ID.
- Prevention: regression tests cover seed31415, unknown and mixed seeds,
  repository mismatch, both valid seed42 repository spellings, and failure
  before authentication or bundle creation.
- Verification: focused submission/scoring/documentation tests passed `37
  passed`; the full offline suite passed `2119 passed / 4 skipped / 22
  deselected`; wheel and sdist passed `twine check --strict` and contained no
  task or reference files.
- Implementation commit: `1f3d6bc`

## Publish the ASI-Bench arXiv link

- Problem: the README carried the paper title but did not link readers to the
  published manuscript.
- Resolution: add the arXiv paper as a header badge, a dedicated Paper section,
  and a Resources-table entry.
- Prevention: README regression coverage locks the exact title and arXiv URL.
- Verification: focused documentation/packaging tests passed `12 passed`; the
  full offline suite passed `2120 passed / 4 skipped / 22 deselected`.
- Implementation commit: `c7dbf6e`

## Gate contributed-task difficulty on B3 and B4

- Problem: the difficulty check applied one default ceiling of 50 to every
  prompt level, although task acceptance should constrain only the low-guidance
  B3/B4 conditions and leave B1/B2 unrestricted.
- Resolution: record B1/B2 as `RECORDED`, gate only B3/B4 at a strict mean-score
  ceiling below 40, reject CLI thresholds above 40, and restrict catalog
  flagging to B3/B4.
- Prevention: terminal, JSON, Markdown, CSV, CLI, persistence, catalog, template,
  and documentation tests cover 100-point B1/B2 acceptance, 39.9-point B3/B4
  acceptance, the strict 40-point failure boundary, and threshold bypass
  rejection.
- Verification: focused policy tests passed `173 passed`; the full offline suite
  passed `2123 passed / 4 skipped / 22 deselected`; wheel and sdist passed
  `twine check --strict` and contained no task or reference files.
- Implementation commit: `d5bd2d6`

## Release evaluator-only scorer runtimes in asibench 0.1.4

- Problem: `asibench==0.1.3` and the corresponding public task tree could not
  import six advertised seed31415 scoring contracts because their sandbox or
  task runtime dependencies were absent from the published artifacts.
- Resolution: release the public submission sandbox and four evaluator-only
  task runtimes introduced in `f77c6929`, then bump the package and runtime
  versions from 0.1.3 to 0.1.4 without changing dependency pins.
- Prevention: public policy tests import all six scorers using only public
  files, enforce exact helper allowlists, reject `generate_gt` imports, and scan
  evaluator runtimes for generator and hidden-reference symbols.
- Verification: the locked Python 3.11 offline suite passed `2128 passed / 2
  skipped / 22 deselected`; `asibench-0.1.4` wheel and sdist both passed
  `twine check --strict`.
- Release preparation commit: `5b172fff`

## 2026-08-26 — BenchFlow seed31415 scorer adapter

- Problem: BenchFlow had no stable single-attempt interface for feeding
  already-materialized seed31415 prediction artifacts into the public scorer,
  and could not receive machine-readable ScoreDetail/provenance output.
- Resolution: add `asibench benchflow-score` and the JSON manifest adapter.
  It validates the seed, instance suffix, reference location, task contract and
  optional scorer revision; hashes prediction artifacts; and emits gate/scorer
  details, status, revisions, framework version and harness/model/effort
  provenance. It never generates instances, imports `generate_gt.py`, or
  accepts seed42.
- Verification: adapter/policy/local-scoring tests passed `30 passed`; the
  complete public offline suite passed `2133 passed, 2 skipped, 22 deselected`;
  the built wheel contains `ai4sci_bench/benchflow.py`.
- Implementation commit: `a209722d`

## 2026-08-27 — Bind BenchFlow scores to strict run evidence

- Problem: `asibench run` returned zero after persisting failed agent attempts,
  and the BenchFlow scorer trusted a caller-supplied prediction directory
  without independently reading its ASI-Bench run result.
- Resolution: add opt-in `run --fail-on-agent-error`, evaluated from the final
  retry attempt after all evidence is saved; require a schema-v2 BenchFlow
  manifest to bind `run_result` to its persisted output directory and report
  separate attempt and evaluation statuses plus both artifact hashes.
- Prevention: CLI tests cover non-zero strict exits and recovered retries;
  adapter tests cover missing/mismatched run evidence and failed attempts.
- Verification: focused CLI/BenchFlow/reporting tests passed `163 passed`; the
  complete public suite passed `2138 passed, 2 skipped, 22 deselected`; the
  previous real seed31415 B1 failure is now reported as `attempt_failed` while
  retaining `evaluation_status=completed` and scorer details.
- Implementation commit: `77c394d`

## 2026-08-27 — Preserve API provenance and clarify BenchFlow status

- Problem: persistence path sanitization restarted its absolute-path match at
  the second slash of an HTTP(S) URL, turning endpoints such as
  `https://api.apexin.ai/v1` into `https:/<abs_path>`; BenchFlow also exposed
  the ambiguous raw attempt value `failed` alongside scorer completion.
- Resolution: protect complete HTTP(S) URL spans while redacting host paths in
  surrounding text, and normalize attempt outcomes to `execution_failed`,
  `execution_timeout`, and `execution_incomplete` while retaining independent
  `evaluation_status`.
- Prevention: persistence tests cover API endpoints, URL paths/query strings,
  mixed URL/host-path text, and final result provenance; BenchFlow tests cover
  failed, timed-out, and incomplete execution states.
- Verification: focused regressions passed `18 passed`; the complete public
  suite passed `2144 passed, 2 skipped, 22 deselected`; the real seed31415 B1
  failure now reports `attempt_status=execution_failed` with
  `evaluation_status=completed`.
- Implementation commit: `db57b7a`

## 2026-08-29 — Native pi/opencode PR audit fixes

- Problem: PR #4's two native adapters validated raw constructor arguments
  before resolving `api_key_env`/`api_base_env`, and persisted provenance could
  expose credential values. The PR also lacked live CLI evidence.
- Resolution: resolve environment-backed credentials before validation and
  native-provider selection; recursively redact credential config keys while
  retaining endpoint values; record verified CLI baselines and document live
  `--version`/`--help` probes.
- Verification: focused additions passed `11 passed`; full public suite passed
  `2239 passed, 2 skipped, 22 deselected`; npx probes returned pi `0.84.3` and
  opencode `1.17.15` with the documented flags. Docker daemon was available,
  but no global CLIs were installed, so no fake container smoke result was
  recorded.
- Commits: implementation and tests are in the local public branch; parent
  submodule remains local and unpushed.

## 2026-08-31 — Native adapter real Docker and seed31415 closure

- Problem: mock tests did not reveal that Docker installed pi through a
  floating npm tag (`0.74.2`), reused stale agent images after CLI/base changes,
  passed OS prompts in argv, used Node 20 although pi `0.84.3` requires Node
  `>=22.19`, installed task packages as an unprivileged build user, and dropped
  native token cost from produce-only result JSON.
- Resolution: pin pi `0.84.3` and opencode `1.17.15`; use Node 22 and Docker
  stdin; bind agent image tags to the base schema and exact install command;
  install overlays as root then restore agent ownership/non-root runtime; carry
  `CostInfo` through the produce-only branch.
- Verification: native/OS regression passed `218 passed`; the built wheel
  contains both native adapters and trajectory extractors. Real Docker
  runs used Node `22.23.2`, exact CLI versions, API-key env injection and image
  SHA provenance. Formal seed31415 B1 completed for both adapters: Pi used 5
  turns/22 tools/74,157 tokens, OpenCode used 11 turns/21 tools/95,728 tokens;
  both produced the required files and scored `93.02/100` locally. Persisted
  results contain no API key.
- Prevention: never use floating CLI tags in evaluator images; every overlay
  cache key must include its parent image and installation recipe; real Docker
  acceptance is required when a native CLI schema changes.
- Implementation commit: `8bc8b8e405d1e97133d26fea7eab425249f5f0c4`.

## 2026-09-03 — Runtime LLM/VLM Judge API routing

- Problem: public Gemini-based scorers could use provider-default credentials,
  but users could not safely select a custom Judge endpoint, key variable, and
  protocol from `score` or `run-score`; gateway secrets risked being placed in
  task configuration or command-line JSON.
- Resolution: add `--judge-api-base`, `--judge-api-key-env`, and
  `--judge-api-protocol` (plus `ASIBENCH_JUDGE_*` equivalents) to public scoring
  entry points. Text, VLM, BenchFlow, and the CMOS custom scorer now share one
  runtime resolver for native providers and OpenAI-compatible gateways such as
  TokenRouter.
- Prevention: only an environment-variable name crosses the CLI; secrets are
  redacted from errors, retry logs, raw responses, and reports. `run-score`
  removes dedicated Judge credentials from the agent-run child without
  disabling an evaluated agent's own dotenv behavior.
- Verification: focused Judge/local-score/BenchFlow tests passed `234 passed`;
  the complete offline suite passed `2305 passed, 2 skipped, 22 deselected`
  after rebasing onto `db94593`.
- Implementation commit: `d64663a`.

## 2026-09-04 — BenchFlow workflow migration

- Problem: the standalone `JiangyuZhou1/ASI-Bench-benchflow` repository carried
  a complete operator workflow, while the public main branch documented only
  the manifest boundary. Directly replacing files from the fork would also
  remove newer Judge routing, run-score, and harness-isolation changes.
- Resolution: confirm that main already contains the BenchFlow schema-v2
  adapter, strict attempt-status handling, CLI, and tests, then migrate only
  the missing end-to-end README workflow. Adapt the clone URL to the canonical
  repository and retain the newer runtime Judge API contract.
- Verification: `tests/test_benchflow.py tests/test_cli.py` passed `152`; CLI
  help and the persisted result/output naming contract were checked against
  the documented commands.
- Prevention: when importing from a long-lived fork, compare commit ancestry
  and feature-specific trees rather than copying the fork wholesale.
- Documentation commit: `ac8bd96`.

## 2026-09-04 — BenchFlow guide split

- Problem: the complete BenchFlow workflow made the top-level README much
  longer than the surrounding feature summaries.
- Resolution: keep a short BenchFlow overview and link in README, move the
  installation, run, manifest, scoring, status, retry, and Judge API details to
  `docs/guide/benchflow.md`, and add the guide to the documentation index.
- Verification: all local Markdown links from README and the guide index
  resolve; BenchFlow/CLI regression tests passed `152`.
- Prevention: README should provide feature discovery and route detailed
  operational workflows to focused guides.
- Documentation commit: `b068978`.
- Follow-up: move the short README overview out of `Contribute a task` and
  place it under Quick start → `Score locally or submit`, where the scoring
  integration belongs. Rename the heading to `BenchFlow integration`.
- Follow-up commit: `547d196`.

## 2026-XX: Harness home isolation (claude host HOME, kimi per-instance home)

- Problem: in host-side run modes (`--sandbox none|task|linux_ns`) the Claude
  Code CLI shared the ambient `HOME` across sequentially executed instances
  (session transcripts, `~/.claude.json` project history, auto-memory, user
  hooks/MCP all visible), and `kimi_code_cli` reused one adapter-level
  `KIMI_CODE_HOME` (with `sessions/` + `logs/`) for every instance, both on
  the host and via the `--sandbox os` rw mount. Only the OS-sandbox path was
  inherently safe (one-shot `--rm` container, fresh `HOME=/home/agent`).
- Resolution: `claude_code_cli` now builds a per-run isolated HOME under
  `.ai4sci-bench/claude_home/<run_key>/` copying only `.credentials.json` +
  `settings.json` (same surface as the OS-sandbox auth mounts) and forcing
  `HOME`/`USERPROFILE`/`CLAUDE_CONFIG_DIR` when `tool_mode != unrestricted`;
  `kimi_code_cli` now keys auto-generated homes by `run_key` under a temp
  root removed at teardown (explicit `kimi_home` stays shared by choice);
  `safe_run_key()` was hoisted into `subprocess_base.py` and codex refactored
  onto it (behavior-equivalent).
- Verification: 9 new unit tests (auth mirroring, memory-surface exclusion,
  per-run keying, `CLAUDE_CONFIG_DIR` override, unrestricted opt-out, kimi
  per-instance env/mounts, teardown); full suite `uv run pytest -q` →
  2258 passed, 2 skipped.
- Prevention: whenever a new harness adapter is added, enumerate every state
  directory it writes (sessions, history, memory, config) and decide per
  directory whether it must be per-run, shared-by-explicit-choice, or
  ephemeral; prefer mirroring the OS-sandbox auth surface so host and
  container runs stay comparable.
- Review pending: upstream issue #5, PR #6 (branch
  `task-harness-home-isolation`, implementation commit
  `12b62b2cbfe5d54d18aae454ac405f13b91d1467`).

## 2026-09: PR #6 execution-level isolation hardening

- Problem: PR #6 keyed Claude homes only by `instance_id + prompt_level`, left
  them on disk, used collision-prone truncated directory names, and initialized
  Kimi's temporary root without synchronization. Normal orchestrator runs also
  never invoked adapter teardown.
- Resolution: add a unique Claude execution root with teardown cleanup, append
  a SHA-256 suffix to sanitized run keys, lock Kimi root creation/cleanup, and
  guarantee orchestrator teardown on success and failure.
- Verification: added regressions for repeated identical run keys, sanitized-key
  collisions, concurrent Kimi first use, and exceptional teardown. Targeted
  adapter/runner tests passed `423`; the full offline suite passed `2263`, with
  `2 skipped` and `22 deselected`.
- Prevention: isolation identities must include both execution and instance
  scope; cleanup code is ineffective unless the workflow owner invokes it in a
  `finally` block; predictable truncated names require a content hash.
- Implementation commit: `bda8c3f`.
- Review follow-up: Copilot correctly identified that a host
  `CLAUDE_CONFIG_DIR` containing `~` was not expanded. Use `expanduser()` and
  cover credential discovery through a user-relative config path; also remove
  the unused Kimi test import. Targeted tests passed `424`; the full offline
  suite passed `2264`, with `2 skipped` and `22 deselected`.
- Review follow-up commit: `1bf84a0`.

## 2026-09: PyPI 0.1.5 release

- Problem: PyPI remained at `0.1.4` while `main` already contained the new
  BenchFlow adapter, Judge API routing, execution isolation hardening, and
  updated BenchFlow documentation.
- Resolution: bump the package and lockfile versions to `0.1.5`, validate the
  complete offline suite, and publish through the repository's tagged GitHub
  Release workflow.
- Verification: `uv run --project . python -m pytest -q` passed with `2305`
  tests, `2 skipped`, and `22 deselected`; `uv build` and strict Twine checks
  passed for both distributions.
- Prevention: publish from an annotated version tag only after the locked test
  suite and distribution metadata checks pass.
- Version bump commit: `1dc79cd`.

## 2026-09: Results-below-expectations troubleshooting

- Problem: user documentation listed `direct_llm` alongside agentic adapters
  without explaining that it is a single-turn, no-tool baseline, so users could
  mistake an unsuitable harness choice for unexpectedly weak model performance.
- Resolution: add a dedicated troubleshooting guide that separates execution,
  evaluation, and genuine low-score cases; explains when iterative CLI agents
  are a better fit; and covers artifacts, provenance, environment, timeout,
  transient API failures, and repeated-run variance. Link it from README,
  Getting Started, and the guide index.
- Verification: the documentation regression test passed and all relative
  Markdown links under README/docs resolved locally; the full offline suite is
  run before integration.
- Prevention: keep adapter capability recommendations tied to regression tests
  so documentation changes when harness behavior or public adapter names do.
- Implementation commit: `147ac23`.

## 2026-09: Agentic difficulty-evidence policy

- Problem: `difficulty-check` silently defaulted to the single-turn
  `direct_llm` adapter (and its legacy default model), allowing an otherwise
  easy iterative task to appear difficult because the baseline could not
  inspect, execute, and repair its work. Task Draft validation checked only the
  OS sandbox and reports did not expose complete effective agent provenance.
- Resolution: require explicit position-matched `--agent` / `--agent-config`
  values with a model; require at least one supported multi-turn harness;
  retain `direct_llm` only as an optional additional baseline; reject
  direct-only Task Draft evidence before authentication; and record agent,
  model, effort, measured CLI/adapter version, framework version, and sandbox
  in difficulty reports. Update the Codex CLI and Codex judge defaults to
  `gpt-5.6-sol` and synchronize the CLI/package version display with 0.1.5.
- Verification: policy, report, submission, adapter, judge, documentation, and
  static-validation tests passed `682`; the full offline suite is run before
  integration.
- Prevention: difficulty evidence must identify both model and harness, and a
  no-tool/single-turn baseline must never be the sole formal difficulty gate.
- Implementation commit: `47bf572`.

## 2026-09: Fix MPSC evaluator solution construction

- Problem: `math.mpsc_safety_filter` raised `TypeError: MPSCSolution() takes no arguments`
  because the public evaluator runtime declared fields on `MPSCSolution` without
  generating an initializer, while all solver paths instantiate it with values.
- Resolution: annotate `MPSCSolution` with `@dataclass`; add a regression test
  that imports the public runtime and constructs the value object using its
  keyword fields.
- Verification: the MPSC public-policy and regression tests pass (`20 passed`);
  full offline suite is run before integration.
- Prevention: whenever evaluator runtimes expose typed result records, test both
  construction and at least one scorer path instead of checking only syntax or
  class presence.
- Implementation commit: `5b40ad0`.

## 2026-09: Difficulty-check terminal progress

- Problem: `asibench difficulty-check` could spend hours inside an agent run
  without visible terminal feedback, leaving authors unable to tell which
  task, harness, prompt level, or instance was currently executing.
- Resolution: add an optional orchestrator instance-progress callback and use
  it to render durable global progress-bar lines before and after every
  difficulty instance. The output identifies task, agent/model preparation,
  B1–B4 level, instance ID, percentage, score, and failed/timeout state.
- Verification: add CLI regressions for level-aware 0–100% output and a runner
  regression for start/completion callbacks; the full offline suite passed
  `2316` tests, with `2 skipped` and `22 deselected`.
- Prevention: long-running author workflows must emit a visible stage before
  blocking external work begins, and progress hooks must remain optional so
  library callers and other CLI commands keep their existing output.
- Implementation commit: `b70cd1c`.

## 2026-09: Replace GPT-5.4 Judge configuration

- Problem: several formal image/text Judge contracts still selected GPT-5.4,
  while the requested review standard is GPT-5.5 at medium reasoning effort.
- Resolution: update all seven GPT Judge task contracts to `openai/gpt-5.5`
  with `reasoning_effort: medium`; add GPT-5.5 model resolution and forward
  reasoning effort through generic text/VLM and CMOS custom Judge paths.
- Verification: targeted Judge, CMOS, NNLS, and public-policy tests passed
  (`105 passed`); full offline suite passed `2320`, with `2 skipped` and
  `22 deselected`.
- Implementation commit: `97fef0e`.

## 2026-09: Remove remaining GPT-5.4 runtime references

- Problem: after migrating formal Judge contracts, CLI examples, compatibility
  mappings, and API test fixtures still referenced GPT-5.4, allowing accidental
  use of the retired Judge model.
- Resolution: migrate those runtime/test references to GPT-5.5 and remove the
  GPT-5.4 model aliases while retaining the existing GPT-5.5 OpenRouter alias.
- Verification: targeted CLI, Judge, CMOS, batch-record, issue-fix, and model
  API tests passed; the full offline suite passed `2320`, with `2 skipped` and
  `22 deselected`.
- Implementation commit: `ee139f0`.

## 2026-09: Synchronize framework version and Codex effort levels

- Problem: run metadata hard-coded framework version `0.1.3` while the package
  and CLI reported `0.1.5`; Codex CLI validation also rejected the installed
  CLI's supported `ultra` effort level.
- Resolution: derive metadata version from `ai4sci_bench.__version__` and add
  `ultra` to Codex adapter validation and regression coverage.
- Verification: targeted metadata, adapter, difficulty, and integration tests
  passed; `asibench --version` and metadata both report `0.1.5`.
- Implementation commit: `c653ec4`.

## 2026-09: Migrate Gemini scoring judges to GPT-5.5

- Problem: formal task scoring still selected Gemini for text and image Judge
  paths, and the generic Judge fallback defaulted to Gemini.
- Resolution: migrate all formal Gemini Judge contracts to `openai/gpt-5.5`
  with `reasoning_effort: medium`; change the generic Judge default likewise.
- Verification: public-policy coverage rejects Gemini Judge configs and checks
  all GPT Judge configs; targeted Judge and policy tests pass.
- Implementation commit: `8e474b7`.

## 2026-09: Add selected-agent overlays for custom task images

- Problem: `runtime.dockerfile` returned the task image directly, so formal task
  Dockerfiles had to bundle agent CLIs and could not provide agent-specific
  cache identities; the CMOS image also lacked regression coverage for Linux
  host UID remapping.
- Resolution: treat a custom Dockerfile as a reusable task base, layer only the
  selected agent CLI on top, include the task-base identity and exact agent
  install command in the overlay cache key, and keep no-agent runs on the task
  base. Remove bundled Claude/Codex CLIs from the CMOS base image.
- Verification: `218 passed, 2 skipped, 1 deselected` across the OS sandbox,
  pi, and opencode suites; the Docker integration test built the CMOS base and
  pi overlay, then passed ngspice, Python-package, pi CLI, agent-isolation, and
  UID/GID `12345:12345` HOME-write probes.
- Prevention: custom task Dockerfiles must provide task dependencies only;
  agent installation belongs to framework-managed, agent-keyed overlays with
  both mock coverage and an opt-in Docker runtime probe.
- Implementation commit: `361f093`.

## 2026-09: Close custom task image PR policy and CI gaps

- Problem: the CMOS `Dockerfile.os` was rejected by the fail-closed public-task
  policy, while its real UID-remapping and agent CLI probes were excluded from
  the required CI workflow; cache tests also did not directly prove that a
  Dockerfile content change invalidates the selected-agent overlay.
- Resolution: add an exact task-level public runtime allowlist, require it to
  match every formal `runtime.dockerfile` declaration, add the focused Docker
  integration suite to `CI required`, and cover both Dockerfile-content cache
  invalidation and the original Claude Code executable smoke path.
- Verification: public policy, task image, and CI workflow tests pass; the
  Docker integration suite validates pi and Claude Code overlays separately.
- Prevention: formal task runtime files must be explicitly allowlisted, and any
  custom-image agent regression must have both offline cache coverage and a
  required Linux Docker smoke test.
- Implementation commit: `41e5615`.

## 2026-09: Distinguish scorer failures from scored zeros

- Problem: the MPSC scorer caught trusted bundle/runtime failures and submitted
  controller failures under the same `setup_error` shape, so infrastructure
  failures could leave `scorer_error_count` at zero and depress aggregate scores
  as if the submission had legitimately scored zero.
- Resolution: classify MPSC evaluator/runtime/reference failures with
  `scorer_internal_error: true` while retaining submitted-code failures as
  scored-zero submission errors. Upgrade the seed31415 local report to schema
  v2, mark invalid evaluations with `final_score: null`, and exclude them from
  aggregate numerators and denominators while preserving legitimate zero scores.
- Verification: focused scoring, CLI, BenchFlow, policy, and fault-isolation
  tests passed (`197 passed`); the full offline suite passed `2350` tests, with
  `2 skipped` and `24 deselected`.
- Prevention: custom scorers must identify failure ownership explicitly, and
  aggregate reports must never serialize evaluator failures as numeric scores.
- Implementation commit: `68b599d`.

## 2026-09: PyPI 0.1.6 release

- Problem: the published `0.1.5` release predated the MPSC solution-constructor
  fix and the distinction between evaluator failures and legitimate zero scores,
  so Issue #2 did not yet have a complete released validation target.
- Resolution: synchronize the package, runtime, lockfile, and version regression
  at `0.1.6`, then publish the current public scorer contracts through the
  repository's tagged GitHub Release workflow.
- Verification: the full locked offline suite passed `2350` tests, with `2
  skipped` and `24 deselected`; wheel and sdist passed strict Twine checks; both
  CLI entry points reported `0.1.6`; and all six Issue #2 custom scorers imported
  against the built wheel in an isolated `asibench[full]` environment.
- Prevention: release validation must compare both package metadata and runtime
  `__version__`, and exercise public task scorers against the built wheel before
  publishing.
- Version bump commit: `80b0968`.

## 2026-09: Preserve model-call lifecycle evidence

- Problem: exported trajectories retained visible messages and tool events but
  did not preserve completion lifecycle boundaries, transport/process evidence,
  retry relationships, or empty-stream coverage gaps, so a non-empty-message
  count could be misread as a zero-percent model-call empty-response rate.
- Resolution: add versioned, adapter-aware model-call records and redacted raw
  sidecars; link provider and benchmark retries; restore records through the
  result loader; and add denominator-aware per-result and aggregate reporting.
  Codex agent turns are retained as coverage evidence but are not treated as
  provider completions unless an explicit provider boundary is emitted.
- Verification: the focused observability suite passed `25` tests; the full
  locked offline suite passed `2375` tests, with `2 skipped` and `24 deselected`.
- Prevention: call metrics must remain separate from visible trajectory metrics,
  unknown or malformed coverage must fail closed, and a numeric empty-response
  rate is valid only when every attempted-call boundary and content state is
  observable.
- Implementation commit: `43cf4ce`.

## 2026-09: Require trusted sandbox timeout markers

- Problem: OS and Linux namespace adapters classified any log containing the
  words `timed out` as a benchmark timeout, so successful agent output or an
  ordinary failed command could be mislabeled as `RunStatus.TIMEOUT`.
- Resolution: require a failed sandbox result plus the complete framework-owned
  timeout marker, and apply the shared check to every OS-capable CLI adapter.
- Verification: focused adapter regressions passed `474` tests; the full locked
  offline suite passed `2385` tests, with `2 skipped` and `24 deselected`.
- Prevention: timeout tests must cover genuine runner markers, successful logs
  containing timeout language, and malformed or unrelated failure messages.
- Implementation commit: `cc01f47`.

## 2026-09: Repair formal task scorer runtime inputs

- Problem: the Levin scorer looked for trusted training data in agent outputs,
  its search result type lacked the decorator required by its positional
  constructors, and the CMOS scorer still loaded the removed monolithic
  `task.yaml` before building its ngspice image.
- Resolution: read Levin training data from the reference bundle, make
  `SearchResult` a dataclass, and load CMOS metadata through the framework's
  split-aware `TaskLoader`.
- Verification: three regression tests failed before the fixes and passed
  afterward; public-policy and CMOS Judge tests passed, and the full locked
  offline suite passed `2388` tests, with `2 skipped` and `24 deselected`.
- Prevention: task scorer regressions must keep trusted inputs separate from
  submission outputs and exercise task runtime metadata through the same
  loader used by the framework.
- Implementation commit: `1dd3922`.

## 2026-09: Preserve Judge temperature for compatible endpoints

- Problem: LiteLLM treated an explicit OpenAI-compatible GPT-5.5 endpoint as
  native OpenAI and rejected the task's `temperature: 0` locally before any
  HTTP request, even though the configured gateway accepts that parameter.
- Resolution: centralize Judge temperature handling and send it in the request
  body for explicit `openai/` compatible endpoints, while retaining the normal
  top-level parameter and LiteLLM validation for native provider calls. Apply
  the same behavior to text, VLM, and both CMOS Judge paths.
- Verification: the actual LLM Judge completed successfully against the
  configured compatible endpoint with GPT-5.5, medium reasoning, and zero
  temperature; focused tests passed `105` tests, and the full locked offline
  suite passed `2390` tests, with `2 skipped` and `24 deselected`.
- Prevention: endpoint-override tests must assert the final LiteLLM request
  shape for every Judge implementation and separately preserve native-provider
  validation coverage.
- Implementation commit: `587b550`.

## 2026-09: Add bounded process-isolated local scoring

- Problem: public seed31415 scoring remained serial even when `run-score` used
  multiple agent workers, while custom scorers mutate process-global state and
  therefore cannot safely share concurrent threads.
- Resolution: add `score --parallel N` with full-batch preflight, fresh spawned
  processes per result, stable result ordering, atomic coordinator-owned report
  writes, fail-closed worker errors, Judge-secret redaction, cancellation
  cleanup, and cross-process Docker image build locking. Keep each result's
  gate/scorer/retry/Judge sequence serial, and place a phase barrier between all
  `run-score` agent jobs and per-repetition scoring to prevent nested fan-out.
- Verification: concurrency, process isolation, source ordering, serial parity,
  Judge override/redaction, crash/start failure, atomic report, CLI forwarding,
  and Docker lock regressions passed; the broader focused suite passed `427`
  tests with `1 skipped` and `2 deselected`; the full locked offline suite
  passed `2411` tests with `2 skipped` and `24 deselected`.
- Prevention: result-level concurrency must use isolated processes, preserve a
  single workflow-wide budget, never parallelize work inside a result
  implicitly, and classify all lost worker results as evaluator failures rather
  than valid zero scores.
- Implementation commit: `c908aed`.

## 2026-09-21 — CAD MCP 三项离线兼容修复

- Commit: `d697881` (`fix: add reproducible CAD MCP compatibility repairs`).
- 问题：SketchUp 的 FastMCP 构造参数不兼容；Fusion stdio 并非标准 MCP；
  CadQuery 同时存在 src 布局导入、Pydantic v2 和自定义 stdio 协议问题。
- 解决：独立上游副本修复，revision/SHA-256 固定补丁与依赖快照；提供先预检所有
  checkout 再应用的 helper；不改凭据配置，不捆绑软件。4 个 live 回归由红转绿，
  各发现 10 个工具；离线项目相关测试 16 passed。完整回归 2415 passed、2 failed，
  两失败在未修改 origin/main 同样复现（macOS linux_ns、临时路径假定），不是全绿。
- 避免复发：从安装包而非源码 cwd 验证导入；测试完整握手/列表/错误后存活；
  区分源代码生成与真实 CAD 执行；原始 CRLF patch 通过局部 attributes 保留字节。
  原始 smoke 安装和历史失败报告保留，不以 patched 本地通过冒称上游已修复。

## 2026-09-24: Preserve evaluator failures and local scorer inputs

- Problem: Judge transport failures lacked `scorer_internal_error`, so local
  scoring counted evaluator outages as valid zeros; local re-scoring also
  passed only persisted outputs to scorers and omitted immutable instance
  `data/` files available during the original run.
- Resolution: standardize evaluator failure kinds, exclude all evaluator
  failures with `evaluation_invalid` and a null final score, and construct a
  fresh per-job scorer workspace from validated instance data plus persisted
  outputs without mutating either source tree.
- Verification: focused Judge/local-scoring/fault-isolation/BenchFlow tests
  passed `101` tests; the full locked suite passed `2424` tests with `2 skipped`
  and `24 deselected`; bytecode compilation, distribution build, and
  `git diff --check` passed. Ruff was unavailable in the locked environment.
- Prevention: local-scoring regressions must cover serial and spawn-parallel
  data staging, bare and `data/`-prefixed input declarations, symlink and input
  overwrite rejection, legitimate submission zeros, and every canonical
  evaluator failure kind.
- Implementation commit: `b8dc871`.

## 2026-09-24: Validate all public task staging contracts

- Problem: local scoring treated every `input.files.name` as a literal path;
  expanded declarations such as `pairs/<ii>/source.npy` do not exist as
  literal files after instance materialization. The Max-3-SAT public scorer
  also classified a missing submitted `solver.py` as an evaluator failure.
- Resolution: score jobs now require only a declared, safe instance `data/`
  root and copy its complete materialized tree into the isolated scorer
  workspace. Missing contestant artifacts are explicitly `submission_error`
  results and remain valid scored zeros.
- Verification: all 60 formal seed31415 task metadata/evaluation contracts
  loaded, all custom scorer imports and scorer registrations passed, all 60
  real instance/reference bundles staged successfully, and a parallel empty
  submission smoke scored all 60 with `scorer_error_count: 0`. The full locked
  suite passed `2426` tests with `2 skipped` and `24 deselected`; `uv build`,
  bytecode compilation, and `git diff --check` passed.
- Implementation commit: `59b9b34`.

## 2026-09-29: Score runs that produced no files instead of aborting the batch

- Problem: when an agent wrote no files, `asibench run` created no `.outputs`
  directory, and `asibench score` preflight rejected the whole batch, so every
  other result went unscored. The documented contract is that missing
  predictions are submission failures (valid zeros).
- Resolution: preflight accepts a missing `.outputs` only when the result JSON
  itself records no outputs (no `persisted_outputs` with empty code/data files,
  or `persisted_outputs.dir: null`) and stages empty outputs; the task's gates
  and scorers then produce an ordinary zero. A recorded but vanished directory
  is still rejected.
- Prevention: `tests/test_local_scoring.py` covers both recorded-empty forms
  and the vanished-directory rejection.
- Implementation commit: `e7093ae`.

## 2026-09-29: Keep every produce-only retry attempt

- Problem: `_run_and_evaluate` returns early in produce-only mode (`score=False`,
  i.e. every `asibench run`) without setting `attempt`, so `--retries N` saved all
  attempts under the attempt-1 file name; only the last attempt survived on
  disk while the summary reported the best one. Existing retry tests all ran
  with scoring enabled and missed it.
- Resolution: set `attempt` on the produce-only `EvalResult`.
- Prevention: `tests/test_retry.py::TestRetryStrategyAll::test_produce_only_run_keeps_every_attempt`
  (red before the fix, green after); confirmed on AWS with `--retries 3`/`5`
  producing one result per attempt.
- Implementation commit: `0e43cdd`.

## 2026-09-29 – 2026-10-07: MCP end-to-end testing (L0–L2), eight servers and nine fake tasks

- Problem: the MCP catalog survey only proved L0 (`initialize` + `tools/list`).
  Nothing showed that an agent inside `asibench run` really calls a tool and
  uses its result — a scorer sees only outputs and references, so a correct
  answer could have been produced any other way.
- Resolution: two levels on top of L0, with framework scoring unchanged.
  L1: `scripts/mcp/e2e/setup.py` clones each server at a pinned revision into
  its own interpreter (install modes `uv-sync-frozen` / `uv-pip-pinned` /
  `conda-explicit`, optional `host_requirements`) and `smoke.py <id>` calls every
  manifest tool over stdio against a reference computed outside the server
  process. L2: fake tasks in `examples/mcp-e2e-tasks/` (`status: test`, outside
  `tasks/`) run the normal generate → run → score path, and `verify_run.py`
  answers the separate question "did the answer come from the tool?" from the run
  artefacts and the task's `e2e_check.json` (`mcp_connected`, `tool_called`,
  `tool_correct`, `tool_chain`, `answer_from_tool`, `no_bypass`).
- Detail deliberately not duplicated here: upstream defects per server in the
  `e2e_smoke/servers/<id>.py` docstrings, install modes and host requirements in
  `scripts/mcp/e2e/README.md`, task design and tolerances in each
  `generate_gt.py` / `task_eval.yaml`, the `e2e_check.json` format in the
  `verify_run.py` docstring, test layout in TEST.md, result tables in both
  READMEs.
- Servers, tasks and implementation commits: pyscf L1 `77f0929`,
  `pyscf_rhf_energy` `3cb28d5` `1a225f1` `fa977ff` `e7093ae` `02d29cd`,
  `pyscf_bond_stretch` (two-tool chain, image result) `39745b3`, Codex CLI
  evidence `7455855`; arxiv L1 `c0fc58f`, `arxiv_search_snippets` (non-numeric
  answers, live data, web tools available) `1d1e6bc`; jsbsim L1 `747409d`,
  `jsbsim_engine_run` (stateful session, one-of tools) `6015365`; s4 L1
  `b513672`, `s4_grating_spectrum` (answers selected from returned arrays)
  `29ba5a5` and structured-output unwrapping `7b78b11`; psi4 L1 `7a9c5d9`,
  `psi4_opt_freq` (geometry-valued chain) `f4c48de`; rdkit L1 `8dc43e6`,
  `rdkit_conformer` (opaque-pickle chain, tool writes a file) `86905ae`;
  build123d L1 `6c8e727` (42 tools, closed-form CAD references),
  `build123d_plate_measure` (stateful CAD session, path arguments, binary
  artefacts) `73ea607`; gpaw L1 `22cd6b1` `929fa2f` `07899d6`,
  `gpaw_mos2_bandgap` (six-tool `run_id` chain, measured DFT reference)
  `fa97782` `2c85892` `ecde93d` `2ea3e7c`.
- Framework and tooling work this produced: docs condensation `fe3651e`,
  verifier golden snapshot `b47c4f7`, verifier split into `e2e_verify/` with a
  strict spec parser `8ffc458`, `setup.py` installer registry `e24e8fc`, smoke
  scripts into `e2e_smoke/` with a shared runner `cbf026f`, tests regrouped into
  `tests/mcp_e2e/` `9fe5843`, persistence path-scrubbing fixed for shell
  commands `879dfc0`, for tool results `86905ae` and for tool arguments
  `913d94e`, `$HOME` pinned in the MCP E2E tests `81d31e2`, `listed_files` /
  `named_statuses` extractors and a `superset` result match `2c85892` `ecde93d`.
- Verification: every task passed B1–B4 on AWS Linux amd64 with both Claude Code
  (`claude-opus-5-5`) and Codex CLI (`gpt-5.6-sol`, effort medium) with every
  verifier check PASS, and at full local score except `build123d_plate_measure`
  on Codex b1 (396/400: it reported the bolt circle's diameter where the bolt
  hole's was asked for, which the prompt now spells out); per-task dates in
  `examples/mcp-e2e-tasks/README.md`, per-server smoke counts in
  `scripts/mcp/e2e/README.md`. Offline coverage is `tests/mcp_e2e/` (661 tests)
  with the golden snapshot pinning every `verify_one` outcome.

Lessons — evidence and observability:

- Check what the persisted artefacts can show before designing a verifier
  check: a correct result the artefacts cannot show looks exactly like a wrong
  one. Where the evidence is missing, report a coverage gap (WARN), never a
  verdict about the agent.
- `asibench run` destroys evidence two ways — it redacts user-role events (where
  Claude's tool results live) and rewrites host paths to placeholders
  (`<home>`, `<workspace>`, `<run_output_dir>`, `<repo_root>`, `<abs_path>`) —
  and everything it carries must be restored from the trajectory by call id.
  Shell commands were fixed in `879dfc0`, tool results in `86905ae` after a
  returned path made every genuine Codex run fail `tool_correct`, and tool
  *arguments* only in `913d94e`, when the first CAD task's own honest run
  failed `tool_chain` on a scrubbed `import_cad_file(path=…)`. A fix for
  "persistence destroyed X" has to enumerate every X the pipeline carries —
  three rounds for the same root cause. Two corollaries: the trajectory keeps
  only `core.trajectory.KEY_ARG_NAMES`, so that list is the contract for what a
  path check can ever see (Codex kept no MCP arguments at all); and a check must
  compare first and only call a mismatch unobservable when a placeholder caused
  it, otherwise a deliberately scrub-tolerant comparison (rdkit pickles) is
  downgraded to WARN.
- A required tool in `e2e_check.json` is a promise every prompt level must make. The
  build123d B4 run built, gated, measured and exported correctly and still failed
  `tool_called`, because only B1–B3 asked for the written STEP to be read back. Put each
  required tool's output in the shared result contract (`reimported_volume_mm3`) instead
  of in level-specific prose, and test prompt-vs-spec coverage offline.
- A test that hard-codes host-like paths is host-dependent: the MCP E2E fixtures
  use `/home/e2e/...`, which the sanitizer rewrites to `<home>/...` when the
  suite runs as that very user — four tests passed in CI and failed on the AWS
  E2E box. Pin `$HOME` in the tests (`81d31e2`) instead of choosing "unlikely"
  literals.
- Compare a tool's report as a superset of its derivable part (`superset`), and
  count what the run really leaves on disk: gpaw writes a text log per SCF, 24
  files where the chain requires 10.
- Build test streams from what the client really shows. A FastMCP tool with a
  return annotation declares an `outputSchema`, so Claude Code passes
  `structuredContent = {"result": …}` instead of the text block; streams built
  from the text block hid that until an AWS run scored 400/400 with every
  provenance check failing. Equally, probe the real event stream before writing a
  parser (Codex item schema, its own `list_mcp_resources*` calls).
- A test fixture standing in for a production code path must call it: six copies
  of a hand-written persistence helper encoded the same blind spot as the code
  under test.

Lessons — agents and prompts:

- Prompts must name the server and tool agent-neutrally, never `mcp__…`. A
  harness-specific name sent the agent looking for ToolSearch, and
  `claude mcp list` then reported no servers because they arrive by
  `--mcp-config`.
- Check which built-in tools a harness gets in the mode MCP forces:
  `--mcp-config` turns on search mode (Claude WebSearch/WebFetch, Codex
  `web_search`) and under `--sandbox none` the shell has network too, so
  "restricted" assumptions do not hold — that is what `no_bypass` is for.
- With a custom gateway the isolated Codex home copies `auth.json` from
  `$CODEX_HOME` but `config.toml` only from the `codex_home` agent-config key.

Lessons — task design:

- Pick cases where the tool's own numerical method makes the answer unique; a
  physically exact quantity proves nothing about provenance. Keep arguments away
  from the server's defaults so that an omitted one changes the result.
- When the reference must not come from the tool's own backend, the sharpness is
  in the method details: fitting basis and isotope masses for a density-fitted
  program, truncation rule and harmonic count for RCWA, seed and library version
  for a conformer.
- Set scorer tolerances from measured margins, with a gap between correct use and
  the plausible procedure errors (calls in the wrong order, default seed, rounded
  values).
- Scorers stay per-artefact: a broken artefact zeroes only the scorers that read
  it, and "the whole instance is zero" belongs to the hard gate. Never tighten a
  comparator shared by every task to make one test fail — choose a difference
  above its detection floor instead.
- When the solver cannot enter a task runtime, carry a measured table (gpaw,
  `fa97782` `ecde93d`): measure only the dimensions the solver depends on, derive
  the rest with the L1-validated pure functions, record the server's own answer
  for every derived field and assert the derivation against it offline. Fix the
  instance grid *after* measuring — ecut 350 eV stopped on the force criterion
  (fmax 0.0495 vs 0.05) and was dropped — and set the zero-credit tolerance from
  the measured distance between neighbouring instances.
- A discrete ground truth must not sit near a decision boundary, and must not
  depend on anything that is not a function of the instance. gpaw's
  `structure_drift` compares raw Cartesian positions with no minimum-image
  convention, so a denormal Mo y (1e-19) wraps to `+a` and the check flips per
  run (`2ea3e7c`); because verdicts are fail > warn > pass it also moved the
  verdict. Keep such a check out of the reference, choose parameters where it
  cannot reach the answer, and let the generator assert that.
- An L1 reference that reimplements an upstream formula reproduces its bugs: the
  smoke's own drift used the same naive formula, agreed with the server in all
  14 runs and could never flag it. L1 green does not mean a quantity is
  derivable.
- Ask for artefacts the tool replies do not already contain (gpaw's figures exist
  only on disk), and gate only the schema, so an agent that drove the chain but
  could not find the server's directory loses those points, not the instance.

Lessons — smoke tests:

- Probe every tool over stdio before designing tasks: docstrings promise fields
  and behaviour the code does not deliver, and L0 can pass with the backend
  absent (psi4 imports it lazily).
- Classify a deliberately probed defect tri-state — PASS for the correct answer,
  WARN only when the output matches the recognised defect exactly, FAIL for
  anything else — so a changed upstream cannot hide behind a WARN.
- A crash probe must not use the FAIL-on-error call wrapper, since any step can
  be the one that dies; tolerances must be absolute rather than scaled by the
  value; and a defect that depends on scheduling order needs repeated probes (12
  calls before `batch_map`'s `fail_fast` race showed up reliably).
- A tool that computes for minutes needs its own `tools/call` timeout: the
  client default of 300 s turned a correct `run_verified_workflow` result into a
  FAIL on a loaded host (gpaw L1, 22cd6b1). Declare `Smoke.call_timeout`, print
  each step before it starts and record `seconds_by_step` in the report —
  otherwise a 30-minute smoke is indistinguishable from a hung one.
- Reference values from the same library and conda lock agree to ~1e-10 eV
  across aarch64 and amd64 (GPAW plane-wave DFT), so cross-platform drift is not
  what a ground truth has to tolerate — upstream *releases* are (2-3 meV between
  gpaw 25.7 and 26.7).

Lessons — process:

- LiteLLM's Anthropic-to-Responses adapter can flatten `thinking` into text and
  discard `redacted_thinking`, reasoning IDs, and encrypted content even when
  the wire API supports those fields. The lossless path added in `5012c87`
  therefore uses an explicit opt-in bridge with execution-keyed, conversation-
  bound replay envelopes and rejects unknown or unverifiable state before the
  upstream call; compatibility translation must never imply semantic parity.
- Hydrology closure scoring must use one consistent post-step interval. The
  seed31415 `water_table.csv` day-0 row is already after the first update, so
  pairing it with pre-step `theta_init` and day-0 fluxes silently mixes ranges.
  Use row 0 through the final state and accumulate fluxes from day 1; validate
  both datum-shift invariance and a deliberately drifting storage series.

- The golden snapshot pins only the paths the fixtures exercise, so check that a
  branch is covered before relying on it, keep the file byte-identical through
  pure refactors, and never let a test behind `importorskip` feed it. Errors from
  an autouse fixture's teardown are reported as ERROR, not FAILED.
- For a refactor of scripts that only run on a remote host, an AST comparison of
  same-named functions against `git show HEAD:` plus a per-check status diff of
  old and new reports is a cheap and complete parity check.
- After delegating work, grep that the new code is actually reached: a sub-agent
  left `parse_spec` unwired and the suite stayed green.
- Traps that cost a round each: `\b--with\b` never matches (no word boundary
  between a space and `-`); modules loaded with `spec_from_file_location` must be
  in `sys.modules` before `exec_module` or dataclasses break on Python 3.14;
  `AGENTS.md` must stay byte-identical to `CLAUDE.md`
  (`tests/test_ci_workflow.py`).
- Diagnose with reads before re-runs: four rounds on the gpaw drift (run
  directories, the two CIFs, upstream source, a fresh-process `verify_run` on two
  run ids) cost no compute, and the first plausible story (an inherited user-site
  ASE) was wrong. In the VM, never `tail` a `git checkout`: without the delete
  permission it half-fails and leaves the old branch's files under the new HEAD.
- When the VM cannot commit and the base branch moves, develop in a scratch clone
  rebased with a throwaway identity and resolve append-only conflicts (PROGRESS,
  README) by keeping both sides; keep scratch directories under the session home,
  since `/tmp/...` can belong to another session's user.

## 2026-10-03: One upstream checkout shared by several manifest ids

- One repository exposing several tool faces (the ToolUniverse SMCP servers) was
  installed once per manifest id: a 689 MB virtualenv and a 19 MB checkout per
  tool face. `cbeb771` adds an optional `checkout` key, so a group installs into
  `<root>/<checkout>` while each id keeps its own `<id>.mcp.json` whose MCP server
  name is still the id — `mcp__<id>__<tool>` and the verifier golden snapshot are
  untouched. arxiv moved to `~/mcp/tooluniverse` and its smoke still reports
  16 / 8 / 0 (amd64, 2026-10-03), so the move changes nothing observable.
- A group must declare the same repository, revision, python, install mode and
  install fields, and its name must not be another manifest id;
  `host_requirements` is deliberately not part of that check, since it gates the
  id being installed rather than the directory's contents.
- "Installing the second id of a group is idempotent" only holds for
  `uv-sync-frozen`: `uv-pip-pinned` runs `uv venv --clear` and `conda-explicit`
  removes the prefix, so there the second id rebuilds the venv. The docs say so.
- Splitting this across two commits (manifest first, `setup.py` second) breaks
  bisect: `load_manifest` rejects the unknown `checkout` key at collection time
  and the whole `tests/mcp_e2e` suite errors out. Keep a new manifest key and its
  validator in one commit.

## 2026-10-03: alphafold_db and ncbi smokes (L1) on the shared ToolUniverse checkout

- Two more tool faces of the ToolUniverse pin (`33ea0be`), so both manifest
  entries only declare `checkout: "tooluniverse"`: no clone, no second venv.
  Measured on AWS amd64: `alphafold_db` 16 / 7 / 0, `ncbi` 13 / 5 / 0.
- A networked API with no date window to pin needs immutable identifiers plus
  recomputation: compare the payload field by field with a raw query of the same
  endpoint made by the smoke (internal consistency alone passes a stale payload),
  recompute pLDDT from the model file's CA B-factors, recompute MUTAGEN values
  from the AlphaMissense CSV the annotation links to, and address NCBI proteins
  by GI number. Everything that moves with a model or annotation version is
  reported, not asserted.
- Rules this run added (all in CLAUDE.md or the module docstrings): `unwrap` must
  treat the runner's `{"_isError": True}` sentinel as an error, otherwise a server
  that *starts* validating input is reported FAIL; every "upstream should reject
  this" probe needs `check_rejected(..., on_accept=…)` so an upstream improvement
  lands as WARN; an empty E-utilities result set is a legitimate "no hits", not
  "invalid input accepted"; a reference the smoke cannot parse is its own `
  [reference]` FAIL with the dependent comparisons skipped, never a value compared
  against `None`; and an in-band error may only be a WARN when its message
  matches the documented empty state (an HTTP 500 is a FAIL).
- Traps that cost a round each: the model file carries two decimals and truncates,
  so a recomputed pLDDT mean needs a ~0.02 tolerance (P04637: 75.0501 vs a
  reported 75.06) and a residue at a bin edge is a WARN; `esummary`
  `chrstart/chrstop` are 0-based with `start > stop` for the minus strand while
  `gene_table` is 1-based; an isoform model carries the isoform accession
  (`P04637-2`), so asserting the base accession accuses the server; `esummary`
  reports an unusable id in a *third* shape (top-level `error`, empty `uids`);
  and `efetch db=gene retmode=xml` is 34 MB for TP53 — use `rettype=gene_table`.
- A stub-server dry run cannot model the client: it predicted a WARN for
  `auto_query_params` overwriting a caller's `type`, but FastMCP rejects
  undeclared arguments, so over MCP that defect is unreachable and the real run
  PASSed. Only a tri-state check survived the difference.

## 2026-10-07: ncbi_gene_protein_card (L2) and the json_scalars extractor

- Framework: `json_scalars` (+ `canon_scalar`) in `e2e_verify/extractors.py`.
  A REST wrapper keys its record by the identifier that was requested
  (`data.result.<uid>.slen`), so no static dotted path reaches it, and the
  values are identifiers, which `Selector.key` cannot read at all (it goes
  through `numbers()`). One generic extractor plus `match: "member"` covers
  every such field; the call-level checks still use plain paths
  (`data.esearchresult.idlist`, `data.result.uids` — digit strings that
  `numbers()` accepts).
- `canon` runs on agent-controlled values (an answer file, a tool argument), so
  it must be total: a nested list or an object reads as "no value", never
  raises. The first version did `{canon_scalar(i) for i in value}`, and a
  `result.json` holding `[[5053]]` raised `TypeError` inside `verify_one` —
  which has no per-check guard, so one malformed answer would have killed the
  report for every result in the run.
- A scorer must not be more forgiving than the verifier. Upper-casing
  identifiers in the scorer made `np_000268.1` score 100/100 while
  `answer_from_tool` reported "does not come from the tool", because the
  verifier compares with the tool's own spelling. Both are case-sensitive now;
  numbers stay lenient on both sides (`452` == `"452"`).
- The hard gate stays structural (seven keys, usable types) like the other fake
  tasks: an unversioned accession or an absurd length is a wrong field worth its
  own weight, not a zeroed instance.
- Fixtures must be measured, not written from memory: `PLUS_STRAND_TABLE` in
  `test_smoke_ncbi.py` invented a `(plus strand)` marker, which hid that a live
  plus-strand `gene_table` has **no marker at all** — so `GENE_TABLE_LOCUS`
  silently failed to parse CFTR, SOD1 and HBA1. The orientation now comes from
  the coordinates (`from > to` ⇔ minus, as for the esummary locus) and the
  marker is only cross-checked: the smoke WARNs on a contradiction, the
  generator raises.
- Generation asserts every scored field against a second endpoint (`gene_table`
  for the id/locus, protein FASTA and the gene_table's `annotated AA length`
  for the length, Datasets `v2alpha` as a soft check) and raises instead of
  writing an instance. All five curated cases verified live, ~6.5 s each.
  Honest scope: the GI freezes the protein facts, but the band, the `NC_…`
  version and the symbol query's uniqueness track the current NCBI build, so
  generate right before a run.
- Offline: `tests/mcp_e2e` + `test_ci_workflow` `683 passed / 3 skipped`;
  `golden.json` 129 → 148 entries, none changed or removed.
- AWS 2026-10-07: Claude Code and Codex B1–B4 each 4 × 100 (400/400), verifier 4/4.
- Commits: `e4ad4ff` (gene_table strand), `4e1586c` (task + extractor).

## 2026-10-07: alphafold_isoform_profile (L2)

- All three `alphafold_db` tools on one accession: isoform models → summary →
  per-residue AlphaMissense annotation. The input file **lists the entry ids**
  to rank, so an isoform model AlphaFold DB adds later cannot change the answer;
  nothing model-version dependent (pLDDT, URLs, `latestVersion`, the growing
  `structures` list) enters the reference. Ranking and the count above the
  threshold are derived values the verifier cannot attribute to a call, so only
  the scorer judges them.
- Measured before curating: the canonical model is **not** always the longest
  (BAX, BID, CDKN2A, RAC1 have a longer isoform); some proteins have a tied top
  score (CDK2, GSK3B) or a value exactly at 0.9 (BCL2L1); a model carries four
  length-like fields. Generation asserts all of it (distinct lengths, agreeing
  length fields, unique top, nothing at the threshold, every value == the
  AlphaMissense CSV mean, payload ≤ 40k characters) and raises otherwise.
- A bypass pattern must never match the tool's own payload: the bare
  `-aa-substitutions\.csv` matched `source_url`/`amAnnotationsUrl`, so an agent
  pasting the annotation into a heredoc or script to count residues would have
  been judged a bypass. Bulk AlphaMissense file names and URL literals handed to
  `read_csv`/`urlretrieve` replace it; tests pin a pasted payload as not a FAIL.
- The scorer compares the score to printing precision: the tool already serves
  the rounded mean, so a tolerance would accept values the verifier rejects.
- Size cap is 40k characters, not 60k: besides SMCP's 100k truncation, Claude
  Code replaces an MCP result above ~25k tokens with a file pointer.
- AlphaFold DB's front end rejects some User-Agents with HTTP 403 (`probe/1`);
  the generator's `asibench-mcp-e2e-generate/1` passes.
- Offline: `tests/mcp_e2e` + `test_ci_workflow` `752 passed / 3 skipped`;
  `golden.json` 148 → 171, none changed.
- AWS 2026-10-07: Claude Code and Codex B1–B4 each 4 × 100 (400/400), verifier 4/4.
- Commit: `e7a8ab3`.

## 2026-10-03: quantum_espresso MCP E2E, stage L0

- Shipped (`aaa6dd0`): manifest entry + `locks/quantum_espresso-linux-{64,aarch64}.txt`
  (174 packages, `qe=7.5`), `e2e_smoke/servers/quantum_espresso.py` with the
  environment checks and `qe_status`, `tests/mcp_e2e/test_smoke_quantum_espresso.py`,
  and the committed-lock test generalised from psi4 to every conda-explicit
  server × platform. 16 PASS / 0 WARN / 0 FAIL on amd64 and aarch64.
- `mcp` must stay below 2 and the pin is load-bearing, not hygiene: conda-forge
  now resolves `mcp` to 2.x, where `mcp.server.fastmcp` is a stub that raises
  `ModuleNotFoundError` (renamed to `MCPServer`). Check this before pinning any
  server that imports `mcp.server.fastmcp`; rdkit hit the same wall.
- Do not assume a conda `qe` needs MPI coaxing: the openmpi build exec'd
  directly with `QE_NPROCS=1` ran clean on both platforms, empty stderr, no
  `OMPI_MCA_*`. And different conda-forge builds of the same `qe` version gave
  bit-identical energies across amd64/aarch64 — platform is not the drift source,
  the package version is.
- Leaving a work-directory variable unset can beat pinning it: with `QE_WORKDIR`
  unset the server writes under its cwd, which is the smoke's temporary directory
  (self-cleaning) and the checkout under an agent run (gitignored upstream).
- Upstream indexes pseudopotentials by raw `glob` order, so which file an element
  gets is host-dependent. Measured on two hosts: Si got the newest on both, Ag
  diverged. A ground truth that depends on the pick must be generated on the host
  that runs the agent — verify the pick, do not assume the highest version.
- `test_smoke_runner.py::test_smoke_covers_every_manifest_tool[quantum_espresso]`
  FAILs by design at this stage: it greps the smoke module for every
  `expected_tool`, and an L0 smoke only calls `qe_status`. Resolve it when L1
  lands (or by declaring the stage on `Smoke`), not by weakening the grep.

## 2026-10-07: quantum_espresso MCP E2E, stage L1

- Shipped (`c71f663`): all 19 tools in the smoke, 44 PASS / 18 WARN / 0 FAIL
  on the aarch64 VM (62 s) and on AWS amd64 (~2 min), every energy identical
  across the two; `test_smoke_covers_every_manifest_tool[quantum_espresso]` is
  green.
- Fast DFT makes the VM a real test host: with a two-atom cell every pw.x call
  is 2–8 s, so the whole smoke, references included, fits inside one sandbox
  command. Size the L1 workload for that before reaching for AWS.
- A reference for a DFT server can be the same binary, run outside the server
  from our own input: write only the physics the server documents (mass,
  prefix and outdir are ours), parse the text ourselves, compare to the printed
  digit. Prove it discriminates with one mutation (`degauss` 0.01 FAILed all
  twelve DFT checks) instead of trusting a green run.
- Read the parser, not the docstring: `re.search` on a multi-step output
  returns the first step (D11), and a "relax then SCF" workflow ran its SCF on
  the input geometry (D12). Both look correct until a reference prints every
  step.
- Re-run upstream's file scan in the host's directory order and require an
  exact match; compare with the newest version separately. Two checks keep
  "the server is consistent" (FAIL if not) apart from "the defect bit here"
  (WARN, naming the elements: 43 of 69 on the VM, 33 on amd64).
- `check_rejected`'s `in_band` hook gets the raw tools/call result; decode it.
  A tolerance comparison must recurse into dicts or every nested reply FAILs.
