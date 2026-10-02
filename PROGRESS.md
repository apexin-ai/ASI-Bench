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

## 2026-09-29: MCP end-to-end (E2E) fake tasks, starting with pyscf

- Problem: the MCP survey only proved L0 (`initialize` + `tools/list`); no
  catalog server had evidence that an agent inside `asibench run` actually
  calls a tool and uses its result. Scorers only see outputs and references, so
  a correct answer could not show whether it came from the MCP tool.
- Resolution: `scripts/mcp/e2e/` installs servers at pinned revisions with
  their own Python (`setup.py`) and checks real `tools/call` against
  independently computed references (`smoke_pyscf.py`, L1).
  `examples/mcp-e2e-tasks/mcp_e2e/pyscf_rhf_energy` (status `test`, outside
  `tasks/`) runs through the normal generate → run → score path, and
  `verify_run.py` checks connection, tool call, tool result, answer provenance
  and bypass from the run artefacts (L2). Framework scoring is unchanged.
- Lessons: (1) `asibench run` redacts every user-role event in the persisted
  stream-json, which also removes tool results; read them from the trajectory
  (a bare JSON list) by `tool_call_id`. (2) Prompts must name the MCP server
  and tool, never a harness-specific name: B1 with Claude's
  `mcp__pyscf__pyscf_rhf_energy` failed 2 of 4 verified runs (agent looked for
  ToolSearch, then `claude mcp list` falsely reported no servers because
  servers are passed by `--mcp-config`); agent-neutral B1 passed 5/5.
  (3) mcp2pyscf needs Python 3.13 and PySCF prints to stdout; neither broke
  Claude Code 2.1.284.
- Verification: AWS Linux amd64, `claude-opus-5-5`: B1–B4 ×3 11/12 verified
  PASS (the failure was the B1 naming issue), agent-neutral B1 ×5 5/5; smoke
  L0/L1 PASS on amd64 and aarch64; new offline tests in
  `tests/test_mcp_e2e_scripts.py` and `tests/test_mcp_e2e_tasks.py`.
- Implementation commits: `3cb28d5`, `1a225f1`, `fa977ff`, `e7093ae`,
  `02d29cd`.

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

## 2026-09-29: pyscf MCP smoke covers all seven tools (L1)

- Problem: the pyscf smoke checked only `pyscf_rhf_energy` values and the atom
  labels of one `generate_pyscf_geom_input` call; five tools were never called,
  so upstream defects were unknown before designing further fake tasks.
- Resolution: `smoke_pyscf.py` now calls every manifest tool and compares
  against references computed in the smoke process (seeded RDKit + UFF,
  PySCF RHF, PySCF + geomeTRIC). Unseeded server geometries are compared with
  rotation/permutation-invariant distances and energies; only rigid molecules
  are used. Wrong values FAIL; upstream defects are WARN: in-band errors
  (`isError=false`), ignored `basis` in the bond scan, missing optimized energy,
  intermittent `PointGroupSymmetryError` when chaining geometry → RHF for
  benzene, visualize writing into the server cwd while returning a hard-coded
  path, and stdout pollution now attributed per tool.
- Lesson: probe every tool over stdio before designing agent tasks; the tool
  docstrings promised fields (`optimized_energy`) and behaviour (`basis`) that
  the code does not deliver.
- Verification: smoke 19 PASS / 8 WARN / 0 FAIL on Linux aarch64 and on AWS
  Linux amd64 (identical values);
  `tests/test_mcp_e2e_scripts.py` adds PySCF-free tests (parsers, invariants,
  PNG header, WARN/FAIL classification, stub-client plot/visualize, every
  manifest tool called). Full suite on macOS / Python 3.13: only the two known
  environment failures that also occur on unmodified main
  (`test_mimo_accepts_all_four_modes` needs Linux for `linux_ns`,
  `test_kimi_host_env_uses_host_path` assumes a temporary path layout).
- Implementation commit: `77f0929`.

## 2026-09-30: L2 pyscf bond-stretch task with a two-tool chain and an image result

- Problem: L2 evidence covered one tool returning one number. Multi-step tool
  use and image results were untested, and the verifier could only check a
  single tool with a scalar answer. The Claude trajectory extractor dropped
  non-text tool_result blocks, and the persisted stream redacts user events, so
  an image returned by an MCP tool was invisible in the run artefacts.
- Resolution: `examples/mcp-e2e-tasks/mcp_e2e/pyscf_bond_stretch` (scan with
  `run_bond_stretch_calculation_mcp`, then plot the returned data with
  `plot_energy_scan_image_mcp`; reference from seeded RDKit + UFF and PySCF,
  rigid molecules only). `verify_run.py` schema 2 checks every required call,
  JSON-field and image results, `tool_chain` (plot inputs equal the scan
  output) and several answers; schema 1 specs are normalised. The Claude
  extractor records `content_types` and `image_media_types` in tool_result
  metadata without copying image data.
- Lesson: check what the persisted artefacts can show before designing a
  verifier check; a correct but unobservable result looks like a failure.
- Verification: 41 seeds × 2 unseeded upstream runs agree with the reference
  within 1.4e-7 Ha; a simulated run from real server outputs passes all checks
  with full score; offline tests in `tests/test_mcp_e2e_bond_stretch.py`.
  AWS Linux amd64, `claude-opus-5-5`, seed 31415, B1–B4 ×3: local score
  1200/1200 and verifier 12/12 PASS on every check, including the image result
  of the plot call (so the extractor metadata matches Claude Code's format).
  macOS full suite: only the two known environment failures plus the timing
  sensitive `test_parallel_local_scoring_is_bounded_isolated_and_ordered`
  (untouched code path).
- Implementation commit: `39745b3`.

## 2026-09-30: MCP E2E evidence for Codex CLI

- Problem: the pyscf MCP E2E tasks were only proven with Claude Code. With
  `codex_cli` the run and score already worked, but nothing could show that
  the answer came from the MCP tool: `verify_run.py` parsed only Claude
  stream-json, and the Codex trajectory extractor ignored `mcp_tool_call`
  items, so MCP calls were missing from the trajectory as well.
- Resolution: `verify_run.py` parses `codex exec --json` (`mcp_tool_call` items
  named `mcp__<server>__<tool>`, `command_execution` as shell evidence). Codex
  emits no list of connected servers or offered tools, so `mcp_connected` is
  PASS only when every required tool returned a result and WARN otherwise. The
  Codex extractor records MCP calls and results with `tool_call_id`,
  `content_types` and `image_media_types`, without image data.
- Lesson: probe the real event stream before writing a parser. A direct
  `codex exec --json` run against the MCP server settled the item schema, the
  image block shape, and that Codex adds its own `list_mcp_resources*` calls
  under the server name (they must not count as tool calls). It also showed
  that the expected problems (server stdout prints, default MCP timeouts) do
  not occur, so no framework change was made for them.
- Lesson: with a custom gateway, the isolated Codex home copies `auth.json`
  from `$CODEX_HOME` but `config.toml` only from the `codex_home` agent-config
  key; pass both or the provider settings are silently dropped.
- Lesson: `AGENTS.md` must stay byte-identical to `CLAUDE.md`
  (`tests/test_ci_workflow.py`); copy it after every `CLAUDE.md` edit. The
  targeted tests did not include that file and the full suite caught it.
- Verification: `tests/test_mcp_e2e_codex.py` (16 offline tests on event shapes
  from real runs). AWS Linux amd64, codex-cli 0.159.2, seed 31415, both tasks,
  B1–B4 ×1: local score 800/800 and verifier 8/8 PASS on every check, after a
  B1-only pass with the same result. macOS full suite: only the two known
  environment failures once `AGENTS.md` was synchronised.
- Implementation commit: `7455855`.

## 2026-09-30: arxiv (ToolUniverse) MCP smoke (L1)

- Problem: the second MCP E2E server, ToolUniverse's arXiv tools, needs live
  network data and a 700 MB general-purpose server; the L0 survey had shown
  only `initialize` and `tools/list`.
- Resolution: manifest entry `arxiv` (Python 3.12, `uv sync --no-dev`) with a
  launch `env`; `setup.py` now supports launch `env`, extra `uv_sync_args` and a
  `{checkout}` placeholder so non-path CLI flags stay literal.
  `smoke_arxiv.py` compares five searches field by field with raw arXiv API
  queries and the snippet tool with snippets cut from the same PDF version,
  converted with the same MarkItDown; all queries use closed 2011 windows.
  Shared report/call helpers moved to `smoke_common.py`.
- Lesson: a `./.tooluniverse` directory in the server cwd makes ToolUniverse
  load its default profile and ignore `--include-tools`, exposing 2718 tools;
  the default result cache (forever, `~/.tooluniverse/cache.sqlite`) creates
  that directory whenever cwd is `$HOME` and would also hide real calls. The
  launch env disables the cache; `TOOLUNIVERSE_HOME`/`--workspace` must not be
  used. FastMCP 3 ignores the catalog's `FASTMCP_NO_BANNER`.
- Lesson: other upstream defects (WARN): an `OR` query with a date range loses
  the date filter (missing parentheses), `truncated` stays false when the cap
  drops matches, old-style IDs containing `v` break, errors are in-band.
- Verification: 16 PASS / 8 WARN / 0 FAIL on Linux aarch64 (twice) and AWS
  Linux amd64 (same WARNs); offline tests in `tests/test_mcp_e2e_scripts.py`.
- Implementation commit: `c0fc58f`.

## 2026-10-02: L2 arxiv search → snippets task and non-numeric verifier checks

- Problem: the verifier compared only numbers, and the pyscf tasks never had
  an alternative data source. For arXiv the agent has one: `--mcp-config`
  forces search mode, so Claude Code gets `WebSearch`/`WebFetch`, Codex keeps
  `web_search`, and the host shell has network access.
- Resolution: `examples/mcp-e2e-tasks/mcp_e2e/arxiv_search_snippets`: search a
  closed date window, pick the paper with the most authors, fetch snippets for
  two terms from its PDF, report IDs, the pick and per-term snippet counts.
  Counts, not snippet text, are scored because MarkItDown garbles two-column
  PDFs. The reference comes from the raw arXiv API and MarkItDown/pdfminer/
  pdfplumber pinned to the server's lockfile, in `generate --sandbox task`.
  `verify_run.py` gained named extractors (`arxiv_ids`, `term_counts`) with
  `match` = equal/subset/member, member chaining (the snippet call must use a
  paper the search returned, by `arxiv_id` or `pdf_url`), merged answers over
  per-term calls, `bypass_tools`/`suspicious_tools` for non-MCP tool calls
  (WebFetch of arxiv.org FAILs, web search WARNs; Codex `web_search` items are
  now parsed), and `server_tools` (WARN when the server offers more tools, i.e.
  the ToolUniverse workspace trap).
- Lesson: check which built-in tools a harness gets in the mode MCP forces;
  "restricted" assumptions do not hold for `--mcp-config` runs.
- Verification: `tests/test_mcp_e2e_arxiv.py` (33 offline tests); the MCP tool
  returned exactly the reference counts and IDs for three of the five cases;
  `asibench generate --sandbox task` + an oracle `--agent-cmd` run scored
  100/100 while `verify_run.py` failed it for lack of MCP evidence.
  AWS agent runs and the implementation commit ID are added with the results.
