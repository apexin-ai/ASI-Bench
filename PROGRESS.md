# Progress

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
