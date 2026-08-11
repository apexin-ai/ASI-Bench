# Progress

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
