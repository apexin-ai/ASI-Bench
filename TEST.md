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
integration:

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

## Web-only task submission

Task submission tests cover the default official Portal, Portal URL overrides and
normalization, browser opening, headless URL output, removal of CLI upload options,
and absence of the retired task-upload module from the public package:

```bash
uv run pytest -q tests/test_cli_task_submit.py tests/test_endpoints.py
```

## Pull filtering and custom pre-submit paths

Regression coverage verifies that the shared seed42 and seed31415
`tasks/<instance-id>/` Hugging Face layout is imported while root-level instance
directories are ignored, grouped `task pull` /
`task create` CLI commands work, retired top-level aliases stay absent, and a
task-filtered pull ignores unrelated
directories retained in a reused Hugging Face snapshot. It also verifies that
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

## Retired internal surfaces

The public distribution intentionally excludes the former internal orchestration
commands (`batch-run`, `eval`, `quickeval`, `fullrun`, `rerun-flagged`, and
`pipeline`), the local benchmark scoring flags/API, and the old GitHub task-PR
automation. Benchmark `run` is always produce-only and reports its results as
awaiting authenticated submission to the ASI-Bench website for scoring. Keep
that boundary covered with:

```bash
uv run pytest -q \
  tests/test_retired_features.py \
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

## Authenticated website submission

Result submission defaults to `https://asibench.apexin.ai/`, requires a submitter
identity, uploads a draft, and directs the user to the website confirmation page.
Public fixed-seed submission bundles do not require
`framework_task_info.json`; bundle tests lock that this framework-only file
remains optional. The tests mock the upload and never create a real online
submission:

```bash
uv run pytest -q \
  tests/test_submission_bundle.py \
  tests/test_device_login_cli.py \
  tests/test_endpoints.py
```

## Public task catalog policy

When updating formal-task scorers or published examples, validate both exact
allowlists and scan the public tree for GT generators, references, private
solver assets, undeclared artifacts, repository identifiers, and secret-like
content. `config/public_scorers.json` locks all 60 formal scoring contracts, 57
custom scorers, their single approved helper, and the private source revision.
Formal `task_eval.yaml` files may contain only scoring/output contracts and must
never contain `generation`. The separate Example policy locks five full public
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
