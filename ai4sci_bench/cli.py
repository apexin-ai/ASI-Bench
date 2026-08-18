"""CLI entry point for ASI-Bench (``asibench``)."""

from __future__ import annotations

import builtins
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from ai4sci_bench.branding import DEFAULT_PULL_DIR, OFFICIAL_HF_REPO_ALIASES
from ai4sci_bench.core.logger import get_logger, setup_logging
from ai4sci_bench.core.result_schema import (
    CURRENT_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION_FIELD,
    ensure_supported_result_schema,
)
from ai4sci_bench.core.types import (
    DEFAULT_TIMEOUT_SECONDS,
    EvalResult,
    PromptLevel,
    RunStatus,
)
from ai4sci_bench.generators.instance_generator import FRAMEWORK_TASK_INFO_FILENAME
from ai4sci_bench.reporting.result_loader import (
    dedupe_results as _dedupe_report_results_impl,
    derive_agent_label as _derive_agent_label_impl,
    is_better_result as _is_better_result_impl,
    is_eval_result_json as _is_eval_result_json_impl,
    load_grouped_results as _load_grouped_results_impl,
    merge_result_groups as _merge_result_groups_impl,
    parse_eval_result as _parse_eval_result_impl,
    report_group_for_result as _report_group_for_result_impl,
)
from ai4sci_bench.runner.task_image import TaskImageBuilder

# Auto-load .env from project root (API keys etc.)
load_dotenv()

logger = get_logger(__name__)


def _load_prediction_task_info(instance_dir: Path, task_id: str) -> dict[str, Any]:
    """Load framework metadata for re-eval, with backward compatibility."""
    for filename in (FRAMEWORK_TASK_INFO_FILENAME, "task_info.json"):
        task_info_path = instance_dir / filename
        if task_info_path.exists():
            return json.loads(task_info_path.read_text(encoding="utf-8"))

    return {
        "task_id": task_id,
        "instance_id": instance_dir.name,
        "prompt_level": "b2",
        "parameters": {},
    }


def _resolve_tool_mode(
    tool_mode: str | None,
    allow_external_tools: bool,
) -> str:
    """Resolve final tool_mode string from CLI flags."""
    if tool_mode is not None:
        return tool_mode
    return "search" if allow_external_tools else "restricted"


def _get_cli_version(binary: str) -> str:
    """Get version string of an external CLI tool."""
    path = shutil.which(binary)
    if not path:
        return "not found"
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            if not version:
                version = result.stderr.strip()
            return f"{version}  ({path})" if version else "unknown"
    except Exception:
        pass
    return "unknown"


_AGENT_CLI_BINARY = {
    "codex_cli": "codex",
    "claude_code_cli": "claude",
    "kimi_code_cli": "kimi",
    "antigravity_cli": "agy",
    "mimo_code_cli": "mimo",
}


def _print_agent_banner(
    adapter: Any,
    agent_name: str | None,
    agent_config: dict[str, Any],
    cli_timeout_seconds: int | None = None,
) -> None:
    """Print a prominent banner showing all reproducibility-critical settings."""
    name = agent_name or adapter.__class__.__name__
    model = getattr(adapter, "model", "N/A")
    effort = getattr(adapter, "effort", "N/A")
    timeout = (
        cli_timeout_seconds
        if cli_timeout_seconds is not None
        else getattr(adapter, "timeout_seconds", "N/A")
    )
    tool_mode = getattr(adapter, "tool_mode", "N/A")
    if hasattr(tool_mode, "value"):
        tool_mode = tool_mode.value

    effort_suffix = ""
    if "effort" not in agent_config:
        effort_suffix = " (default)"

    cli_binary = _AGENT_CLI_BINARY.get(name or "")
    cli_version = _get_cli_version(cli_binary) if cli_binary else "N/A"

    lines = [
        ("Agent", name),
        ("Model", str(model)),
        ("Effort", f"{effort}{effort_suffix}"),
        ("Timeout", f"{timeout}s"),
        ("Tools", str(tool_mode)),
    ]
    if cli_binary:
        lines.append(("CLI", cli_version))

    max_key = max(len(k) for k, _ in lines)
    border = "=" * 62

    click.echo(f"\n{border}")
    click.echo("  Agent Configuration (affects reproducibility)")
    click.echo(border)
    for key, value in lines:
        click.echo(f"  {key:<{max_key}}  {value}")
    click.echo(f"{border}\n")


def _build_agent_metadata(
    agent_cmd: str | None,
    agent_name: str | None,
    agent_config: dict[str, Any],
    *,
    allow_external_tools: bool = False,
    tool_mode: str | None = None,
) -> dict[str, Any]:
    """Build provenance metadata describing the requested agent configuration."""
    resolved_mode = _resolve_tool_mode(tool_mode, allow_external_tools)
    metadata: dict[str, Any] = {
        "agent_name": agent_name,
        "cmd_template": agent_cmd,
        "config": dict(agent_config),
        "allow_external_tools": allow_external_tools,
        "tool_mode": resolved_mode,
    }
    if agent_cmd:
        metadata["adapter_class"] = "CLIAgentAdapter"
    elif agent_name == "claude_code_cli":
        metadata["adapter_class"] = "ClaudeCodeCLIAdapter"
    elif agent_name == "direct_llm":
        metadata["adapter_class"] = "DirectLLMAdapter"
    elif agent_name == "codex_cli":
        metadata["adapter_class"] = "CodexCLIAdapter"
    elif agent_name == "openhands":
        metadata["adapter_class"] = "OpenHandsAdapter"
    elif agent_name == "hermes":
        metadata["adapter_class"] = "HermesAgentAdapter"
    elif agent_name == "codewhale":
        metadata["adapter_class"] = "CodeWhaleAdapter"
    else:
        metadata["adapter_class"] = "CLIAgentAdapter"
    return metadata


@click.group()
@click.version_option(version="0.1.3")
def cli():
    """ASI-Bench: LLM Agent benchmark for AI for Science."""
    pass



@cli.command("review")
@click.argument("result_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output-dir", default=None, help="Scan this directory for results (batch mode, requires --tasks)")
@click.option("--tasks", default=None,
              help="Task ID(s) to review in batch mode (comma-separated)")
@click.option("--tasks-dir", default="tasks/", help="Tasks directory with task definitions")
@click.option("--threshold", default=10.0, type=float,
              help="In batch mode, only diagnose instances below this score (default: 10.0)")
@click.option("--prompt-levels", default=None,
              help="Only review these prompt levels (comma-separated, e.g. 'b1,b2')")
@click.option("--report-dir", default=None,
              help="Where to write diagnosis reports (default: next to results)")
@click.option("--all-scores", is_flag=True,
              help="Diagnose ALL matched instances regardless of score")
def review_cmd(
    result_files: tuple[str, ...],
    output_dir: str | None,
    tasks: str | None,
    tasks_dir: str,
    threshold: float,
    prompt_levels: str | None,
    report_dir: str | None,
    all_scores: bool,
):
    """Review benchmark results — diagnose why agent scores are low.

    \b
    Two modes:

    1. Direct file mode — pass one or more result JSON paths:
       asibench review path/to/result__b1.json path/to/result__b2.json

    2. Batch mode — scan a directory for specific tasks:
       asibench review --output-dir results/agent_a --tasks domain.task_name

    \b
    Each result JSON uniquely identifies one task + one model + one B-level.
    The command collects task definitions + agent trajectories and calls the
    local Claude Code CLI to produce a structured diagnosis.

    \b
    Examples:
      # Diagnose one specific result file
      asibench review results/agent_a/task_b1/domain.task_name/instance__b1.json

      # Diagnose multiple files
      asibench review results/agent_a/task_b1/*__b1.json results/agent_b/task_b1/*__b1.json

      # Batch: scan directory for a task, only low scores
      asibench review --output-dir results/agent_a --tasks domain.task_name

      # Batch: scan directory, diagnose all scores
      asibench review --output-dir results/agent_b --tasks domain.another_task --all-scores
    """
    from ai4sci_bench.analysis.reviewer import (
        DiagnosisSummary,
        run_diagnosis,
        run_gradient_analysis,
        write_diagnosis_report,
    )

    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        click.echo(f"Error: tasks-dir '{tasks_dir}' does not exist.", err=True)
        raise SystemExit(1)

    if not shutil.which("claude"):
        click.echo("Error: Claude Code CLI ('claude') not found in PATH.", err=True)
        raise SystemExit(1)

    level_filter = set(prompt_levels.split(",")) if prompt_levels else None
    candidates: list[dict] = []

    def _load_result(json_file: Path) -> dict | None:
        if json_file.name in (
            "run_metadata.json", "eval_config.json",
            "task_info.json", "framework_task_info.json",
        ):
            return None
        if ".trajectory." in json_file.name or ".agent_model_output." in json_file.name:
            return None
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if "final_score" not in data or "task_id" not in data:
            return None
        data["_json_path"] = str(json_file)
        data["_combo_label"] = data.get("agent_name", "unknown")
        return data

    # --- Mode 1: Direct file paths ---
    if result_files:
        for fpath in result_files:
            p = Path(fpath)
            data = _load_result(p)
            if data is None:
                click.echo(f"Skipping {fpath} (not a valid result JSON)", err=True)
                continue
            level = data.get("prompt_level", "")
            if level_filter and level not in level_filter:
                continue
            if not all_scores and data["final_score"] >= threshold:
                continue
            candidates.append(data)

    # --- Mode 2: Batch directory scan ---
    elif output_dir:
        if not tasks:
            click.echo("Error: --tasks is required when using --output-dir (batch mode).", err=True)
            raise SystemExit(1)
        output_path = Path(output_dir)
        if not output_path.exists():
            click.echo(f"Error: output-dir '{output_dir}' does not exist.", err=True)
            raise SystemExit(1)

        task_filter = set(t.strip() for t in tasks.split(","))

        for json_file in sorted(output_path.rglob("*.json")):
            data = _load_result(json_file)
            if data is None:
                continue
            tid = data["task_id"]
            level = data.get("prompt_level", "")
            if tid not in task_filter:
                continue
            if level_filter and level not in level_filter:
                continue
            if not all_scores and data["final_score"] >= threshold:
                continue
            candidates.append(data)
    else:
        click.echo("Error: provide result JSON file(s) as arguments, or use --output-dir + --tasks.", err=True)
        raise SystemExit(1)

    if not candidates:
        click.echo("Nothing to diagnose. Check your paths/filters, or use --all-scores / raise --threshold.")
        return

    # Show what will be diagnosed
    click.echo(f"Found {len(candidates)} instance(s) to diagnose:")
    for c in candidates:
        agent = c.get("agent_name", c.get("_combo_label", "?"))
        click.echo(f"  {c['task_id']} {c.get('prompt_level','?')} [{agent}] — score: {c['final_score']:.1f}")
        click.echo(f"    {c['_json_path']}")

    # Determine report output location
    if report_dir:
        rpath = Path(report_dir)
    elif output_dir:
        rpath = Path(output_dir)
    else:
        rpath = Path(candidates[0]["_json_path"]).parent

    summary = DiagnosisSummary(
        total_instances=len(candidates),
        reviewed_count=len(candidates),
    )

    for i, result_data in enumerate(candidates, 1):
        tid = result_data.get("task_id", "?")
        level = result_data.get("prompt_level", "?")
        score = result_data.get("final_score", 0)
        click.echo(f"\n[{i}/{len(candidates)}] Reviewing {tid} {level} (score: {score:.1f}) ...")
        diagnosis = run_diagnosis(result_data, tasks_path, rpath)
        summary.diagnoses.append(diagnosis)
        click.echo(f"  -> {diagnosis.cause_bucket}: {diagnosis.classification}")
        click.echo(f"     {diagnosis.bottom_line}")

    # Cross-level gradient analysis when multiple levels are present
    task_combo_scores: dict[tuple[str, str], dict[str, float]] = {}
    for c in candidates:
        key = (c["task_id"], c.get("_combo_label", "unknown"))
        lvl = c.get("prompt_level", "")
        if lvl:
            task_combo_scores.setdefault(key, {})[lvl] = c["final_score"]

    for (tid, label), level_scores in task_combo_scores.items():
        if len(level_scores) >= 2:
            click.echo(f"\nGradient analysis: {tid} — {label}")
            ga = run_gradient_analysis(tid, label, level_scores, tasks_path)
            summary.gradient_analyses.append(ga)

    write_diagnosis_report(summary, rpath)
    diag_dir = rpath / "diagnosis"
    click.echo(f"\nReview complete. Reports written to {diag_dir}/")
    click.echo(f"  Summary: {diag_dir / 'summary.md'}")


@cli.group("task")
def task_group():
    """Create, download, and submit benchmark tasks."""


@cli.command("list")
@click.option("--domain", help="Filter by domain")
@click.option("--include-test", is_flag=True, help="Include test-status tasks")
@click.option("--include-sample", is_flag=True, help="Include sample-status tasks")
@click.option("--include-dev", is_flag=True, help="Include in-development tasks")
@click.option("--include-abandoned", is_flag=True, help="Include abandoned tasks")
@click.option("--tasks-dir", default="tasks/", help="Tasks directory")
def list_tasks(domain: str | None, include_test: bool, include_sample: bool,
               include_dev: bool, include_abandoned: bool, tasks_dir: str):
    """List available benchmark tasks."""
    from ai4sci_bench.core.task import TaskLoader

    loader = TaskLoader(Path(tasks_dir))
    tasks = loader.discover_tasks(
        include_test=include_test,
        include_sample=include_sample,
        include_dev=include_dev,
        include_abandoned=include_abandoned,
    )

    if domain:
        tasks = [t for t in tasks if t.get("domain") == domain]

    if not tasks:
        click.echo("No tasks found.")
        return

    click.echo(f"Found {len(tasks)} task(s):\n")
    for t in tasks:
        status = t.get("status", "unknown")
        line = f"  [{status}] {t['id']}: {t.get('name', 'N/A')}"
        abandoned_reason = t.get("abandoned_reason")
        if status == "abandoned" and abandoned_reason:
            line += f"  (reason: {abandoned_reason})"
        click.echo(line)
        click.echo(f"         Domain: {t.get('domain', 'N/A')} / {t.get('subdomain', 'N/A')}")


@task_group.command("pull")
@click.option(
    "--repo",
    default=None,
    type=click.Choice(OFFICIAL_HF_REPO_ALIASES, case_sensitive=True),
    metavar="seed42|seed31415",
    help="Official fixed-seed HF dataset; must be selected explicitly.",
)
@click.option("--tasks", default=None,
              help="Comma-separated task IDs to pull (default: all in the repo)")
@click.option("--output-dir", "output_dir", default=DEFAULT_PULL_DIR,
              help="Where to place pulled <instance_id>/ dirs (feed to run --instances-dir)")
@click.option("--token", default=None,
              help="HF token (default: $HF_TOKEN). Public repos need none.")
@click.option("--revision", default="main", help="Git revision to pull")
@click.option("--overwrite", is_flag=True, help="Replace existing instance dirs")
def pull(repo: str | None, tasks: str | None, output_dir: str, token: str | None,
         revision: str, overwrite: bool):
    """Download pre-generated instances from HuggingFace for local runs.

    \b
    Official datasets:
      seed42 / seed31415 → the only supported benchmark contracts

    \b
    Examples:
      asibench task pull --repo seed42 --output-dir hf_instances_seed42/
      asibench task pull --repo seed31415 --output-dir hf_instances_seed31415/
      asibench run --instances-dir hf_instances/ --tasks domain.task_name ...
    """
    from ai4sci_bench.hf import pull_instances

    if repo is None:
        raise click.UsageError(
            "--repo is required; choose exactly one official contract: "
            "seed42 or seed31415."
        )

    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None
    try:
        result = pull_instances(
            repo,
            output_dir=output_dir,
            tasks=task_ids,
            token=token,
            revision=revision,
            overwrite=overwrite,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # network / auth / missing repo
        raise click.ClickException(f"Pull from '{repo}' failed: {exc}") from exc

    click.echo(f"Pulled from {result.repo_id} (rev {revision})")
    click.echo(f"  tasks:     {len(result.task_ids)}")
    click.echo(f"  instances: {len(result.instance_ids)} → {result.output_dir}/")
    if repo == "seed31415":
        click.echo("  references: public (local scoring enabled)")
    else:
        click.echo("  references: private (excluded; use submit for scoring)")
    if result.skipped:
        click.echo(f"  skipped (already present, use --overwrite): {len(result.skipped)}")
    if not result.instance_ids and not result.skipped:
        click.echo(
            "  (no instances matched — expected tasks/<instance-id>/; "
            "also check --repo / --tasks / token access)"
        )


def _format_file_size(size_bytes: int) -> str:
    """Format archive sizes without rounding non-empty small files to 0.0 MB."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes / (1024 * 1024):.1f} MiB"


@cli.command("score")
@click.option(
    "--repo",
    required=True,
    type=click.Choice(OFFICIAL_HF_REPO_ALIASES, case_sensitive=True),
    metavar="seed42|seed31415",
    help="Dataset contract. Only seed31415 supports public local scoring.",
)
@click.option("--results-dir", required=True,
              help="Produce-only run directory from `asibench run`")
@click.option("--instances-dir", required=True,
              help="Pulled seed31415 instances containing public reference/")
@click.option("--tasks-dir", default="tasks/", show_default=True,
              help="GitHub task catalog containing task_eval.yaml and scorers")
@click.option("--output", "output_path", default=None,
              help="Local score report JSON (default: <results-dir>/local_score_seed31415.json)")
def score_cmd(repo: str, results_dir: str, instances_dir: str,
              tasks_dir: str, output_path: str | None):
    """Score seed31415 locally with its public references and GitHub scorers.

    \b
    seed31415 publishes its GT and supports reproducible local scores. These
    scores are explicitly non-official and do not alter the source run.
    seed42 GT stays private; use `asibench submit` for seed42 scoring.
    """
    from ai4sci_bench.local_scoring import (
        LocalScoringError,
        PRIVATE_SCORING_REPO,
        score_seed31415_results,
    )

    if repo == PRIVATE_SCORING_REPO:
        raise click.ClickException(
            "seed42 references are private and local scoring is not available. "
            "Use `asibench submit --results-dir ...` for official scoring."
        )
    try:
        report, destination = score_seed31415_results(
            results_dir,
            instances_dir,
            tasks_dir,
            output_path=output_path,
        )
    except LocalScoringError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("ASI-Bench public local scoring (seed31415; non-official)")
    for item in report["results"]:
        click.echo(
            f"  {item['instance_id']} {item['prompt_level']}: "
            f"{item['final_score']:.2f} / {item['max_score']:.2f}"
        )
    click.echo(
        f"Total: {report['total_score']:.2f} / "
        f"{report['total_max_score']:.2f} "
        f"({report['mean_percent']:.2f}%)"
    )
    click.echo(f"Report: {destination}")
    if report["scorer_error_count"]:
        raise click.ClickException(
            f"{report['scorer_error_count']} instance(s) had internal scorer errors; "
            f"inspect {destination}."
        )


@cli.command("submit")
@click.option("--results-dir", "results_dir", required=True,
              help="A produce-only run dir (from `asibench run ...`)")
@click.option("--output", "output_path", default=None,
              help="Bundle path: a dir, or a *.tar.gz archive. "
                   "Default: <results-dir>/submission_<timestamp>.tar.gz")
@click.option("--note", default=None, help="Free-form submitter note (team / model)")
@click.option("--benchmark-repo", default=None,
              help="Official seed42 HF repo, recorded for provenance")
@click.option("--endpoint", default=None,
              help="ASI-Bench website URL (default: $ASIBENCH_SUBMIT_ENDPOINT, then official site)")
@click.option("--token", default=None,
              help="Scoring-service auth token (default: $ASIBENCH_SUBMIT_TOKEN)")
@click.option("--no-upload", "no_upload", is_flag=True, default=False,
              help="Build the bundle only; never upload even if an endpoint is set")
@click.option("--no-archive", "no_archive", is_flag=True, default=False,
              help="Keep the bundle directory; skip the .tar.gz (implies --no-upload)")
def submit(results_dir, output_path, note, benchmark_repo, endpoint, token,
           no_upload, no_archive):
    """Submit a seed42 produce-only run for official website scoring.

    \b
    The bundle carries everything the website scoring service needs — each
    instance's result JSON (identity + params + provenance) and the agent's
    actual output artifacts. A framework_task_info snapshot is included only
    when present; public fixed-seed runs do not require it. The authenticated
    website submission enters its private scoring workflow.

    \b
    Only seed42 results are accepted. Score seed31415 locally with
    `asibench score --repo seed31415`; it cannot enter official submission.

    \b
    Examples:
      asibench run --instances-dir hf_instances/ --agent ... --output-dir out/
      asibench login
      asibench submit --results-dir out/
    """
    from ai4sci_bench.branding import submit_endpoint
    from ai4sci_bench.submission import build_submission, upload_bundle

    archive = not no_archive
    try:
        bundle = build_submission(
            results_dir,
            output_path=output_path,
            note=note,
            benchmark_repo=benchmark_repo,
            archive=archive,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Bundle built: {bundle.bundle_dir}")
    click.echo(f"  instances: {bundle.instance_count}")
    if bundle.run_metadata_included:
        click.echo("  run_metadata.json: included")
    if bundle.archive_path:
        size = _format_file_size(bundle.archive_path.stat().st_size)
        click.echo(f"  archive:   {bundle.archive_path} ({size})")

    flagged = bundle.instances_with_missing_outputs
    if flagged:
        click.echo(
            click.style(
                f"  WARNING: {len(flagged)} instance(s) are missing declared "
                "output files (the agent did not produce them). These will be "
                "scored as incomplete:",
                fg="yellow",
            )
        )
        for entry in flagged:
            click.echo(f"    - {entry.base_name}: missing {entry.missing_outputs}")

    # Upload to the official website unless explicitly suppressed.
    resolved_endpoint = submit_endpoint(endpoint)
    if no_upload or no_archive:
        reason = "--no-upload" if no_upload else "--no-archive (no tarball to send)"
        click.echo(f"\nUpload skipped ({reason}). Bundle is ready above.")
        return
    # Credential order (§2 of the device-login design): --token > env >
    # ~/.asibench/credentials > interactive device-flow login.
    from ai4sci_bench.auth import resolve_token
    from ai4sci_bench.submission.endpoints import bundle_upload_url, portal_api_base

    credential_base = portal_api_base(resolved_endpoint) or resolved_endpoint.rstrip("/")
    tok, _tok_src = resolve_token(credential_base, token,
                                  interactive=_stdio_interactive(), echo=click.echo)
    if not tok:
        raise click.ClickException(
            "Submission requires an ASI-Bench identity. Run `asibench login` "
            "and approve the browser sign-in, or pass --token / set "
            "ASIBENCH_SUBMIT_TOKEN in headless environments."
        )

    # Tolerate web-page/origin URLs (users copy them from the browser): rewrite
    # to the bundle API instead of letting the web server answer 405.
    upload_url, endpoint_note = bundle_upload_url(resolved_endpoint)
    if endpoint_note:
        click.echo(click.style(f"\n  note: {endpoint_note}", fg="yellow"))
    click.echo(f"\nUploading to {upload_url} ...")
    result = upload_bundle(
        bundle.archive_path,
        upload_url,
        token=tok,
    )
    if result.ok:
        click.echo(click.style(f"  OK (HTTP {result.status_code})", fg="green"))
        rj = result.response_json or {}
        sub_id = rj.get("submission_id") or rj.get("id")
        if sub_id:
            click.echo(f"  submission_id: {sub_id}")
        if rj.get("status"):
            click.echo(f"  status: {rj['status']}")
        # Portal parked the bundle as a draft; surface integrity + next step.
        if rj.get("integrity_ok") is False:
            click.echo(click.style(
                "  integrity check FAILED on the server — re-run submit and re-upload.",
                fg="red"))
        incomplete = rj.get("incomplete_instances")
        if incomplete:
            click.echo(click.style(
                f"  {len(incomplete)} instance(s) missing declared outputs "
                "(will be scored incomplete).", fg="yellow"))
        if rj.get("message"):
            click.echo(f"  {rj['message']}")
        if rj.get("confirm_url"):
            # Two-step submit: upload parks a DRAFT; the runner must open the
            # confirm page in a browser, review the completeness summary, and
            # click Confirm to enter the scoring queue. Make it unmistakable
            # that upload success alone does not confirm the submission.
            click.echo(click.style(
                "\n  UPLOAD COMPLETE — confirmation is still required.\n"
                "  Your run is saved on the ASI-Bench website as a DRAFT.\n"
                "  Open this page in your browser, review the\n"
                "  completeness summary, and click Confirm to enter the scoring queue:\n"
                f"    {rj['confirm_url']}", fg="cyan"))
    else:
        raise click.ClickException(
            f"Upload failed: {result.error or f'HTTP {result.status_code}'}"
            + (f"\n{result.response_body[:500]}" if result.response_body else "")
        )


def _stdio_interactive() -> bool:
    """True when both stdin and stdout are a TTY (a human at a terminal). Used to
    decide whether it's safe to prompt for login vs. fail-loud in CI. Seam for tests."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _resolve_api_base(endpoint: str | None) -> str:
    """Resolve the portal API base from --endpoint / env / official default."""
    from ai4sci_bench.branding import submit_endpoint
    from ai4sci_bench.submission.endpoints import portal_api_base

    resolved = submit_endpoint(endpoint)
    if not resolved:
        raise click.ClickException(
            "Could not resolve the ASI-Bench website endpoint.")
    api = portal_api_base(resolved)
    if api is None:
        raise click.ClickException(
            "The configured endpoint is not a recognized ASI-Bench Portal URL."
        )
    return api


@cli.command("login")
@click.option("--endpoint", default=None,
              help="Portal URL (default: $ASIBENCH_SUBMIT_ENDPOINT, then official site)")
@click.option("--device", is_flag=True, default=False,
              help="Use the optional browser device-code flow instead of pasting a PAT")
def login_cmd(endpoint: str | None, device: bool):
    """Save a Portal login for future Task and Run submissions.

    By default the CLI opens Portal Settings and securely prompts for the PAT
    shown there. The token is validated before it is stored and is never put in
    shell history. ``--device`` retains the optional no-copy authorization flow.
    """
    from ai4sci_bench.auth.device_login import DeviceLoginError, device_login

    api = _resolve_api_base(endpoint)
    if not device:
        _prompt_and_save_portal_token(api)
        return
    if not _stdio_interactive():
        raise click.ClickException(
            "login needs an interactive terminal; in CI set ASIBENCH_SUBMIT_TOKEN.")
    try:
        grant = device_login(api, echo=click.echo,
                             open_browser=not os.environ.get("ASIBENCH_NO_BROWSER"))
    except DeviceLoginError as exc:
        raise click.ClickException(str(exc)) from exc
    user = grant.get("user") or {}
    path = save_credential(api, grant["access_token"], user)
    who = user.get("email") or user.get("display_name") or "you"
    click.echo(click.style(f"✅ Signed in as {who} — credential saved to {path}",
                           fg="green"))


def _prompt_and_save_portal_token(api_base: str, *, attempts: int = 3) -> str:
    """Prompt without echo, validate, then save a manually-created Portal PAT."""
    import webbrowser

    from ai4sci_bench.auth import (
        TokenLoginError,
        save_credential,
        validate_portal_token,
    )
    from ai4sci_bench.submission.endpoints import token_settings_url

    settings_url = token_settings_url(api_base)
    if not settings_url:
        raise click.ClickException("Could not derive the Portal token settings URL.")
    click.echo("No saved ASI-Bench login was found.")
    click.echo("Sign in, create a CLI token, and copy it from:\n"
               f"  {settings_url}\n")
    if not os.environ.get("ASIBENCH_NO_BROWSER"):
        try:
            webbrowser.open(settings_url, new=2)
        except Exception:
            pass
    last_error = "Token validation failed."
    for _ in range(attempts):
        token = click.prompt("Paste token", hide_input=True).strip()
        try:
            user = validate_portal_token(api_base, token)
        except TokenLoginError as exc:
            last_error = str(exc)
            click.echo(click.style(
                f"Token not saved: {last_error}\n"
                "The clipboard may have changed; copy the full token and try again.",
                fg="yellow",
            ))
            continue
        path = save_credential(api_base, token, user)
        who = user.get("primary_email") or user.get("email") \
            or user.get("display_name") or user.get("id")
        click.echo(click.style(
            f"✅ Signed in as {who} — credential saved to {path}", fg="green",
        ))
        return token
    raise click.ClickException(last_error)


@cli.command("logout")
@click.option("--endpoint", default=None,
              help="Portal URL (default: $ASIBENCH_SUBMIT_ENDPOINT; use --all for every one)")
@click.option("--all", "all_endpoints", is_flag=True, default=False,
              help="Remove saved credentials for every endpoint")
def logout_cmd(endpoint: str | None, all_endpoints: bool):
    """Forget the locally saved login (the server-side token stays until revoked
    in Settings → CLI tokens — this only clears ~/.asibench/credentials)."""
    from ai4sci_bench.auth import clear_credential
    removed = clear_credential(None if all_endpoints else _resolve_api_base(endpoint))
    click.echo("Signed out." if removed else "No saved login to remove.")


@cli.group("auth")
def auth_group():
    """Inspect CLI authentication state."""


@auth_group.command("status")
def auth_status_cmd():
    """Show which credential each portal endpoint would use."""
    from ai4sci_bench.auth.credentials import list_credentials

    env_tok = os.environ.get("ASIBENCH_SUBMIT_TOKEN")
    if env_tok:
        click.echo(f"env    ASIBENCH_SUBMIT_TOKEN = {env_tok[:12]}…  (takes precedence)")
    saved = list_credentials()
    if not saved and not env_tok:
        click.echo("Not logged in — run `asibench login`, or set ASIBENCH_SUBMIT_TOKEN.")
        return
    for ep, entry in saved.items():
        who = entry["user"].get("primary_email") or entry["user"].get("email") \
            or entry["user"].get("display_name") or "?"
        click.echo(f"stored {ep}  as {who}  ({entry['token_prefix']}, "
                   f"saved {entry.get('created_at') or '?'})")


@task_group.command("submit")
@click.option(
    "--task-dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    help="Local Task root containing task_meta.yaml",
)
@click.option("--endpoint", default=None,
              help="Portal URL (default: $ASIBENCH_SUBMIT_ENDPOINT, then official Portal)")
@click.option(
    "--force-file-sync",
    is_flag=True,
    default=False,
    help="Replace conflicting server files after reviewing the Draft",
)
def task_submit_cmd(task_dir: Path, endpoint: str | None, force_file_sync: bool):
    """Upload a local Task as a Draft, then open Portal for final confirmation."""
    import webbrowser

    from ai4sci_bench.auth import clear_credential
    from ai4sci_bench.auth.resolve import resolve_token
    from ai4sci_bench.branding import submit_endpoint
    from ai4sci_bench.submission.endpoints import token_settings_url
    from ai4sci_bench.submission.task_submit import submit_task

    resolved = submit_endpoint(endpoint)
    api = _resolve_api_base(resolved)
    token, source = resolve_token(api, None, interactive=False, echo=click.echo)
    if not token:
        if not _stdio_interactive():
            settings_url = token_settings_url(api) or "the Portal Settings page"
            raise click.ClickException(
                "No saved ASI-Bench token is available in this non-interactive "
                "environment. Create one at " + settings_url + " and set "
                "ASIBENCH_SUBMIT_TOKEN, or run `asibench login` in a terminal."
            )
        token = _prompt_and_save_portal_token(api)
        source = "pasted"

    click.echo(f"Validating and uploading Task from {task_dir} ...")
    result = submit_task(
        task_dir, resolved, token, force_file_sync=force_file_sync,
    )
    if result.status_code == 401 and source == "stored" and _stdio_interactive():
        clear_credential(api)
        click.echo("The saved token is no longer valid; paste a new Portal token.")
        token = _prompt_and_save_portal_token(api)
        result = submit_task(
            task_dir, resolved, token, force_file_sync=force_file_sync,
        )
    if not result.ok:
        message = result.error or "Task Draft upload failed"
        if result.web_url:
            message += f"\nA recoverable Draft may exist here: {result.web_url}"
        raise click.ClickException(message)
    if not result.files_ok:
        raise click.ClickException(
            "The Draft was created, but exact file synchronization failed: "
            + (result.file_error or "unknown error")
            + (f"\nReview the recoverable Draft: {result.web_url}" if result.web_url else "")
        )

    click.echo(click.style(
        f"\nDRAFT UPLOAD COMPLETE — {len(result.uploaded)} files synchronized.",
        fg="green",
    ))
    click.echo("The Task has not entered review; final confirmation is still required.")
    if result.completeness_percent is not None:
        click.echo(f"Portal completeness: {result.completeness_percent}%")
    if result.missing:
        click.echo(click.style("Missing: " + ", ".join(result.missing), fg="yellow"))
    if result.web_url:
        click.echo("Review the exact file list and imported fields, then submit:\n"
                   f"  {result.web_url}")
    if not result.web_url or os.environ.get("ASIBENCH_NO_BROWSER"):
        return
    try:
        opened = webbrowser.open(result.web_url, new=2)
    except Exception:
        opened = False
    if not opened:
        click.echo("A browser could not be opened automatically. Open the URL above manually.")


def _ensure_run_sandbox_available(sandbox: str) -> None:
    """Fail before a run when its host-level sandbox backend is unavailable."""
    if sandbox == "linux_ns":
        from ai4sci_bench.runner.linux_ns_sandbox import check_linux_ns_available

        available, reason = check_linux_ns_available()
        if not available:
            raise click.ClickException(
                f"--sandbox linux_ns is unavailable: {reason}"
            )
    elif sandbox == "os":
        from ai4sci_bench.runner.task_image import TaskImageBuilder
        from ai4sci_bench.runner.runtime_root import resolve_runtime_root

        try:
            TaskImageBuilder(resolve_runtime_root()).ensure_docker_available()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--tasks", help="Comma-separated task IDs (or 'all')")
@click.option("--prompt-levels", default="b1,b2,b3,b4", show_default=True,
              help="Comma-separated prompt levels")
@click.option("--instances-per-task", default=1, type=int, help="Instances per task")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("--agent-cmd", help="Agent command template (file-exchange mode)")
@click.option("--agent", help="Built-in agent name (direct_llm, claude_code_cli, codex_cli)")
@click.option("--agent-config", default="{}", help="Agent config JSON")
@click.option("--output-dir", default="results/", help="Output directory")
@click.option("--parallel", default=1, type=int, help="Parallel workers")
@click.option("--timeout", default=DEFAULT_TIMEOUT_SECONDS, show_default=True, type=int,
              help="Timeout per instance. Task metadata cannot override this value.")
@click.option("--sandbox", default="none", help="Sandbox mode: none (default), task (Python venv), os (Docker container), linux_ns (Linux namespaces)")
@click.option("--instances-dir", help="Pre-generated instances directory")
@click.option("--params", default=None, help="Fixed params JSON for one debug instance")
@click.option("--include-test", is_flag=True, help="Include test-status tasks")
@click.option("--include-sample", is_flag=True, help="Include sample-status tasks")
@click.option("--include-dev", is_flag=True, help="Include in-development tasks")
@click.option("--include-abandoned", is_flag=True, help="Include abandoned tasks")
@click.option("--analyze", is_flag=True, help="Enable error analysis")
@click.option("--analyze-backend", default="llm_api", help="Analysis backend")
@click.option("--analyze-model", default="gemini/gemini-2.0-flash", help="Analysis model")
@click.option("--resume", help="Resume from a prior output directory")
@click.option("--retries", default=1, type=int, help="Attempts per instance (1=no retry)")
@click.option("--retry-strategy", default="all", type=click.Choice(["all", "until_success"]),
              help="all: always run N times; until_success: stop on first success")
@click.option("--tasks-dir", default="tasks/", help="Tasks directory")
@click.option("--allow-external-tools", is_flag=True, default=False,
              help="Allow agents to use external tools (web search, MCP). Alias for --tool-mode search.")
@click.option("--tool-mode", type=click.Choice(["restricted", "search", "unrestricted"]),
              default=None,
              help="Agent tool isolation level. Default: restricted.")
@click.option("--write-batch-records", is_flag=True, default=False,
              help="Write derived batch_records/ artifacts after the run.")
@click.option("--batch-records-root", default=None,
              help="Shared results root to scan when writing batch_records/ (defaults to --output-dir).")
@click.option("--diagnose", is_flag=True, default=False,
              help="Run trajectory review after evaluation completes")
@click.option("--diagnose-threshold", default=10.0, type=float,
              help="Score threshold for diagnosis (only used when --diagnose without full review)")
def run(
    tasks: str | None,
    prompt_levels: str,
    instances_per_task: int,
    seed: int,
    agent_cmd: str | None,
    agent: str | None,
    agent_config: str,
    output_dir: str,
    parallel: int,
    timeout: int,
    sandbox: str,
    instances_dir: str | None,
    params: str | None,
    include_test: bool,
    include_sample: bool,
    include_dev: bool,
    include_abandoned: bool,
    analyze: bool,
    analyze_backend: str,
    analyze_model: str,
    resume: str | None,
    retries: int,
    retry_strategy: str,
    tasks_dir: str,
    allow_external_tools: bool,
    tool_mode: str | None,
    write_batch_records: bool,
    batch_records_root: str | None,
    diagnose: bool,
    diagnose_threshold: float,
):
    """Run agents on benchmark instances and collect outputs for submission."""
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
    from ai4sci_bench.runner.parallel import auto_limit_workers
    from ai4sci_bench.runner.sandbox_support import print_sandbox_banner

    if tool_mode and allow_external_tools:
        raise click.ClickException(
            "--tool-mode and --allow-external-tools are mutually exclusive"
        )
    resolved_tool_mode = _resolve_tool_mode(tool_mode, allow_external_tools)

    print_sandbox_banner(sandbox)
    _ensure_run_sandbox_available(sandbox)

    # Auto-limit parallel workers based on host resources for --sandbox os
    parallel = auto_limit_workers(parallel, sandbox=sandbox)

    parsed_agent_config = json.loads(agent_config)
    # Build agent adapter
    try:
        adapter = _build_agent(agent_cmd, agent, dict(parsed_agent_config),
                               allow_external_tools=allow_external_tools,
                               tool_mode=resolved_tool_mode)
    except (ValueError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print_agent_banner(
        adapter,
        agent,
        parsed_agent_config,
        cli_timeout_seconds=timeout,
    )

    task_ids = None
    if tasks and tasks != "all":
        task_ids = [t.strip() for t in tasks.split(",")]

    fixed_params = None
    if params is not None:
        if not task_ids or len(task_ids) != 1:
            raise click.ClickException("--params requires exactly one task via --tasks")
        if instances_dir:
            raise click.ClickException("--params cannot be combined with --instances-dir")
        if instances_per_task != 1:
            raise click.ClickException("--params requires --instances-per-task 1")
        try:
            fixed_params = json.loads(params)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid --params JSON: {exc}") from exc
        if not isinstance(fixed_params, dict):
            raise click.ClickException("--params must decode to a JSON object")

    if batch_records_root and not write_batch_records:
        raise click.ClickException("--batch-records-root requires --write-batch-records")

    config = RunConfig(
        agent=adapter,
        tasks=task_ids,
        include_test=include_test,
        include_sample=include_sample,
        include_dev=include_dev,
        include_abandoned=include_abandoned,
        prompt_levels=[l.strip() for l in prompt_levels.split(",")],
        seed=seed,
        instances_per_task=instances_per_task,
        instances_dir=instances_dir,
        fixed_params=fixed_params,
        parallel=parallel,
        timeout=timeout,
        sandbox=sandbox,
        output_dir=output_dir,
        resume=resume,
        retries=retries,
        retry_strategy=retry_strategy,
        analyze=analyze,
        analyze_backend=analyze_backend,
        analyze_model=analyze_model,
        tasks_dir=tasks_dir,
        score=False,
        agent_metadata=_build_agent_metadata(
            agent_cmd,
            agent,
            parsed_agent_config,
            allow_external_tools=allow_external_tools,
            tool_mode=resolved_tool_mode,
        ),
    )

    try:
        orchestrator = BenchmarkOrchestrator(config)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    report = orchestrator.run(task_ids)
    click.echo(report.format_summary())

    if diagnose:
        click.echo(f"\n{'='*60}")
        click.echo("REVIEW MODULE — Trajectory Review")
        click.echo(f"{'='*60}\n")

        from ai4sci_bench.analysis.reviewer import (
            DiagnosisSummary,
            find_all_instances,
            run_diagnosis,
            run_gradient_analysis,
            write_diagnosis_report,
        )

        levels = [l.strip() for l in prompt_levels.split(",")]
        out = Path(output_dir)

        candidates: list[dict] = []
        for json_file in sorted(out.rglob("*.json")):
            if json_file.name in (
                "run_metadata.json", "eval_config.json",
                "task_info.json", "framework_task_info.json",
            ):
                continue
            if ".trajectory." in json_file.name or ".agent_model_output." in json_file.name:
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "final_score" not in data or "task_id" not in data:
                continue
            level = data.get("prompt_level", "")
            if level not in levels:
                continue
            data["_json_path"] = str(json_file)
            data["_combo_label"] = data.get("agent_name", agent or "unknown")
            candidates.append(data)

        if not candidates:
            click.echo("No result instances found for trajectory review.")
        else:
            summary = DiagnosisSummary(
                total_instances=len(candidates),
                reviewed_count=len(candidates),
            )

            for i, result_data in enumerate(candidates, 1):
                tid = result_data.get("task_id", "?")
                lvl = result_data.get("prompt_level", "?")
                score = result_data.get("final_score", 0)
                click.echo(f"[{i}/{len(candidates)}] Reviewing {tid} {lvl} (score: {score:.1f}) ...")
                diagnosis = run_diagnosis(result_data, Path(tasks_dir), out)
                summary.diagnoses.append(diagnosis)
                click.echo(f"  -> {diagnosis.cause_bucket}: {diagnosis.classification}")

            task_combo_scores: dict[tuple[str, str], dict[str, float]] = {}
            for d in candidates:
                key = (d["task_id"], d["_combo_label"])
                lvl = d.get("prompt_level", "")
                if lvl:
                    task_combo_scores.setdefault(key, {})[lvl] = d["final_score"]

            for (tid, label), level_scores in task_combo_scores.items():
                if len(level_scores) >= 2:
                    click.echo(f"Gradient analysis: {tid} — {label}")
                    ga = run_gradient_analysis(tid, label, level_scores, Path(tasks_dir))
                    summary.gradient_analyses.append(ga)

            write_diagnosis_report(summary, out)

            click.echo(f"\nReviewed instances: {summary.reviewed_count}")
            click.echo(f"Diagnoses completed: {len(summary.diagnoses)}")
            click.echo(f"Gradient analyses: {len(summary.gradient_analyses)}")
            if summary.diagnoses:
                click.echo("\nBucket distribution:")
                for bucket, count in sorted(
                    summary.bucket_distribution.items(), key=lambda x: -x[1]
                ):
                    pct = count / len(summary.diagnoses) * 100
                    click.echo(f"  {bucket}: {count} ({pct:.0f}%)")
            click.echo(f"\nDiagnosis report: {out / 'diagnosis' / 'summary.md'}")

    _refresh_batch_records_for_root(
        enabled=write_batch_records,
        batch_records_root=batch_records_root,
        default_root=output_dir,
    )



def _refresh_batch_records_for_root(
    *,
    enabled: bool,
    batch_records_root: str | None,
    default_root: str,
) -> list[Path]:
    """Refresh derived batch record artifacts when requested."""
    if not enabled:
        return []

    from ai4sci_bench.reporting.batch_records import refresh_batch_records

    target_root = Path(batch_records_root) if batch_records_root else Path(default_root)
    record_paths = refresh_batch_records(target_root)
    if record_paths:
        click.echo("Batch record artifacts:")
        for record_path in record_paths:
            click.echo(f"  - {record_path}")
    return record_paths


def _derive_agent_label(agent_name: str, agent_config: dict) -> str:
    """Derive a short directory-safe label from agent name + config."""
    return _derive_agent_label_impl(agent_name, agent_config)


def _dedupe_report_results(results: list[EvalResult]) -> list[EvalResult]:
    """Keep the best retry attempt per (instance_id, prompt_level)."""
    return _dedupe_report_results_impl(results)


def _report_group_from_agent_metadata(agent_metadata: dict[str, Any]) -> tuple[str, str]:
    """Build a stable grouping key and display label from saved agent provenance."""
    key = f"provenance:{json.dumps(agent_metadata, sort_keys=True, separators=(',', ':'))}"
    agent_name = agent_metadata.get("agent_name")
    agent_config = agent_metadata.get("config")
    if isinstance(agent_name, str) and agent_name:
        label = _derive_agent_label(
            agent_name,
            agent_config if isinstance(agent_config, dict) else {},
        )
        return key, label
    adapter_class = agent_metadata.get("adapter_class")
    if isinstance(adapter_class, str) and adapter_class:
        return key, adapter_class
    return key, "unknown"


def _report_group_for_result(
    *,
    results_path: Path,
    json_file: Path,
    data: dict[str, Any],
    result: EvalResult,
) -> tuple[str, str]:
    """Return a stable group key and display label for report bucketing."""
    return _report_group_for_result_impl(
        results_path=results_path,
        json_file=json_file,
        data=data,
        result=result,
    )


def _copy_instances_clean(src_dir: Path, dst_dir: Path) -> None:
    """Copy instances directory without workspace subdirectories."""
    import shutil

    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if not item.is_dir():
            shutil.copy2(item, dst_dir / item.name)
            continue
        # Instance directory: copy everything except workspace* dirs
        dst_instance = dst_dir / item.name
        dst_instance.mkdir(parents=True, exist_ok=True)
        for sub in item.iterdir():
            if sub.is_dir() and sub.name.startswith("workspace"):
                continue
            if sub.is_dir():
                shutil.copytree(sub, dst_instance / sub.name, dirs_exist_ok=True)
            else:
                shutil.copy2(sub, dst_instance / sub.name)


@cli.command()
@click.option("--task", required=True, help="Task ID to generate instances for")
@click.option("--instances-per-task", default=1, type=int)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("--strategy", default="random", help="Sampling strategy")
@click.option("--params", default=None, help="Specific params JSON")
@click.option("--output-dir", default="instances/", help="Output directory")
@click.option("--tasks-dir", default="tasks/", help="Tasks directory")
@click.option("--precompute", is_flag=True, help="Precompute GT for all finite settings")
@click.option("--sandbox", default="none", help="Sandbox mode: none (default), task (Python venv), os (Docker container), linux_ns (Linux namespaces)")
def generate(
    task: str,
    instances_per_task: int,
    seed: int,
    strategy: str,
    params: str | None,
    output_dir: str,
    tasks_dir: str,
    precompute: bool,
    sandbox: str,
):
    """Generate task instances."""
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.core.types import GenerationMode
    from ai4sci_bench.generators.instance_generator import InstanceGenerator
    from ai4sci_bench.generators.param_space import ParamSpace
    from ai4sci_bench.runner.sandbox_support import print_sandbox_banner

    print_sandbox_banner(sandbox)

    from ai4sci_bench.runner.orchestrator import resolve_instance_timeout

    loader = TaskLoader(Path(tasks_dir))
    metadata = loader.load_task_by_id(task)
    try:
        generator = InstanceGenerator(Path(tasks_dir), sandbox=sandbox)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    out = Path(output_dir)
    eff_timeout = resolve_instance_timeout(metadata, None)

    if precompute:
        # Precompute GT for all finite settings
        mode = metadata.get("_generation_mode", GenerationMode.INFINITE)
        if mode != GenerationMode.FINITE:
            click.echo("ERROR: --precompute is only valid for tasks with generation.mode: finite")
            sys.exit(1)

        settings = metadata.get("_generation_settings", [])
        if not settings:
            click.echo("ERROR: No settings defined in task.yaml generation.settings")
            sys.exit(1)

        task_dir = metadata["_task_dir"]
        for i, setting in enumerate(settings):
            setting_id = setting.get("id", f"setting_{i}")
            setting_params = {k: v for k, v in setting.items() if k != "id"}
            ref_output = task_dir / "reference" / setting_id
            ref_output.mkdir(parents=True, exist_ok=True)

            # Run generate_gt.py for this setting
            generator._run_generate_gt(metadata, ref_output, setting_params)
            click.echo(f"  [{i+1}/{len(settings)}] Precomputed: {setting_id}")

        click.echo(f"Precomputed GT for {len(settings)} settings in {task_dir / 'reference'}")
        return

    if params:
        # Single instance with specific params
        p = json.loads(params)
        instance = generator.generate_instance(
            metadata, p, out, effective_timeout_seconds=eff_timeout,
        )
        click.echo(f"Generated: {instance.instance_id}")
    else:
        # Multiple instances
        mode = metadata.get("_generation_mode", GenerationMode.INFINITE)
        if mode == GenerationMode.FINITE:
            instances = generator.generate_instances_on_the_fly(
                metadata,
                seed=seed,
                count=instances_per_task,
                output_dir=out,
                effective_timeout_seconds=eff_timeout,
            )
            for i, instance in enumerate(instances, start=1):
                click.echo(f"  [{i}/{instances_per_task}] {instance.instance_id}")
        else:
            param_space = ParamSpace(metadata.get("generation", {}).get("parameters", {}))
            param_sets = param_space.sample(instances_per_task, strategy=strategy, seed=seed)
            for i, p in enumerate(param_sets):
                instance = generator.generate_instance(
                    metadata, p, out, effective_timeout_seconds=eff_timeout,
                )
                click.echo(f"  [{i+1}/{instances_per_task}] {instance.instance_id}")

    click.echo(f"Instances saved to: {out}")


@cli.command()
@click.option("--task", default=None, help="Task ID to validate (required unless --gt-selfcheck, --static, or --pre-submit)")
@click.option(
    "--tasks-dir",
    default=None,
    help="Tasks directory (inferred from --static/--pre-submit path; otherwise tasks/)",
)
@click.option("--preflight", is_flag=True, help="Generate a sample instance and validate runtime contracts")
@click.option("--sandbox", default="none", help="Sandbox mode for preflight: none (default), task (Python venv), os (Docker container), linux_ns (Linux namespaces)")
@click.option("--agent-cmd", default=None, help="Optional agent command template to validate sandbox compatibility")
@click.option("--agent", default=None, help="Optional built-in agent name to validate sandbox compatibility")
@click.option("--agent-config", default="{}", help="Optional agent config JSON for preflight setup checks")
@click.option("--gt-selfcheck", is_flag=True, help="Run GT self-evaluation: feed reference outputs back through scorers and verify 100/100")
@click.option("--gt-timeout", default=300, type=int, help="Timeout in seconds for GT generation during selfcheck (default: 300)")
@click.option("--static", "static_dir", default=None, type=click.Path(exists=True), help="Run static checks on a task directory (path to task dir)")
@click.option("--pre-submit", "pre_submit_dir", default=None, type=click.Path(exists=True), help="Run full pre-submit checks (static + GT generation + GT self-check) on a task directory")
@click.option("--scores-dir", default="scores/", show_default=True, help="Directory holding difficulty-check score files; used to warn when a status:final task has no passing record")
def validate(
    task: str | None,
    tasks_dir: str | None,
    preflight: bool,
    sandbox: str,
    agent_cmd: str | None,
    agent: str | None,
    agent_config: str,
    gt_selfcheck: bool,
    gt_timeout: int,
    static_dir: str | None,
    pre_submit_dir: str | None,
    scores_dir: str,
):
    """Validate a task definition.

    Modes:
      --task ID              Original mode: validate by task ID (with optional --preflight)
      --gt-selfcheck         Feed GT reference outputs back through scorers (100/100 check)
      --static <dir>         Static-only checks on a task directory
      --pre-submit <dir>     Full pre-submit: static + GT generation + GT self-check
    """
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.generators.instance_generator import InstanceGenerator
    from ai4sci_bench.runner.orchestrator import _resolve_config_templates

    if static_dir or pre_submit_dir:
        target_dir = Path(pre_submit_dir or static_dir)  # type: ignore[arg-type]
        path_tasks_dir = tasks_dir or str(target_dir.parent.parent)
        _run_static_or_presubmit(
            static_dir=static_dir,
            pre_submit_dir=pre_submit_dir,
            tasks_dir=path_tasks_dir,
            gt_timeout=gt_timeout,
            scores_dir=scores_dir,
        )
        return

    resolved_tasks_dir = tasks_dir or "tasks/"

    if gt_selfcheck:
        _run_gt_selfcheck_command(
            task_id=task,
            tasks_dir=resolved_tasks_dir,
            timeout=gt_timeout,
        )
        return

    if task is None:
        raise click.ClickException("--task is required (unless using --gt-selfcheck, --static, or --pre-submit)")

    loader = TaskLoader(Path(resolved_tasks_dir))
    errors = []

    try:
        metadata = loader.load_task_by_id(task)
    except ValueError as e:
        click.echo(f"FAIL: {e}")
        sys.exit(1)

    task_dir = metadata["_task_dir"]
    click.echo(f"Validating: {task} ({task_dir})")

    # Check required fields
    for field in ["id", "name", "domain", "status"]:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")

    warnings = []
    task_status = metadata.get("status")
    abandoned_reason = metadata.get("abandoned_reason")
    is_abandoned = task_status == "abandoned"

    if is_abandoned and not abandoned_reason:
        warnings.append("Task is abandoned but missing 'abandoned_reason' field")
    if not is_abandoned and abandoned_reason:
        warnings.append("Task has 'abandoned_reason' but status is not 'abandoned'")

    if not is_abandoned:
        # Check prompt files (b1-b3 required, b4 optional but validated if declared)
        for level in ["b1", "b2", "b3"]:
            prompt_file = task_dir / f"prompt_{level}.md"
            if not prompt_file.exists():
                errors.append(f"Missing prompt file: prompt_{level}.md")
            elif prompt_file.stat().st_size == 0:
                errors.append(f"Empty prompt file: prompt_{level}.md")

        # B4 is optional, but if declared in task.yaml prompts, it must exist
        if "b4" in metadata.get("prompts", {}):
            prompt_b4 = task_dir / "prompt_b4.md"
            if not prompt_b4.exists():
                errors.append("prompt_b4.md declared in task.yaml but file not found")
            elif prompt_b4.stat().st_size == 0:
                errors.append("Empty prompt file: prompt_b4.md")

        # Check generate_gt.py
        gt_path = task_dir / "generate_gt.py"
        if not gt_path.exists():
            errors.append("Missing generate_gt.py")
        else:
            try:
                module = loader.load_generate_gt_module(task_dir)
                if not hasattr(module, "generate"):
                    errors.append("generate_gt.py missing generate() function")
                if not hasattr(module, "INPUT_SPEC"):
                    errors.append("generate_gt.py missing INPUT_SPEC")
                if not hasattr(module, "OUTPUT_SPEC"):
                    errors.append("generate_gt.py missing OUTPUT_SPEC")
                if not hasattr(module, "DEFAULT_PARAMS"):
                    errors.append("generate_gt.py missing DEFAULT_PARAMS")
                try:
                    loader.load_custom_scorers(task_dir)
                except Exception as e:
                    errors.append(f"Error loading custom_scorer.py: {e}")
            except Exception as e:
                errors.append(f"Error loading generate_gt.py: {e}")
    else:
        click.echo("  (abandoned task — skipping prompt and generate_gt.py checks)")

    if warnings:
        click.echo("WARNINGS:")
        for w in warnings:
            click.echo(f"  - {w}")

    if errors:
        click.echo("VALIDATION FAILED:")
        for e in errors:
            click.echo(f"  - {e}")
        sys.exit(1)
    else:
        click.echo("VALIDATION PASSED")

    if preflight:
        click.echo("Running preflight checks...")
        try:
            module = loader.load_generate_gt_module(task_dir)
            params = _build_preflight_params(metadata, module)
            try:
                parsed_agent_config = json.loads(agent_config)
            except json.JSONDecodeError as exc:
                raise click.ClickException(f"Invalid --agent-config JSON: {exc}") from exc
            if not isinstance(parsed_agent_config, dict):
                raise click.ClickException("--agent-config must decode to a JSON object")

            _validate_task_contracts(metadata, module, errors)
            _validate_evaluation_templates(metadata, params, _resolve_config_templates, errors)
            _validate_registered_scorers(metadata, errors)
            _validate_preflight_sandbox_request(metadata, sandbox, errors)
            _validate_preflight_os_sandbox(
                metadata=metadata,
                sandbox=sandbox,
                repo_root=Path(__file__).resolve().parents[1],
                errors=errors,
            )
            _validate_preflight_linux_ns(
                sandbox=sandbox,
                errors=errors,
            )
            _validate_preflight_agent_setup(
                agent_cmd=agent_cmd,
                agent_name=agent,
                agent_config=parsed_agent_config,
                sandbox=sandbox,
                errors=errors,
            )

            if not errors:
                with tempfile.TemporaryDirectory(prefix="ai4sci_preflight_") as tmp:
                    from ai4sci_bench.runner.orchestrator import resolve_instance_timeout
                    generator = InstanceGenerator(Path(resolved_tasks_dir), sandbox=sandbox)
                    eff_timeout = resolve_instance_timeout(metadata, None)
                    instance = generator.generate_instance(
                        metadata, params, Path(tmp),
                        effective_timeout_seconds=eff_timeout,
                    )
                    _run_task_preflight_hook(module, instance, metadata, errors)
                    _validate_generated_instance(instance, metadata, errors)
        except Exception as e:
            errors.append(f"Preflight error: {e}")

        if errors:
            click.echo("PREFLIGHT FAILED:")
            for e in errors:
                click.echo(f"  - {e}")
            sys.exit(1)
        click.echo("PREFLIGHT PASSED")




@cli.command("clean")
@click.option("--sandbox-images", is_flag=True, help="Remove cached Docker images for --sandbox os")
@click.option("--include-base", is_flag=True, help="Also remove the base image (slower rebuild next time)")
def clean(sandbox_images: bool, include_base: bool):
    """Clean up cached resources."""
    if not sandbox_images:
        click.echo("Nothing to clean. Use --sandbox-images to remove cached Docker images.")
        return

    builder = TaskImageBuilder(Path(__file__).resolve().parents[1])
    try:
        removed = builder.clean_images(include_base=include_base)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e

    if removed:
        click.echo(f"Removed {len(removed)} image(s):")
        for tag in removed:
            click.echo(f"  - {tag}")
    else:
        click.echo("No cached sandbox images found.")


@cli.group("sandbox")
def sandbox_group():
    """Sandbox management commands."""
    pass


@sandbox_group.command("shell")
@click.option("--task", required=True, help="Task ID")
@click.option("--workspace", required=True, type=click.Path(exists=True), help="Workspace directory")
@click.option("--tasks-dir", default="tasks/", help="Tasks directory")
@click.option("--allow-external-tools", is_flag=True, default=False, help="Allow network access")
def sandbox_shell(task: str, workspace: str, tasks_dir: str, allow_external_tools: bool):
    """Open an interactive shell in the task's Docker container.

    Useful for debugging agent behavior in the sandbox environment.
    """
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.runner.os_sandbox import OSSandbox
    from ai4sci_bench.runner.runtime_root import resolve_runtime_root

    loader = TaskLoader(Path(tasks_dir))
    metadata = loader.load_task_by_id(task)

    repo_root = resolve_runtime_root(Path(tasks_dir))
    sandbox = OSSandbox(repo_root)
    cmd = sandbox.shell(
        metadata,
        workspace=Path(workspace),
        allow_external_tools=allow_external_tools,
    )
    click.echo(f"Running: {' '.join(cmd)}")
    import os
    os.execvp(cmd[0], cmd)


@sandbox_group.command("test")
@click.option("--tasks-dir", default="tasks/", help="Tasks directory")
def sandbox_test(tasks_dir: str):
    """Test Docker sandbox setup (build base image, validate tools)."""
    from ai4sci_bench.runner.runtime_root import resolve_runtime_root

    repo_root = resolve_runtime_root(Path(tasks_dir))
    builder = TaskImageBuilder(repo_root)

    click.echo("Checking Docker availability...")
    try:
        builder.ensure_docker_available()
        click.echo("  Docker daemon: OK")
    except RuntimeError as e:
        click.echo(f"  Docker daemon: FAILED ({e})")
        sys.exit(1)

    click.echo("Building/checking base image...")
    try:
        tag = builder.ensure_base_image()
        click.echo(f"  Base image: {tag}")
    except RuntimeError as e:
        click.echo(f"  Base image: FAILED ({e})")
        sys.exit(1)

    click.echo("Validating base image tools...")
    errors = builder.validate_image(tag)
    if errors:
        for err in errors:
            click.echo(f"  FAIL: {err}")
        sys.exit(1)
    click.echo("  All required tools present: OK")

    click.echo("Checking NVIDIA Container Toolkit...")
    if builder.check_nvidia_toolkit_installed():
        click.echo("  nvidia-container-cli: installed")
    else:
        click.echo("  nvidia-container-cli: not found (GPU tasks will not work)")

    click.echo("\nSandbox test PASSED")


@task_group.command("create")
@click.option("--domain", required=True, help="Domain (physics, chemistry, etc.)")
@click.option("--name", required=True, help="Task name (snake_case)")
@click.option("--tasks-dir", default="tasks/", help="Tasks directory")
def new_task(domain: str, name: str, tasks_dir: str):
    """Scaffold a new task from template."""
    import shutil

    template_dir = Path(tasks_dir) / "_template"
    target_dir = Path(tasks_dir) / domain / name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy template files
    if template_dir.exists():
        for f in template_dir.iterdir():
            if f.is_file():
                dest = target_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
    else:
        # Create minimal template
        _create_minimal_template(target_dir, domain, name)

    # Benchmark prompts are distributed with HF instances, not tracked in this
    # repository. Task authors still need local starter prompts in a new scaffold.
    for level in ("b1", "b2", "b3", "b4"):
        prompt_path = target_dir / f"prompt_{level}.md"
        if not prompt_path.exists():
            prompt_path.write_text(
                f"# {name}\n\nTODO: Write prompt for level {level}.\n",
                encoding="utf-8",
            )

    click.echo(f"Created task scaffold: {target_dir}")
    click.echo("Next steps:")
    click.echo(
        f"  1. Edit {target_dir}/task_meta.yaml + task_eval.yaml "
        "(generation remains private after acceptance)"
    )
    click.echo(f"  2. Complete {target_dir}/task_submission.yaml (private Portal evidence)")
    click.echo(f"  3. Write prompts (prompt_b1.md, prompt_b2.md, prompt_b3.md, prompt_b4.md)")
    click.echo(f"  4. Implement generate_gt.py")
    click.echo(f"  5. Run: asibench validate --task {domain}.{name}")


@cli.command("analyze")
@click.option("--results-dir", required=True, help="Results directory (from a prior run or eval)")
@click.option("--only-failed", is_flag=True, help="Only analyze instances that scored below max")
@click.option("--analyze-backend", default="llm_api", help="Analysis backend (llm_api or claude_code)")
@click.option("--analyze-model", default="gemini/gemini-2.0-flash", help="Analysis model")
@click.option("--output-dir", default=None, help="Output directory (defaults to results-dir)")
def analyze_cmd(
    results_dir: str,
    only_failed: bool,
    analyze_backend: str,
    analyze_model: str,
    output_dir: str | None,
):
    """Post-hoc error analysis on existing evaluation results.

    Loads previously saved evaluation results, runs LLM-powered error diagnosis
    on failed (or all) instances, and saves updated results with analysis reports.
    """
    from ai4sci_bench.analysis.error_analyzer import ErrorAnalyzer
    from ai4sci_bench.core.types import AgentOutput, RunStatus

    results_path = Path(results_dir)
    out_path = Path(output_dir) if output_dir else results_path

    if not results_path.exists():
        click.echo(f"Results directory not found: {results_dir}")
        sys.exit(1)

    analyzer = ErrorAnalyzer(
        enabled=True,
        backend=analyze_backend,
        model=analyze_model,
    )

    analyzed = 0
    skipped = 0

    for json_file in sorted(results_path.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if not _is_eval_result_json(data):
                continue
            ensure_supported_result_schema(data)
        except Exception as exc:
            if isinstance(exc, ValueError):
                logger.warning("Skipping unsupported result file %s: %s", json_file, exc)
            continue

        # Skip non-representative retry attempts (only keep best attempt = attempt 1 result)
        if data.get("attempt", 1) > 1:
            continue

        eval_result = _parse_eval_result(data)

        # Skip already-analyzed or perfect scores
        if only_failed and eval_result.final_score >= eval_result.max_possible_score:
            skipped += 1
            continue
        if data.get("error_analysis") and not output_dir:
            skipped += 1
            continue

        # Reconstruct AgentOutput from workspace path
        # The workspace is typically at predictions/<task_id>/<instance_id>/
        workspace_dir = json_file.parent / eval_result.instance_id
        if not workspace_dir.is_dir():
            workspace_dir = json_file.parent

        code_files = [f.name for f in workspace_dir.glob("*.py")]
        agent_output = AgentOutput(
            instance_id=eval_result.instance_id,
            output_dir=workspace_dir,
            code_files=code_files,
            data_files=[f.name for f in workspace_dir.glob("*.npy")],
            log="",
            execution_time_seconds=0.0,
            status=RunStatus.COMPLETED,
        )

        report = analyzer.analyze(eval_result, agent_output)
        if report:
            eval_result.error_analysis = report
            # Save updated result
            if output_dir:
                save_path = out_path / json_file.relative_to(results_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_path = json_file
            _save_eval_result(eval_result, save_path, existing_data=data)
            analyzed += 1
            click.echo(
                f"  [{analyzed}] {eval_result.instance_id}: "
                f"{report.error_category}/{report.error_subcategory} "
                f"(confidence={report.confidence:.0%})"
            )

    click.echo(f"\nAnalyzed {analyzed} instance(s), skipped {skipped}.")


def _save_eval_result(
    eval_result,
    path: Path,
    *,
    existing_data: dict[str, Any] | None = None,
) -> None:
    """Save an EvalResult to a JSON file."""
    from ai4sci_bench.core.types import EvalResult

    data = dict(existing_data or {})
    data.update({
        RESULT_SCHEMA_VERSION_FIELD: CURRENT_RESULT_SCHEMA_VERSION,
        "instance_id": eval_result.instance_id,
        "task_id": eval_result.task_id,
        "prompt_level": eval_result.prompt_level.value,
        "agent_name": eval_result.agent_name,
        "parameters": eval_result.parameters,
        "gates_passed": eval_result.gates_passed,
        "hard_gates_passed": eval_result.hard_gates_passed,
        "soft_gate_failures": eval_result.soft_gate_failures,
        "gate_results": [
            {
                "scorer_name": r.scorer_name,
                "score": r.score,
                "max_score": r.max_score,
                "passed": r.passed,
                "details": r.details,
                "message": r.message,
                "severity": r.severity,
            }
            for r in eval_result.gate_results
        ],
        "score_results": [
            {
                "scorer_name": r.scorer_name,
                "score": r.score,
                "max_score": r.max_score,
                "passed": r.passed,
                "details": r.details,
                "message": r.message,
            }
            for r in eval_result.score_results
        ],
        "final_score": eval_result.final_score,
        "max_possible_score": eval_result.max_possible_score,
        "execution_time_seconds": eval_result.execution_time_seconds,
        "status": eval_result.status.value,
    })

    if eval_result.error_analysis:
        data["error_analysis"] = {
            "error_category": eval_result.error_analysis.error_category,
            "error_subcategory": eval_result.error_analysis.error_subcategory,
            "root_cause": eval_result.error_analysis.root_cause,
            "evidence": eval_result.error_analysis.evidence,
            "fix_suggestions": eval_result.error_analysis.fix_suggestions,
            "confidence": eval_result.error_analysis.confidence,
        }
    else:
        data.pop("error_analysis", None)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@cli.command()
@click.option(
    "--results-dir", "results_dirs", multiple=True, required=True,
    help="Results directory (repeatable; when multiple dirs are given, results are merged and the best score per instance×level is kept)",
)
@click.option("--csv", "csv_path", default=None, help="Export per-task×level scores to a CSV file")
@click.option("--per-task", is_flag=True, default=False, help="Show detailed per-task score table")
def report(results_dirs: tuple[str, ...], csv_path: str | None, per_task: bool):
    """View aggregated report from results.

    Accepts one or more --results-dir options.  When multiple directories are
    provided, results are merged across them and deduplicated: for the same
    (instance_id, prompt_level) the entry with the highest score is kept.
    """
    from ai4sci_bench.reporting.aggregator import aggregate_results

    all_groups: list[list] = []
    for rd in results_dirs:
        rp = Path(rd)
        if not rp.exists():
            click.echo(f"Results directory not found: {rd}")
            sys.exit(1)
        groups = _load_grouped_results_impl(rp)
        if groups:
            all_groups.append(groups)

    if not all_groups:
        click.echo("No results found.")
        return

    if len(all_groups) == 1:
        ordered_groups = all_groups[0]
    else:
        ordered_groups = _merge_result_groups_impl(all_groups)

    if len(results_dirs) > 1:
        click.echo(f"Merged results from {len(results_dirs)} directories.\n")

    if len(ordered_groups) > 1:
        click.echo(f"Detected {len(ordered_groups)} agent groups in results root.\n")

    for idx, group in enumerate(ordered_groups):
        run_report = aggregate_results(group.results)
        if len(ordered_groups) > 1:
            run_report.agent_name = group.label
        click.echo(run_report.format_summary())
        if per_task:
            click.echo(run_report.format_task_table())
        if csv_path and idx == len(ordered_groups) - 1:
            csv_content = run_report.format_csv()
            Path(csv_path).write_text(csv_content + "\n", encoding="utf-8")
            click.echo(f"\nCSV written to {csv_path}")
        elif csv_path and len(ordered_groups) > 1:
            csv_file = Path(csv_path)
            suffix = f"_{group.label}" if group.label else f"_{idx}"
            actual = csv_file.with_stem(csv_file.stem + suffix)
            csv_content = run_report.format_csv()
            actual.write_text(csv_content + "\n", encoding="utf-8")
            click.echo(f"  CSV written to {actual}")
        if idx < len(ordered_groups) - 1:
            click.echo()


@cli.command("batch-report")
@click.option("--results-dir", required=True, help="Batch results root to scan")
def batch_report(results_dir: str):
    """Regenerate derived batch_records/ artifacts from saved result JSON files."""
    results_path = Path(results_dir)
    if not results_path.exists():
        click.echo(f"Results directory not found: {results_dir}")
        sys.exit(1)

    record_paths = _refresh_batch_records_for_root(
        enabled=True,
        batch_records_root=results_dir,
        default_root=results_dir,
    )
    if not record_paths:
        click.echo("No results found.")


def _build_agent(
    agent_cmd: str | None,
    agent_name: str | None,
    agent_config: dict,
    allow_external_tools: bool = False,
    tool_mode: str | None = None,
) -> Any:
    """Build an agent adapter from CLI arguments."""
    agent_config.setdefault("allow_external_tools", allow_external_tools)
    if tool_mode is not None:
        agent_config.setdefault("tool_mode", tool_mode)
    if agent_cmd:
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        agent_config.pop("allow_external_tools", None)
        agent_config.pop("tool_mode", None)
        return CLIAgentAdapter(cmd_template=agent_cmd, **agent_config)
    elif agent_name == "claude_code_cli":
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        return ClaudeCodeCLIAdapter(**agent_config)
    elif agent_name == "direct_llm":
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        agent_config.pop("allow_external_tools", None)
        agent_config.pop("tool_mode", None)
        return DirectLLMAdapter(**agent_config)
    elif agent_name == "codex_cli":
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        return CodexCLIAdapter(**agent_config)
    elif agent_name == "openhands":
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        return OpenHandsAdapter(**agent_config)
    elif agent_name == "hermes":
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        return HermesAgentAdapter(**agent_config)
    elif agent_name == "codewhale":
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        return CodeWhaleAdapter(**agent_config)
    elif agent_name == "kimi_code_cli":
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        return KimiCodeCLIAdapter(**agent_config)
    elif agent_name == "antigravity_cli":
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        return AntigravityCLIAdapter(**agent_config)
    elif agent_name == "mimo_code_cli":
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        return MiMoCodeCLIAdapter(**agent_config)
    else:
        # Default to a dummy adapter for testing
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        return CLIAgentAdapter(cmd_template="echo 'no agent configured'")


def _is_better_result(candidate: Any, existing: Any) -> bool:
    """Return True if candidate is a better result than existing (same selection as orchestrator)."""
    return _is_better_result_impl(candidate, existing)


def _is_eval_result_json(data: dict[str, Any]) -> bool:
    """Return whether a JSON blob matches the saved eval-result schema."""
    return _is_eval_result_json_impl(data)


def _create_minimal_template(target_dir: Path, domain: str, name: str):
    """Create minimal task template files."""
    task_id = f"{domain}.{name}"

    (target_dir / "task_meta.yaml").write_text(f"""# task_meta.yaml (PUBLIC) — public metadata only; NO evaluation/generation
id: {task_id}
name: "{name}"
version: "1.0"
status: in_development
domain: {domain}
subdomain: ""
tags: []

difficulty:
  estimated_lines: 100-300
  estimated_time_minutes: 30-60
  requires_gpu: false
  requires_network: false

prompts:
  b1: prompt_b1.md
  b2: prompt_b2.md
  b3: prompt_b3.md
  b4: prompt_b4.md

input:
  files: []

output:
  files: []
""")

    (target_dir / "task_eval.yaml").write_text(f"""# task_eval.yaml — scoring + generation during authoring.
# Accepted formal tasks publish scoring/output fields only; generation stays private.
task_id: {task_id}

evaluation:
  gates: []
  scoring: []

generation:
  script: generate_gt.py
  mode: infinite
  parameters: {{}}
  # For finite mode with precomputed GT:
  # mode: finite
  # precomputed: true
  # settings:
  #   - {{id: s1, param1: val1}}
  #   - {{id: s2, param1: val2}}
  timeout_seconds: 300
""")

    (target_dir / "task_submission.yaml").write_text("""# task_submission.yaml (PRIVATE, PORTAL-ONLY)
schema_version: 1
scientific_goal: ""
core_method: ""
why_difficult: ""
difficulty_assessment: ""
feasibility_checklist:
  version: 2
  items:
    - {id: deterministic_ground_truth, status: unsure}
    - {id: runtime_budget, status: unsure}
    - {id: offline_runtime, status: unsure}
    - {id: self_contained_inputs, status: unsure}
    - {id: machine_checkable_scoring, status: unsure}
local_testing_done: false
local_test_results: []
""")

    for level in ["b1", "b2", "b3", "b4"]:
        (target_dir / f"prompt_{level}.md").write_text(f"# {name}\n\nTODO: Write prompt for level {level}.\n")

    (target_dir / "generate_gt.py").write_text(f'''"""Parametric ground-truth generator for {task_id}."""

import argparse
import json
import time
from pathlib import Path

INPUT_SPEC = []
OUTPUT_SPEC = []
DEFAULT_PARAMS = {{}}


def generate(output_dir: Path, params: dict) -> dict:
    """Generate one task instance."""
    p = {{**DEFAULT_PARAMS, **params}}
    t0 = time.time()

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = output_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement generation logic

    for level in ["b1", "b2", "b3", "b4"]:
        (output_dir / f"prompt_{{level}}.md").write_text(
            f"# {{p}}\\n\\nTODO: Render prompt for level {{level}}."
        )

    meta = {{
        "params_used": p,
        "input_files": [s["name"] for s in INPUT_SPEC],
        "reference_files": [s["name"] for s in OUTPUT_SPEC],
        "generation_time_seconds": round(time.time() - t0, 2),
    }}
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{{}}")
    args = parser.parse_args()
    result = generate(args.output_dir, json.loads(args.params))
    print(json.dumps(result, indent=2))
''')


def _parse_eval_result(data: dict) -> "EvalResult":
    """Parse an EvalResult from a JSON dict."""
    return _parse_eval_result_impl(data)


def main():
    """Entry point for the CLI."""
    from dotenv import load_dotenv

    load_dotenv(override=True)  # auto-load .env; override stale shell vars
    cli()


def _build_preflight_params(metadata: dict[str, Any], module: Any) -> dict[str, Any]:
    """Choose one representative parameter set for validate --preflight."""
    generation = metadata.get("generation", {})
    if generation.get("mode", "infinite") == "finite":
        settings = generation.get("settings", [])
        if not settings:
            raise ValueError("finite generation mode requires at least one setting for preflight")
        return {k: v for k, v in settings[0].items() if k != "id"}

    params = {
        name: spec.get("default")
        for name, spec in generation.get("parameters", {}).items()
        if "default" in spec
    }
    default_params = getattr(module, "DEFAULT_PARAMS", {})
    merged = {**default_params, **params}
    if generation.get("seed") and "seed" not in merged:
        merged["seed"] = 0
    return merged


def _validate_task_contracts(metadata: dict[str, Any], module: Any, errors: list[str]) -> None:
    """Check task.yaml input/output names against generate_gt specs."""
    input_declared = [f["name"] for f in metadata.get("input", {}).get("files", [])]
    input_spec = [s["name"] for s in getattr(module, "INPUT_SPEC", [])]
    if input_declared != input_spec:
        errors.append(
            f"Input contract mismatch: task.yaml declares {input_declared}, "
            f"generate_gt.INPUT_SPEC declares {input_spec}"
        )

    output_declared = [
        f["name"]
        for f in metadata.get("output", {}).get("files", [])
        if f.get("type") != "code"
    ]
    output_spec = [
        s["name"]
        for s in getattr(module, "OUTPUT_SPEC", [])
        if s.get("type") != "code"
    ]
    if output_declared != output_spec:
        errors.append(
            f"Output contract mismatch: non-code task.yaml outputs declare {output_declared}, "
            f"generate_gt.OUTPUT_SPEC declares {output_spec}"
        )


def _validate_evaluation_templates(
    metadata: dict[str, Any],
    params: dict[str, Any],
    resolve_fn,
    errors: list[str],
) -> None:
    """Ensure parameterized evaluation config resolves for representative params."""
    evaluation = metadata.get("evaluation", {})
    for layer_name in ["gates", "scoring"]:
        for idx, scorer_cfg in enumerate(evaluation.get(layer_name, [])):
            if layer_name == "gates":
                severity = str(scorer_cfg.get("severity", "hard")).strip().lower()
                if severity not in {"hard", "soft"}:
                    errors.append(
                        f"Invalid gate severity in {layer_name}[{idx}] "
                        f"({scorer_cfg.get('scorer', 'unknown')}): {severity}"
                    )

            scorer_name = scorer_cfg.get("scorer", "unknown")
            scoring_mode = (
                scorer_cfg.get("config", {}) or {}
            ).get("scoring_mode", "all_or_nothing")
            if scorer_name in {"file_match", "code_analysis"}:
                if scoring_mode not in {"all_or_nothing", "per_check"}:
                    errors.append(
                        f"Invalid scoring_mode in {layer_name}[{idx}] "
                        f"({scorer_name}): {scoring_mode}"
                    )

            try:
                resolved = resolve_fn(scorer_cfg.get("config", {}), params)
            except Exception as e:
                errors.append(
                    f"Evaluation template resolution failed in {layer_name}[{idx}] "
                    f"({scorer_cfg.get('scorer', 'unknown')}): {e}"
                )
                continue

            unresolved = _find_unresolved_templates(resolved)
            if unresolved:
                errors.append(
                    f"Unresolved evaluation templates in {layer_name}[{idx}] "
                    f"({scorer_cfg.get('scorer', 'unknown')}): {unresolved}"
                )


def _validate_registered_scorers(metadata: dict[str, Any], errors: list[str]) -> None:
    """Ensure every scorer referenced by task.yaml is registered/importable."""
    import ai4sci_bench.scorers  # noqa: F401  # trigger built-in scorer registration
    from ai4sci_bench.core.scorer import get_scorer

    evaluation = metadata.get("evaluation", {})
    for layer_name in ["gates", "scoring"]:
        for idx, scorer_cfg in enumerate(evaluation.get(layer_name, [])):
            scorer_name = scorer_cfg.get("scorer", "unknown")
            try:
                get_scorer(scorer_name)
            except Exception as e:
                errors.append(
                    f"Unknown or unloadable scorer in {layer_name}[{idx}] "
                    f"({scorer_name}): {e}"
                )
                continue

            if scorer_name == "agent_judge":
                from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
                preflight_agent_judge(scorer_cfg.get("config", {}) or {}, errors)


def _validate_preflight_sandbox_request(
    metadata: dict[str, Any],
    sandbox: str,
    errors: list[str],
) -> None:
    """Ensure preflight actually validates the runtime the task declares."""
    runtime_python = metadata.get("_runtime_python")
    runtime_packages = metadata.get("_runtime_packages", [])
    if sandbox in ("task", "os", "linux_ns"):
        return
    if runtime_python or runtime_packages:
        details = []
        if runtime_python:
            details.append(f"runtime.python={runtime_python}")
        if runtime_packages:
            details.append(f"runtime.packages={runtime_packages}")
        errors.append(
            "Task declares runtime requirements "
            f"({', '.join(details)}), but preflight is running with "
            f"--sandbox {sandbox}. Use --sandbox task to validate the real task runtime."
        )


def _validate_preflight_os_sandbox(
    *,
    metadata: dict[str, Any],
    sandbox: str,
    repo_root: Path,
    errors: list[str],
) -> None:
    """Ensure OS sandbox prerequisites are available before a real run."""
    if sandbox != "os":
        return

    try:
        builder = TaskImageBuilder(repo_root)
        image = builder.ensure_image(metadata)
    except Exception as e:
        errors.append(f"OS sandbox preflight failed: {e}")
        return

    # Validate custom image tools if using a custom image
    runtime = metadata.get("runtime", {}) if isinstance(metadata.get("runtime"), dict) else {}
    if runtime.get("image") or runtime.get("dockerfile"):
        validation_errors = builder.validate_image(image)
        for err in validation_errors:
            errors.append(f"Custom image validation: {err}")

    # GPU preflight: check NVIDIA Container Toolkit
    requires_gpu = bool(metadata.get("difficulty", {}).get("requires_gpu"))
    if requires_gpu:
        if not builder.check_nvidia_toolkit_installed():
            errors.append(
                "Task requires GPU but nvidia-container-cli (NVIDIA Container Toolkit) "
                "is not installed. Install nvidia-container-toolkit to use GPU in containers."
            )


def _validate_preflight_linux_ns(
    *,
    sandbox: str,
    errors: list[str],
) -> None:
    """Ensure linux namespace sandbox prerequisites are available."""
    if sandbox != "linux_ns":
        return

    from ai4sci_bench.runner.linux_ns_sandbox import check_linux_ns_available

    available, reason = check_linux_ns_available()
    if not available:
        errors.append(f"Linux namespace sandbox preflight failed: {reason}")


def _validate_preflight_agent_setup(
    *,
    agent_cmd: str | None,
    agent_name: str | None,
    agent_config: dict[str, Any],
    sandbox: str,
    errors: list[str],
) -> None:
    """Optionally validate agent/sandbox compatibility without running the agent."""
    if not agent_cmd and not agent_name:
        return

    adapter = _build_agent(agent_cmd, agent_name, dict(agent_config))
    timeout = agent_config.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    try:
        adapter.setup(
            {
                "timeout": int(timeout),
                "sandbox": sandbox,
                "repo_root": str(Path(__file__).resolve().parents[1]),
            }
        )
    except Exception as e:
        target = agent_name or "agent-cmd"
        errors.append(f"Preflight agent sandbox validation failed for {target}: {e}")
    finally:
        try:
            adapter.teardown()
        except Exception:
            pass


def _find_unresolved_templates(value: Any, path: str = "config") -> list[str]:
    """Find unresolved template-like strings after config resolution."""
    issues: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            issues.extend(_find_unresolved_templates(inner, f"{path}.{key}"))
    elif isinstance(value, builtins.list):
        for idx, inner in enumerate(value):
            issues.extend(_find_unresolved_templates(inner, f"{path}[{idx}]"))
    elif isinstance(value, str):
        stripped = value.strip()
        if "{{" in value or (stripped.startswith("{") and stripped.endswith("}")):
            issues.append(f"{path}={value}")
    return issues


def _validate_generated_instance(instance, metadata: dict[str, Any], errors: list[str]) -> None:
    """Check generated instance layout and prompt rendering."""
    instance_dir = instance.reference_dir.parent
    workspace = instance.workspace_dir
    reference_dir = instance.reference_dir

    if not reference_dir.exists():
        errors.append("Preflight instance missing reference directory")
    if not (workspace / "prompt.md").exists():
        errors.append("Preflight workspace missing prompt.md")
    if not (workspace / "task_info.json").exists():
        errors.append("Preflight workspace missing task_info.json")
    if not (instance_dir / FRAMEWORK_TASK_INFO_FILENAME).exists():
        errors.append(f"Preflight instance missing {FRAMEWORK_TASK_INFO_FILENAME}")
    if metadata.get("input", {}).get("files") and not (workspace / "data").is_dir():
        errors.append("Preflight workspace missing data/ directory")
    if not (instance_dir / "instance_meta.json").exists():
        errors.append("Preflight instance missing instance_meta.json")

    workspace_prompt = workspace / "prompt.md"
    workspace_prompt_text = _read_preflight_prompt_text(
        workspace_prompt,
        "workspace prompt.md",
        errors,
    )
    if workspace_prompt_text is not None and "{{" in workspace_prompt_text:
        errors.append("Rendered prompt still contains template placeholders: workspace prompt.md")

    for prompt_name in metadata.get("prompts", {}).values():
        prompt_path = instance_dir / prompt_name
        if not prompt_path.exists():
            errors.append(f"Preflight instance missing rendered prompt: {prompt_name}")
            continue
        text = _read_preflight_prompt_text(prompt_path, prompt_name, errors)
        if text is None:
            continue
        if "{{" in text:
            errors.append(f"Rendered prompt still contains template placeholders: {prompt_name}")

    task_info = json.loads((workspace / "task_info.json").read_text(encoding="utf-8"))
    framework_task_info = json.loads(
        (instance_dir / FRAMEWORK_TASK_INFO_FILENAME).read_text(encoding="utf-8")
    )
    expected_outputs = [entry["name"] for entry in task_info.get("expected_outputs", [])]
    declared_outputs = [entry["name"] for entry in metadata.get("output", {}).get("files", [])]
    if expected_outputs != declared_outputs:
        errors.append(
            f"task_info expected_outputs mismatch: got {expected_outputs}, expected {declared_outputs}"
        )
    for redacted_field in ("instance_id", "parameters", "requires_gpu"):
        if redacted_field in task_info:
            errors.append(f"task_info should not expose redacted field: {redacted_field}")
    if (workspace / FRAMEWORK_TASK_INFO_FILENAME).exists():
        errors.append(f"Preflight workspace should not expose {FRAMEWORK_TASK_INFO_FILENAME}")
    if framework_task_info.get("instance_id") != instance.instance_id:
        errors.append(
            "framework_task_info instance_id mismatch: "
            f"got {framework_task_info.get('instance_id')}, expected {instance.instance_id}"
        )
    if framework_task_info.get("parameters") != instance.parameters:
        errors.append(
            f"framework_task_info parameters mismatch: got {framework_task_info.get('parameters')}, "
            f"expected {instance.parameters}"
        )

    instance_meta = json.loads((instance_dir / "instance_meta.json").read_text(encoding="utf-8"))
    declared_reference_files = instance_meta.get("reference_files", [])
    available_reference_files = {
        str(path.relative_to(reference_dir))
        for path in reference_dir.rglob("*")
        if path.is_file()
    }
    for ref_file in declared_reference_files:
        if ref_file not in available_reference_files:
            errors.append(
                f"Preflight missing declared reference file in reference/: {ref_file}"
            )

    _validate_preflight_scoring_files(
        metadata,
        instance.parameters,
        reference_dir,
        declared_outputs,
        errors,
    )
    _validate_data_ref_file_no_collision(metadata, instance.parameters, workspace, errors)


def _read_preflight_prompt_text(path: Path, label: str, errors: list[str]) -> str | None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        errors.append(f"Preflight could not read rendered prompt {label}: {exc}")
        return None

    if b"\x00" in data:
        errors.append(f"Rendered prompt contains NUL byte: {label}")
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"Rendered prompt is not valid UTF-8: {label}: {exc}")
        return None


def _validate_preflight_scoring_files(
    metadata: dict[str, Any],
    params: dict[str, Any],
    reference_dir: Path,
    declared_outputs: list[str],
    errors: list[str],
) -> None:
    """Check scorer file references against generated references and declared outputs."""
    from ai4sci_bench.runner.orchestrator import _resolve_config_templates

    available_reference_files = {
        str(path.relative_to(reference_dir))
        for path in reference_dir.rglob("*")
        if path.is_file()
    }
    evaluation = metadata.get("evaluation", {})
    for layer_name in ["gates", "scoring"]:
        for idx, scorer_cfg in enumerate(evaluation.get(layer_name, [])):
            resolved = _resolve_config_templates(scorer_cfg.get("config", {}), params)
            scorer_name = scorer_cfg.get("scorer", "unknown")

            for pred_key in ["pred_file", "file", "target_file"]:
                pred_file = resolved.get(pred_key)
                if isinstance(pred_file, str) and pred_key != "file":
                    if pred_file not in declared_outputs:
                        errors.append(
                            f"{layer_name}[{idx}] ({scorer_name}) references undeclared output file: {pred_file}"
                        )

            if scorer_name == "file_match":
                for check in resolved.get("checks", []):
                    pred_file = check.get("file")
                    if isinstance(pred_file, str) and pred_file not in declared_outputs:
                        errors.append(
                            f"{layer_name}[{idx}] ({scorer_name}) references undeclared output file: {pred_file}"
                        )

            ref_file = resolved.get("ref_file")
            if isinstance(ref_file, str) and ref_file not in available_reference_files:
                errors.append(
                    f"{layer_name}[{idx}] ({scorer_name}) missing reference file in generated instance: {ref_file}"
                )


def _validate_data_ref_file_no_collision(
    metadata: dict[str, Any],
    params: dict[str, Any],
    workspace: Path,
    errors: list[str],
) -> None:
    """Check that data/ filenames do not collide with evaluation ref_file names.

    If a file in ``data/`` (agent-visible) has the same name as a
    ``ref_file`` used by a scorer, an agent could potentially infer
    evaluation details or confuse the framework by overwriting it.
    """
    from ai4sci_bench.runner.orchestrator import _resolve_config_templates

    data_dir = workspace / "data"
    if not data_dir.is_dir():
        return
    data_filenames = {f.name for f in data_dir.iterdir() if f.is_file()}
    if not data_filenames:
        return

    evaluation = metadata.get("evaluation", {})
    for layer_name in ["gates", "scoring"]:
        for idx, scorer_cfg in enumerate(evaluation.get(layer_name, [])):
            resolved = _resolve_config_templates(scorer_cfg.get("config", {}), params)
            scorer_name = scorer_cfg.get("scorer", "unknown")
            ref_file = resolved.get("ref_file")
            if isinstance(ref_file, str) and ref_file in data_filenames:
                errors.append(
                    f"{layer_name}[{idx}] ({scorer_name}) ref_file '{ref_file}' "
                    f"collides with a file in data/ — this may confuse evaluation "
                    f"or leak reference info to the agent"
                )


def _run_task_preflight_hook(
    module: Any,
    instance,
    metadata: dict[str, Any],
    errors: list[str],
) -> None:
    """Run an optional task-defined preflight/self-check hook."""
    hook = getattr(module, "preflight_check", None) or getattr(module, "self_check", None)
    if hook is None:
        return
    if not callable(hook):
        errors.append("Task preflight hook must be callable")
        return

    instance_dir = instance.reference_dir.parent
    workspace_dir = instance.workspace_dir
    params = instance.parameters

    try:
        signature = inspect.signature(hook)
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(
            param.kind == inspect.Parameter.VAR_POSITIONAL
            for param in signature.parameters.values()
        )
        argc = len(positional)
        if has_varargs or argc >= 4:
            result = hook(instance_dir, workspace_dir, params, metadata)
        elif argc == 3:
            result = hook(instance_dir, workspace_dir, params)
        elif argc == 2:
            result = hook(instance_dir, params)
        elif argc == 1:
            result = hook(instance_dir)
        else:
            result = hook()
    except Exception as e:
        errors.append(f"Task preflight hook failed: {e}")
        return

    _normalize_task_preflight_result(result, errors)


def _normalize_task_preflight_result(result: Any, errors: list[str]) -> None:
    """Normalize optional task preflight hook output into validation errors."""
    if result is None or result is True:
        return
    if result is False:
        errors.append("Task preflight hook reported failure")
        return
    if isinstance(result, str):
        if result.strip():
            errors.append(result)
        return
    if isinstance(result, dict):
        messages = result.get("errors", [])
        if isinstance(messages, str):
            messages = [messages]
        if isinstance(messages, list):
            for message in messages:
                if str(message).strip():
                    errors.append(str(message))
            return
    if isinstance(result, (list, tuple, set)):
        for message in result:
            if str(message).strip():
                errors.append(str(message))
        return
    errors.append(
        f"Task preflight hook returned unsupported result type: {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# GT self-check: feed reference outputs back through scorers → must be 100/100
# ---------------------------------------------------------------------------

def _run_gt_selfcheck_command(
    task_id: str | None,
    tasks_dir: str,
    timeout: int,
) -> None:
    """Entry point for ``validate --gt-selfcheck``."""
    from ai4sci_bench.core.task import TaskLoader

    loader = TaskLoader(Path(tasks_dir))

    if task_id:
        task_ids = [task_id]
    else:
        all_tasks = loader.discover_tasks(
            include_test=True, include_sample=True, include_dev=True,
            include_abandoned=False,
        )
        task_ids = [t["id"] for t in all_tasks]

    click.echo(f"GT self-check: {len(task_ids)} task(s)")
    click.echo("(Skipped scorers in self-check: multimodal/llm_judge/agent_judge (API required), code_analysis (agent-style), or skip_in_gt_selfcheck=true)\n")

    summary: list[dict[str, Any]] = []
    for tid in sorted(task_ids):
        result = _gt_selfcheck_one_task(tid, tasks_dir, timeout)
        summary.append(result)
        status = result["status"]
        max_p = result.get("max_possible", 100)
        score_str = (
            f"{result['score']:.2f}/{max_p:.2f}"
            if result["score"] is not None
            else "N/A"
        )
        skipped = result.get("skipped_llm_weight", 0)
        skip_note = f"  (skipped {skipped:.2f} agent-only pts)" if skipped > 0 else ""
        if status == "PASS":
            click.echo(f"  PASS  {tid}  {score_str}{skip_note}")
        elif status == "FAIL":
            click.echo(f"  FAIL  {tid}  {score_str}{skip_note}")
            for detail in result.get("details", []):
                click.echo(f"        {detail}")
        else:
            click.echo(f"  ERROR {tid}  {result.get('error', 'unknown')}")

    click.echo("\n" + "=" * 72)
    click.echo("GT Self-Check Summary")
    click.echo("=" * 72)

    passed = [r for r in summary if r["status"] == "PASS"]
    failed = [r for r in summary if r["status"] == "FAIL"]
    errored = [r for r in summary if r["status"] == "ERROR"]

    click.echo(f"  PASS:  {len(passed)}")
    click.echo(f"  FAIL:  {len(failed)}")
    click.echo(f"  ERROR: {len(errored)}")

    if failed:
        click.echo("\nFailed tasks:")
        for r in failed:
            max_p = r.get("max_possible", 100)
            click.echo(f"  {r['task_id']}  {r['score']:.2f}/{max_p:.2f}")
            for detail in r.get("details", []):
                click.echo(f"    {detail}")

    if errored:
        click.echo("\nErrored tasks:")
        for r in errored:
            click.echo(f"  {r['task_id']}  {r.get('error', '')}")

    click.echo()
    if failed or errored:
        sys.exit(1)


# Scorers skipped in GT self-check:
#   - LLM-based scorers need API credentials
#   - code_analysis checks agent code style (forbidden imports, line counts,
#     "_ref" references). The GT generator legitimately violates these because
#     it IS the reference. code_analysis is meant for agent-produced analysis.py.
_LLM_SCORER_NAMES = frozenset({"multimodal", "llm_judge", "agent_judge", "code_analysis"})


def _run_generate_with_timeout(module: Any, output_dir: Path, params: dict, timeout: int) -> None:
    """Run module.generate() with a wall-clock timeout.

    Unix-like platforms use ``signal.alarm`` to preserve the historical
    in-process behavior. Windows does not provide SIGALRM, so it falls back to
    an isolated subprocess that can be terminated by ``subprocess.run``.
    """
    import signal

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "alarm"):
        _run_generate_with_subprocess_timeout(module, output_dir, params, timeout)
        return

    class _Timeout(TimeoutError):
        pass

    def _handler(signum, frame):
        raise _Timeout(f"generate() timed out after {timeout}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        module.generate(output_dir, params)
    except _Timeout:
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_generate_with_subprocess_timeout(
    module: Any,
    output_dir: Path,
    params: dict,
    timeout: int,
) -> None:
    """Cross-platform generate() timeout fallback for platforms without SIGALRM."""
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(
            "Cannot run generate() timeout fallback because generate_gt module "
            "has no __file__"
        )

    payload = json.dumps(params)
    code = """
import importlib.util
import json
import sys
from pathlib import Path

module_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
params = json.loads(sys.argv[3])
spec = importlib.util.spec_from_file_location("ai4sci_gt_selfcheck_generate", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load generate_gt module from {module_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.generate(output_dir, params)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(module_file), str(output_dir), payload],
            cwd=str(Path(module_file).parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"generate() timed out after {timeout}s") from e

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"generate() failed in subprocess: {detail}")


def _gt_selfcheck_one_task(
    task_id: str,
    tasks_dir: str,
    timeout: int,
) -> dict[str, Any]:
    """Run GT self-check for a single task. Returns a result dict."""
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.runner.orchestrator import (
        _evaluate_gates_and_scores,
        _resolve_config_templates,
    )
    from ai4sci_bench.scorers.custom import load_custom_scorer

    loader = TaskLoader(Path(tasks_dir))
    result: dict[str, Any] = {"task_id": task_id, "score": None, "status": "ERROR"}

    try:
        metadata = loader.load_task_by_id(task_id)
    except ValueError as e:
        result["error"] = f"task not found: {e}"
        return result

    task_dir: Path = metadata["_task_dir"]
    if metadata.get("status") == "abandoned":
        result["error"] = "abandoned task"
        return result

    # Load generate_gt module
    try:
        module = loader.load_generate_gt_module(task_dir)
    except Exception as e:
        result["error"] = f"cannot load generate_gt.py: {e}"
        return result

    if not hasattr(module, "generate"):
        result["error"] = "generate_gt.py missing generate()"
        return result

    # Load custom scorers
    try:
        load_custom_scorer(task_dir)
    except Exception as e:
        result["error"] = f"cannot load custom_scorer.py: {e}"
        return result

    # Build params
    params = _build_preflight_params(metadata, module)

    # Strip LLM-dependent scorers from evaluation for self-check
    evaluation = _strip_llm_scorers(metadata.get("evaluation", {}))

    # Generate instance in temp dir
    try:
        with tempfile.TemporaryDirectory(prefix="ai4sci_gt_selfcheck_") as tmp:
            tmp_path = Path(tmp)
            _run_generate_with_timeout(module, tmp_path, params, timeout)

            ref_dir = tmp_path / "reference"
            if not ref_dir.exists() or not any(ref_dir.iterdir()):
                result["error"] = "generate() produced no reference/ directory"
                return result

            # Read resolved params from instance_meta.json if available
            meta_path = tmp_path / "instance_meta.json"
            if meta_path.exists():
                try:
                    instance_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    params_used = instance_meta.get("params_used")
                    if isinstance(params_used, dict):
                        params = params_used
                except json.JSONDecodeError:
                    pass

            # Build pred_dir: simulate a perfect agent
            pred_dir = tmp_path / "_pred"
            pred_dir.mkdir()

            # Copy data/ directory into pred_dir (agents always have data/ in workspace)
            data_dir = tmp_path / "data"
            if data_dir.exists():
                shutil.copytree(data_dir, pred_dir / "data")

            warnings = _build_gt_pred_dir(
                pred_dir=pred_dir,
                ref_dir=ref_dir,
                task_dir=task_dir,
                metadata=metadata,
                module=module,
            )

            # Run evaluation (with LLM scorers stripped)
            gate_results, hard_gates_passed, soft_gate_failures, score_results, final_score = (
                _evaluate_gates_and_scores(evaluation, pred_dir, ref_dir, params)
            )

            # Compute max possible score (after stripping LLM scorers)
            max_possible = sum(
                s.get("weight", 1.0) for s in evaluation.get("scoring", [])
            )
            skipped_weight = _skipped_llm_weight(metadata.get("evaluation", {}))

            result["score"] = final_score
            result["max_possible"] = max_possible
            result["skipped_llm_weight"] = skipped_weight
            details: list[str] = []
            if warnings:
                for w in warnings:
                    details.append(f"[warning] {w}")

            if skipped_weight > 0:
                details.append(
                    f"[info] skipped {skipped_weight:.0f} pts of agent-only scorers (LLM/code_analysis)"
                )

            if not hard_gates_passed:
                details.append("[hard gate FAILED]")
                for g in gate_results:
                    if not g.passed and g.severity == "hard":
                        details.append(f"  gate {g.scorer_name}: {g.message}")

            if soft_gate_failures > 0:
                for g in gate_results:
                    if not g.passed and g.severity == "soft":
                        details.append(f"[soft gate warning] {g.scorer_name}: {g.message}")

            for s in score_results:
                if s.score < s.max_score:
                    details.append(
                        f"  scorer {s.scorer_name}: {s.score:.2f}/{s.max_score:.2f} — {s.message}"
                    )

            result["details"] = details
            if hard_gates_passed and (
                max_possible == 0 or abs(final_score - max_possible) < 0.01
            ):
                result["status"] = "PASS"
            else:
                result["status"] = "FAIL"

    except Exception as e:
        result["error"] = f"generate failed: {type(e).__name__}: {e}"

    return result


def _strip_llm_scorers(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the evaluation config with LLM-dependent scorers removed."""
    import copy
    ev = copy.deepcopy(evaluation)
    ev["gates"] = [
        g for g in ev.get("gates", [])
        if not _is_gt_selfcheck_skipped_scorer(g)
    ]
    ev["scoring"] = [
        s for s in ev.get("scoring", [])
        if not _is_gt_selfcheck_skipped_scorer(s)
    ]
    return ev


def _is_gt_selfcheck_skipped_scorer(entry: dict[str, Any]) -> bool:
    """Return whether a gate/score entry should be skipped in GT self-check."""
    return bool(entry.get("skip_in_gt_selfcheck")) or entry.get("scorer") in _LLM_SCORER_NAMES


def _skipped_llm_weight(evaluation: dict[str, Any]) -> float:
    """Sum the weight of LLM-dependent scorers that were skipped."""
    total = 0.0
    for s in evaluation.get("scoring", []):
        if _is_gt_selfcheck_skipped_scorer(s):
            total += s.get("weight", 1.0)
    return total


def _build_gt_pred_dir(
    pred_dir: Path,
    ref_dir: Path,
    task_dir: Path,
    metadata: dict[str, Any],
    module: Any,
) -> list[str]:
    """Populate pred_dir with files as if a perfect agent produced them.

    Returns a list of warnings (e.g., missing figure files).
    """
    warnings: list[str] = []
    output_files = metadata.get("output", {}).get("files", [])
    evaluation = metadata.get("evaluation", {})

    # Step 1: Build pred_file → ref_file mapping from scorer configs
    pred_to_ref = _extract_pred_ref_mapping(evaluation)

    # Build reverse mapping: ref_name → list of pred_names
    ref_to_pred: dict[str, list[str]] = {}
    for pf, rf in pred_to_ref.items():
        ref_to_pred.setdefault(rf, []).append(pf)

    # Step 2: Copy files for each declared output
    for output_file in output_files:
        name = output_file["name"]
        file_type = output_file.get("type", "data")
        target = pred_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)

        if file_type == "code":
            gt_path = task_dir / "generate_gt.py"
            if gt_path.exists():
                shutil.copy2(gt_path, target)
            else:
                warnings.append(f"code file {name}: generate_gt.py not found")
        elif file_type == "figure":
            placed = _try_place_figure(name, pred_dir, ref_dir, pred_to_ref)
            if not placed:
                try:
                    _create_placeholder_png(target)
                    warnings.append(f"figure {name}: no reference found, using placeholder")
                except Exception:
                    target.write_bytes(b"")
                    warnings.append(f"figure {name}: no reference found, empty placeholder")
        else:
            # type: data — find corresponding reference file
            if not _place_data_file(name, target, ref_dir, pred_to_ref):
                warnings.append(f"data file {name}: no matching reference found")

    # Step 3: Copy any extra files scorers reference from pred_dir
    _copy_extra_scorer_files(pred_dir, ref_dir, evaluation, pred_to_ref)

    # Step 4: Copy any reference files NOT ending in _ref* to pred_dir.
    # These are "checkpoint"/exploration artifacts that the GT also produces
    # (e.g., exploration_summary.json, diagnostic_report.json) that agent
    # scorers look up in pred_dir.
    _ref_only_suffixes = ("_ref",)
    for src in ref_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(ref_dir)
        stem = src.stem
        # Skip files whose stem ends with _ref (these are reference-only)
        if any(stem.endswith(suffix) for suffix in _ref_only_suffixes):
            continue
        # Skip if pred_to_ref already maps something to this name
        if str(rel) in pred_to_ref.values():
            continue
        target = pred_dir / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    return warnings


def _place_data_file(
    name: str,
    target: Path,
    ref_dir: Path,
    pred_to_ref: dict[str, str],
) -> bool:
    """Try to find and copy the right reference file for a data output. Returns True on success."""
    # 1. Explicit scorer mapping
    ref_name = pred_to_ref.get(name)
    if ref_name and (ref_dir / ref_name).exists():
        shutil.copy2(ref_dir / ref_name, target)
        return True

    # 2. Same name in reference
    if (ref_dir / name).exists():
        shutil.copy2(ref_dir / name, target)
        return True

    # 3. Common _ref naming patterns
    for candidate in _ref_name_candidates(name):
        if (ref_dir / candidate).exists():
            shutil.copy2(ref_dir / candidate, target)
            return True

    # 4. Basename match (strip subdirectory prefix)
    basename = Path(name).name
    stem = Path(basename).stem
    suffix = Path(basename).suffix

    # Try basename directly in ref_dir
    if (ref_dir / basename).exists():
        shutil.copy2(ref_dir / basename, target)
        return True

    # Try basename with _ref suffix
    ref_basename = f"{stem}_ref{suffix}"
    if (ref_dir / ref_basename).exists():
        shutil.copy2(ref_dir / ref_basename, target)
        return True

    # 5. Fuzzy match: any ref file whose stem contains our stem or vice versa
    for rf in sorted(ref_dir.rglob("*")):
        if rf.is_file() and rf.suffix == suffix and (stem in rf.stem or rf.stem in stem):
            shutil.copy2(rf, target)
            return True

    return False


def _extract_pred_ref_mapping(evaluation: dict[str, Any]) -> dict[str, str]:
    """Extract pred_file → ref_file mappings from all scorer configs.

    Handles standard keys (pred_file/ref_file) and custom scorer patterns
    where paired keys like ``curve_file`` / ``reference_curve_file`` define
    the mapping.
    """
    mapping: dict[str, str] = {}

    all_configs: list[dict] = []
    for gate in evaluation.get("gates", []):
        all_configs.append(gate.get("config", {}))
    for scorer_cfg in evaluation.get("scoring", []):
        all_configs.append(scorer_cfg.get("config", {}))
        for sub in scorer_cfg.get("config", {}).get("sub_scorers", []):
            all_configs.append(sub.get("config", {}))

    for config in all_configs:
        # Standard pred_file / ref_file
        pred_file = config.get("pred_file")
        ref_file = config.get("ref_file")
        if pred_file and ref_file and pred_file != ref_file:
            mapping[pred_file] = ref_file

        single_file = config.get("file")
        if single_file and single_file not in mapping:
            mapping[single_file] = single_file

        # Detect paired keys: <X>_file / reference_<X>_file
        file_keys = [k for k in config if k.endswith("_file") and isinstance(config[k], str)]
        for key in file_keys:
            val = config[key]
            if key.startswith("reference_"):
                # reference_curve_file → curve_file
                pred_key = key[len("reference_"):]
                if pred_key in config and isinstance(config[pred_key], str):
                    mapping[config[pred_key]] = val
            elif f"reference_{key}" in config:
                ref_val = config[f"reference_{key}"]
                if isinstance(ref_val, str):
                    mapping[val] = ref_val

        # Also handle pred_image / ref_image (multimodal scorer)
        pred_img = config.get("pred_image")
        ref_img = config.get("ref_image")
        if pred_img and ref_img:
            mapping[pred_img] = ref_img

    return mapping


def _ref_name_candidates(pred_name: str) -> list[str]:
    """Generate candidate reference file names for a given output file name."""
    p = Path(pred_name)
    stem = p.stem
    suffix = p.suffix
    parent = p.parent

    candidates = [
        str(parent / f"{stem}_ref{suffix}") if str(parent) != "." else f"{stem}_ref{suffix}",
        str(parent / f"ref_{stem}{suffix}") if str(parent) != "." else f"ref_{stem}{suffix}",
        str(parent / f"{stem}_reference{suffix}") if str(parent) != "." else f"{stem}_reference{suffix}",
    ]
    # Also try without subdirectory prefix
    if str(parent) != ".":
        candidates.append(f"{stem}{suffix}")
        candidates.append(f"{stem}_ref{suffix}")
    return candidates


def _try_place_figure(
    name: str,
    pred_dir: Path,
    ref_dir: Path,
    pred_to_ref: dict[str, str],
) -> bool:
    """Try to find and copy a reference figure. Returns True if successful."""
    target = pred_dir / name

    # Check explicit mapping
    ref_name = pred_to_ref.get(name)
    if ref_name and (ref_dir / ref_name).exists():
        shutil.copy2(ref_dir / ref_name, target)
        return True

    # Check same name in reference
    if (ref_dir / name).exists():
        shutil.copy2(ref_dir / name, target)
        return True

    # Check common ref naming
    for candidate in _ref_name_candidates(name):
        if (ref_dir / candidate).exists():
            shutil.copy2(ref_dir / candidate, target)
            return True

    # Search for any image file with similar name
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf"):
        for rf in ref_dir.rglob(f"*{ext}"):
            if rf.is_file() and stem in rf.stem:
                shutil.copy2(rf, target)
                return True

    return False


def _copy_extra_scorer_files(
    pred_dir: Path,
    ref_dir: Path,
    evaluation: dict[str, Any],
    pred_to_ref: dict[str, str],
) -> None:
    """Copy any additional files that scorers expect in pred_dir but are not in output.files."""
    # For each pred_file referenced in scorer configs, ensure it exists in pred_dir
    for pred_name, ref_name in pred_to_ref.items():
        target = pred_dir / pred_name
        if target.exists():
            continue
        source = ref_dir / ref_name
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _create_placeholder_png(path: Path) -> None:
    """Create a minimal valid 1x1 white PNG file."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        raw = chunk_type + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_data = b"\x00\xff\xff\xff"  # filter byte + RGB white
    idat = _chunk(b"IDAT", zlib.compress(raw_data))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


def _run_static_or_presubmit(
    static_dir: str | None,
    pre_submit_dir: str | None,
    tasks_dir: str,
    gt_timeout: int = 300,
    scores_dir: str | None = "scores/",
) -> None:
    """Handle --static and --pre-submit validate modes."""
    import tempfile

    from ai4sci_bench.validators.static_validator import validate_task_static

    target_dir = Path(pre_submit_dir or static_dir)  # type: ignore[arg-type]
    tasks_root = Path(tasks_dir)
    scores_root = Path(scores_dir) if scores_dir else None

    from ai4sci_bench.core.task import resolve_task_sources
    try:
        resolve_task_sources(target_dir)
    except FileNotFoundError:
        click.echo(f"FAIL: No task.yaml / task_meta.yaml found in {target_dir}")
        sys.exit(1)

    click.echo(f"{'Pre-submit' if pre_submit_dir else 'Static'} validation: {target_dir}")

    click.echo("\n── Static Checks ──")
    result = validate_task_static(target_dir, tasks_root=tasks_root, scores_dir=scores_root)

    for w in result.warnings:
        click.echo(f"  WARN: {w}")
    for e in result.errors:
        click.echo(f"  FAIL: {e}")

    if result.passed:
        click.echo("  Static checks: PASSED")
    else:
        click.echo(f"\n  Static checks: FAILED ({len(result.errors)} error(s))")
        if pre_submit_dir:
            click.echo("  Skipping GT generation and self-check due to static failures.")
        sys.exit(1)

    if not pre_submit_dir:
        return

    click.echo("\n── GT Generation ──")
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.generators.instance_generator import InstanceGenerator

    loader = TaskLoader(tasks_root)
    from ai4sci_bench.core.task import resolve_task_sources
    try:
        meta_path, eval_path = resolve_task_sources(target_dir)
        metadata = loader.load_task_metadata(meta_path)
    except Exception as exc:
        click.echo(f"  FAIL: Could not load task metadata: {exc}")
        sys.exit(1)

    import random as _random

    default_params = {}
    gen_cfg = metadata.get("generation", {})
    if isinstance(gen_cfg, dict):
        for pname, pdef in gen_cfg.get("parameters", {}).items():
            if isinstance(pdef, dict) and "default" in pdef:
                default_params[pname] = pdef["default"]

    seeds = [default_params.get("seed", 42)]
    rng = _random.Random(12345)
    while len(seeds) < 3:
        s = rng.randint(0, 999999)
        if s not in seeds:
            seeds.append(s)

    gt_errors: list[str] = []
    for seed in seeds:
        params = {**default_params, "seed": seed}
        click.echo(f"  Generating GT with seed={seed} ...")
        try:
            with tempfile.TemporaryDirectory(prefix="ai4sci_presubmit_") as tmp:
                generator = InstanceGenerator(tasks_root)
                from ai4sci_bench.runner.orchestrator import resolve_instance_timeout

                eff_timeout = resolve_instance_timeout(metadata, None)
                generator.generate_instance(
                    metadata, params, Path(tmp),
                    effective_timeout_seconds=eff_timeout,
                )
                click.echo(f"    seed={seed}: OK")
        except Exception as exc:
            gt_errors.append(f"GT generation failed (seed={seed}): {exc}")
            click.echo(f"    seed={seed}: FAIL — {exc}")

    if gt_errors:
        click.echo(f"\n  GT Generation: FAILED ({len(gt_errors)} error(s))")
        sys.exit(1)
    click.echo("  GT Generation: PASSED")

    click.echo("\n── GT Self-check ──")
    task_id = metadata.get("id", "unknown")
    try:
        result_code = subprocess.run(
            [sys.executable, "-m", "ai4sci_bench.cli", "validate",
             "--gt-selfcheck", "--task", task_id,
             "--tasks-dir", str(tasks_root),
             "--gt-timeout", str(gt_timeout)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=gt_timeout + 60,
        )
        if result_code.stdout.strip():
            for line in result_code.stdout.strip().splitlines():
                click.echo(f"  {line}")
        if result_code.returncode == 0:
            click.echo("  GT Self-check: PASSED")
            click.echo(f"\nPre-submit validation: PASSED for {task_id}")
        else:
            if result_code.stderr.strip():
                click.echo(result_code.stderr.strip())
            click.echo("  GT Self-check: FAILED")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        click.echo("  GT Self-check: TIMEOUT")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"  GT Self-check: ERROR — {exc}")
        sys.exit(1)




def _parse_agent_config(raw: str) -> dict:
    """Parse agent config JSON with fallback for PowerShell quote stripping."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        import yaml

        result = yaml.safe_load(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    raise click.BadParameter(
        f"Cannot parse agent config: {raw!r}\n"
        "Hint: on PowerShell, escape inner double-quotes with backslash:\n"
        '  --agent \'codex_cli:{\\"model\\":\\"gpt-5.4\\"}\'',
        param_hint="--agent",
    )


@cli.command("difficulty-check")
@click.option("--task", "task_id", required=False, default=None,
              help="Task ID, e.g. physics.new_task. Mutually exclusive with --status.")
@click.option("--status", "status_filter", default=None,
              type=click.Choice(["in_development", "test", "sample", "final"]),
              help="Run difficulty-check against every task of the given status (batch re-eval).")
@click.option("--agent", "agents", multiple=True, default=("direct_llm",),
              show_default=True,
              help="Agent(s) to use for difficulty evaluation (repeatable).")
@click.option("--agent-config", "agent_configs", multiple=True,
              help="Agent config JSON, one per --agent (matched by position).")
@click.option("--prompt-levels", default="b1,b2,b3,b4", show_default=True,
              help="Prompt levels to evaluate.")
@click.option("--threshold", default=50, show_default=True, type=int,
              help="Difficulty threshold (max score an LLM should get).")
@click.option("--instances-per-task", default=1, type=int, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--sandbox", default="none", show_default=True,
              help="Sandbox mode for the evaluated agent: none, task, os, linux_ns.")
@click.option("--timeout", default=DEFAULT_TIMEOUT_SECONDS, show_default=True, type=int,
              help="Agent timeout in seconds. Task metadata cannot override this value.")
@click.option("--tasks-dir", default="tasks/", show_default=True)
@click.option("--output", "output_path", default=None, type=click.Path(),
              help="Write structured JSON report to this path.")
@click.option("--markdown", "markdown_path", default=None, type=click.Path(),
              help="Write Markdown report to this path.")
@click.option("--csv-output", "csv_path", default=None, type=click.Path(),
              help="Write per-row CSV summary (batch mode).")
@click.option("--save-scores/--no-save-scores", default=True, show_default=True,
              help="Append evaluation records to scores/<task_id>.json.")
@click.option("--scores-dir", default="scores/", show_default=True,
              help="Directory for score records.")
@click.option("--trigger", default="manual", show_default=True,
              help="Trigger label stored in the score record (manual|ci|re-evaluation).")
@click.option("--trigger-ref", default=None,
              help="Trigger reference (e.g. pr#142) stored in the score record.")
@click.option("--no-color", is_flag=True, default=False,
              help="Disable color in terminal output.")
def difficulty_check(
    task_id, status_filter, agents, agent_configs, prompt_levels, threshold,
    instances_per_task, seed, sandbox, timeout, tasks_dir, output_path,
    markdown_path, csv_path, save_scores, scores_dir, trigger, trigger_ref,
    no_color,
):
    """Check whether a task is hard enough for the benchmark.

    Runs the specified agent(s) without external tools, compares mean scores
    against the threshold, and prints a pass/fail verdict.

    Exit codes:
      0 — all tasks passed (hard enough)
      1 — at least one task failed (too easy)
      2 — at least one task ABORTED due to agent infrastructure failure
          (subprocess error, API failure, timeout). Aborted tasks are NOT
          scored, persisted, or reported — the evaluation effectively did
          not happen, so no verdict is implied.

    A task PASSES if EVERY (agent, prompt_level) combination scores strictly
    below the threshold. With --status, runs the check against every task of
    the given status and prints a summary table.
    """
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.reporting.difficulty_report import (
        DifficultyReport,
        build_report,
        format_batch_summary,
        format_markdown,
        format_terminal,
        write_batch_csv,
        write_json,
        write_markdown,
    )
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
    from ai4sci_bench.tracking.difficulty_scores import append_evaluation

    if (task_id is None) == (status_filter is None):
        raise click.ClickException("Provide exactly one of --task or --status.")

    parsed_levels = [lvl.strip() for lvl in prompt_levels.split(",") if lvl.strip()]
    if not parsed_levels:
        raise click.ClickException("--prompt-levels must list at least one level.")

    if agent_configs and len(agent_configs) != len(agents):
        raise click.ClickException(
            f"--agent-config count ({len(agent_configs)}) must match --agent count ({len(agents)})"
        )
    if not agent_configs:
        agent_configs = tuple("{}" for _ in agents)

    parsed_configs: list[dict[str, Any]] = []
    for raw in agent_configs:
        parsed_configs.append(_parse_agent_config(raw))

    # Always restrict tool access during difficulty checks
    resolved_tool_mode = "restricted"

    # Determine which tasks to evaluate
    loader = TaskLoader(Path(tasks_dir))
    if task_id:
        task_id_list = [task_id]
    else:
        task_id_list = []
        for yaml_path in loader._iter_task_source_files():
            if yaml_path.parent.name == "_template":
                continue
            try:
                meta = loader.load_task_metadata(yaml_path)
            except Exception:
                continue
            if meta.get("status") == status_filter and meta.get("id"):
                task_id_list.append(str(meta["id"]))
        if not task_id_list:
            raise click.ClickException(f"No tasks found with status={status_filter}")

    # Build (label, name, config, adapter) for each agent — adapters are stateless
    # enough that we rebuild them per task; that keeps RunConfig.agent.setup() clean.
    def _build_agents_for_run() -> list[tuple[str, str, dict[str, Any], Any]]:
        out = []
        for name, cfg in zip(agents, parsed_configs):
            cfg_copy = dict(cfg)
            try:
                adapter = _build_agent(
                    None, name, cfg_copy,
                    allow_external_tools=False,
                    tool_mode=resolved_tool_mode,
                )
            except (ValueError, TypeError) as exc:
                raise click.ClickException(f"Agent '{name}': {exc}") from exc
            label = _derive_agent_label(name, cfg)
            out.append((label, name, cfg, adapter))
        return out

    summary_rows: list[tuple[str, bool, str]] = []  # (task_id, passed, terminal_text)
    batch_reports: list[DifficultyReport] = []
    failures = 0
    aborts: list[tuple[str, str]] = []  # (task_id, reason) for infra-failure aborts

    for tid in task_id_list:
        try:
            task_meta = loader.load_task_by_id(tid)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
        task_version = str(task_meta.get("version", ""))
        task_status = str(task_meta.get("status", ""))

        # status_filter mode already gated by status; for --task, allow any status.
        include_test = task_status in ("test", "final")
        include_sample = task_status == "sample"
        include_dev = task_status == "in_development"
        include_abandoned = task_status == "abandoned"

        agent_runs: list[tuple[str, str, dict[str, Any], list]] = []
        agent_specs = _build_agents_for_run()

        with tempfile.TemporaryDirectory(prefix=f"difficulty_{tid}_") as tmp_root:
            for label, name, cfg, adapter in agent_specs:
                agent_out = Path(tmp_root) / label
                config = RunConfig(
                    agent=adapter,
                    tasks=[tid],
                    include_test=include_test,
                    include_sample=include_sample,
                    include_dev=include_dev,
                    include_abandoned=include_abandoned,
                    prompt_levels=parsed_levels,
                    seed=seed,
                    instances_per_task=instances_per_task,
                    sandbox=sandbox,
                    timeout=timeout,
                    output_dir=str(agent_out),
                    tasks_dir=tasks_dir,
                    agent_metadata=_build_agent_metadata(
                        None, name, dict(cfg),
                        allow_external_tools=False,
                        tool_mode=resolved_tool_mode,
                    ),
                )
                try:
                    orchestrator = BenchmarkOrchestrator(config)
                except ValueError as exc:
                    raise click.ClickException(str(exc)) from exc
                run_report = orchestrator.run([tid])
                agent_runs.append((label, name, dict(cfg), list(run_report.results)))

        # Detect agent infrastructure failure — if any instance crashed (subprocess
        # error, API failure, timeout), this evaluation didn't actually happen.
        # Treat as ABORT: skip score persistence, skip report writing, no verdict.
        infra_failed = []
        for label, name, cfg, results in agent_runs:
            for r in results:
                if r.status in (RunStatus.FAILED, RunStatus.TIMEOUT):
                    err = "unknown"
                    if r.agent_output and r.agent_output.error_message:
                        err = r.agent_output.error_message
                    infra_failed.append((label, str(r.prompt_level.value) if hasattr(r.prompt_level, "value") else str(r.prompt_level), r.status.value, err))

        if infra_failed:
            abort_msg = (
                f"  ✗ ABORT: {tid} — agent infrastructure failure on "
                f"{len(infra_failed)} instance(s); evaluation did not run.\n"
                f"    First failure: agent={infra_failed[0][0]} level={infra_failed[0][1]} "
                f"status={infra_failed[0][2]}\n"
                f"    Error: {infra_failed[0][3][:200]}"
            )
            if not no_color:
                abort_msg = click.style(abort_msg, fg="red")
            click.echo(abort_msg)
            reason = f"{infra_failed[0][2]}: {infra_failed[0][3][:120]}"
            aborts.append((tid, reason))
            continue  # Skip scoring, persistence, report writing for this task.

        report = build_report(
            tid,
            agent_runs,
            threshold,
            tool_mode=resolved_tool_mode,
            sandbox=sandbox,
            instances_per_task=instances_per_task,
            seeds=[seed],
            task_version=task_version,
        )

        terminal_text = format_terminal(report, use_color=not no_color)
        click.echo(terminal_text)

        if save_scores:
            try:
                saved_path = append_evaluation(
                    scores_dir=scores_dir,
                    task_id=tid,
                    task_version=task_version,
                    results=report.to_per_agent_results(),
                    threshold=threshold,
                    verdict="pass" if report.overall_pass else "fail",
                    trigger=trigger,
                    trigger_ref=trigger_ref,
                    tool_mode=resolved_tool_mode,
                    sandbox=sandbox,
                    instances_per_task=instances_per_task,
                    seeds=[seed],
                )
                click.echo(f"  Score record updated: {saved_path}")
            except Exception as exc:
                click.echo(click.style(
                    f"  WARNING: failed to persist scores ({exc})", fg="yellow"
                ))

        if output_path and (task_id or len(task_id_list) == 1):
            write_json(report, output_path)
            click.echo(f"  JSON report: {output_path}")
        elif output_path:
            # In batch mode, fan out: <output>/<task_id>.json
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(report, out_dir / f"{tid}.json")

        if markdown_path and (task_id or len(task_id_list) == 1):
            write_markdown(report, markdown_path)
            click.echo(f"  Markdown report: {markdown_path}")
        elif markdown_path:
            md_dir = Path(markdown_path)
            md_dir.mkdir(parents=True, exist_ok=True)
            write_markdown(report, md_dir / f"{tid}.md")

        summary_rows.append((tid, report.overall_pass, terminal_text))
        batch_reports.append(report)
        if not report.overall_pass:
            failures += 1

    if status_filter and len(summary_rows) > 1:
        click.echo()
        click.echo("=" * 63)
        click.echo(f"  Batch summary — {status_filter} ({len(summary_rows)} tasks)")
        click.echo("=" * 63)
        for tid, passed, _ in summary_rows:
            word = "PASS" if passed else "FAIL"
            if not no_color:
                word = click.style(word, fg="green" if passed else "red", bold=True)
            click.echo(f"  {word}  {tid}")
        click.echo()
        click.echo(f"  {len(summary_rows) - failures}/{len(summary_rows)} passed")

        click.echo()
        click.echo(format_batch_summary(
            batch_reports,
            status_filter=status_filter,
            threshold=threshold,
            use_color=not no_color,
            scores_dir=scores_dir if save_scores else None,
        ))

    if csv_path:
        write_batch_csv(batch_reports, csv_path)
        click.echo(f"  CSV summary: {csv_path}")

    if aborts:
        click.echo()
        msg = f"  ABORTED ({len(aborts)} task[s] — infrastructure failure, no scores recorded):"
        if not no_color:
            msg = click.style(msg, fg="red", bold=True)
        click.echo(msg)
        for tid, reason in aborts:
            click.echo(f"    {tid}: {reason}")

    # Exit codes: 0 = all passed, 1 = some failed, 2 = aborted (infra failure)
    if aborts:
        sys.exit(2)
    sys.exit(0 if failures == 0 else 1)


def _latest_evaluation_date(score_data: dict[str, Any] | None) -> str | None:
    """Return the most recent evaluation date (YYYY-MM-DD) or None."""
    if not score_data:
        return None
    evaluations = score_data.get("evaluations") or []
    if not evaluations:
        return None
    raw = evaluations[-1].get("date")
    if not raw:
        return None
    return str(raw).split("T", 1)[0]


def _collect_domain_summary(
    tasks: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Aggregate tasks by domain and stitch in latest-check dates.

    Returns (rows, totals). Each row is one domain; ``totals`` is a flat
    ``{status: count}`` dict for the table footer.
    """
    by_domain: dict[str, dict[str, Any]] = {}
    for t in tasks:
        domain = str(t.get("domain") or "unknown")
        row = by_domain.setdefault(domain, {
            "domain": domain,
            "final": 0,
            "test": 0,
            "sample": 0,
            "in_development": 0,
            "abandoned": 0,
            "total": 0,
            "last_checked": None,
        })
        status = str(t.get("status") or "in_development")
        if status in row:
            row[status] += 1
        row["total"] += 1

        task_id = t.get("id")
        if task_id:
            date = _latest_evaluation_date(scores.get(str(task_id)))
            if date:
                current = row["last_checked"]
                if current is None or date > current:
                    row["last_checked"] = date

    rows = sorted(by_domain.values(), key=lambda r: -r["total"])
    totals = {
        status: sum(r[status] for r in rows)
        for status in ("final", "test", "sample", "in_development", "abandoned")
    }
    totals["total"] = sum(r["total"] for r in rows)
    return rows, totals


def _format_summary_table(
    rows: list[dict[str, Any]],
    totals: dict[str, int],
    *,
    show_abandoned: bool = False,
) -> str:
    """Build the --summary terminal table (returns a multi-line string)."""
    if not rows:
        return "No tasks found.\n"

    headers = ["Domain", "Final", "Test", "Sample", "Dev"]
    keys = ["domain", "final", "test", "sample", "in_development"]
    if show_abandoned:
        headers.append("Aband")
        keys.append("abandoned")
    headers += ["Total", "Last Checked"]
    keys += ["total", "last_checked"]

    widths = [max(len(h), 8) for h in headers]
    widths[0] = max(widths[0], max(len(r["domain"]) for r in rows))
    widths[-1] = max(widths[-1], 12)

    def _fmt_row(values: list[str]) -> str:
        return "  " + " │ ".join(v.ljust(w) for v, w in zip(values, widths))

    out_lines = [
        "AI4Sci-Bench Task Catalog",
        "=" * 65,
        "",
        _fmt_row(headers),
        "  " + "─┼─".join("─" * w for w in widths),
    ]
    for r in rows:
        row_vals = []
        for k in keys:
            v = r.get(k)
            row_vals.append("never" if (k == "last_checked" and not v) else str(v))
        out_lines.append(_fmt_row(row_vals))
    out_lines.append("  " + "─┼─".join("─" * w for w in widths))
    totals_row = ["Total"]
    for k in keys[1:]:
        if k == "last_checked":
            totals_row.append("")
        elif k == "total":
            totals_row.append(str(totals["total"]))
        else:
            totals_row.append(str(totals[k]))
    out_lines.append(_fmt_row(totals_row))
    out_lines.append("")

    gap_alerts = [r["domain"] for r in rows if r["final"] == 0]
    weak = [r["domain"] for r in rows if r["final"] == 1]
    if gap_alerts or weak:
        parts = []
        if gap_alerts:
            parts.append(", ".join(f"{d} (0 final)" for d in gap_alerts))
        if weak:
            parts.append(", ".join(f"{d} (1 final)" for d in weak))
        out_lines.append("Gap alert: " + "; ".join(parts))
    return "\n".join(out_lines) + "\n"


def _format_task_scores(task_id: str, score_data: dict[str, Any] | None) -> str:
    """Render the --task <id> --scores history for one task."""
    if not score_data or not score_data.get("evaluations"):
        return f"Task: {task_id}\nNo difficulty check on record.\n"

    evaluations = score_data["evaluations"]
    task_version = score_data.get("task_version", "")
    header = f"Task: {task_id}"
    if task_version:
        header += f" (v{task_version})"
    out_lines = [header, ""]
    out_lines.append("Difficulty Check History")
    out_lines.append("─" * 70)
    out_lines.append("  Date        │ Model                          │ Level │ Mean   │ Verdict")
    out_lines.append("  " + "─" * 68)
    for evaluation in reversed(evaluations):
        date = str(evaluation.get("date", "")).split("T", 1)[0] or "?"
        verdict = str(evaluation.get("verdict", "?")).upper()
        for agent_block in evaluation.get("results", []):
            model = (
                agent_block.get("agent_config", {}).get("model")
                or agent_block.get("agent")
                or "unknown"
            )
            for level, scores in (agent_block.get("scores") or {}).items():
                mean = scores.get("mean")
                mean_str = f"{float(mean):>5.1f}" if mean is not None else "  —  "
                out_lines.append(
                    f"  {date:<11} │ {str(model):<30} │ {str(level).upper():<5} │ {mean_str} │ {verdict}"
                )
    out_lines.append("")
    return "\n".join(out_lines)


def _format_flagged(
    flagged: list[dict[str, Any]], threshold: int,
) -> str:
    """Render the --flagged listing."""
    if not flagged:
        return f"No flagged tasks (no final task scored >= {threshold}).\n"
    out_lines = [
        f"Flagged Tasks (latest score >= {threshold})",
        "─" * 70,
        "  Task                                │ Model                  │ Level │ Score",
        "  " + "─" * 68,
    ]
    for row in flagged:
        out_lines.append(
            f"  {row['task_id']:<35} │ {row['model']:<22} │ "
            f"{str(row['prompt_level']).upper():<5} │ {row['mean_score']:>5.1f}"
        )
    out_lines.append("")
    out_lines.append("Action needed: harden these tasks or change status to 'test'.")
    return "\n".join(out_lines) + "\n"


def _format_gaps(
    rows: list[dict[str, Any]],
    tasks_dir: Path,
    min_final: int,
) -> str:
    """Render the --gaps listing of under-covered domains."""
    under = [r for r in rows if r["final"] < min_final]
    out_lines = [
        f"Domain Coverage Gaps (final < {min_final})",
        "─" * 70,
    ]
    if not under:
        out_lines.append(f"  All domains have at least {min_final} final task(s). No gaps.")
        return "\n".join(out_lines) + "\n"

    under.sort(key=lambda r: (r["final"], r["domain"]))
    for r in under:
        candidates: list[str] = []
        domain_dir = tasks_dir / r["domain"]
        if domain_dir.exists():
            for path in sorted(domain_dir.glob("candidates*.md")):
                candidates.append(path.name)
        suffix = f"  (candidates: {', '.join(candidates)})" if candidates else ""
        out_lines.append(
            f"  {r['domain']:<22} │ final={r['final']} test={r['test']} "
            f"dev={r['in_development']}{suffix}"
        )
    out_lines.append("")
    out_lines.append(
        "Tip: review the listed candidates_*.md files for topic ideas, "
        "then run `asibench task create --domain <d> --name <task>`."
    )
    return "\n".join(out_lines) + "\n"


@cli.command("catalog")
@click.option("--summary", is_flag=True, default=False,
              help="Show domain coverage summary (default if no other flag given).")
@click.option("--task", "task_id", default=None,
              help="Show details for a specific task ID.")
@click.option("--scores", "show_scores", is_flag=True, default=False,
              help="Include difficulty-check history. Requires --task.")
@click.option("--gaps", is_flag=True, default=False,
              help="List domains with too few final tasks.")
@click.option("--flagged", is_flag=True, default=False,
              help="List final tasks whose latest score >= threshold.")
@click.option("--threshold", default=50, show_default=True, type=int,
              help="Score threshold for --flagged.")
@click.option("--min-final", default=3, show_default=True, type=int,
              help="Minimum final tasks per domain for --gaps; below counts as a gap.")
@click.option("--include-abandoned", is_flag=True, default=False,
              help="Include abandoned tasks in summary aggregation.")
@click.option("--tasks-dir", default="tasks/", show_default=True)
@click.option("--scores-dir", default="scores/", show_default=True)
@click.option("--output-json", "output_json", default=None, type=click.Path(),
              help="Also write structured JSON to this path.")
def catalog(
    summary: bool,
    task_id: str | None,
    show_scores: bool,
    gaps: bool,
    flagged: bool,
    threshold: int,
    min_final: int,
    include_abandoned: bool,
    tasks_dir: str,
    scores_dir: str,
    output_json: str | None,
):
    """Inspect the task catalog: coverage summary, per-task history, gaps, flagged.

    Modes (use exactly one; --summary is default):

    \b
      --summary               Domain coverage table.
      --task <id> [--scores]  Task details with optional difficulty history.
      --gaps                  Domains with fewer than --min-final tasks.
      --flagged               Final tasks scoring >= --threshold on latest check.
    """
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.tracking.difficulty_scores import (
        find_flagged_tasks,
        get_all_task_scores,
        load_scores,
    )

    modes = [bool(summary), bool(task_id), bool(gaps), bool(flagged)]
    if sum(modes) > 1:
        raise click.ClickException(
            "Provide at most one of --summary / --task / --gaps / --flagged."
        )
    if show_scores and not task_id:
        raise click.ClickException("--scores can only be used together with --task.")

    tasks_path = Path(tasks_dir)
    scores_path = Path(scores_dir)

    if task_id:
        score_data = None
        if scores_path.exists():
            score_data = load_scores(scores_path, task_id)
        meta: dict[str, Any] | None = None
        if tasks_path.exists():
            loader = TaskLoader(tasks_path)
            for yaml_path in loader._iter_task_source_files():
                if yaml_path.parent.name == "_template":
                    continue
                try:
                    candidate = loader.load_task_metadata(yaml_path)
                except Exception:
                    continue
                if str(candidate.get("id", "")) == task_id:
                    meta = candidate
                    break

        if meta is None and score_data is None:
            raise click.ClickException(
                f"No task or score record found for '{task_id}'."
            )

        if meta:
            status = meta.get("status", "unknown")
            domain = meta.get("domain", "?")
            version = meta.get("version", "")
            click.echo(f"Task: {task_id}  (v{version}, status: {status}, domain: {domain})")
            click.echo(f"Name: {meta.get('name', '')}")
            click.echo("")
        if show_scores or score_data is not None:
            click.echo(_format_task_scores(task_id, score_data))
        if output_json:
            payload = {
                "task_id": task_id,
                "metadata": {k: v for k, v in (meta or {}).items() if not k.startswith("_")},
                "scores": score_data,
            }
            Path(output_json).write_text(
                json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
            )
            click.echo(f"JSON report: {output_json}")
        return

    loader = TaskLoader(tasks_path) if tasks_path.exists() else None
    tasks_list: list[dict[str, Any]] = []
    if loader:
        try:
            tasks_list = loader.discover_tasks(
                include_test=True,
                include_sample=True,
                include_dev=True,
                include_abandoned=include_abandoned,
            )
        except FileNotFoundError:
            tasks_list = []
    all_scores = get_all_task_scores(scores_path) if scores_path.exists() else {}
    rows, totals = _collect_domain_summary(tasks_list, all_scores)

    if flagged:
        flagged_rows = find_flagged_tasks(
            scores_path, tasks_dir=tasks_path, threshold=threshold, only_final=True,
        ) if scores_path.exists() else []
        click.echo(_format_flagged(flagged_rows, threshold))
        if output_json:
            Path(output_json).write_text(
                json.dumps({"threshold": threshold, "flagged": flagged_rows},
                           indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            click.echo(f"JSON report: {output_json}")
        return

    if gaps:
        click.echo(_format_gaps(rows, tasks_path, min_final=min_final))
        if output_json:
            under = [r for r in rows if r["final"] < min_final]
            Path(output_json).write_text(
                json.dumps({"min_final": min_final, "gaps": under},
                           indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            click.echo(f"JSON report: {output_json}")
        return

    click.echo(_format_summary_table(rows, totals, show_abandoned=include_abandoned))
    if output_json:
        Path(output_json).write_text(
            json.dumps({"rows": rows, "totals": totals}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        click.echo(f"JSON report: {output_json}")


if __name__ == "__main__":
    main()
