"""Agent-as-Judge scorer — uses agentic CLI (Codex / Claude Code) for evaluation."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai4sci_bench.adapters.claude_code_cli import CLAUDE_CORE_TOOLS, CLAUDE_SEARCH_TOOLS
from ai4sci_bench.adapters.codex_cli import CODEX_RESTRICTED_DISABLE_FEATURES
from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail, ToolMode
from ai4sci_bench.scorers._judge_common import aggregate_judge_scores
from ai4sci_bench.scorers._parse_utils import parse_judge_json

logger = get_logger(__name__)

_DEFAULT_MODELS = {"codex": "gpt-5.6-sol", "claude_code": "claude-opus-4-6"}
_SUPPORTED_AGENT_TYPES = frozenset(_DEFAULT_MODELS.keys())

JUDGE_INSTRUCTIONS_TEMPLATE = """\
# Judge Instructions

You are an expert scientific computing evaluator.

## Evaluation Workspace

- `agent_output/` — the agent's output files to evaluate
- `reference/` — reference/ground-truth files (if available)

## Rubric

{rubric}

## Output Requirements

After completing your evaluation, create a file called `score.json` in the current directory:

```json
{{"score": <0-{max_score}>, "reasoning": "<your evaluation reasoning>"}}
```

This is your only scoring output. Do not modify files in `agent_output/` or `reference/`.
"""


@register_scorer("agent_judge")
class AgentJudgeScorer(Scorer):
    """Use an agentic CLI (Codex / Claude Code) as judge.

    The agent runs in a workspace containing pred and ref files,
    follows a rubric, and writes score.json.
    """

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        agent_type = config.get("agent", "codex")
        if agent_type not in _SUPPORTED_AGENT_TYPES:
            raise ValueError(
                f"Unsupported agent type: {agent_type!r}. "
                f"Supported: {sorted(_SUPPORTED_AGENT_TYPES)}"
            )

        model = config.get("model", _DEFAULT_MODELS.get(agent_type, "gpt-5.6-sol"))
        rubric = config.get("rubric", "")
        timeout = config.get("timeout", 300)
        max_score_value = config.get("max_score_value", 10)
        weight = config.get("weight", 1.0)
        threshold = config.get("threshold", 0.5)
        num_judges = config.get("num_judges", 3)

        if num_judges < 3:
            logger.warning(
                "num_judges=%d provides no parse-failure tolerance; consider >= 3",
                num_judges,
            )

        scores: list[float | None] = []
        raw_logs: list[str] = []
        score_sources: list[str] = []

        for i in range(num_judges):
            workspace: Path | None = None
            try:
                workspace = self._prepare_workspace(pred_dir, ref_dir, rubric, max_score_value)
                stdout, stderr, returncode = self._run_agent(
                    agent_type, model, workspace, timeout, config,
                )
                score, source = self._extract_score(workspace, stdout, max_score_value)
                raw_log = stdout[:5000] if stdout else "(empty)"
                score_sources.append(source)
            except subprocess.TimeoutExpired:
                score = None
                raw_log = "(agent timed out)"
                score_sources.append("timeout")
            except Exception as e:
                score = None
                raw_log = f"(agent error: {e})"
                score_sources.append("error")
            finally:
                if workspace is not None:
                    self._cleanup_workspace(workspace)

            scores.append(score)
            raw_logs.append(raw_log)

        tool_mode_str = config.get("tool_mode", "unrestricted")

        return aggregate_judge_scores(
            scores=scores,
            raw_responses=raw_logs,
            scorer_name="agent_judge",
            model=f"{agent_type}/{model}",
            num_judges=num_judges,
            max_score_value=max_score_value,
            weight=weight,
            threshold=threshold,
            extra_details={
                "agent_type": agent_type,
                "tool_mode": tool_mode_str,
                "score_sources": score_sources,
            },
        )

    def _prepare_workspace(
        self,
        pred_dir: Path,
        ref_dir: Path,
        rubric: str,
        max_score_value: float,
    ) -> Path:
        workspace = Path(tempfile.mkdtemp(prefix="agent_judge_"))
        try:
            agent_output_link = workspace / "agent_output"
            agent_output_link.symlink_to(pred_dir.resolve())

            if ref_dir.exists() and any(ref_dir.iterdir()):
                reference_link = workspace / "reference"
                reference_link.symlink_to(ref_dir.resolve())

            instructions = JUDGE_INSTRUCTIONS_TEMPLATE.format(
                rubric=rubric,
                max_score=int(max_score_value),
            )
            (workspace / "JUDGE_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise

        return workspace

    def _run_agent(
        self,
        agent_type: str,
        model: str,
        workspace: Path,
        timeout: int,
        config: dict,
    ) -> tuple[str, str, int]:
        prompt = "Read JUDGE_INSTRUCTIONS.md and follow the instructions exactly."
        tool_mode = ToolMode(config.get("tool_mode", "unrestricted"))

        if agent_type == "codex":
            sandbox = config.get("sandbox", "workspace-write")
            reasoning_effort = config.get("reasoning_effort", "xhigh")
            cmd = [
                "codex", "exec",
                "--model", model,
                "--cd", str(workspace),
                "--sandbox", sandbox,
                "--skip-git-repo-check",
                "-c", f'model_reasoning_effort="{reasoning_effort}"',
                "--json",
            ]
            self._apply_codex_tool_isolation(cmd, tool_mode)
            cmd += ["--", prompt]
            stdin_input = ""
        elif agent_type == "claude_code":
            cmd = [
                "claude",
                "--model", model,
                "--output-format", "stream-json",
                "--print", "--verbose",
            ]
            pm = config.get("permission_mode", "bypassPermissions")
            if pm:
                cmd.extend(["--permission-mode", pm])
            self._apply_claude_tool_isolation(cmd, tool_mode)
            stdin_input = prompt
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        result = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            input=stdin_input,
        )
        return result.stdout, result.stderr, result.returncode

    def _extract_score(
        self, workspace: Path, stdout: str, max_score: float
    ) -> tuple[float | None, str]:
        """Extract score via dual-channel: score.json (primary) then stdout (fallback).

        Returns (score, source) where source is one of:
        "score.json", "stdout_parse", "parse_failed".
        """
        score_file = workspace / "score.json"
        if score_file.exists():
            try:
                data = json.loads(score_file.read_text(encoding="utf-8"))
                raw_score = float(data["score"])
                return max(0.0, min(raw_score, max_score)), "score.json"
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                logger.warning("score.json exists but is invalid, falling back to stdout")

        parsed = parse_judge_json(stdout, max_score=max_score)
        if parsed is not None:
            return parsed, "stdout_parse"

        return None, "parse_failed"

    def _cleanup_workspace(self, workspace: Path) -> None:
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _apply_claude_tool_isolation(cmd: list[str], tool_mode: ToolMode) -> None:
        if tool_mode == ToolMode.UNRESTRICTED:
            return
        tools = CLAUDE_SEARCH_TOOLS if tool_mode == ToolMode.SEARCH else CLAUDE_CORE_TOOLS
        cmd += ["--tools", tools, "--strict-mcp-config", "--disable-slash-commands"]

    @staticmethod
    def _apply_codex_tool_isolation(cmd: list[str], tool_mode: ToolMode) -> None:
        if tool_mode == ToolMode.UNRESTRICTED:
            return
        cmd += ["--ignore-user-config"]
        for feature in CODEX_RESTRICTED_DISABLE_FEATURES:
            cmd += ["--disable", feature]
        if tool_mode == ToolMode.RESTRICTED:
            cmd += ["--config", 'web_search="disabled"']


def preflight_agent_judge(scorer_config: dict, errors: list[str]) -> None:
    """Preflight checks for agent_judge scorer configuration."""
    agent_type = scorer_config.get("agent", "codex")

    if agent_type not in _SUPPORTED_AGENT_TYPES:
        errors.append(
            f"agent_judge: unsupported agent type {agent_type!r}. "
            f"Supported: {sorted(_SUPPORTED_AGENT_TYPES)}"
        )
        return

    cli_name = "codex" if agent_type == "codex" else "claude"
    if shutil.which(cli_name) is None:
        errors.append(
            f"agent_judge: {cli_name} CLI not found in PATH "
            f"(required for agent={agent_type!r})"
        )

    rubric = scorer_config.get("rubric", "")
    if not rubric or not rubric.strip():
        errors.append("agent_judge: rubric must not be empty")

    timeout = scorer_config.get("timeout", 300)
    if timeout < 60:
        errors.append(
            f"agent_judge: timeout={timeout}s is too low (minimum 60s)"
        )

    tool_mode_str = scorer_config.get("tool_mode", "unrestricted")
    valid_tool_modes = {"restricted", "search", "unrestricted"}
    if tool_mode_str not in valid_tool_modes:
        errors.append(
            f"agent_judge: invalid tool_mode {tool_mode_str!r}. "
            f"Valid: {sorted(valid_tool_modes)}"
        )

    num_judges = scorer_config.get("num_judges", 3)
    if num_judges < 1:
        errors.append(
            f"agent_judge: num_judges={num_judges} must be >= 1"
        )
