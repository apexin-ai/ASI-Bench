"""LLM-powered error diagnosis engine."""

from __future__ import annotations

import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.types import AgentOutput, AnalysisReport, EvalResult

logger = get_logger(__name__)


class AnalyzerBackend(Enum):
    LLM_API = "llm_api"
    CLAUDE_CODE = "claude_code"


class ErrorAnalyzer:
    """LLM-powered error diagnosis engine.

    Supports two backends:
      llm_api:       call any LLM via litellm (stateless, fast)
      claude_code:   invoke Claude Code CLI (agentic: can read code, run tools)
    """

    def __init__(
        self,
        enabled: bool = False,
        backend: str = "llm_api",
        model: str = "gemini/gemini-2.0-flash",
        domain_prompts_dir: Path | None = None,
    ):
        self.enabled = enabled
        self.backend = AnalyzerBackend(backend)
        self.model = model
        self.domain_prompts_dir = domain_prompts_dir

    def analyze(
        self,
        eval_result: EvalResult,
        agent_output: AgentOutput,
        reference_specs: str | None = None,
    ) -> AnalysisReport | None:
        if not self.enabled:
            return None
        if eval_result.final_score >= eval_result.max_possible_score:
            return None

        if self.backend == AnalyzerBackend.LLM_API:
            return self._analyze_via_api(eval_result, agent_output, reference_specs)
        else:
            return self._analyze_via_claude_code(eval_result, agent_output)

    def _analyze_via_api(
        self,
        eval_result: EvalResult,
        agent_output: AgentOutput,
        reference_specs: str | None,
    ) -> AnalysisReport:
        import litellm

        prompt = self._build_prompt(eval_result, agent_output, reference_specs)
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        raw = response.choices[0].message.content
        return self._parse_report(raw, eval_result.instance_id)

    def _analyze_via_claude_code(
        self,
        eval_result: EvalResult,
        agent_output: AgentOutput,
    ) -> AnalysisReport:
        workspace = agent_output.output_dir
        analysis_prompt = self._build_claude_code_analysis_prompt(eval_result)

        cmd = [
            "claude",
            "--model", "claude-sonnet-4-6",
            "--max-turns", "20",
            "--print",
            "--output-format", "text",
            "--disallow-tools", "WebSearch,WebFetch",
            "--", analysis_prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=300,
            )
            raw = result.stdout.strip()
        except Exception as e:
            raw = f"Analysis failed: {e}"

        return self._parse_report(raw, eval_result.instance_id)

    def _build_prompt(
        self,
        eval_result: EvalResult,
        agent_output: AgentOutput,
        reference_specs: str | None,
    ) -> str:
        score_summary = self._format_score_breakdown(eval_result)

        # Read agent code
        agent_code = ""
        for code_file in agent_output.code_files:
            code_path = agent_output.output_dir / code_file
            if code_path.exists():
                content = code_path.read_text(encoding="utf-8")
                agent_code += f"\n## {code_file}:\n```python\n{content[:6000]}\n```\n"

        # Execution log
        execution_log = agent_output.log[-2000:] if agent_output.log else "No log available"

        return f"""You are a scientific computing expert analyzing a failed benchmark submission.

## Task: {eval_result.task_id}
## Agent score: {eval_result.final_score:.0f}/100

## Scoring breakdown:
{score_summary}

## Agent-generated code:
{agent_code}

## Execution log:
```
{execution_log}
```

## Reference implementation specs:
{reference_specs or "Not available"}

Identify root cause(s) and output JSON:
{{"error_category": "...", "error_subcategory": "...",
  "root_cause": "...", "evidence": [...], "fix_suggestions": [...],
  "confidence": 0.0-1.0}}

Error category taxonomy:
  algorithm_error / implementation_bug / misunderstanding / quality_issue / resource_issue
"""

    def _build_claude_code_analysis_prompt(self, eval_result: EvalResult) -> str:
        score_summary = "\n".join(
            f"  - {r.scorer_name}: {r.score}/{r.max_score} (passed={r.passed})"
            for r in eval_result.gate_results + eval_result.score_results
        )
        return f"""You are analyzing a failed benchmark submission for task '{eval_result.task_id}'.

The agent scored {eval_result.final_score:.0f}/100.

Scoring breakdown:
{score_summary}

Please:
1. Read the agent's generated code files in this directory
2. Read the execution log (run_log.txt if present)
3. Identify the specific root cause(s) of failure
4. Point to exact line numbers or functions where issues occur
5. Explain why each issue causes the observed failure pattern

Output a structured diagnosis in this JSON format:
{{"error_category": "<see taxonomy>",
  "error_subcategory": "<specific sub-type>",
  "root_cause": "<concise description>",
  "evidence": ["<line X: ...", "..."],
  "fix_suggestions": ["...", "..."],
  "confidence": 0.0-1.0}}

Error category taxonomy:
  algorithm_error / implementation_bug / misunderstanding / quality_issue / resource_issue
"""

    def _format_score_breakdown(self, eval_result: EvalResult) -> str:
        lines = []
        for r in eval_result.gate_results:
            lines.append(f"  Gate - {r.scorer_name}: {'PASS' if r.passed else 'FAIL'} ({r.score}/{r.max_score})")
        for r in eval_result.score_results:
            lines.append(f"  Score - {r.scorer_name}: {r.score}/{r.max_score}")
        return "\n".join(lines)

    def _parse_report(self, raw: str, instance_id: str) -> AnalysisReport:
        """Parse LLM output into an AnalysisReport."""
        try:
            # Try to extract JSON from the response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(raw[json_start:json_end])
                return AnalysisReport(
                    instance_id=instance_id,
                    error_category=data.get("error_category", "unknown"),
                    error_subcategory=data.get("error_subcategory", "unknown"),
                    root_cause=data.get("root_cause", "Unable to determine"),
                    evidence=data.get("evidence", []),
                    fix_suggestions=data.get("fix_suggestions", []),
                    raw_analysis=raw,
                    confidence=data.get("confidence", 0.0),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return AnalysisReport(
            instance_id=instance_id,
            error_category="unknown",
            error_subcategory="parse_failure",
            root_cause="Failed to parse LLM analysis output",
            evidence=[],
            fix_suggestions=[],
            raw_analysis=raw,
            confidence=0.0,
        )

    def summarize_for_human(self, report: AnalysisReport) -> str:
        """Format an AnalysisReport as a human-readable summary."""
        lines = [
            f"Error Analysis for {report.instance_id}",
            f"  Category: {report.error_category}/{report.error_subcategory}",
            f"  Root Cause: {report.root_cause}",
            f"  Confidence: {report.confidence:.0%}",
        ]
        if report.evidence:
            lines.append("  Evidence:")
            for e in report.evidence:
                lines.append(f"    - {e}")
        if report.fix_suggestions:
            lines.append("  Suggestions:")
            for s in report.fix_suggestions:
                lines.append(f"    - {s}")
        return "\n".join(lines)
