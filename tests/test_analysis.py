"""Tests for the error analysis module."""

import json
from pathlib import Path

import pytest

from ai4sci_bench.analysis.error_analyzer import AnalyzerBackend, ErrorAnalyzer
from ai4sci_bench.core.types import (
    AgentOutput,
    AnalysisReport,
    EvalResult,
    PromptLevel,
    RunStatus,
    ScoreDetail,
)


def _make_eval_result(score=50.0):
    return EvalResult(
        instance_id="test_instance",
        task_id="physics.test",
        prompt_level=PromptLevel.B2,
        agent_name="TestAgent",
        parameters={},
        gate_results=[
            ScoreDetail(scorer_name="file_match", score=1, max_score=1, passed=True, details={}),
        ],
        gates_passed=True,
        score_results=[
            ScoreDetail(scorer_name="numerical:l2", score=score, max_score=100, passed=True, details={}),
        ],
        final_score=score,
    )


def _make_agent_output(tmp_dir):
    output_dir = tmp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "simulation.py").write_text("print('hello')")
    return AgentOutput(
        instance_id="test_instance",
        output_dir=output_dir,
        code_files=["simulation.py"],
        data_files=[],
        log="Test log output\nLine 2",
        execution_time_seconds=5.0,
        status=RunStatus.COMPLETED,
    )


class TestErrorAnalyzer:
    def test_disabled(self, tmp_dir):
        analyzer = ErrorAnalyzer(enabled=False)
        result = analyzer.analyze(
            _make_eval_result(), _make_agent_output(tmp_dir)
        )
        assert result is None

    def test_perfect_score_skipped(self, tmp_dir):
        analyzer = ErrorAnalyzer(enabled=True)
        result = analyzer.analyze(
            _make_eval_result(score=100.0), _make_agent_output(tmp_dir)
        )
        assert result is None

    def test_parse_report_valid_json(self):
        analyzer = ErrorAnalyzer()
        raw = json.dumps({
            "error_category": "algorithm_error",
            "error_subcategory": "wrong_formula",
            "root_cause": "Used wrong equation",
            "evidence": ["line 42: formula is incorrect"],
            "fix_suggestions": ["Use the correct formula"],
            "confidence": 0.9,
        })
        report = analyzer._parse_report(raw, "test_id")
        assert report.error_category == "algorithm_error"
        assert report.root_cause == "Used wrong equation"
        assert report.confidence == 0.9

    def test_parse_report_json_embedded_in_text(self):
        analyzer = ErrorAnalyzer()
        raw = 'Here is my analysis:\n{"error_category": "implementation_bug", "error_subcategory": "off_by_one", "root_cause": "test", "evidence": [], "fix_suggestions": [], "confidence": 0.5}\nThat is all.'
        report = analyzer._parse_report(raw, "test_id")
        assert report.error_category == "implementation_bug"

    def test_parse_report_invalid_json(self):
        analyzer = ErrorAnalyzer()
        raw = "This is not JSON at all"
        report = analyzer._parse_report(raw, "test_id")
        assert report.error_category == "unknown"
        assert report.error_subcategory == "parse_failure"

    def test_build_prompt(self, tmp_dir):
        analyzer = ErrorAnalyzer()
        eval_result = _make_eval_result(50.0)
        agent_output = _make_agent_output(tmp_dir)
        prompt = analyzer._build_prompt(eval_result, agent_output, "Test specs")
        assert "physics.test" in prompt
        assert "50/100" in prompt
        assert "Test specs" in prompt
        assert "simulation.py" in prompt

    def test_build_claude_code_analysis_prompt(self):
        analyzer = ErrorAnalyzer()
        eval_result = _make_eval_result(30.0)
        prompt = analyzer._build_claude_code_analysis_prompt(eval_result)
        assert "30/100" in prompt
        assert "error_category" in prompt

    def test_format_score_breakdown(self):
        analyzer = ErrorAnalyzer()
        eval_result = _make_eval_result(50.0)
        breakdown = analyzer._format_score_breakdown(eval_result)
        assert "Gate" in breakdown
        assert "Score" in breakdown

    def test_summarize_for_human(self):
        analyzer = ErrorAnalyzer()
        report = AnalysisReport(
            instance_id="test",
            error_category="algorithm_error",
            error_subcategory="wrong_formula",
            root_cause="Wrong equation used",
            evidence=["line 10: incorrect formula"],
            fix_suggestions=["Use correct formula"],
            raw_analysis="",
            confidence=0.85,
        )
        summary = analyzer.summarize_for_human(report)
        assert "algorithm_error" in summary
        assert "Wrong equation used" in summary
        assert "85%" in summary

    def test_backend_enum(self):
        assert AnalyzerBackend("llm_api") == AnalyzerBackend.LLM_API
        assert AnalyzerBackend("claude_code") == AnalyzerBackend.CLAUDE_CODE
