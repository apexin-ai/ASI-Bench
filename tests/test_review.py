"""Tests for the trajectory review and diagnosis module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from ai4sci_bench.analysis.reviewer import (
    Diagnosis,
    DiagnosisSummary,
    GradientAnalysis,
    _format_gradient_report,
    _format_single_diagnosis,
    _format_summary,
    _parse_diagnosis_output,
    _parse_gradient_output,
    _parse_review_output,
    find_all_instances,
    find_low_score_instances,
    write_diagnosis_report,
)

# ─────────────────────── reviewer ─────────────────────────────────────


class TestFindLowScores:
    def test_finds_low_scores(self, tmp_path):
        combo_dir = tmp_path / "combo_a" / "physics.task_a"
        combo_dir.mkdir(parents=True)
        low = {"task_id": "physics.task_a", "prompt_level": "b3", "final_score": 5.0}
        high = {"task_id": "physics.task_a", "prompt_level": "b1", "final_score": 80.0}
        (combo_dir / "b3.json").write_text(json.dumps(low), encoding="utf-8")
        (combo_dir / "b1.json").write_text(json.dumps(high), encoding="utf-8")
        results = find_low_score_instances(tmp_path, ["combo_a"], threshold=10.0)
        assert len(results) == 1
        assert results[0]["final_score"] == 5.0

    def test_threshold_boundary(self, tmp_path):
        combo_dir = tmp_path / "combo_a"
        combo_dir.mkdir(parents=True)
        exact = {"task_id": "t1", "prompt_level": "b1", "final_score": 10.0}
        below = {"task_id": "t2", "prompt_level": "b1", "final_score": 9.99}
        (combo_dir / "exact.json").write_text(json.dumps(exact), encoding="utf-8")
        (combo_dir / "below.json").write_text(json.dumps(below), encoding="utf-8")
        results = find_low_score_instances(tmp_path, ["combo_a"], threshold=10.0)
        assert len(results) == 1
        assert results[0]["task_id"] == "t2"

    def test_skips_metadata(self, tmp_path):
        combo_dir = tmp_path / "combo_a"
        combo_dir.mkdir(parents=True)
        (combo_dir / "eval_config.json").write_text(
            json.dumps({"final_score": 0, "task_id": "x"}), encoding="utf-8"
        )
        results = find_low_score_instances(tmp_path, ["combo_a"])
        assert len(results) == 0


class TestParseDiagnosisOutput:
    def test_full_parse(self):
        raw = """
WHAT_IMPLEMENTED: Used explicit Euler method for 2D simulation
MAIN_MISMATCH: Reference uses leapfrog, agent uses forward Euler
CLASSIFICATION: Agent求解代码算法实现错误
CAUSE_BUCKET: Agent
SCORER_BREAKDOWN: numerical=0.0, code_analysis=10.0
EVIDENCE: Trajectory shows agent chose Euler "for simplicity"
BOTTOM_LINE: Wrong numerical method caused divergence
"""
        d = _parse_diagnosis_output(raw, {
            "task_id": "physics.test",
            "prompt_level": "b3",
            "_combo_label": "combo_a",
            "final_score": 0.0,
        })
        assert d.what_implemented == "Used explicit Euler method for 2D simulation"
        assert d.classification == "Agent求解代码算法实现错误"
        assert d.cause_bucket == "Agent"
        assert d.task_id == "physics.test"

    def test_partial_parse(self):
        raw = "CAUSE_BUCKET: Environment\nBOTTOM_LINE: sandbox crashed"
        d = _parse_diagnosis_output(raw, {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 0,
        })
        assert d.cause_bucket == "Environment"
        assert d.bottom_line == "sandbox crashed"
        assert d.what_implemented == ""

    def test_empty_output(self):
        d = _parse_diagnosis_output("", {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 0,
        })
        assert d.cause_bucket == ""


class TestDiagnosisSummary:
    def test_bucket_distribution(self):
        s = DiagnosisSummary(diagnoses=[
            Diagnosis("t1", "b1", "c1", 0, cause_bucket="Agent"),
            Diagnosis("t2", "b1", "c1", 0, cause_bucket="Agent"),
            Diagnosis("t3", "b1", "c1", 0, cause_bucket="Task"),
        ])
        assert s.bucket_distribution == {"Agent": 2, "Task": 1}

    def test_category_distribution(self):
        s = DiagnosisSummary(diagnoses=[
            Diagnosis("t1", "b1", "c1", 0, classification="Agent求解代码算法实现错误"),
            Diagnosis("t2", "b1", "c1", 0, classification="Agent求解代码算法实现错误"),
        ])
        assert s.category_distribution == {"Agent求解代码算法实现错误": 2}

    def test_empty(self):
        s = DiagnosisSummary()
        assert s.bucket_distribution == {}
        assert s.category_distribution == {}


class TestDiagnosisReportFormatting:
    def test_single_diagnosis_format(self):
        d = Diagnosis(
            task_id="physics.test",
            prompt_level="b3",
            combo_label="combo_a",
            final_score=0.0,
            what_implemented="Used Euler method",
            main_mismatch="Should have used leapfrog",
            classification="Agent求解代码算法实现错误",
            cause_bucket="Agent",
            bottom_line="Wrong method",
        )
        text = _format_single_diagnosis(d)
        assert "physics.test" in text
        assert "Agent求解代码算法实现错误" in text
        assert "Wrong method" in text

    def test_summary_format(self):
        s = DiagnosisSummary(
            total_instances=60,
            reviewed_count=5,
            diagnoses=[
                Diagnosis("t1", "b1", "c1", 0, cause_bucket="Agent",
                          classification="Agent求解代码算法实现错误", bottom_line="x"),
                Diagnosis("t2", "b2", "c1", 0, cause_bucket="Task",
                          classification="Prompt隐藏了与评分有关的信息", bottom_line="y"),
            ],
        )
        text = _format_summary(s)
        assert "总评测实例: 60" in text
        assert "已审核实例: 5" in text
        assert "Agent" in text
        assert "Task" in text
        assert "需关注的 Task" in text


# ─────────────────────── CLI registration ─────────────────────────────


class TestEndToEndDiagnoserFlow:
    """Test the full diagnoser flow: find low scores → format reports."""

    def test_full_flow(self, tmp_path):
        # Create mock results with low and high scores
        combo_dir = tmp_path / "combo_a" / "physics_task"
        combo_dir.mkdir(parents=True)
        for level, score in [("b1", 85.0), ("b2", 60.0), ("b3", 5.0), ("b4", 0.0)]:
            result = {
                "task_id": "physics.task_a",
                "prompt_level": level,
                "final_score": score,
                "gates_passed": score > 0,
                "status": "completed",
                "gate_results": [],
                "score_results": [],
            }
            (combo_dir / f"result__{level}.json").write_text(
                json.dumps(result), encoding="utf-8"
            )

        low = find_low_score_instances(tmp_path, ["combo_a"], threshold=10.0)
        assert len(low) == 2  # b3=5.0, b4=0.0

        # Test summary formatting
        summary = DiagnosisSummary(
            total_instances=4,
            reviewed_count=2,
            diagnoses=[
                Diagnosis("physics.task_a", "b3", "combo_a", 5.0,
                          cause_bucket="Agent", classification="Agent求解代码算法实现错误",
                          bottom_line="bad method"),
                Diagnosis("physics.task_a", "b4", "combo_a", 0.0,
                          cause_bucket="Environment", classification="Agent求解代码算法实现错误",
                          bottom_line="timeout"),
            ],
        )
        text = _format_summary(summary)
        assert "总评测实例: 4" in text
        assert "已审核实例: 2" in text


# ─────────────────────── New diagnoser tests (expanded) ─────────────────


def _make_result_json(tmp_path, combo, task_id, level, score, **extra):
    """Helper: create a result JSON in the expected directory structure."""
    combo_dir = tmp_path / combo / task_id.replace(".", "_")
    combo_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "task_id": task_id,
        "prompt_level": level,
        "final_score": score,
        "gates_passed": score > 0,
        "status": "completed",
        "gate_results": [],
        "score_results": [],
        **extra,
    }
    fname = f"{task_id}__{level}.json"
    (combo_dir / fname).write_text(json.dumps(data), encoding="utf-8")
    return combo_dir / fname


class TestFindAllInstances:
    """Tests for find_all_instances with prompt_levels filtering."""

    def test_finds_all_without_filter(self, tmp_path):
        for level in ["b1", "b2", "b3", "b4"]:
            _make_result_json(tmp_path, "combo_a", "physics.task_a", level, 50.0)
        results = find_all_instances(tmp_path, ["combo_a"])
        assert len(results) == 4

    def test_filter_by_prompt_levels(self, tmp_path):
        for level in ["b1", "b2", "b3", "b4"]:
            _make_result_json(tmp_path, "combo_a", "physics.task_a", level, 50.0)
        results = find_all_instances(tmp_path, ["combo_a"], prompt_levels=["b1", "b2"])
        assert len(results) == 2
        levels = {r["prompt_level"] for r in results}
        assert levels == {"b1", "b2"}

    def test_filter_excludes_b4(self, tmp_path):
        for level in ["b1", "b2", "b3", "b4"]:
            _make_result_json(tmp_path, "combo_a", "physics.task_a", level, 50.0)
        results = find_all_instances(tmp_path, ["combo_a"], prompt_levels=["b1", "b2", "b3"])
        assert len(results) == 3
        assert all(r["prompt_level"] != "b4" for r in results)

    def test_filter_single_level(self, tmp_path):
        for level in ["b1", "b2", "b3"]:
            _make_result_json(tmp_path, "combo_a", "physics.task_a", level, 50.0)
        results = find_all_instances(tmp_path, ["combo_a"], prompt_levels=["b2"])
        assert len(results) == 1
        assert results[0]["prompt_level"] == "b2"

    def test_multiple_combos(self, tmp_path):
        for combo in ["combo_a", "combo_b"]:
            for level in ["b1", "b2"]:
                _make_result_json(tmp_path, combo, "physics.task_a", level, 50.0)
        results = find_all_instances(tmp_path, ["combo_a", "combo_b"], prompt_levels=["b1"])
        assert len(results) == 2

    def test_empty_prompt_levels_returns_all(self, tmp_path):
        for level in ["b1", "b2"]:
            _make_result_json(tmp_path, "combo_a", "physics.task_a", level, 50.0)
        results = find_all_instances(tmp_path, ["combo_a"], prompt_levels=None)
        assert len(results) == 2

    def test_nonexistent_combo(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "physics.task_a", "b1", 50.0)
        results = find_all_instances(tmp_path, ["nonexistent"])
        assert len(results) == 0

    def test_skips_metadata_files(self, tmp_path):
        combo_dir = tmp_path / "combo_a"
        combo_dir.mkdir(parents=True)
        for fname in ["run_metadata.json", "eval_config.json", "task_info.json", "framework_task_info.json"]:
            (combo_dir / fname).write_text(
                json.dumps({"final_score": 0, "task_id": "x"}), encoding="utf-8"
            )
        results = find_all_instances(tmp_path, ["combo_a"])
        assert len(results) == 0

    def test_skips_invalid_json(self, tmp_path):
        combo_dir = tmp_path / "combo_a"
        combo_dir.mkdir(parents=True)
        (combo_dir / "bad.json").write_text("{{not json", encoding="utf-8")
        results = find_all_instances(tmp_path, ["combo_a"])
        assert len(results) == 0

    def test_skips_missing_required_fields(self, tmp_path):
        combo_dir = tmp_path / "combo_a"
        combo_dir.mkdir(parents=True)
        (combo_dir / "no_score.json").write_text(
            json.dumps({"task_id": "t1"}), encoding="utf-8"
        )
        (combo_dir / "no_task.json").write_text(
            json.dumps({"final_score": 5.0}), encoding="utf-8"
        )
        results = find_all_instances(tmp_path, ["combo_a"])
        assert len(results) == 0

    def test_attaches_json_path_and_combo_label(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "physics.task_a", "b1", 50.0)
        results = find_all_instances(tmp_path, ["combo_a"])
        assert results[0]["_json_path"].endswith(".json")
        assert results[0]["_combo_label"] == "combo_a"

    def test_filter_with_no_matching_levels(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "physics.task_a", "b1", 50.0)
        results = find_all_instances(tmp_path, ["combo_a"], prompt_levels=["b4"])
        assert len(results) == 0


class TestFindLowScoreInstancesBackwardCompat:
    """Ensure find_low_score_instances still works as before."""

    def test_uses_find_all_internally(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "t1", "b1", 5.0)
        _make_result_json(tmp_path, "combo_a", "t2", "b1", 50.0)
        results = find_low_score_instances(tmp_path, ["combo_a"], threshold=10.0)
        assert len(results) == 1
        assert results[0]["task_id"] == "t1"

    def test_custom_threshold(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "t1", "b1", 30.0)
        _make_result_json(tmp_path, "combo_a", "t2", "b1", 50.0)
        results = find_low_score_instances(tmp_path, ["combo_a"], threshold=40.0)
        assert len(results) == 1
        assert results[0]["task_id"] == "t1"

    def test_zero_threshold_finds_nothing(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "t1", "b1", 0.0)
        results = find_low_score_instances(tmp_path, ["combo_a"], threshold=0.0)
        assert len(results) == 0


class TestParseReviewOutput:
    """Tests for _parse_review_output with the new 5-dimension fields."""

    def test_full_5_dimension_parse(self):
        raw = """
TASK_UNDERSTANDING: Agent correctly understood the Sod shock tube problem
WHAT_IMPLEMENTED: Implemented Godunov method with exact Riemann solver
PROCESS_SCORE_MATCH: Correct implementation but low score due to strict tolerance
MAIN_MISMATCH: Reference uses relaxation tolerance 1e-6, agent used 1e-4
ANOMALY_DETECTED: None detected
CLASSIFICATION: Scorer与Prompt要求不一致
CAUSE_BUCKET: Task
SCORER_BREAKDOWN: density_L2=0.03, velocity_L2=0.08, pressure_L2=0.05
EVIDENCE: Agent output matches reference within 1% but scorer requires 0.01%
BOTTOM_LINE: Scorer tolerance too strict for the declared B2 difficulty
"""
        d = _parse_review_output(raw, {
            "task_id": "physics.sod_shock_tube",
            "prompt_level": "b2",
            "_combo_label": "codex_ds4pro",
            "final_score": 8.5,
        })
        assert d.task_understanding == "Agent correctly understood the Sod shock tube problem"
        assert d.process_score_match == "Correct implementation but low score due to strict tolerance"
        assert d.anomaly_detected == "None detected"
        assert d.classification == "Scorer与Prompt要求不一致"
        assert d.cause_bucket == "Task"
        assert d.task_id == "physics.sod_shock_tube"
        assert d.final_score == 8.5

    def test_anomaly_detected(self):
        raw = """
TASK_UNDERSTANDING: Understood correctly
ANOMALY_DETECTED: Agent searched for ../reference/ directory; attempted to read framework_task_info.json
CLASSIFICATION: Agent尝试寻找GT/数据泄露
CAUSE_BUCKET: Agent
BOTTOM_LINE: GT leakage attempt detected
"""
        d = _parse_review_output(raw, {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 95.0,
        })
        assert "framework_task_info" in d.anomaly_detected
        assert d.classification == "Agent尝试寻找GT/数据泄露"

    def test_process_score_mismatch_high_score_wrong(self):
        raw = """
PROCESS_SCORE_MATCH: Agent used wrong physical model but scored 90/100 — scorer only checks output format
CLASSIFICATION: Scorer存在漏洞(做错但高分)
CAUSE_BUCKET: Task
BOTTOM_LINE: Scorer missing physics validation
"""
        d = _parse_review_output(raw, {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 90.0,
        })
        assert "wrong physical model" in d.process_score_match
        assert d.classification == "Scorer存在漏洞(做错但高分)"

    def test_all_10_categories_parseable(self):
        categories = [
            "Prompt隐藏了与评分有关的信息",
            "Prompt信息模糊可能歧义",
            "Scorer与Prompt要求不一致",
            "Scorer存在漏洞(做错但高分)",
            "Agent物理边界条件理解错误",
            "Agent过于依赖标准模型",
            "Agent求解代码算法实现错误",
            "Agent取巧采用参数拟合/经典答案",
            "Agent尝试寻找GT/数据泄露",
            "环境/框架问题",
        ]
        for cat in categories:
            raw = f"CLASSIFICATION: {cat}\nCAUSE_BUCKET: Agent"
            d = _parse_review_output(raw, {
                "task_id": "t1", "prompt_level": "b1",
                "_combo_label": "c1", "final_score": 0,
            })
            assert d.classification == cat

    def test_colon_in_value(self):
        raw = "BOTTOM_LINE: Error in step 3: boundary condition was wrong"
        d = _parse_review_output(raw, {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 0,
        })
        assert d.bottom_line == "Error in step 3: boundary condition was wrong"

    def test_backward_compat_alias(self):
        raw = "CAUSE_BUCKET: Agent"
        d1 = _parse_review_output(raw, {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 0,
        })
        d2 = _parse_diagnosis_output(raw, {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 0,
        })
        assert d1.cause_bucket == d2.cause_bucket


class TestParseGradientOutput:
    """Tests for _parse_gradient_output."""

    def test_full_parse(self):
        raw = """
IS_MONOTONIC: yes
GRADIENT_ISSUES: None — gradient is reasonable
B4_DISTRACTOR_EFFECTIVE: yes — B4 scored 15 points lower than B2
BOTTOM_LINE: Score gradient follows expected B1>B2>B3 pattern
"""
        g = _parse_gradient_output(raw, "physics.task_a", "combo_a", {"b1": 90, "b2": 70, "b3": 50, "b4": 55})
        assert g.is_monotonic is True
        assert "None" in g.gradient_issues
        assert "yes" in g.b4_distractor_effective
        assert g.task_id == "physics.task_a"
        assert g.combo_label == "combo_a"
        assert g.level_scores == {"b1": 90, "b2": 70, "b3": 50, "b4": 55}

    def test_non_monotonic(self):
        raw = """
IS_MONOTONIC: no
GRADIENT_ISSUES: B2 (85.0) > B1 (70.0) — B1 prompt may be providing misleading info
B4_DISTRACTOR_EFFECTIVE: not_applicable
"""
        g = _parse_gradient_output(raw, "t1", "c1", {"b1": 70, "b2": 85})
        assert g.is_monotonic is False
        assert "B2" in g.gradient_issues

    def test_monotonic_case_insensitive(self):
        for val in ["Yes", "YES", "True", "true"]:
            raw = f"IS_MONOTONIC: {val}"
            g = _parse_gradient_output(raw, "t1", "c1", {"b1": 90, "b2": 70})
            assert g.is_monotonic is True

        for val in ["no", "No", "false", "False"]:
            raw = f"IS_MONOTONIC: {val}"
            g = _parse_gradient_output(raw, "t1", "c1", {"b1": 90, "b2": 70})
            assert g.is_monotonic is False

    def test_empty_output(self):
        g = _parse_gradient_output("", "t1", "c1", {"b1": 90})
        assert g.is_monotonic is True  # default
        assert g.gradient_issues == ""


class TestGradientAnalysis:
    """Tests for GradientAnalysis dataclass."""

    def test_defaults(self):
        g = GradientAnalysis(task_id="t1", combo_label="c1")
        assert g.is_monotonic is True
        assert g.gradient_issues == ""
        assert g.b4_distractor_effective == ""
        assert g.level_scores == {}

    def test_with_scores(self):
        g = GradientAnalysis(
            task_id="t1",
            combo_label="c1",
            level_scores={"b1": 90.0, "b2": 70.0, "b3": 50.0},
            is_monotonic=True,
        )
        assert len(g.level_scores) == 3


class TestDiagnosisNewFields:
    """Test the new fields in the Diagnosis dataclass."""

    def test_new_fields_default_empty(self):
        d = Diagnosis("t1", "b1", "c1", 50.0)
        assert d.task_understanding == ""
        assert d.process_score_match == ""
        assert d.anomaly_detected == ""

    def test_new_fields_set(self):
        d = Diagnosis(
            "t1", "b1", "c1", 50.0,
            task_understanding="Understood correctly",
            process_score_match="Score matches process",
            anomaly_detected="None",
        )
        assert d.task_understanding == "Understood correctly"
        assert d.process_score_match == "Score matches process"


class TestDiagnosisSummaryExpanded:
    """Extended tests for DiagnosisSummary with gradient analyses."""

    def test_with_gradient_analyses(self):
        s = DiagnosisSummary(
            total_instances=8,
            reviewed_count=8,
            diagnoses=[
                Diagnosis("t1", "b1", "c1", 90, cause_bucket="Agent", classification="Agent求解代码算法实现错误"),
                Diagnosis("t1", "b2", "c1", 70, cause_bucket="Task", classification="Scorer与Prompt要求不一致"),
            ],
            gradient_analyses=[
                GradientAnalysis("t1", "c1", {"b1": 90, "b2": 70}, is_monotonic=True),
            ],
        )
        assert len(s.gradient_analyses) == 1
        assert s.reviewed_count == 8

    def test_category_distribution_with_all_10(self):
        from ai4sci_bench.analysis.reviewer import SCIENTIFIC_CATEGORIES
        diagnoses = [
            Diagnosis(f"t{i}", "b1", "c1", 0, classification=cat)
            for i, cat in enumerate(SCIENTIFIC_CATEGORIES)
        ]
        s = DiagnosisSummary(diagnoses=diagnoses)
        assert len(s.category_distribution) == 10

    def test_mixed_buckets(self):
        s = DiagnosisSummary(diagnoses=[
            Diagnosis("t1", "b1", "c1", 0, cause_bucket="Environment"),
            Diagnosis("t2", "b1", "c1", 0, cause_bucket="Task"),
            Diagnosis("t3", "b1", "c1", 0, cause_bucket="Agent"),
            Diagnosis("t4", "b1", "c1", 0, cause_bucket="Agent"),
            Diagnosis("t5", "b1", "c1", 0, cause_bucket=""),
        ])
        dist = s.bucket_distribution
        assert dist["Agent"] == 2
        assert dist["Task"] == 1
        assert dist["Environment"] == 1
        assert dist.get("Unknown", 0) == 1


class TestFormatGradientReport:
    """Tests for _format_gradient_report."""

    def test_basic_format(self):
        g = GradientAnalysis(
            task_id="physics.task_a",
            combo_label="combo_a",
            level_scores={"b1": 90.0, "b2": 70.0, "b3": 50.0},
            is_monotonic=True,
            gradient_issues="None — gradient is reasonable",
            b4_distractor_effective="not_applicable",
        )
        text = _format_gradient_report(g)
        assert "physics.task_a" in text
        assert "B1: 90.0" in text
        assert "B2: 70.0" in text
        assert "B3: 50.0" in text
        assert "单调性: 是" in text

    def test_non_monotonic(self):
        g = GradientAnalysis(
            task_id="t1", combo_label="c1",
            level_scores={"b1": 50.0, "b2": 80.0},
            is_monotonic=False,
            gradient_issues="B2 > B1 inversion",
        )
        text = _format_gradient_report(g)
        assert "单调性: 否" in text
        assert "B2 > B1 inversion" in text


class TestFormatSingleDiagnosisExpanded:
    """Test the expanded single diagnosis format with new fields."""

    def test_includes_new_sections(self):
        d = Diagnosis(
            task_id="physics.test",
            prompt_level="b2",
            combo_label="combo_a",
            final_score=8.5,
            task_understanding="Correctly identified as shock tube problem",
            what_implemented="Godunov scheme",
            process_score_match="Process correct but scorer too strict",
            main_mismatch="Tolerance difference",
            anomaly_detected="None detected",
            classification="Scorer与Prompt要求不一致",
            cause_bucket="Task",
            scorer_breakdown="density=0.03",
            evidence="Output within 1%",
            bottom_line="Scorer tolerance issue",
        )
        text = _format_single_diagnosis(d)
        assert "任务理解" in text
        assert "Correctly identified" in text
        assert "过程-分数匹配" in text
        assert "Process correct" in text
        assert "异常行为" in text
        assert "None detected" in text


class TestFormatSummaryExpanded:
    """Test _format_summary with gradient analyses and scorer issues."""

    def test_gradient_table_in_summary(self):
        s = DiagnosisSummary(
            total_instances=4,
            reviewed_count=4,
            diagnoses=[
                Diagnosis("t1", "b1", "c1", 90, cause_bucket="Agent", classification="X"),
            ],
            gradient_analyses=[
                GradientAnalysis(
                    "t1", "c1",
                    {"b1": 90, "b2": 70, "b3": 50},
                    is_monotonic=True,
                    gradient_issues="None",
                    b4_distractor_effective="N/A",
                ),
            ],
        )
        text = _format_summary(s)
        assert "分数梯度分析" in text
        assert "✓" in text  # monotonic checkmark
        assert "梯度分析: 1" in text

    def test_scorer_issues_section(self):
        s = DiagnosisSummary(
            total_instances=2,
            reviewed_count=2,
            diagnoses=[
                Diagnosis("t1", "b1", "c1", 95, cause_bucket="Task",
                          classification="Scorer存在漏洞(做错但高分)",
                          bottom_line="Scorer only checks format"),
                Diagnosis("t2", "b1", "c1", 5, cause_bucket="Agent",
                          classification="Agent求解代码算法实现错误",
                          bottom_line="Wrong algorithm"),
            ],
        )
        text = _format_summary(s)
        assert "Scorer 漏洞" in text
        assert "Scorer only checks format" in text

    def test_task_issues_section(self):
        s = DiagnosisSummary(
            total_instances=2,
            reviewed_count=2,
            diagnoses=[
                Diagnosis("t1", "b1", "c1", 5, cause_bucket="Task",
                          classification="Prompt隐藏了与评分有关的信息",
                          bottom_line="Hidden scoring rule"),
            ],
        )
        text = _format_summary(s)
        assert "需关注的 Task 问题" in text
        assert "Hidden scoring rule" in text

    def test_process_score_match_in_index(self):
        s = DiagnosisSummary(
            total_instances=1,
            reviewed_count=1,
            diagnoses=[
                Diagnosis("t1", "b1", "c1", 50, cause_bucket="Agent",
                          classification="X",
                          process_score_match="Correct but low"),
            ],
        )
        text = _format_summary(s)
        assert "过程-分数匹配" in text
        assert "Correct but low" in text

    def test_empty_summary_no_crash(self):
        s = DiagnosisSummary(total_instances=0, reviewed_count=0)
        text = _format_summary(s)
        assert "总评测实例: 0" in text
        assert "已审核实例: 0" in text


class TestWriteDiagnosisReport:
    """Test write_diagnosis_report with gradient analyses."""

    def test_writes_all_files(self, tmp_path):
        summary = DiagnosisSummary(
            total_instances=2,
            reviewed_count=2,
            diagnoses=[
                Diagnosis("physics.task_a", "b1", "combo_a", 90,
                          cause_bucket="Agent", classification="X", bottom_line="ok"),
                Diagnosis("physics.task_a", "b2", "combo_a", 70,
                          cause_bucket="Task", classification="Y", bottom_line="issue"),
            ],
            gradient_analyses=[
                GradientAnalysis(
                    "physics.task_a", "combo_a",
                    {"b1": 90, "b2": 70},
                    is_monotonic=True,
                    gradient_issues="None",
                ),
            ],
        )
        write_diagnosis_report(summary, tmp_path)
        diag_dir = tmp_path / "diagnosis"
        assert diag_dir.exists()
        assert (diag_dir / "summary.md").exists()

        # Per-instance files
        md_files = list(diag_dir.glob("physics.task_a__b1__combo_a.md"))
        assert len(md_files) == 1
        md_files = list(diag_dir.glob("physics.task_a__b2__combo_a.md"))
        assert len(md_files) == 1

        # Gradient file
        grad_files = list(diag_dir.glob("gradient__physics.task_a__combo_a.md"))
        assert len(grad_files) == 1

        # Summary content
        summary_text = (diag_dir / "summary.md").read_text()
        assert "physics.task_a" in summary_text
        assert "梯度分析" in summary_text

    def test_empty_summary(self, tmp_path):
        summary = DiagnosisSummary()
        write_diagnosis_report(summary, tmp_path)
        assert (tmp_path / "diagnosis" / "summary.md").exists()


class TestRunAllDiagnosesUnit:
    """Unit tests for run_all_diagnoses with mocked Claude calls."""

    def test_review_all_finds_all_instances(self, tmp_path):
        for level, score in [("b1", 90), ("b2", 70), ("b3", 50)]:
            _make_result_json(tmp_path, "combo_a", "physics.task_a", level, score)

        # Create minimal task structure
        tasks_dir = tmp_path / "tasks" / "physics" / "task_a"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "task.yaml").write_text("id: physics.task_a\n")
        (tasks_dir / "prompt_b1.md").write_text("Prompt B1")
        (tasks_dir / "prompt_b2.md").write_text("Prompt B2")
        (tasks_dir / "prompt_b3.md").write_text("Prompt B3")

        from ai4sci_bench.analysis.reviewer import run_all_diagnoses

        with patch("ai4sci_bench.analysis.reviewer._call_claude") as mock_claude:
            mock_claude.return_value = (
                "TASK_UNDERSTANDING: ok\n"
                "WHAT_IMPLEMENTED: something\n"
                "PROCESS_SCORE_MATCH: matches\n"
                "MAIN_MISMATCH: none\n"
                "ANOMALY_DETECTED: None detected\n"
                "CLASSIFICATION: Agent求解代码算法实现错误\n"
                "CAUSE_BUCKET: Agent\n"
                "SCORER_BREAKDOWN: x=1\n"
                "EVIDENCE: none\n"
                "BOTTOM_LINE: fine\n"
            )

            summary = run_all_diagnoses(
                output_dir=tmp_path,
                combo_labels=["combo_a"],
                tasks_dir=tmp_path / "tasks",
                review_all=True,
                prompt_levels=["b1", "b2", "b3"],
            )

        assert summary.reviewed_count == 3
        assert len(summary.diagnoses) == 3
        assert len(summary.gradient_analyses) == 1  # 1 task × 1 combo, 3 levels >= 2

    def test_review_all_false_only_low_scores(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "physics.t1", "b1", 90)
        _make_result_json(tmp_path, "combo_a", "physics.t2", "b1", 5)

        task_dir1 = tmp_path / "tasks" / "physics" / "t1"
        task_dir1.mkdir(parents=True, exist_ok=True)
        (task_dir1 / "task.yaml").write_text("id: physics.t1")
        task_dir2 = tmp_path / "tasks" / "physics" / "t2"
        task_dir2.mkdir(parents=True, exist_ok=True)
        (task_dir2 / "task.yaml").write_text("id: physics.t2")

        from ai4sci_bench.analysis.reviewer import run_all_diagnoses

        with patch("ai4sci_bench.analysis.reviewer._call_claude") as mock_claude:
            mock_claude.return_value = "CAUSE_BUCKET: Agent\nBOTTOM_LINE: x"

            summary = run_all_diagnoses(
                output_dir=tmp_path,
                combo_labels=["combo_a"],
                tasks_dir=tmp_path / "tasks",
                threshold=10.0,
                review_all=False,
            )

        assert len(summary.diagnoses) == 1
        assert len(summary.gradient_analyses) == 0  # no gradient when not review_all

    def test_prompt_levels_filter_in_review_all(self, tmp_path):
        for level in ["b1", "b2", "b3", "b4"]:
            _make_result_json(tmp_path, "combo_a", "physics.task_a", level, 50)

        tasks_dir = tmp_path / "tasks" / "physics" / "task_a"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "task.yaml").write_text("id: physics.task_a\n")

        from ai4sci_bench.analysis.reviewer import run_all_diagnoses

        with patch("ai4sci_bench.analysis.reviewer._call_claude") as mock_claude:
            mock_claude.return_value = "CAUSE_BUCKET: Agent\nBOTTOM_LINE: x"

            summary = run_all_diagnoses(
                output_dir=tmp_path,
                combo_labels=["combo_a"],
                tasks_dir=tmp_path / "tasks",
                review_all=True,
                prompt_levels=["b1", "b2"],
            )

        assert len(summary.diagnoses) == 2
        levels_reviewed = {d.prompt_level for d in summary.diagnoses}
        assert levels_reviewed == {"b1", "b2"}

    def test_claude_timeout_handled(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "physics.t1", "b1", 5)

        task_dir = tmp_path / "tasks" / "physics" / "t1"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("id: physics.t1")
        tasks_dir = tmp_path / "tasks"

        from ai4sci_bench.analysis.reviewer import run_all_diagnoses

        with patch("ai4sci_bench.analysis.reviewer._call_claude") as mock_claude:
            mock_claude.return_value = "DIAGNOSIS TIMEOUT"

            summary = run_all_diagnoses(
                output_dir=tmp_path,
                combo_labels=["combo_a"],
                tasks_dir=tasks_dir,
                review_all=False,
                threshold=10.0,
            )

        assert len(summary.diagnoses) == 1

    def test_claude_not_found_handled(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "physics.t1", "b1", 5)

        task_dir = tmp_path / "tasks" / "physics" / "t1"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("id: physics.t1")
        tasks_dir = tmp_path / "tasks"

        from ai4sci_bench.analysis.reviewer import run_all_diagnoses

        with patch("ai4sci_bench.analysis.reviewer._call_claude") as mock_claude:
            mock_claude.return_value = "SKIPPED: Claude Code CLI not found"

            summary = run_all_diagnoses(
                output_dir=tmp_path,
                combo_labels=["combo_a"],
                tasks_dir=tasks_dir,
                review_all=False,
                threshold=10.0,
            )

        assert len(summary.diagnoses) == 1
        assert "SKIPPED" in summary.diagnoses[0].bottom_line


class TestCLIRunDiagnoseOption:
    """Test that --diagnose is properly registered on the run command."""

    def test_run_help_shows_diagnose(self):
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--diagnose" in result.output
        assert "--diagnose-threshold" in result.output

    def test_review_help_shows_prompt_levels(self):
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["review", "--help"])
        assert result.exit_code == 0
        assert "--prompt-levels" in result.output
        assert "--all-scores" in result.output


class TestCollectMaterials:
    """Test _collect_materials function."""

    def test_collects_task_files(self, tmp_path):
        from ai4sci_bench.analysis.reviewer import _collect_materials

        # Create task structure
        task_dir = tmp_path / "tasks" / "physics" / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("id: physics.task_a\nname: Test")
        (task_dir / "prompt_b2.md").write_text("Solve the problem")
        (task_dir / "custom_scorer.py").write_text("def score(): pass")
        (task_dir / "generate_gt.py").write_text("def generate(): pass")

        # Create result data
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        result_path = result_dir / "result__b2.json"
        result_data = {
            "task_id": "physics.task_a",
            "prompt_level": "b2",
            "final_score": 50.0,
            "gates_passed": True,
            "status": "completed",
            "gate_results": [],
            "score_results": [{"score": 50}],
            "_json_path": str(result_path),
        }
        result_path.write_text(json.dumps(result_data))

        materials = _collect_materials(result_data, tmp_path / "tasks", tmp_path)
        assert "task.yaml" in materials
        assert "prompt_b2.md" in materials
        assert "custom_scorer.py" in materials
        assert "generate_gt.py" in materials
        assert "score_details.json" in materials

    def test_missing_optional_files(self, tmp_path):
        from ai4sci_bench.analysis.reviewer import _collect_materials

        task_dir = tmp_path / "tasks" / "physics" / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("id: physics.task_a")

        result_dir = tmp_path / "results"
        result_dir.mkdir()
        result_path = result_dir / "result__b1.json"
        result_data = {
            "task_id": "physics.task_a",
            "prompt_level": "b1",
            "final_score": 0,
            "_json_path": str(result_path),
        }
        result_path.write_text(json.dumps(result_data))

        materials = _collect_materials(result_data, tmp_path / "tasks", tmp_path)
        assert "task.yaml" in materials
        assert "custom_scorer.py" not in materials
        assert "score_details.json" in materials

    def test_trajectory_files_collected(self, tmp_path):
        from ai4sci_bench.analysis.reviewer import _collect_materials

        task_dir = tmp_path / "tasks" / "physics" / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("id: physics.task_a")

        result_dir = tmp_path / "results"
        result_dir.mkdir()
        result_path = result_dir / "result__b1.json"
        result_data = {
            "task_id": "physics.task_a",
            "prompt_level": "b1",
            "final_score": 0,
            "_json_path": str(result_path),
        }
        result_path.write_text(json.dumps(result_data))

        # Create trajectory files
        (result_dir / "result__b1.agent_stdout.jsonl").write_text('{"line": 1}\n{"line": 2}')
        (result_dir / "result__b1.agent_stderr.log").write_text("error log")

        materials = _collect_materials(result_data, tmp_path / "tasks", tmp_path)
        assert "trajectory.agent_stdout.jsonl" in materials
        assert "trajectory.agent_stderr.log" in materials

    def test_long_trajectory_truncated(self, tmp_path):
        from ai4sci_bench.analysis.reviewer import _collect_materials

        task_dir = tmp_path / "tasks" / "physics" / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("id: physics.task_a")

        result_dir = tmp_path / "results"
        result_dir.mkdir()
        result_path = result_dir / "result__b1.json"
        result_data = {
            "task_id": "physics.task_a",
            "prompt_level": "b1",
            "final_score": 0,
            "_json_path": str(result_path),
        }
        result_path.write_text(json.dumps(result_data))

        # Create a very long trajectory
        long_content = "x" * 100000
        (result_dir / "result__b1.agent_stdout.jsonl").write_text(long_content)

        materials = _collect_materials(result_data, tmp_path / "tasks", tmp_path)
        traj = materials["trajectory.agent_stdout.jsonl"]
        assert len(traj) < 60000  # truncated
        assert "TRUNCATED" in traj


class TestBuildReviewPrompt:
    """Test _build_review_prompt construction."""

    def test_contains_all_dimensions(self):
        from ai4sci_bench.analysis.reviewer import _build_review_prompt

        result_data = {
            "task_id": "physics.sod",
            "prompt_level": "b2",
            "_combo_label": "combo_a",
            "final_score": 50.0,
        }
        materials = {"task.yaml": "id: physics.sod", "prompt_b2.md": "Solve it"}
        prompt = _build_review_prompt(result_data, materials)

        assert "Dimension 1" in prompt
        assert "Dimension 2" in prompt
        assert "Dimension 3" in prompt
        assert "Dimension 4" in prompt
        assert "Dimension 5" in prompt
        assert "TASK_UNDERSTANDING:" in prompt
        assert "PROCESS_SCORE_MATCH:" in prompt
        assert "ANOMALY_DETECTED:" in prompt

    def test_includes_materials(self):
        from ai4sci_bench.analysis.reviewer import _build_review_prompt

        result_data = {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 50.0,
        }
        materials = {"task.yaml": "content_here", "custom_scorer.py": "def score(): pass"}
        prompt = _build_review_prompt(result_data, materials)
        assert "content_here" in prompt
        assert "def score(): pass" in prompt


class TestBuildGradientPrompt:
    """Test _build_gradient_prompt construction."""

    def test_includes_all_level_scores(self, tmp_path):
        from ai4sci_bench.analysis.reviewer import _build_gradient_prompt

        task_dir = tmp_path / "physics" / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "prompt_b1.md").write_text("B1 prompt")
        (task_dir / "prompt_b2.md").write_text("B2 prompt")

        prompt = _build_gradient_prompt(
            "physics.task_a", "combo_a",
            {"b1": 90.0, "b2": 70.0},
            tmp_path,
        )
        assert "B1: 90.0" in prompt
        assert "B2: 70.0" in prompt
        assert "B1 prompt" in prompt
        assert "IS_MONOTONIC:" in prompt

    def test_long_prompt_truncated(self, tmp_path):
        from ai4sci_bench.analysis.reviewer import _build_gradient_prompt

        task_dir = tmp_path / "physics" / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "prompt_b1.md").write_text("x" * 20000)

        prompt = _build_gradient_prompt(
            "physics.task_a", "combo_a",
            {"b1": 90.0},
            tmp_path,
        )
        assert "TRUNCATED" in prompt


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_unicode_in_result_data(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "physics.量子", "b1", 50.0)
        results = find_all_instances(tmp_path, ["combo_a"])
        assert len(results) == 1
        assert results[0]["task_id"] == "physics.量子"

    def test_zero_score(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "t1", "b1", 0.0)
        results = find_all_instances(tmp_path, ["combo_a"])
        assert len(results) == 1
        assert results[0]["final_score"] == 0.0

    def test_negative_score(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "t1", "b1", -5.0)
        results = find_low_score_instances(tmp_path, ["combo_a"], threshold=0)
        assert len(results) == 1

    def test_very_high_score(self, tmp_path):
        _make_result_json(tmp_path, "combo_a", "t1", "b1", 100.0)
        results = find_low_score_instances(tmp_path, ["combo_a"], threshold=10)
        assert len(results) == 0

    def test_multiple_tasks_multiple_combos_multiple_levels(self, tmp_path):
        for combo in ["combo_a", "combo_b", "combo_c"]:
            for task in ["physics.t1", "math.t2", "chem.t3"]:
                for level in ["b1", "b2", "b3", "b4"]:
                    score = 50.0 + hash(f"{combo}{task}{level}") % 50
                    _make_result_json(tmp_path, combo, task, level, score)

        results = find_all_instances(tmp_path, ["combo_a", "combo_b", "combo_c"])
        assert len(results) == 36  # 3 combos × 3 tasks × 4 levels

        results_filtered = find_all_instances(
            tmp_path, ["combo_a", "combo_b", "combo_c"],
            prompt_levels=["b1", "b3"],
        )
        assert len(results_filtered) == 18  # 3 combos × 3 tasks × 2 levels

    def test_diagnosis_with_special_chars_in_output(self):
        raw = 'BOTTOM_LINE: Agent used "magic" number 3.14159 — π reference?'
        d = _parse_review_output(raw, {
            "task_id": "t1", "prompt_level": "b1",
            "_combo_label": "c1", "final_score": 0,
        })
        assert "magic" in d.bottom_line
        assert "π" in d.bottom_line

    def test_empty_combo_labels_list(self, tmp_path):
        results = find_all_instances(tmp_path, [])
        assert len(results) == 0

    def test_write_report_with_slashes_in_names(self, tmp_path):
        summary = DiagnosisSummary(
            diagnoses=[
                Diagnosis("domain/task", "b1", "combo/name", 50,
                          cause_bucket="Agent", classification="X", bottom_line="y"),
            ],
        )
        write_diagnosis_report(summary, tmp_path)
        diag_dir = tmp_path / "diagnosis"
        assert diag_dir.exists()
        md_files = list(diag_dir.glob("*.md"))
        assert len(md_files) >= 2  # at least per-instance + summary
