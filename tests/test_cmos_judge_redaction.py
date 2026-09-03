"""Credential redaction tests for the CMOS task-local Judge scorers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from tasks.electrical_engineering.cmos_opamp_design import custom_scorer


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _vector_response(reasoning: str) -> str:
    return json.dumps(
        {
            "score": 10,
            "reasoning": reasoning,
            "subscores": {metric: 2 for metric in custom_scorer.TRUST_METRICS},
        }
    )


def test_vector_judge_redacts_key_from_raw_and_parsed_reasoning(tmp_path):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()
    (pred_dir / "simulation.py").write_text("print('ok')", encoding="utf-8")
    secret = "cmos-vector-secret-123"

    with patch.object(custom_scorer.litellm, "completion") as completion:
        completion.return_value = _completion(
            _vector_response(f"The request key was {secret}.")
        )
        result = custom_scorer._call_vector_judges(
            pred_dir=pred_dir,
            ref_dir=ref_dir,
            pred_file="simulation.py",
            ref_file="",
            rubric="score the simulation",
            model="gpt-5.4",
            num_judges=1,
            temperature=0.0,
            max_tokens=100,
            max_chars=1000,
            api_key=secret,
        )

    serialized = json.dumps(result)
    assert secret not in serialized
    assert "<redacted>" in result["raw_responses"][0]
    assert "<redacted>" in result["parsed_responses"][0]["reasoning"]


def test_strict_gate_redacts_key_from_raw_and_parsed_reasoning(tmp_path):
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "opamp_netlist.cir").write_text(
        ".subckt opamp vdd gnd inp inn out\n"
        "M1 n1 inp gnd gnd nmos\n"
        "M2 n2 inn gnd gnd nmos\n"
        "M3 out n1 vdd vdd pmos\n"
        "M4 out n2 vdd vdd pmos\n"
        ".ends opamp\n",
        encoding="utf-8",
    )
    secret = "cmos-strict-secret-456"

    with patch.object(custom_scorer.litellm, "completion") as completion:
        completion.return_value = _completion(
            json.dumps(
                {
                    "score": 8,
                    "pass": True,
                    "reasoning": f"Validated with {secret}.",
                }
            )
        )
        result = custom_scorer._netlist_strict_gate_report(
            pred_dir,
            {
                "strict_netlist_gate": {
                    "model": "gpt-5.4",
                    "num_judges": 1,
                    "api_key": secret,
                    "api_base": "https://judge.example.test/v1",
                    "api_protocol": "openai",
                }
            },
        )

    serialized = json.dumps(result)
    assert secret not in serialized
    assert "<redacted>" in result["raw_responses"][0]
    assert "<redacted>" in result["parsed_responses"][0]["reasoning"]


def test_metric_trust_report_returns_non_grounded_judge_payload(tmp_path):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()
    vector = {
        "available": True,
        "error": None,
        "median_score": 12.0,
        "median_subscores": {metric: 2.0 for metric in custom_scorer.TRUST_METRICS},
        "raw_responses": ["{}"],
        "parsed_responses": [{"score": 12.0, "reasoning": "ok", "subscores": {}}],
        "model": "gpt-5.4",
    }

    with patch.object(custom_scorer, "_call_vector_judges", return_value=vector):
        result = custom_scorer._metric_trust_report(
            pred_dir,
            ref_dir,
            {
                "trust_judge": {
                    "agent": "llm",
                    "model": "gpt-5.4",
                    "num_judges": 1,
                }
            },
        )

    assert result["failed"] is False
    assert result["testbench"] is vector
    assert result["extraction"] is vector
    assert all(result["trusted_metrics"].values())
