from __future__ import annotations

import base64
import importlib

import pytest


@pytest.fixture
def executor_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASIBENCH_HARBOR_EXECUTOR_TOKEN", "test-token")
    return importlib.import_module("ai4sci_bench.runner.harbor_executor_server")


def test_solver_execution_returns_artifact_and_process_result(
    executor_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    def fake_execute(command, *, cwd, timeout):
        seen.update(mode=cwd.stat().st_mode & 0o777, timeout=timeout)
        assert command == [
            "python", "solver.py", "--input", "instance.tsp", "--output", "tour.npy",
            "--seed", "271828182", "--time-limit-seconds", "3600",
        ]
        assert timeout == 3600
        (cwd / "tour.npy").write_bytes(b"tour")
        return False, 0, b"solver stdout", b""

    monkeypatch.setattr(executor_server, "_execute", fake_execute)
    payload = {
        "kind": "solver",
        "task_kind": "tsp",
        "seed": 271828182,
        "timeout_seconds": 3600,
        "files": {
            "solver.py": base64.b64encode(b"pass").decode("ascii"),
            "instance.tsp": base64.b64encode(b"instance").decode("ascii"),
        },
    }

    result = executor_server._run(payload)

    assert result["ok"] is True
    assert result["timed_out"] is False
    assert result["exit_code"] == 0
    assert result["stdout_tail"] == "solver stdout"
    assert base64.b64decode(result["artifacts"]["tour.npy"]) == b"tour"
    assert result["infrastructure_error"] is None
    assert seen["mode"] == 0o777


def test_executor_rejects_path_traversal(executor_server) -> None:
    with pytest.raises(ValueError, match="unsafe input path"):
        executor_server._safe_relative("../secret")


def test_ngspice_execution_returns_only_generated_files(
    executor_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_execute(command, *, cwd, timeout):
        assert command == ["ngspice", "-b", "bench.cir"]
        assert timeout == 45
        (cwd / "waveform.dat").write_text("0 1\n", encoding="ascii")
        return False, 0, b"simulation complete", b""

    monkeypatch.setattr(executor_server, "_execute", fake_execute)
    payload = {
        "kind": "ngspice",
        "netlist": "bench.cir",
        "timeout_seconds": 45,
        "files": {
            "bench.cir": base64.b64encode(b".end\n").decode("ascii"),
            "model.lib": base64.b64encode(b"* model\n").decode("ascii"),
        },
    }

    result = executor_server._run(payload)

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["stdout_tail"] == "simulation complete"
    assert set(result["artifacts"]) == {"waveform.dat"}
    assert base64.b64decode(result["artifacts"]["waveform.dat"]) == b"0 1\n"
