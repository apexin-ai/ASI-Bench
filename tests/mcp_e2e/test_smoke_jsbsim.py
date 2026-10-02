"""e2e_smoke/servers/jsbsim.py without JSBSim: frames, aircraft scan, telemetry comparison, trim digest, crash probe."""


from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, runner

jsb = support.smoke_module("jsbsim")


def test_frames_for_rounds_and_never_returns_zero():
    assert jsb.frames_for(10.0) == 600
    assert jsb.frames_for(5.0) == 300
    assert jsb.frames_for(0.0) == 1
    assert jsb.frames_for(1.0, 1 / 120) == 120


def test_aircraft_dirs_requires_name_xml(tmp_path):
    for name, files in {"c172x": ["c172x.xml"], "odd": ["other.xml"], "empty": []}.items():
        (tmp_path / "aircraft" / name).mkdir(parents=True)
        for f in files:
            (tmp_path / "aircraft" / name / f).write_text("<fdm_config/>")
    (tmp_path / "aircraft" / "aircraft_template.xml").write_text("x")
    assert jsb.aircraft_dirs(tmp_path) == ["c172x"]


def test_jsbsim_in_band_error_detection():
    assert jsb.in_band_error({"ok": False, "error": "unknown-session"}) == "unknown-session"
    assert jsb.in_band_error({"ok": False, "note": "script not found"}) == "script not found"
    assert jsb.in_band_error({"ok": True, "path": "x", "value": 1.0}) is None
    assert jsb.in_band_error([1, 2]) is None


def _telemetry_reference(**overrides):
    ref = {}
    for path, decimals, real in jsb.TELEMETRY_FIELDS.values():
        ref[path] = None if real else 1.0
        if real:
            ref[real] = 5.0
    ref.update(overrides)
    return ref


def test_compare_telemetry_separates_wrong_values_from_dead_fields():
    ref = _telemetry_reference(**{"position/h-sl-ft": 4115.051741, "velocities/mach": 0.1234567})
    frame = {name: (1.0 if decimals is not None else True) for name, (_, decimals, _) in jsb.TELEMETRY_FIELDS.items()}
    frame.update(alt_ft=4115.05, mach=0.123, pitch_deg=0.0, rpm=0)
    mismatches, dead = jsb.compare_telemetry(frame, ref)
    assert mismatches == []
    assert len(dead) == sum(1 for _, _, real in jsb.TELEMETRY_FIELDS.values() if real)
    assert any(d.startswith("pitch_deg<-attitude/pitch-deg (real attitude/theta-deg=5") for d in dead)

    frame.update(alt_ft=4115.07, mach=float("nan"))
    mismatches, _ = jsb.compare_telemetry(frame, ref)
    assert [m.split("=")[0] for m in mismatches] == ["alt_ft", "mach"]


def test_compare_telemetry_dead_field_without_known_real_property():
    ref = _telemetry_reference(**{"forces/lift-lbs": None})
    frame = {name: 1.0 for name in jsb.TELEMETRY_FIELDS}
    frame["lift_lbs"] = 0.0
    _, dead = jsb.compare_telemetry(frame, ref)
    assert "lift_lbs<-forces/lift-lbs (server 0.0)" in dead


def test_trim_digest_ignores_mode_and_ok():
    a = {"ok": True, "mode": "longitudinal", "throttle": 0.7, "elevator": 0.0}
    b = {"ok": True, "mode": "none", "throttle": 0.7, "elevator": 0.0}
    assert jsb.trim_digest(a) == jsb.trim_digest(b)
    assert jsb.trim_digest(a) != jsb.trim_digest({**b, "throttle": 0.8})


def test_jsbsim_missing_property_and_unknown_session_classification():
    report = jsb.Report()
    client = StubClient({"get_property": rpc_json({"path": "no/such/property", "value": 0.0, "present": True})})
    jsb.check_missing_property(jsb.Caller(client, report), report, "s1")
    assert report_statuses(report) == {"get_property[missing path]": "WARN"}

    report = jsb.Report()
    client = StubClient({"get_property": rpc_json({"path": "no/such/property", "value": None, "present": False})})
    jsb.check_missing_property(jsb.Caller(client, report), report, "s1")
    assert report_statuses(report) == {"get_property[missing path]": "PASS"}

    report = jsb.Report()
    client = StubClient({
        "create_session": rpc_text("Error executing tool create_session: Failed to load aircraft", is_error=True),
        "step": rpc_json({"ok": False, "error": "unknown-session"}),
        "get_telemetry": rpc_text("unknown session", is_error=True),
        "trim": rpc_json({"ok": True, "mode": "longitudinal"}),
        "__nonexistent__": rpc_text("Unknown tool", is_error=True),
    })
    jsb.check_errors(jsb.Caller(client, report), report)
    assert report_statuses(report) == {"create_session[unknown aircraft]": "PASS", "step[unknown session]": "WARN",
                                 "get_telemetry[unknown session]": "PASS", "trim[unknown session]": "FAIL"}


def test_list_aircraft_compared_with_directory_scan(tmp_path):
    for name in ("c172x", "737"):
        (tmp_path / "aircraft" / name).mkdir(parents=True)
        (tmp_path / "aircraft" / name / f"{name}.xml").write_text("x")
    good = rpc_json({"aircraft": ["737", "c172x"], "count": 2})
    report = jsb.Report()
    jsb.check_list_aircraft(jsb.Caller(StubClient({"list_aircraft": good}), report), report, tmp_path)
    assert report_statuses(report) == {"list_aircraft": "PASS"}
    report = jsb.Report()
    bad = rpc_json({"aircraft": ["737"], "count": 2})
    jsb.check_list_aircraft(jsb.Caller(StubClient({"list_aircraft": bad}), report), report, tmp_path)
    assert report_statuses(report) == {"list_aircraft": "FAIL"}


class _DyingClient(StubClient):
    """Like StubClient, but the server process dies at `die_at`."""

    class _Proc:
        def poll(self):
            return -11

    def __init__(self, responses, die_at, chatter=0):
        super().__init__(responses, chatter)
        self.die_at = die_at
        self.proc = self._Proc()

    def call_tool(self, name, arguments, timeout=300.0):
        if name == self.die_at:
            raise jsb.MCPError("server closed stdout (exit=-11) while waiting for tools/call")
        return super().call_tool(name, arguments, timeout)


def test_stock_script_probe_turns_crashes_into_warn():
    responses = {"execute_script": rpc_json({"ok": True, "note": "queued c1722.xml"}),
                 "step": rpc_text("Error executing tool step: Trim Failed", is_error=True),
                 "get_property": rpc_text('{"path": "position/h-sl-ft", "value": NaN, "present": true}'),
                 "close_session": rpc_json({"ok": True}), "list_aircraft": rpc_json({"aircraft": []})}
    status, detail = jsb.stock_script_probe(_DyingClient(responses, die_at="close_session", chatter=3), "s", 0)
    assert status == "WARN"
    assert "died during close_session" in detail and "h=nan" in detail and "next step fails" in detail
    status, detail = jsb.stock_script_probe(_DyingClient(responses, die_at="execute_script"), "s", 0)
    assert status == "WARN" and "died during execute_script" in detail
    healthy = dict(responses, step=rpc_json({"frames": 60}),
                   get_property=rpc_json({"path": "position/h-sl-ft", "value": 4000.0, "present": True}))
    assert jsb.stock_script_probe(_DyingClient(healthy, die_at=None), "s", 0)[0] == "PASS"


def test_jsbsim_server_env_is_minimal_and_applies_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    env = runner.server_env({"command": "/m/jsbsim/.venv/bin/python",
                             "env": {"JBSIM_ROOT": "/m/jsbsim/jsbsim_data", "JSBSIM_DEBUG": "0"}},
                            tmp_path / "home", tmp_path / "tmp", pass_proxies=jsb.SMOKE.pass_proxies)
    assert env["TMPDIR"] == str(tmp_path / "tmp") and env["PATH"].startswith("/m/jsbsim/.venv/bin:")
    assert env["JSBSIM_DEBUG"] == "0" and env["JBSIM_ROOT"] == "/m/jsbsim/jsbsim_data"
    assert "ANTHROPIC_API_KEY" not in env
