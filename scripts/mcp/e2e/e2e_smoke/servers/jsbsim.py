"""Direct (agent-free) E2E smoke test for the pinned jsbsim-mcp flight-dynamics tools.

Run with the server's own virtualenv so that the ``jsbsim`` Python module (the
same JSBSim build the server uses) is importable for the references::

    ~/mcp/jsbsim/.venv/bin/python scripts/mcp/e2e/smoke.py jsbsim --config ~/mcp/jsbsim.mcp.json

No network is needed. References are computed here, in this process, outside
the server: a separate ``FGFDMExec`` is driven through the same scenario with
JSBSim's own property names (``ic/*``, ``attitude/theta-deg`` ...) and an exact
ft/s -> kt conversion, then compared with what the MCP tools return.

L1: real tools/call of all 10 tools: aircraft catalogue, initial conditions,
property writes/reads, stepping and telemetry against the reference;
repeatability (a second session reaching the same state through the other
initial-condition path and chunked steps); defect probes for trim,
execute_script, partial initial conditions and missing properties (the shared
L0/L1 checks are in ``e2e_smoke/runner.py``).

Upstream defects that do not make a correctly used tool wrong (in-band errors,
ignored parameters, misleading results, fields read from non-existent
properties, a crash in execute_script, stdout chatter) are WARN; wrong numbers
for correct inputs are FAIL.

execute_script, which can crash the whole server, runs in a second server
process; two more short-lived servers show why the launch env (JBSIM_ROOT,
JSBSIM_DEBUG=0) is required.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from ..client import MCPError, StdioMCP, text_of
from ..runner import Caller, Report, Session, Smoke, check_rejected

AIRCRAFT = "c172x"
DT = 1.0 / 60.0
FPS_PER_KT = 6076.115485564304 / 3600.0      # 1 international nautical mile per hour in ft/s
# Every key of set_initial_conditions, all given explicitly (unspecified keys are reset to 0).
IC = {"altitude_ft": 4000.0, "latitude_deg": 37.0, "longitude_deg": -122.0, "airspeed_fps": 168.78,
      "heading_deg": 90.0, "pitch_deg": 2.0, "roll_deg": 0.0}
# JSBSim's own initial-condition properties, for the reference (airspeed_fps is a calibrated airspeed).
IC_REFERENCE = {"ic/h-sl-ft": IC["altitude_ft"], "ic/lat-geod-deg": IC["latitude_deg"],
                "ic/long-gc-deg": IC["longitude_deg"], "ic/vc-kts": IC["airspeed_fps"] / FPS_PER_KT,
                "ic/psi-true-deg": IC["heading_deg"], "ic/theta-deg": IC["pitch_deg"], "ic/phi-deg": IC["roll_deg"]}
# Readback right after the initial conditions: (property, expected, tolerance).
IC_READBACK = (("position/h-sl-ft", IC["altitude_ft"], 1e-6), ("velocities/vc-fps", IC["airspeed_fps"], 1e-3),
               ("attitude/psi-deg", IC["heading_deg"], 1e-6), ("attitude/theta-deg", IC["pitch_deg"], 1e-6),
               ("attitude/phi-deg", IC["roll_deg"], 1e-6), ("position/lat-geod-deg", IC["latitude_deg"], 1e-6),
               ("position/long-gc-deg", IC["longitude_deg"], 1e-6))
# Applied after the initial conditions, in this order. c172x needs propulsion/set-running=-1
# (propulsion/engine[0]/set-running does not start it).
SETTINGS = (("propulsion/set-running", -1.0), ("fcs/mixture-cmd-norm", 1.0), ("fcs/throttle-cmd-norm", 0.8))
RUN_S = 10.0
# Full-precision get_property comparisons after RUN_S (the reference converts ft/s -> kt
# exactly, the server with 0.592484; that alone moves the state by < 1e-4).
STATE_TOL = {"position/h-sl-ft": 2e-3, "velocities/vc-kts": 2e-4, "attitude/theta-deg": 1e-4,
             "attitude/psi-deg": 1e-4, "aero/alpha-deg": 1e-4, "propulsion/engine[0]/thrust-lbs": 1e-2}
TELEMETRY_SLACK = 2e-3       # absolute, added to half a unit in the last printed decimal
ACCELERATIONS = ("accelerations/udot-ft_sec2", "accelerations/wdot-ft_sec2", "accelerations/qdot-rad_sec2")
TRIM_MODES = ("longitudinal", "none", "no_such_mode")
SCRIPT_LITERAL = '<run start="0" end="1" dt="0.01"></run>'   # the form the tool description suggests
SCRIPT_FILE = "scripts/c1722.xml"                               # a stock JSBSim script for c172x
MISSING_PROPERTY = "no/such/property"
REQUIRED_ENV_KEYS = ("JBSIM_ROOT", "JSBSIM_DEBUG")

# get_telemetry fields: friendly name -> (property the server reads, printed decimals,
# JSBSim property that actually holds the quantity, if the server's one does not exist).
TELEMETRY_FIELDS = {
    "lat_deg": ("position/latitude-deg", 6, "position/lat-geod-deg"),
    "lon_deg": ("position/longitude-deg", 6, "position/long-gc-deg"),
    "alt_ft": ("position/h-sl-ft", 2, None),
    "altitude_agl_ft": ("position/altitude-agl-ft", 2, "position/h-agl-ft"),
    "pitch_deg": ("attitude/pitch-deg", 2, "attitude/theta-deg"),
    "roll_deg": ("attitude/roll-deg", 2, "attitude/phi-deg"),
    "heading_deg": ("attitude/heading-true-deg", 2, "attitude/psi-deg"),
    "airspeed_fps": ("velocities/vc-fps", 2, None),
    "airspeed_kt": ("velocities/vc-kts", 2, None),
    "ground_speed_fps": ("velocities/vg-fps", 2, None),
    "mach": ("velocities/mach", 3, None),
    "u_fps": ("velocities/u-fps", 2, None),
    "v_fps": ("velocities/v-fps", 2, None),
    "w_fps": ("velocities/w-fps", 2, None),
    "alpha_deg": ("aero/alpha-deg", 2, None),
    "beta_deg": ("aero/beta-deg", 2, None),
    "cl": ("aero/cl-squared", 4, None),
    "lift_lbs": ("forces/lift-lbs", 2, None),
    "drag_lbs": ("forces/drag-lbs", 2, None),
    "side_lbs": ("forces/side-lbs", 2, None),
    "thrust_lbs": ("propulsion/engine[0]/thrust-lbs", 2, None),
    "n1": ("propulsion/engine[0]/n1", 1, None),
    "rpm": ("propulsion/engine[0]/rpm", 0, "propulsion/engine[0]/engine-rpm"),
    "engine_running": ("propulsion/engine[0]/running", None, None),
    "fuel_remaining_lbs": ("propulsion/total-fuel-lbs", 2, None),
    "nz_g": ("accelerations/nz", 2, "accelerations/n-pilot-z-norm"),
    "wind_north_fps": ("velocities/wind-north-fps", 2, "atmosphere/wind-north-fps"),
    "wind_east_fps": ("velocities/wind-east-fps", 2, "atmosphere/wind-east-fps"),
    "wind_down_fps": ("velocities/wind-down-fps", 2, "atmosphere/wind-down-fps"),
    "wow_nose": ("gear/gear[0]/wow", None, "gear/unit[0]/WOW"),
    "wow_main_l": ("gear/gear[1]/wow", None, "gear/unit[1]/WOW"),
    "wow_main_r": ("gear/gear[2]/wow", None, "gear/unit[2]/WOW"),
    "gear_comp_main_ft": ("gear/gear[1]/compression-ft", 3, "gear/unit[1]/compression-ft"),
}


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def frames_for(seconds: float, dt_s: float = DT) -> int:
    """Frames the step tool documents for `seconds` at `dt_s` (rounded, at least one)."""
    return max(1, round(seconds / dt_s))


def aircraft_dirs(root: Path) -> list[str]:
    """Aircraft JSBSim can load by name: aircraft/<name>/<name>.xml."""
    base = Path(root) / "aircraft"
    return sorted(d.name for d in base.iterdir() if d.is_dir() and (d / f"{d.name}.xml").is_file())


def in_band_error(payload) -> str | None:
    """The error message of a normal (isError=false) result that reports failure."""
    if isinstance(payload, dict) and (payload.get("ok") is False or "error" in payload):
        return str(payload.get("error") or payload.get("note") or payload)
    return None


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_telemetry(frame: dict, reference: dict[str, float | None]) -> tuple[list[str], list[str]]:
    """(mismatches, dead fields) of a get_telemetry frame.

    ``reference`` maps every property named in TELEMETRY_FIELDS to its value in
    the reference FDM, or None if the property does not exist there. A field
    whose server property exists must equal the reference up to the printed
    rounding; a field whose property does not exist is "dead" (the server
    silently reports 0/false).
    """
    mismatches, dead = [], []
    for name, (path, decimals, real) in TELEMETRY_FIELDS.items():
        value = frame.get(name)
        ref = reference.get(path)
        if ref is None:
            real_value = reference.get(real) if real else None
            dead.append(f"{name}<-{path}" + (f" (real {real}={real_value:.4g}, server {value})"
                                              if real_value is not None else f" (server {value})"))
            continue
        if decimals is None:
            if bool(value) != bool(ref):
                mismatches.append(f"{name}={value} vs {bool(ref)}")
            continue
        tol = 0.5 * 10 ** -decimals + TELEMETRY_SLACK
        if not finite(value) or abs(value - ref) > tol:
            mismatches.append(f"{name}={value} vs reference {ref:.6g} (tol {tol:.2g})")
    return mismatches, dead


def trim_digest(payload: dict) -> tuple:
    """The numeric part of a trim result, for comparing modes."""
    return tuple((k, payload.get(k)) for k in sorted(payload) if k not in {"mode", "ok"})


# --------------------------------------------------------------------------
# Independent reference (JSBSim driven directly, outside the server)
# --------------------------------------------------------------------------

class Reference:
    def __init__(self, root: Path) -> None:
        os.environ["JSBSIM_DEBUG"] = "0"        # read by FGFDMExec's constructor; keeps our stdout clean
        import jsbsim
        self.jsbsim = jsbsim
        self.root = Path(root)

    def fdm(self, ic: dict | None = IC_REFERENCE, settings=SETTINGS):
        fdm = self.jsbsim.FGFDMExec(str(self.root))
        fdm.set_dt(DT)
        if not fdm.load_model(AIRCRAFT):
            raise RuntimeError(f"reference could not load {AIRCRAFT} from {self.root}")
        if ic is not None:
            for path, value in ic.items():
                fdm.set_property_value(path, float(value))
            fdm.run_ic()
        for path, value in settings:
            fdm.set_property_value(path, float(value))
        return fdm

    @staticmethod
    def advance(fdm, seconds: float) -> None:
        for _ in range(frames_for(seconds)):
            fdm.run()

    @staticmethod
    def get(fdm, path: str) -> float | None:
        """Property value, or None if JSBSim has no such property (get_property_value would say 0.0)."""
        if fdm.get_property_manager().get_node(path) is None:
            return None
        return float(fdm.get_property_value(path))

    def telemetry_reference(self, fdm) -> dict[str, float | None]:
        paths = {p for path, _, real in TELEMETRY_FIELDS.values() for p in (path, real) if p}
        return {p: self.get(fdm, p) for p in paths}

    def loads_script(self, xml: str) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
            handle.write(xml)
        try:
            return bool(self.fdm().load_script(handle.name))
        finally:
            os.unlink(handle.name)

    def jsbsim_trim(self) -> dict:
        """JSBSim's own longitudinal trim (FGTrim via do_trim(0)) from the same state."""
        fdm = self.fdm()
        try:
            fdm.do_trim(0)
            ok = True
        except Exception:  # JSBSim raises TrimFailureError
            ok = False
        return {"ok": ok, **{p.split("/")[1]: self.get(fdm, p) for p in ACCELERATIONS}}


# --------------------------------------------------------------------------
# Tool-call helpers
# --------------------------------------------------------------------------

def call_json(call: Caller, report: Report, check: str, tool: str, arguments: dict, *, allow_error=False):
    """tools/call + JSON decode; None (with a FAIL) on transport/tool/JSON errors."""
    return call.json(check, tool, arguments, allow_error=allow_error)


def get_value(call: Caller, report: Report, check: str, sid: str, path: str):
    payload = call_json(call, report, check, "get_property", {"session_id": sid, "path": path})
    if not isinstance(payload, dict) or "value" not in payload:
        if payload is not None:
            report.add("L1", check, "FAIL", f"unexpected get_property result {str(payload)[:200]}")
        return None
    return payload["value"]


def new_session(call: Caller, report: Report, check: str, **arguments) -> str | None:
    payload = call_json(call, report, check, "create_session", {"aircraft": AIRCRAFT, **arguments})
    if not isinstance(payload, dict) or not payload.get("session_id"):
        if payload is not None:
            report.add("L1", check, "FAIL", f"no session_id: {str(payload)[:200]}")
        return None
    return payload["session_id"]


def apply_settings(call: Caller, report: Report, check: str, sid: str) -> bool:
    for path, value in SETTINGS:
        payload = call_json(call, report, f"{check}[{path}]", "set_property",
                            {"session_id": sid, "path": path, "value": value})
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            if payload is not None:
                report.add("L1", f"{check}[{path}]", "FAIL", f"write refused: {str(payload)[:200]}")
            return False
    return True


# --------------------------------------------------------------------------
# Checks (main server)
# --------------------------------------------------------------------------

def check_list_aircraft(call: Caller, report: Report, root: Path) -> None:
    name = "list_aircraft"
    payload = call_json(call, report, name, "list_aircraft", {})
    if payload is None:
        return
    expected = aircraft_dirs(root)
    got = payload.get("aircraft") if isinstance(payload, dict) else None
    problems = []
    if got != expected:
        problems.append(f"server {got and len(got)} names vs {len(expected)} loadable dirs; "
                        f"extra {sorted(set(got or []) - set(expected))[:5]}, missing {sorted(set(expected) - set(got or []))[:5]}")
    if isinstance(payload, dict) and payload.get("count") != len(got or []):
        problems.append(f"count {payload.get('count')} != {len(got or [])}")
    if AIRCRAFT not in (got or []):
        problems.append(f"{AIRCRAFT} not listed")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else f"{len(got)} aircraft = aircraft/<name>/<name>.xml under JBSIM_ROOT")


def check_main_flight(call: Caller, report: Report, ref: Reference, root: Path) -> dict | None:
    """create_session -> set_initial_conditions -> set_property x3 -> step -> get_telemetry/get_property."""
    created = call_json(call, report, "create_session", "create_session", {"aircraft": AIRCRAFT})
    if not isinstance(created, dict) or not created.get("session_id"):
        report.add("L1", "create_session", "FAIL", f"no session_id: {str(created)[:200]}")
        return None
    sid = created["session_id"]
    problems = [f"{k}={created.get(k)!r} (expected {v!r})" for k, v in
                (("aircraft", AIRCRAFT), ("sim_time", 0.0), ("root", str(root))) if created.get(k) != v]
    if not finite(created.get("dt")) or abs(created["dt"] - DT) > 1e-12:
        problems.append(f"dt={created.get('dt')} (expected default {DT})")
    report.add("L1", "create_session", "FAIL" if problems else "PASS",
               "; ".join(problems) or f"session {sid}: {AIRCRAFT}, dt=1/60 s, t=0, root=JBSIM_ROOT")
    payload = call_json(call, report, "set_initial_conditions", "set_initial_conditions", {"session_id": sid, **IC})
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        report.add("L1", "set_initial_conditions", "FAIL", f"not accepted: {str(payload)[:200]}")
        return None
    problems = []
    for path, expected, tol in IC_READBACK:
        value = get_value(call, report, f"get_property[{path}]", sid, path)
        if not finite(value) or abs(value - expected) > tol:
            problems.append(f"{path}={value} (expected {expected})")
    report.add("L1", "set_initial_conditions[readback via get_property]", "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               "altitude, calibrated airspeed (airspeed_fps), heading, pitch, roll, lat, lon all applied")

    if not apply_settings(call, report, "set_property", sid):
        return None
    problems = []
    for path, value in SETTINGS[1:]:          # set-running is a command, not a readable state
        got = get_value(call, report, f"get_property[{path}]", sid, path)
        if got != value:
            problems.append(f"{path}={got} (wrote {value})")
    report.add("L1", "set_property[readback]", "FAIL" if problems else "PASS", "; ".join(problems))

    step = call_json(call, report, "step", "step", {"session_id": sid, "seconds": RUN_S})
    want = frames_for(RUN_S)
    ok = isinstance(step, dict) and step.get("frames") == want and finite(step.get("sim_time")) \
        and abs(step["sim_time"] - want * DT) < 1e-9 and step.get("session_id") == sid
    report.add("L1", "step", "PASS" if ok else "FAIL",
               f"{want} frames of {DT:.6f} s -> t={step.get('sim_time') if isinstance(step, dict) else None}")

    fdm = ref.fdm()
    ref.advance(fdm, RUN_S)
    expected = ref.telemetry_reference(fdm)
    thrust = expected.get("propulsion/engine[0]/thrust-lbs") or 0.0
    if thrust <= 0.0:
        report.add("L1", "reference scenario", "FAIL", "engine not running in the reference; fix the smoke inputs")

    frame = call_json(call, report, "get_telemetry", "get_telemetry", {"session_id": sid})
    if isinstance(frame, dict) and "alt_ft" in frame:
        mismatches, dead = compare_telemetry(frame, expected)
        t_ok = finite(frame.get("t")) and abs(frame["t"] - want * DT) < 1e-6
        if not t_ok:
            mismatches.append(f"t={frame.get('t')}")
        live = len(TELEMETRY_FIELDS) - len(dead)
        report.add("L1", "get_telemetry[live fields vs reference]", "FAIL" if mismatches else "PASS",
                   "; ".join(mismatches[:6]) if mismatches else
                   f"{live} fields equal the reference to printed precision (alt {frame['alt_ft']} ft, "
                   f"{frame['airspeed_kt']} kt, thrust {frame['thrust_lbs']} lbf)")
        report.add("L1", "get_telemetry[dead fields]", "WARN" if dead else "PASS",
                   f"{len(dead)}/{len(TELEMETRY_FIELDS)} fields read JSBSim properties that do not exist and are "
                   f"always 0/false: " + "; ".join(dead) if dead else "", dead_fields=dead)
        if "cl" in frame and expected.get("aero/cl-squared") is not None:
            report.add("L1", "get_telemetry[field naming]", "WARN",
                       f"'cl' is aero/cl-squared (CL^2 = {frame['cl']}), not the lift coefficient")
    elif frame is not None:
        report.add("L1", "get_telemetry", "FAIL", f"not a telemetry frame: {str(frame)[:200]}")

    problems = []
    for path, tol in STATE_TOL.items():
        value = get_value(call, report, f"get_property[{path}]", sid, path)
        want_value = ref.get(fdm, path)
        if not finite(value) or want_value is None or abs(value - want_value) > tol:
            problems.append(f"{path}={value} vs reference {want_value}")
    report.add("L1", "get_property[state after step vs reference]", "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"{len(STATE_TOL)} properties at full precision within {max(STATE_TOL.values())}")
    return {"sid": sid, "h": ref.get(fdm, "position/h-sl-ft")}


def check_repeatability(call: Caller, report: Report, first_sid: str) -> None:
    """Second session: IC through create_session(initial_conditions=...), steps of RUN_S/2 twice."""
    sid = new_session(call, report, "create_session[initial_conditions]", initial_conditions=dict(IC))
    if sid is None or not apply_settings(call, report, "set_property[second session]", sid):
        return
    for k in range(2):
        call_json(call, report, f"step[second session {k + 1}/2]", "step", {"session_id": sid, "seconds": RUN_S / 2})
    paths = ("position/h-sl-ft", "velocities/vc-kts", "attitude/theta-deg", "attitude/psi-deg")
    first = [get_value(call, report, f"get_property[first {p}]", first_sid, p) for p in paths]
    second = [get_value(call, report, f"get_property[second {p}]", sid, p) for p in paths]
    same = first == second and all(finite(v) for v in first)
    report.add("L1", "repeatability[create_session IC + chunked step]", "PASS" if same else "FAIL",
               f"bit-identical state ({dict(zip(paths, first))})" if same else f"{first} vs {second}")
    closed = call_json(call, report, "close_session", "close_session", {"session_id": sid})
    after = call_json(call, report, "close_session[then get_telemetry]", "get_telemetry", {"session_id": sid})
    again = call_json(call, report, "close_session[twice]", "close_session", {"session_id": sid})
    ok = isinstance(closed, dict) and closed.get("ok") is True and in_band_error(after) == "unknown-session" \
        and isinstance(again, dict) and again.get("ok") is False
    report.add("L1", "close_session", "PASS" if ok else "FAIL",
               "closed; the id is unknown afterwards; closing twice returns ok=false" if ok else
               f"close={closed} after={str(after)[:120]} again={again}")


def check_missing_property(call: Caller, report: Report, sid: str) -> None:
    payload = call_json(call, report, "get_property[missing path]", "get_property",
                        {"session_id": sid, "path": MISSING_PROPERTY})
    if not isinstance(payload, dict):
        return
    if payload.get("present") is False or payload.get("value") is None:
        report.add("L1", "get_property[missing path]", "PASS", f"present={payload.get('present')}")
    else:
        report.add("L1", "get_property[missing path]", "WARN",
                   f"{MISSING_PROPERTY} -> value={payload.get('value')}, present={payload.get('present')}: "
                   "JSBSim returns 0.0 for unknown paths, so typos read as a real zero")


def check_defaults_and_partial_ic(call: Caller, report: Report) -> None:
    sid = new_session(call, report, "create_session[no initial conditions]")
    if sid is None:
        return
    one = call_json(call, report, "step[1 s without initial conditions]", "step", {"session_id": sid, "seconds": 1.0})
    h = get_value(call, report, "get_property[h without initial conditions]", sid, "position/h-sl-ft")
    if finite(h) and h > -1e3:
        report.add("L1", "create_session[without initial conditions]", "PASS", f"h={h:.1f} ft after 1 s")
    else:
        report.add("L1", "create_session[without initial conditions]", "WARN",
                   f"run_ic() is never called, so after {one and one.get('frames')} frames the state is "
                   f"invalid (h={h} ft); always set initial conditions first")
    zero = call_json(call, report, "step[seconds=0]", "step", {"session_id": sid, "seconds": 0})
    if isinstance(zero, dict):
        report.add("L1", "step[seconds=0]", "PASS" if zero.get("frames") == 0 else "WARN",
                   f"frames={zero.get('frames')}" + ("" if zero.get("frames") == 0 else
                                                     ": a zero-length step still integrates one frame"))
    call_json(call, report, "set_initial_conditions[altitude only]", "set_initial_conditions",
              {"session_id": sid, "altitude_ft": IC["altitude_ft"]})
    vc = get_value(call, report, "get_property[vc after altitude-only IC]", sid, "velocities/vc-fps")
    lat = get_value(call, report, "get_property[lat after altitude-only IC]", sid, "position/lat-geod-deg")
    if finite(vc) and vc == 0.0:
        report.add("L1", "set_initial_conditions[partial]", "WARN",
                   f"unspecified keys are reset to 0 (vc={vc} ft/s, lat={lat}) instead of being kept; "
                   "always pass all seven keys")
    else:
        report.add("L1", "set_initial_conditions[partial]", "PASS", f"vc={vc}")
    typo = call_json(call, report, "create_session[unknown IC key]", "create_session",
                     {"aircraft": AIRCRAFT, "initial_conditions": {"altitude": 4000.0}}, allow_error=True)
    if isinstance(typo, dict) and typo.get("_isError"):
        report.add("L1", "create_session[unknown IC key]", "PASS", "isError=true")
    else:
        report.add("L1", "create_session[unknown IC key]", "WARN",
                   "an unknown initial_conditions key ('altitude') is silently ignored")
        if isinstance(typo, dict) and typo.get("session_id"):
            call_json(call, report, "close_session[typo session]", "close_session", {"session_id": typo["session_id"]})
    call_json(call, report, "close_session[defaults session]", "close_session", {"session_id": sid})


def check_trim(call: Caller, report: Report, ref: Reference) -> None:
    results = {}
    for mode in TRIM_MODES:
        sid = new_session(call, report, f"create_session[trim {mode}]", initial_conditions=dict(IC))
        if sid is None or not apply_settings(call, report, f"set_property[trim {mode}]", sid):
            return
        payload = call_json(call, report, f"trim[{mode}]", "trim", {"session_id": sid, "mode": mode})
        t = get_value(call, report, f"get_property[t after trim {mode}]", sid, "simulation/sim-time-sec")
        residual = {p.split("/")[1]: get_value(call, report, f"get_property[{p} after trim]", sid, p)
                    for p in ACCELERATIONS}
        theta = get_value(call, report, f"get_property[theta after trim {mode}]", sid, "attitude/theta-deg")
        results[mode] = {"payload": payload, "t": t, "residual": residual, "theta": theta}
        call_json(call, report, f"close_session[trim {mode}]", "close_session", {"session_id": sid})
    long = results["longitudinal"]
    payload = long["payload"] if isinstance(long["payload"], dict) else {}
    if not payload or payload.get("_isError"):
        report.add("L1", "trim", "FAIL", f"no trim result: {str(payload)[:200]}")
        return
    jsb = ref.jsbsim_trim()
    worst = max(abs(v) for v in long["residual"].values() if finite(v)) if any(
        finite(v) for v in long["residual"].values()) else math.inf
    jsb_worst = max(abs(v) for k, v in jsb.items() if k != "ok" and finite(v))
    notes = []
    if payload.get("ok") is True and worst > 10 * max(jsb_worst, 1e-3):
        notes.append(f"ok=true but not trimmed: max |udot,wdot,qdot| = {worst:.3g} vs {jsb_worst:.3g} after "
                     f"JSBSim's own do_trim(0) (converged={jsb['ok']}) from the same state")
    if finite(long["t"]) and long["t"] > DT * 1.5:
        notes.append(f"advances the simulation to t={long['t']:.4f} s")
    if payload.get("throttle") == 0.7:
        notes.append("forces throttle 0.7 (overwrites the commanded 0.8)")
    if finite(payload.get("pitch_deg")) and finite(long["theta"]) and abs(payload["pitch_deg"] - long["theta"]) > 0.01:
        notes.append(f"reports pitch_deg={payload['pitch_deg']} while attitude/theta-deg={long['theta']:.3f}")
    digests = {m: trim_digest(r["payload"]) if isinstance(r["payload"], dict) else None for m, r in results.items()}
    if len(set(digests.values())) == 1:
        notes.append(f"mode is ignored: {', '.join(TRIM_MODES)} give identical results (incl. 'none')")
    if any(not finite(v) for k, v in payload.items() if k not in {"mode", "ok"} and not isinstance(v, str)):
        notes.append("non-finite values in the result")
    report.add("L1", "trim", "WARN" if notes else "PASS", "; ".join(notes) or "trimmed",
               trim_results={m: r["payload"] for m, r in results.items()}, jsbsim_do_trim=jsb)


def check_errors(call: Caller, report: Report) -> None:
    check_rejected(call, "create_session[unknown aircraft]", "create_session", {"aircraft": "no_such_aircraft"})
    for tool, arguments in (("step", {"seconds": 1.0}), ("get_telemetry", {}), ("trim", {})):
        check_rejected(call, f"{tool}[unknown session]", tool, {"session_id": "nosuchsession", **arguments},
                       in_band=_in_band)


def _in_band(result: dict) -> str | None:
    try:
        return in_band_error(json.loads(text_of(result)))
    except json.JSONDecodeError:
        return None


def run_l1(session: Session) -> None:
    call, report, ref, root = session.call, session.report, session.state["ref"], session.state["root"]
    check_list_aircraft(call, report, root)
    main = check_main_flight(call, report, ref, root)
    if main:
        check_repeatability(call, report, main["sid"])
        check_missing_property(call, report, main["sid"])
        call_json(call, report, "close_session[main]", "close_session", {"session_id": main["sid"]})
    check_defaults_and_partial_ic(call, report)
    check_trim(call, report, ref)
    check_errors(call, report)


# --------------------------------------------------------------------------
# Extra server processes
# --------------------------------------------------------------------------

def check_execute_script(session: Session) -> None:
    """execute_script in its own server: a stock script crashes the process."""
    report, ref, scratch = session.report, session.state["ref"], session.scratch
    before = set(scratch.iterdir())
    with session.spawn("script-cwd") as client:
        try:
            _execute_script(client, report, ref, scratch, before)
        except MCPError as exc:
            report.add("L1", "execute_script", "FAIL", f"server failed before the crash probe: {exc}")


def _execute_script(client: StdioMCP, report: Report, ref: Reference, scratch: Path, before: set) -> None:
    call = Caller(client, report)
    client.initialize()
    sid = new_session(call, report, "create_session[script]", initial_conditions=dict(IC))
    if sid is None:
        return
    t0 = get_value(call, report, "get_property[t before literal script]", sid, "simulation/sim-time-sec")
    payload = call_json(call, report, "execute_script[<run> literal]", "execute_script",
                        {"session_id": sid, "script": SCRIPT_LITERAL})
    t1 = get_value(call, report, "get_property[t after literal script]", sid, "simulation/sim-time-sec")
    loads = ref.loads_script(SCRIPT_LITERAL)
    if isinstance(payload, dict):
        if payload.get("ok") is True and not loads:
            report.add("L1", "execute_script[<run> literal]", "WARN",
                       f"ok=true ({payload.get('note')}) although JSBSim rejects this document "
                       "(load_script returns False; the tool ignores the return value); "
                       f"t {t0} -> {t1}: one ordinary frame, no script")
        else:
            report.add("L1", "execute_script[<run> literal]", "PASS" if payload.get("ok") == loads else "FAIL",
                       f"ok={payload.get('ok')}, JSBSim load_script={loads}")
    leaked = sorted(p.name for p in set(scratch.iterdir()) - before)
    report.add("L1", "execute_script[temp file]", "WARN" if leaked else "PASS",
               f"literal scripts are written to $TMPDIR and never deleted: {leaked}" if leaked else "")
    noise_before = len(client.non_json_stdout)
    report.add("L1", f"execute_script[{SCRIPT_FILE}]", *stock_script_probe(client, sid, noise_before))


def stock_script_probe(client: StdioMCP, sid: str, noise_before: int) -> tuple[str, str]:
    """Run a stock script, step, read, close; the server may die at any of these (WARN, not FAIL)."""
    steps = (("execute_script", {"session_id": sid, "script": SCRIPT_FILE}),
             ("step", {"session_id": sid, "seconds": 1.0}),
             ("get_property", {"session_id": sid, "path": "position/h-sl-ft"}),
             ("close_session", {"session_id": sid}),
             ("list_aircraft", {}))
    seen, notes = [], []
    for tool, arguments in steps:
        try:
            response = client.call_tool(tool, arguments, timeout=60)
        except MCPError as exc:
            notes.append(f"the server died during {tool} (exit {client.proc.poll()}): {exc}")
            break
        result = response.get("result", {})
        text = text_of(result)
        if result.get("isError"):
            seen.append(f"{tool}: error {text[:80]!r}")
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        if tool == "execute_script":
            seen.append(f"execute_script ok={payload.get('ok') if isinstance(payload, dict) else payload}")
        elif tool == "get_property" and isinstance(payload, dict):
            seen.append(f"h={payload.get('value')}")
            if not finite(payload.get("value")):
                notes.append(f"the script re-loads a model into the live session: h={payload.get('value')}")
    noise = len(client.non_json_stdout) - noise_before
    if noise:
        notes.append(f"{noise} non-JSON stdout line(s) despite JSBSIM_DEBUG=0")
    if any(s.startswith("step: error") for s in seen):
        notes.append("the next step fails")
    detail = "; ".join(notes) + (f" [{'; '.join(seen)}]" if seen else "")
    return ("WARN" if notes else "PASS"), detail


def probe_launch_env(session: Session, drop: str) -> tuple[str, str]:
    """A short-lived server without one launch env key: what goes wrong."""
    reduced = {k: v for k, v in session.env.items() if k != drop}
    try:
        with session.spawn(f"no-{drop}", env=reduced) as client:
            client.initialize()
            response = client.call_tool("create_session", {"aircraft": AIRCRAFT})
            result = response.get("result", {})
            noise = len(client.non_json_stdout)
    except MCPError as exc:
        return "FAIL", f"server without {drop} failed: {exc}"
    if drop == "JBSIM_ROOT":
        if result.get("isError"):
            return "WARN", (f"without JBSIM_ROOT the data root is looked up from the cwd and create_session "
                            f"fails ({text_of(result)[:240]}); the launch env sets it")
        return "PASS", "aircraft found without JBSIM_ROOT"
    if noise:
        return "WARN", (f"without JSBSIM_DEBUG=0 one create_session writes {noise} non-JSON lines to the "
                        "JSON-RPC stdout; the launch env sets it")
    return "PASS", "no stdout chatter without JSBSIM_DEBUG"


# --------------------------------------------------------------------------
# Declaration
# --------------------------------------------------------------------------

def prepare(session: Session) -> None:
    root = Path(session.server.get("env", {}).get("JBSIM_ROOT", session.checkout / "jsbsim_data"))
    session.state.update(root=root, ref=Reference(root))


def after(session: Session) -> None:
    check_execute_script(session)
    if not session.args.skip_env_probes:
        for key in REQUIRED_ENV_KEYS:
            session.report.add("L0", f"server without {key}", *probe_launch_env(session, key))


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-env-probes", action="store_true",
                        help="do not start the servers without JBSIM_ROOT / JSBSIM_DEBUG")


SMOKE = Smoke(
    server="jsbsim",
    run_l1=run_l1,
    packages=("jsbsim", "mcp", "pydantic", "numpy"),
    add_arguments=_arguments,
    prepare=prepare,
    after=after,
    report_fields=lambda session: {"scenario": {"aircraft": AIRCRAFT, "dt": DT, "initial_conditions": IC,
                                                "settings": SETTINGS, "run_s": RUN_S}},
)
