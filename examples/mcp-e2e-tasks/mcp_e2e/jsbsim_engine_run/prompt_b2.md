# Powered flight of a Cessna 172 via the jsbsim MCP server

`data/scenario.json` describes a short powered flight of the JSBSim `{{aircraft}}` model (a Cessna 172) near {{place}}: the aircraft, seven initial conditions (`altitude_ft`, `latitude_deg`, `longitude_deg`, `airspeed_fps` — a calibrated airspeed in ft/s —, `heading_deg`, `pitch_deg`, `roll_deg`), a list of JSBSim property `settings` (`path`, `value`) that start the engine and set mixture and throttle, and the run time `duration_s` ({{duration_s}} s).

Using the tools of the `jsbsim` MCP server, create one simulation session for the aircraft, set its initial conditions, apply the property settings in order, advance the simulation by `duration_s`, and read the resulting state.

## Output

Write `result.json` in the current working directory:

```json
{
  "altitude_ft": <float>,
  "airspeed_kt": <float>,
  "thrust_lbs": <float>,
  "sim_time_s": <float>
}
```

All four values are read from the simulation **after** advancing it by `duration_s`:

- `altitude_ft`: altitude above mean sea level in ft (JSBSim property `position/h-sl-ft`).
- `airspeed_kt`: calibrated airspeed in knots (`velocities/vc-kts`).
- `thrust_lbs`: engine thrust in lbf (`propulsion/engine[0]/thrust-lbs`).
- `sim_time_s`: simulation time in s (`simulation/sim-time-sec`).

Copy each value exactly as the simulator returns it (full precision where available); do not round, convert or compute them.

## Rules

- JSBSim is available **only** through the MCP server attached to this session. Do not install, import or run JSBSim or any other flight-dynamics package yourself (no `pip`/`uv` installs, no `import jsbsim`, no `jsbsim` command, no Python interpreter or data files belonging to the MCP server), and do not estimate values from memory or by hand.
- Use one simulation session for the whole run. Set **all seven** initial conditions from `data/scenario.json` (also the ones that are 0) before anything else, then apply the `settings` in the listed order, then advance the simulation by exactly `duration_s` at the default timestep (do not change the timestep).
- Do not run a trim/balancing routine or a simulation script: both change the aircraft state.
- Do not modify `data/scenario.json`.
