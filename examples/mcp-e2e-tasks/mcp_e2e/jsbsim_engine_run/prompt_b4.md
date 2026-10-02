# Powered flight of a Cessna 172

Fly the scenario in `data/scenario.json` (aircraft, initial conditions, property settings, run time) in a flight-dynamics simulation and report the resulting state. Use the flight-dynamics MCP tools available in this session.

Background: JSBSim is a six-degree-of-freedom flight-dynamics model driven through a property tree. Initial conditions take effect when the simulation is initialised, so they must be set before the engine and controls are commanded; `propulsion/set-running = -1` starts all engines, and `fcs/mixture-cmd-norm` / `fcs/throttle-cmd-norm` are normalised commands in [0, 1]. The simulation advances in fixed timesteps, so the run time is a whole number of frames.

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
