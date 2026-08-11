# Piecewise Polynomial Flight Reconstruction

## Goal

The workspace contains a timestamped 3D flight log, approximate timing windows
for latent knots, segment duration bounds, public query times, and additional
one-sided derivative requirements.  Construct a piecewise polynomial trajectory
and choose the waypoint times.

The trajectory should explain the reliable observations, avoid being pulled by
bad measurements, satisfy the listed derivative requirements, use the declared
total time and polynomial degree, and follow the continuity requirements and quality
quality metric described in `data/task_info.json`.

Read all files under `data/` before choosing an implementation strategy.  The
submitted times, coefficients, query predictions, and outlier scores must use
the declared conventions and ordering.

## Output

Write `analysis.py` and create `results/` containing:

1. `results/knot_times.csv` with columns `waypoint_id,t`.
2. `results/segment_coefficients.npy` with shape `[n_segments, 3, degree+1]`.
3. `results/knot_derivatives.csv` with columns `waypoint_id,axis,side,derivative_order,value`.
4. `results/query_predictions.csv` with columns `query_id,t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz`.
5. `results/outlier_scores.csv` with columns `observation_id,outlier_score`, one row per observation, where larger values indicate more likely outliers.
6. `results/objective.json` with keys `metric`, `basis`, `per_axis`, and `total` (the numeric values are self-reported and not numerically scored; the file must exist and contain finite values).
7. `results/trajectory_diagnostics.png`.

Use only the Python standard library, NumPy, and Matplotlib for the diagnostic
figure. Do not import SciPy, external optimization/modeling packages, symbolic
math packages, high-precision arithmetic packages, machine-learning frameworks,
or dedicated trajectory-generation packages. Do not use dynamic imports such as `importlib` or `__import__`.
