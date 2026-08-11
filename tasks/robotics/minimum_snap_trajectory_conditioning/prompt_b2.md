# Minimum-Snap Flight-Log Reconstruction

## Goal

Reconstruct the latent high-order piecewise polynomial flight trajectory from
the noisy 3D observations in `data/observations.csv`. Some observations are
outliers and some time intervals are under-sampled. The submitted trajectory
must use the coefficient convention, total time, continuity requirements, and
quality metric declared in `data/task_info.json`; knot times must lie inside
`data/waypoint_windows.csv`, and each segment duration must satisfy
`data/segment_time_bounds.csv`.

Treat this as a robust minimum-snap trajectory reconstruction with
tracking-dropout deviations. The observation stream has dropout gaps; within
each gap (in the segment of duration `d`, local time `delta`) the vehicle
departs from the smooth plan by the deterministic bump
`A * 256 * delta^4 * (d-delta)^4 / d^8 * (1 + s*(delta/d - 0.5)) * u`
with skew `s = {{ residual_skew }}`, which vanishes together with its first
three derivatives at both bounding waypoints. The direction `u`: normalize the chord's xy projection to a UNIT vector first, rotate that
unit vector about the vertical axis by {{ residual_azimuth_deg }} degrees,
then append {{ residual_z_comp }} as the third (vertical) component and
normalize the resulting 3-vector. Observations are unavailable
wherever the deviation magnitude exceeds {{ residual_occlusion_level_m }} m
(each gap is additionally widened by a 0.12 s guard margin per side) - so
each observation gap's extent is the level crossing of its bump, and the
gap geometry determines the amplitude `A`. Recover each gap's amplitude,
include the reconstructed bumps in the submitted polynomials, keep the plan
itself smooth through each gap (the gap interior is unobserved - restrict
it to the knot-state Hermite family or regularize the 4th derivative
strongly, and let the disclosed bump carry ALL of the in-gap deviation),
compute the submitted objective from the final coefficients WITH the bumps
included (they dominate the snap integral), enforce the declared continuity and the fixed
derivative constraints in `data/constraints.csv`, use a robust reweighting
scheme so that outlying measurements do not distort the curve, and optimize
the knot times within their windows and duration bounds. Scoring emphasizes
position accuracy at hidden held-out times, with the dropout intervals
carrying most of the weight. Read all files under `data/` before
choosing an implementation; the submitted times, coefficients, query
predictions, and outlier scores must use the declared conventions and
ordering, and the reported artifacts must be consistent with the saved
coefficients.

## Output

Write `analysis.py` and create `results/` containing:

1. `results/knot_times.csv` with columns `waypoint_id,t`.
2. `results/segment_coefficients.npy` with shape `[n_segments, 3, degree+1]`.
3. `results/knot_derivatives.csv` with columns `waypoint_id,axis,side,derivative_order,value`.
4. `results/query_predictions.csv` with columns `query_id,t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz`.
5. `results/outlier_scores.csv` with columns `observation_id,outlier_score`, one row per observation, where larger values indicate more likely outliers.
6. `results/objective.json` with keys `metric`, `basis`, `per_axis`, and `total`.
7. `results/trajectory_diagnostics.png`.

Use only the Python standard library, NumPy, and Matplotlib for the diagnostic
figure. Do not import SciPy, CVXPY, CasADi, OSQP, SymPy, mpmath, machine-learning
frameworks, or dedicated trajectory-generation packages. Do not use dynamic imports such as `importlib` or `__import__`.
