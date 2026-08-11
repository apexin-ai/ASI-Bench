# Minimum-Snap Flight-Log Reconstruction

## Goal

Reconstruct the latent high-order piecewise polynomial flight trajectory from
the noisy 3D observations in `data/observations.csv`. Some observations are
outliers and some time intervals are under-sampled. The submitted trajectory
must use the coefficient convention, total time, continuity requirements, and
quality metric declared in `data/task_info.json`; knot times must lie inside
`data/waypoint_windows.csv`, and each segment duration must satisfy
`data/segment_time_bounds.csv`.

The latent trajectory is an imperfectly tracked realization of a
minimum-snap flight plan, and the tracker loses lock during aggressive
maneuvers: the observation stream has dropout gaps, and inside each gap the
vehicle departs from the smooth plan by a deterministic deviation bump
before re-anchoring at the bounding waypoints. Fit piecewise polynomials
that explain the reliable observations subject to C0-C3 continuity at
interior knots and the fixed derivative constraints in
`data/constraints.csv`, then reconstruct the in-gap deviations from the
disclosed law below. Down-weight suspected
outliers with an iteratively reweighted robust loss and map the final
normalized residuals monotonically into `[0, 1]` for the outlier scores (the
evaluator clips scores to `[0, 1]` before ranking). Choose the knot times yourself:
segment durations are free within their bounds and should be optimized against
the combined data-fit plus quality-metric objective.

The tracking-deviation law for this instance:

- Dropout mechanism: observations are unavailable wherever the deviation
  magnitude exceeds {{ residual_occlusion_level_m }} m, and each gap is
  widened by a 0.12 s guard margin on both sides.
- Bump shape: within an affected segment of duration `d` (local time
  `delta`), the deviation is
  `A * 256 * delta^4 * (d-delta)^4 / d^8 * (1 + s*(delta/d - 0.5)) * u`
  with skew `s = {{ residual_skew }}`. It vanishes together with its first
  three derivatives at both knots, so knot states, continuity, and the
  fixed constraints are unaffected.
- Direction `u`: take the segment's knot-to-knot chord and normalize the chord's xy projection to a UNIT vector first, rotate that
  unit vector about the vertical axis by {{ residual_azimuth_deg }} degrees,
  then append {{ residual_z_comp }} as the third (vertical) component and
  normalize the resulting 3-vector.
  (Order matters: the xy projection is normalized BEFORE the vertical
  component is appended - appending it to the raw meter-scale chord would
  dilute the vertical part several-fold.)
- Amplitude `A` (one per gap): the gap edges are the level crossings of the
  bump at the dropout threshold. Estimate the dilated window as the
  observed gap minus one local sampling interval, strip the two 0.12 s
  margins, and solve `A` from the crossing width of the disclosed shape by
  bisection.
- Two facts that decide success: (i) the gap interiors are unobserved -
  any fit freedom there beyond the disclosed structure will swing freely,
  so keep the plan itself smooth through each gap (strong snap
  regularization, or restrict the plan to the knot-state Hermite family)
  and let the disclosed bump carry ALL of the in-gap deviation; (ii) the
  bumps dominate the snap integral - compute the submitted objective from
  the final coefficients WITH the bumps included.

Key facts:

- The quality metric (undisclosed in `data/task_info.json`) is the time
  integral of squared derivative order 4. If
  `segment_coefficients.npy` stores ascending powers of elapsed segment time
  `delta = t - t_start`, then for powers `p,r >= 4` each segment contributes
  `c[p] * c[r] * falling(p,4) * falling(r,4) * duration**(p+r-7) / (p+r-7)`,
  where `falling(p,k) = p*(p-1)*...*(p-k+1)`.
- Observation coordinates carry constant offsets of order `1e6` while the
  local motion spans only a few meters. Do the linear algebra in centered
  coordinates and add the offset back when writing outputs, or the solves lose
  all precision.
- Evaluation compares POSITIONS against the latent reference at hidden
  held-out times across the full duration; times inside the dropout gaps
  dominate (about 70% of the trajectory component, which itself carries
  ~41% of the score, and much of the ~30% query component). In-gap position
  RMS earns full credit below 0.05 m and zero above 0.16 m; outside the
  gaps full credit below 0.02 m and zero above 0.10 m. Higher derivatives
  are not compared against the reference, but the submitted artifacts
  (knot derivatives, query predictions, objective) must be recomputed
  from the saved coefficients so the reports stay consistent.

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
