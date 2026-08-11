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

Sorry for the wall of text that follows — I promised the team I would forward
the whole context, so here it is.  This log landed on my desk after the flight
test group wrapped up their spring campaign; the folder also contained about
forty gigabytes of camera footage, a spreadsheet of battery voltages, and a
README that just said "ask Marek", except Marek left the company in March.
The original plan was for the summer intern to handle the reconstruction, but
the intern got pulled onto the demo for the investor visit, then the demo got
rescheduled twice, and now the deadline for this analysis is somehow before
the demo it was supposed to support.  Classic.  A colleague of mine tried a
related log-reconstruction problem last quarter and spent most of a week on
it, flip-flopping between two versions of his script; I never heard how it
ended because he went on parental leave, and his branch was deleted in the
last repo cleanup.  I also vaguely remember a professor warning our class
about a subtle failure mode in exactly this kind of task — something about
units, maybe?  Or time zones?  It bugs me that I cannot remember.  If you hit
a wall, my manager suggested we could "just do a simpler version first and
iterate", but the last time we said that, the simple version shipped to the
customer and we spent a quarter apologizing for it, so take that suggestion
with a grain of salt.  Also ignore the numbering scheme in the data folder if
it seems odd — an earlier pipeline prefixed everything with dates, someone
stripped those in a migration, and the observation IDs you see now are the
post-migration ones, which are the correct ones to use, as declared.  Budget
note: the workstation reserved for this analysis is shared with the CFD group
on weekdays, so if something takes hours, expect it to take longer.  And if
anyone asks, the reason the earlier draft report was withdrawn is unrelated
to this dataset — that was the other vehicle.  Good luck; I need to run to a
design review that should have been an email.

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
