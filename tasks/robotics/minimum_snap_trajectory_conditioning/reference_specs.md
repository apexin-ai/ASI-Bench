# Minimum-Snap Flight-Log Reconstruction Reference

This task implements RB1 from the robotics supplement table as a trajectory
reconstruction problem rather than a clean waypoint-optimization exercise.  The
public workspace contains noisy 3D observations, approximate knot-time windows,
segment-duration bounds, public query times, and selected derivative
constraints.  Exact waypoint positions, clean query states, and outlier labels
are reference-only.

The scientific target is still a high-order piecewise polynomial trajectory with
the public coefficient convention, C0-C3 continuity, and the public trajectory
quality index.  The added reconstruction stage prevents low-information prompts
from being solved by directly applying the standard fixed-waypoint
minimum-snap/Hermite recipe to exact waypoints.  A successful solution must infer
the latent trajectory from contaminated data, identify unreliable measurements,
choose knot times, and produce stable coefficients despite large coordinate
offsets and small local motion.

The reference generator first creates a hidden minimum-snap trajectory using the
same stable NumPy reduced-knot-derivative solver as the original RB1 draft.  It
then samples a flight log with Gaussian noise, sparse intervals, and structured
outliers.  Public query times are held out from fitting labels; reference query
states include position, velocity, acceleration, and jerk.

Scoring is result-based:

- query-state reconstruction against hidden clean states;
- coefficient trajectory agreement at hidden audit times;
- knot-time agreement plus feasibility (waypoint windows, duration bounds,
  monotonicity, total time);
- outlier-score ranking against hidden outlier labels;
- public continuity and derivative-constraint residuals;
- objective/report consistency and diagnostic figure presence.

The evaluation mechanism is publicly disclosed to the agent: the `evaluation`
section of `data/task_info.json` describes every scoring component, including
the fact that trajectory agreement is measured at hidden held-out times inside
every segment for derivative orders 0-4. Only the concrete sample times,
reference states, and outlier labels stay hidden.

Scoring is per-component fault isolated: a malformed prediction file zeroes
only the components that depend on it (a broken outlier file cannot erase the
trajectory credit), while missing or malformed reference files still fail the
scorer as an infrastructure error.

Band calibration (2026-07-03): the trajectory/query/objective error bands are
anchored by oracle experiments, not by GT self-similarity. With observation
noise sigma ~0.02-0.06 m and segment durations down to ~1e-2 s, the latent
high-order derivatives are noise-limited: the intended robust-IRLS + reduced
Hermite pipeline achieves scale-normalized trajectory RMS ~0.11-0.14 whether
knot times are estimated from windows (0.113), refined by search (0.137), or
taken from the reference itself (0.133). Naive or partial attempts measured
from real agent runs sit at 0.9-2.5. Full credit is therefore placed at 0.15
and zero credit at 0.8 (objective: 0.08/0.8; query zero credit is tighter at
0.4 because query states at public times can be approximated by direct
local smoothing without recovering the latent structure), separating "executed the
intended pipeline" from "produced structurally valid but wrong output".
Earlier bands (full 1.5e-3) demanded ~100x more accuracy than the data
information content allows; every agent and the oracle scored only the ~8
structural points, making levels indistinguishable.

Design note on GT de-idealization (v0.13): the latent trajectory is no
longer an exact constrained minimizer of the smoothness functional. The free
knot derivatives receive multiplicative Gaussian noise (eps=0.05) after the
optimal solve. Rationale: an exact minimizer collapses the trajectory to ~36
identifiable dof; structure-aware solvers recovered it to sample_rel 1.7e-4
(600x below the observation-noise floor) and scored 100. With the
perturbation, the structure-assuming shortcut bottoms out at sample_rel
~0.14-0.18 (calibrated across instance shapes) and the general robust-fit
path at ~0.11, so ~0.11 is a hard physical floor for every approach. Scoring
bands map floor-level performance to roughly 25-30 total points and even a
hypothetical 30%-beyond-floor solution to <47, making the 50-point difficulty
threshold unreachable by construction. Full credit remains reachable only by
the reference itself (self-check).

Design note on metric non-disclosure (v0.12): the quality-metric formula is
no longer stated in `data/task_info.json`. Frontier reasoning models were
reliably reconstructing the full intended pipeline from the phrase "time
integral of squared derivative order 4" (measured: every no-hint agent run
built the same snap-penalized robust fit), collapsing the B1->B3 information
gradient. task_info now declares the metric as an undisclosed property of the
data-generating process to be inferred from the data; the formula moves to
prompt_b1 only. Consequently the numeric content of results/objective.json is
NOT scored (agents without the formula cannot compute it - scoring it would
recreate the S6 unfairness); objective_agreement is still scored because it is
a pure functional of the submitted coefficients and times. The file itself
remains a required artifact for B1/B2-level self-checks.

Design note on waypoint windows: each public window contains the latent knot
time but uses independently drawn left/right margins, so window midpoints do
not encode the reference knot times. (Symmetric windows in versions <= 0.6
leaked the exact answer at the midpoint.)

The hard static gate forbids SciPy, external optimization/modeling packages,
symbolic/high-precision packages, ML frameworks, dedicated trajectory-generation
packages, and reference-file access.


## Difficulty-gate convention (plan A)

The difficulty gate for this task is judged on **B3 and B4 only** (mean
score < 50 for codex gpt-5.5 xhigh under the standard restricted
protocol). B1 and B2 are the solvability and knowledge-gradient reference
levels: B1 discloses the per-instance deviation law (rendered into the
prompt) and is expected to score high; its score measures disclosure
quality plus agent execution reliability, not gate compliance. This
follows the plan-A convention established for
computer_science.calibrated_association_transfer (v0.6 evaluation note).
Rationale: the B3/B4 ceiling is pinned by information (the deviation
structure leaves <= ~2.5 sigma of trace at observed times and the
hypothesis space is CV-degenerate), so the gate holds structurally; the
B1 premium proves the task is solvable with the disclosed knowledge
(oracle range 80-98, ground truth scores 100.0).
