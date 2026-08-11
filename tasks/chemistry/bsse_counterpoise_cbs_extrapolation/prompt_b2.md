You are given a synthetic precomputed energy ledger for a weak two-fragment
dimer across a four-level ladder. The numbers are modeled after HF/reference
and correlated total basis-set convergence, but they are not live
quantum-chemistry calculations. The visible inputs are:

- `data/case_index.csv`
- `data/ladder_spec.json`
- `data/channel_ledger.csv`

Write `analysis.py` and produce the required files in `results/`.

This task is about counterpoise-style balancing and asymptotic extrapolation, but the ledger has been intentionally anonymized.
The `level_index` values `2, 3, 4, 5` are the benchmark's simulated
correlation-consistent basis cardinal numbers `X`.

What you must recover from the data:

- among the five row codes at each case/level, one row is the combined system
- the other four rows form two fragment families
- within each family, one row is a native/local fragment evaluation
- the partner row in that family is an expanded/balanced fragment evaluation in the combined-center representation
- one input energy column is the lower-order reference-like channel
- the other input energy column is the higher-order total channel

Use those recovered roles to build:

- `path_0`: the interaction from the two native/local rows
- `path_1`: the interaction from the two expanded/balanced rows
- `path_gap_kcal_mol = path_1_kcal_mol - path_0_kcal_mol`

Also report the within-family stabilization shifts:

- for each case/channel/level, compute the two positive shifts between the local row and the expanded row inside each fragment family
- sort those two shifts ascending
- write them as `shift_small_kcal_mol` and `shift_large_kcal_mol`

For asymptotic limits:

- treat the reference-like channel with the benchmark reference convention
  `E_ref(X) = E_inf + A * exp(-alpha * X)` on all four levels `X = 2, 3, 4, 5`
- use one shared `alpha` across all cases and both paths; for a fixed `alpha`,
  fit each case/path series' own `E_inf` and `A` by ordinary unweighted least
  squares, and choose the shared `alpha` that minimizes the total unweighted
  SSE over all reference-like case/path series. Search alpha over `[0.25, 3.5]`
- form a residual channel as `total - reference`
- extrapolate that residual channel with `delta(X) = delta_inf + B * X^-3`
  by ordinary unweighted least squares on the upper ladder, meaning
  `X = 3, 4, 5`
- assemble the total-channel limit from the reference-like limit plus the residual limit

Required outputs:

1. `results/path_scan.csv`
   Exact columns:
   `case_id,channel_role,level_label,level_index,path_0_kcal_mol,path_1_kcal_mol,path_gap_kcal_mol`

2. `results/pair_shift_scan.csv`
   Exact columns:
   `case_id,channel_role,level_label,level_index,shift_small_kcal_mol,shift_large_kcal_mol`

3. `results/asymptotic_summary.csv`
   Exact columns:
   `case_id,reference_limit_path_0_kcal_mol,reference_limit_path_1_kcal_mol,delta_limit_path_0_kcal_mol,delta_limit_path_1_kcal_mol,total_limit_path_0_kcal_mol,total_limit_path_1_kcal_mol`

4. `results/convergence_plot.png`

Implementation requirements:

- Read the provided files instead of hardcoding values.
- Keep scan rows sorted by `case_id`, then `channel_role` (`reference`, `total`), then `level_index`.
- Keep pair-shift rows sorted by `case_id`, then `channel_role` (`reference`, `total`), then `level_index`.
- Write a complete runnable script with no placeholders and no command-line arguments.
