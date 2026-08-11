You are given a synthetic precomputed energy ledger for a weak two-fragment
system across a four-level ladder. The numbers are modeled after HF/reference
and correlated total basis-set convergence, but they are not live
quantum-chemistry calculations. The visible inputs are:

- `data/case_index.csv`
- `data/ladder_spec.json`
- `data/channel_ledger.csv`

Write `analysis.py` and produce the required files in `results/`.

The `level_index` values `2, 3, 4, 5` are the benchmark's simulated
correlation-consistent basis cardinal numbers `X`.

Goal:

- reconstruct two interaction pathways from the ledger rows at each case and ladder level
- one row at each case/level is the combined-system row
- the remaining four rows form two fragment families
- within each family, one row is a local/native evaluation and one row is an expanded evaluation in the combined-system representation
- `path_0` uses the two local/native rows and `path_1` uses the two expanded rows
- report those pathways for both reported channel roles
- compute the public diagnostic `path_gap_kcal_mol = path_1_kcal_mol - path_0_kcal_mol`
- compute the two positive within-family stabilization shifts for each case/channel/level, sort them ascending, and report them as `shift_small_kcal_mol` and `shift_large_kcal_mol`
- infer an asymptotic summary that separates a lower-order reference contribution, a residual contribution, and the reported total contribution
- use the benchmark asymptotic convention: fit every reference-like case/path
  series to `E_ref(X) = E_inf + A * exp(-alpha * X)` over `X = 2, 3, 4, 5`,
  with one shared `alpha` for all cases and both paths. For fixed `alpha`,
  fit each series' own `E_inf` and `A` by ordinary unweighted least squares;
  choose the shared `alpha` minimizing total unweighted SSE over all
  reference-like case/path series. Search alpha over `[0.25, 3.5]`
- form residuals as `delta(X) = total(X) - reference(X)`, then fit
  `delta(X) = delta_inf + B * X^-3` by ordinary unweighted least squares on
  the upper ladder `X = 3, 4, 5`; set
  `total_limit_path_i = reference_limit_path_i + delta_limit_path_i`

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
