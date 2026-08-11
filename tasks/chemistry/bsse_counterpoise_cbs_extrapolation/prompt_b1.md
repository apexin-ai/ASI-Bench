You are given a synthetic precomputed energy ledger for a weak two-fragment
dimer. The numbers are modeled after HF/reference and correlated total
basis-set convergence, but they are not live quantum-chemistry calculations.
The visible inputs are:

- `data/case_index.csv`
- `data/ladder_spec.json`
- `data/channel_ledger.csv`

Write `analysis.py` and produce the required files in `results/`.

The four ladder levels have `level_index` values `2, 3, 4, 5`; treat these as
the simulated correlation-consistent basis cardinal numbers `X` for the
extrapolation formulas below.

Use the instance-specific decoding block that appears above this prompt. For this task:

- `path_0` is the interaction built from the native/local fragment pair
- `path_1` is the interaction built from the balanced/expanded fragment pair
- `path_gap_kcal_mol = path_1_kcal_mol - path_0_kcal_mol`

Required outputs:

1. `results/path_scan.csv`
   Exact columns:
   `case_id,channel_role,level_label,level_index,path_0_kcal_mol,path_1_kcal_mol,path_gap_kcal_mol`

2. `results/pair_shift_scan.csv`
   Exact columns:
   `case_id,channel_role,level_label,level_index,shift_small_kcal_mol,shift_large_kcal_mol`
   At each case/channel/level:
   - compute the two positive within-family stabilization shifts
   - sort them ascending
   - write them as `shift_small_kcal_mol` and `shift_large_kcal_mol`

3. `results/asymptotic_summary.csv`
   Exact columns:
   `case_id,reference_limit_path_0_kcal_mol,reference_limit_path_1_kcal_mol,delta_limit_path_0_kcal_mol,delta_limit_path_1_kcal_mol,total_limit_path_0_kcal_mol,total_limit_path_1_kcal_mol`

4. `results/convergence_plot.png`
   Make a readable scientific plot comparing `path_0` and `path_1` across the ladder for the reference and total channels.

Asymptotic-limit convention:

- For every case/path reference-like series, fit
  `E_ref(X) = E_inf + A * exp(-alpha * X)` using all four levels
  `X = 2, 3, 4, 5`.
- Use one shared `alpha` for all reference-like series across every case and
  both paths. For a fixed `alpha`, fit each series' own `E_inf` and `A` by
  ordinary unweighted least squares. Choose the shared `alpha` that minimizes
  the total unweighted SSE over all reference-like case/path series. Search
  alpha over `[0.25, 3.5]`.
- Form each residual series as `delta(X) = total(X) - reference(X)`.
- Fit `delta(X) = delta_inf + B * X^-3` by ordinary unweighted least squares
  on the upper ladder only, meaning `X = 3, 4, 5`.
- Report `total_limit_path_i = reference_limit_path_i + delta_limit_path_i`.

Implementation requirements:

- Read the provided files instead of hardcoding values.
- Keep scan rows sorted by `case_id`, then `channel_role` (`reference`, `total`), then `level_index`.
- Keep pair-shift rows sorted by `case_id`, then `channel_role` (`reference`, `total`), then `level_index`.
- Write a complete runnable script with no placeholders and no command-line arguments.
