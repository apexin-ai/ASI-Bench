# Synthetic BSSE Counterpoise Ledger + Asymptotic-Limit Reconstruction

Used by ErrorAnalyzer to diagnose agent failures.

## Key Implementation Details

- This task uses a synthetic precomputed energy ledger modeled after HF-like
  reference convergence and correlated total-energy convergence. It is not a
  live quantum-chemistry runtime task and not a full reproduction of the
  original papers.
- The Boys-Bernardi contribution used here is the function-counterpoise idea: one interaction pathway uses native fragment rows, while the other uses fragment rows evaluated in the combined-center basis.
- The ladder `level_index` is defined by this benchmark as a simulated
  correlation-consistent basis cardinal number `X` with labels
  `level_02`, `level_03`, `level_04`, `level_05` corresponding to
  `X = 2, 3, 4, 5`.
- Helgaker et al. motivate the inverse-cubic correlation/residual CBS
  extrapolation. The reference-like exponential form, and especially the
  single shared-alpha bundle fit, are explicit benchmark conventions rather
  than a literal attribution to that paper.
- The visible input is a synthetic precomputed single-point ledger, not a live quantum-chemistry runtime task.
- The ledger is intentionally anonymized:
  - row codes do not reveal which row is the combined system
  - the two energy columns do not reveal which one is the reference-like channel
  - however, the public contract does state that the four non-combined rows form two fragment families, each with a native/local row and an expanded/balanced row
- `path_gap_kcal_mol` must equal `path_1_kcal_mol - path_0_kcal_mol`.
- `pair_shift_scan.csv` reports the two positive within-family stabilization shifts for each case/channel/level, sorted ascending.
- The asymptotic summary must report:
  - the reference-like limits for both paths
  - the residual limits for both paths
  - the total-channel limits assembled from reference plus residual
- The intended reference-channel workflow in v3 is a shared-alpha exponential
  fit across the whole reference-path bundle, not an exact three-point
  closed-form reconstruction:
  - for each reference-like series, use `E_ref(X) = E_inf + A * exp(-alpha * X)`
    on all four levels `X = 2, 3, 4, 5`;
  - use one shared `alpha` for every case and both paths;
  - for each fixed `alpha`, fit each series' own `E_inf` and `A` by ordinary
    unweighted least squares;
  - choose the shared `alpha` that minimizes the total unweighted SSE over all
    reference-like case/path series. The generator searches alpha in `[0.25, 3.5]`.
- The intended residual-channel workflow in v3 is an upper-ladder inverse-cubic
  regression, not a single two-point shortcut:
  - form the residual series as `delta(X) = total(X) - reference(X)` for each
    case/path;
  - fit `delta(X) = delta_inf + B * X^-3` by ordinary unweighted least squares
    using only the upper ladder `X = 3, 4, 5`;
  - assemble `total_limit_path_i = reference_limit_path_i + delta_limit_path_i`.

## Common Failure Modes

- Treating the most negative interaction pathway as automatically more physical.
- Recovering the right path gap but pairing the fragment-family rows incorrectly.
- Mistaking the balanced fragment rows for ordinary isolated-fragment rows.
- Failing to identify which anonymous energy column is the lower-order reference-like channel.
- Extrapolating the total channel directly with a single formula instead of separating reference and residual parts.
- Using a per-series exact shortcut on the reference channel instead of the shared dominant-alpha fit implied by the visible ladder.
- Applying an inverse-power model to the reference-like channel or an exponential model to the residual channel.
- Using the wrong path ordering, so `path_0` and `path_1` are swapped.
- Computing `path_gap_kcal_mol` with the wrong sign.
