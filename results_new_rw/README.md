# Results with the new RW (26 Nov 2026)

All RW experiments rerun with `RW_OPTIONS` on (frozen per-dataset learned weights from
`configs/rw_learned_weights_<ds>.json` + size prefilter + revisit + up to 10 steps). RW rows only; brute force,
Greedy and the other algorithms were not rerun (their verdicts do not depend on RW) and stay in `results/`.

- `<ds>_homogeneity_results.xlsx`: main experiment (RW and the matching Random baseline rows).
- `<ds>_scalability_rows.csv`, `<ds>_scalability_attributes.csv`: scalability, RW only.
- `all_ablation_rows_new_rw.csv`: ε / δ ablation, RW only, all jobs (column `job`).

Old RW vs new RW: `calibration_results/rw_old_vs_new_full_runs.csv` (`experiments/compare_old_new_rw.py`).
Weight-method test (5 methods, 10 seeds, non-homogeneous rules): `calibration_results/rw_methods10_<ds>.csv`,
`rw_methods10_so_rest.csv` + `rw_seeds10_so.csv` (SO).
