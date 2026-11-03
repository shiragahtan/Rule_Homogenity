# Homogeneity project (organized repo)

Self-contained repo: clone and run from the repo root. All paths are relative to the repo so nothing outside the clone is required for the default (ACS) dataset.

## Quick start (clone and run)

```bash
git clone <this-repo>
cd homogeneity_repo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**macOS (required for XGBoost):** install the OpenMP runtime once:

```bash
brew install libomp
```

Then run the main script from the repo root:

```bash
python algorithms/main.py
```

If you want the main script to append to the result csv files and not override the ones from previous runs add the flag '-d':
```bash
python algorithms/main.py -d
```

## Layout
- **ablation_results/** – ablation result CSVs and reports.
- **algorithms/** – main runner script and BruteForce, RW, greedy, random, CasualForest and WTE algorithms.
- **article_figures/** – notebook to generate figures from result CSVs.
- **configs/** – dataset and run configuration, and the rules we check for (paths are inside the repo).
- **datasets/** – datasets used by experiments (see `configs/config.json`).
- **experiments/** – experiment scripts, in particular ablation study and scalability test.
- **graphs/** - scripts that create graphs based on the csv files created from the main algorithm/from the experiments like calculating accuracy/average runtimes on main script or the scalability graphs.
- **problem_2_3_algorithms/** – benchmarks for largest δ and smallest ε.
- **results/** - CSV files used to generate the scripts in  `article_figures/generate_figures.ipynb`.
- **utils/** – CATE estimation (`ATE_update.py`, DoWhy `backdoor.linear_regression`) and the one-time dataset cleaning (`preprocessing.py`).
- **tests/** – synthetic-data checks for the estimator and the preprocessing (`python tests/test_cate_estimator.py`).

## Effect estimation and data cleaning
- **CATE** (overall rule population and every subgroup) is computed with DoWhy `CausalModel.estimate_effect(method_name="backdoor.linear_regression")` on exactly the rows of the group (the filtered subgroup is passed to DoWhy; no effect modifiers). No outlier trimming and no effect caps.
- **Confounders** are a fixed parameter: the dataset's `IMMUTABLE_ATTRIBUTES` from `configs/config.json` are passed to DoWhy as `common_causes` in every call (`CONFOUNDERS` in the config may override them). Only names that are not columns of the rule's data (the rule's condition/treatment columns are removed) are left out, because DoWhy raises an error on them; a column that is constant inside a subgroup is passed as is.
- **Subgroups whose effect cannot be computed or is unreliable** are ignored: no treated or no control rows, fewer rows than regression terms, a DoWhy error, or (subgroups only, not the rule's overall effect) a treated or control side with fewer than `delta / MIN_ARM_DELTA_DIVISOR` rows (config, default 10; the instructor chose this over "10% of the subgroup"). They are not counterexamples to homogeneity. Skips are counted (`ATE_update.get_skip_summary()`). If the overall rule population cannot be computed, the rule is reported as not evaluable. The instructor decided against filtering subgroups by DoWhy's p-value (it only tests "effect = 0"); `estimate_cate_with_p` still returns it for analysis.
- **Missing values** are removed once, when the dataset is loaded (`utils/preprocessing.py::clean_dataset`). The algorithms do not clean data while running.
- **Calibration of epsilon / delta after the estimator change** (`calibration_results/`, scripts in `experiments/`): `rw_min_epsilon.py` finds the smallest epsilon at which RW calls each rule homogeneous, for the old and the new estimator (`--repo` points at a checkout of the old code); `calibrate_epsilon_delta.py` + `analyze_calibration.py` compare per-subgroup deviations; `confounder_sensitivity.py` shows how the confounder choice moves each rule's overall effect. `generate_candidate_rules.py` is the original rule search (old repo, `generate_treatments.py`) ported to the current estimator: it lists every (immutable value, mutable value) rule whose condition covers more than the dataset's `COVERAGE_THRESHOLD` percent, ranks them by effect, and with `--replace-negative` proposes replacements for rules whose effect is negative (candidate lists in `calibration_results/candidates/`). `exact_min_epsilon.py` enumerates every subgroup (exact verdicts) and stores effect, p-value and arm sizes unfiltered; `analyze_exact.py` applies the subgroup-size filter and reports the exact smallest epsilon per rule.
