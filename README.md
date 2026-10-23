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
- **CATE** (overall rule population and every subgroup) is computed with DoWhy `CausalModel.estimate_effect(method_name="backdoor.linear_regression")` on exactly the rows of the group: no outlier trimming, no effect caps, no minimum arm sizes.
- **Confounders** are a fixed parameter: the `CONFOUNDERS` list of each dataset in `configs/config.json` (currently equal to its `IMMUTABLE_ATTRIBUTES`). Columns of the list that are not in a rule's data (the rule's condition/treatment columns are removed) or are constant inside a subgroup are not used.
- **Subgroups whose effect cannot be computed** (no treated or no control rows, fewer rows than regression terms, DoWhy error) are ignored; they are not counterexamples to homogeneity. Skips are counted (`ATE_update.get_skip_summary()`). If the overall rule population cannot be computed, the rule is reported as not evaluable.
- **Missing values** are removed once, when the dataset is loaded (`utils/preprocessing.py::clean_dataset`). The algorithms do not clean data while running.
