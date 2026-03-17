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
- **utils/** – ATE calculation algorithm.
