"""
Make the paper's figures (the code of article_figures/generate_figures.ipynb) without Jupyter, in a separate folder
(the repo's article_figures/ is not touched), and save a PNG next to every PDF (for web pages).

    python scripts/make_figures.py --out ~/figs/oldrw                        # the results of this repo
    python scripts/make_figures.py --out ~/figs/adapt --rw-run ~/rw_adapt     # RW replaced by another run's RW

--rw-run: a copy of the repo in which only the RW jobs were rerun (parallel_runner.py plan --algos RW), e.g. with
another RW_OPTIONS. Its RW rows replace the RW rows of this repo's ablation and scalability results; the brute-force
rows stay, because they do not depend on RW. The benchmark figures (Problems 2 and 3) use brute force only and are
read from this repo.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASETS = ["so", "acs"]  # the datasets of the paper's figures

PNG_PATCH = '''
# --- added by scripts/make_figures.py: also save a PNG next to every PDF ---
import matplotlib.figure as _mf
_orig_savefig = _mf.Figure.savefig
def _savefig_png_too(self, fname, *a, **k):
    r = _orig_savefig(self, fname, *a, **k)
    if isinstance(fname, str) and fname.endswith(".pdf"):
        _orig_savefig(self, fname[:-4] + ".png", *a, **dict(k, format="png", dpi=130))
    return r
_mf.Figure.savefig = _savefig_png_too
'''


def combine_ablation(ds, rw_run, out_dir):
    """Brute-force rows of this repo + RW rows of rw_run -> <ds>_ablation_summary.csv (same code as the merge)."""
    import pandas as pd
    sys.path.insert(0, str(REPO / "experiments"))
    import ablation_study as A
    assert A.CHOSEN_DS == ds, (A.CHOSEN_DS, ds)
    old = pd.read_excel(REPO / "ablation_results" / f"{ds}_ablation_raw_results.xlsx")
    frames = [old[old.algorithm != "RW_Direct"]]
    files = sorted((Path(rw_run) / "par_runs" / "jobs").glob(f"{ds}-ablation-*/ablation_rows.csv"))
    for f in files:
        d = pd.read_csv(f)
        frames.append(d[d.algorithm == "RW_Direct"])
    df = pd.concat(frames, ignore_index=True)
    print(f"{ds} ablation: {len(frames[0])} brute-force rows + {len(df) - len(frames[0])} RW rows "
          f"from {len(files)} jobs of {rw_run}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_dir / f"{ds}_ablation_raw_results.xlsx", index=False)
    A.generate_visualizations(df, out_dir)


def combine_scalability(rw_run, out_dir):
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        for kind in ("rows", "attributes"):
            name = f"{ds}_scalability_{kind}.csv"
            old = pd.read_csv(REPO / "results" / name)
            if rw_run:
                new = pd.read_csv(Path(rw_run) / "results" / name)
                old = pd.concat([old[old.algorithm != "RW"], new[new.algorithm == "RW"]], ignore_index=True)
            old.to_csv(out_dir / name, index=False)
            print(f"{name}: " + ", ".join(f"{a} {n}" for a, n in old.algorithm.value_counts().items()), flush=True)


def run_notebook(out):
    fig_dir = out / "article_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    nb_path = REPO / "article_figures" / "generate_figures.ipynb"
    shutil.copy2(nb_path, fig_dir)  # the notebook finds its folder by this file
    cells = ["".join(c["source"]) for c in json.load(open(nb_path))["cells"] if c["cell_type"] == "code"]
    code = cells[0] + "\n" + PNG_PATCH + "\n\n".join(cells[1:])
    (fig_dir / "generate_figures_run.py").write_text(code)
    subprocess.run([sys.executable, "generate_figures_run.py"], cwd=fig_dir, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rw-run", default=None)
    ap.add_argument("--ablation-one", default=None, help=argparse.SUPPRESS)  # internal: one dataset per process
    args = ap.parse_args()
    out = Path(args.out).expanduser().resolve()
    if args.ablation_one:
        combine_ablation(args.ablation_one, args.rw_run, out / "ablation_results")
        return
    if out.exists():
        raise SystemExit(f"{out} exists already; choose a new --out")
    (out / "ablation_results").mkdir(parents=True)
    for ds in DATASETS:
        if args.rw_run:  # ablation_study reads the dataset when it is imported, so one process per dataset
            subprocess.run([sys.executable, __file__, "--out", str(out), "--rw-run", args.rw_run,
                            "--ablation-one", ds], env=dict(os.environ, HOMOGENEITY_DATASET=ds), check=True)
        else:
            shutil.copy2(REPO / "ablation_results" / f"{ds}_ablation_summary.csv", out / "ablation_results")
    combine_scalability(args.rw_run, out / "results")
    (out / "problem_2_3_algorithms").symlink_to(REPO / "problem_2_3_algorithms")
    run_notebook(out)
    print("figures ->", out / "article_figures")


if __name__ == "__main__":
    main()
