"""
Evidence for how epsilon and delta must move when the CATE estimator changes.

For every rule of a dataset, the SAME random subgroups (conjunctions of 1-3 attribute=value
predicates, size >= the configured delta) are estimated with the previous estimator (taken from git
history) and with the current DoWhy estimator. For each subgroup we store its effect and its
deviation |effect(subgroup) - effect(rule population)| under both estimators. Homogeneity is
exactly "every subgroup of size >= delta deviates by at most epsilon", so these deviations show
how the epsilon needed for the same verdicts, and the noise at a given delta, change.

    python experiments/calibrate_epsilon_delta.py --dataset german --n 1500
    python experiments/calibrate_epsilon_delta.py --dataset so --n 800
    python experiments/calibrate_epsilon_delta.py --dataset acs --n 200

Writes <out>/<dataset>_subgroup_deviations.csv (one row per sampled subgroup). Sampling only
finds a LOWER bound on the largest deviation (it does not enumerate every subgroup), but the
sample is paired, so old-vs-new comparisons are like for like.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402

OLD_COMMIT = "d2af6f5"  # last commit before the DoWhy estimator


def load_old_estimator(dataset):
    """The pre-DoWhy utils/ATE_update.py from git history, configured for `dataset`."""
    root = Path(tempfile.mkdtemp(prefix="old_estimator_"))
    archive = subprocess.run(["git", "-C", str(REPO), "archive", OLD_COMMIT, "utils", "configs"],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    cfg_path = root / "configs" / "config.json"
    cfg = json.load(open(cfg_path))
    cfg["CHOSEN_DATASET"] = dataset
    json.dump(cfg, open(cfg_path, "w"))
    old_cwd = os.getcwd()
    os.chdir(root / "utils")  # the old module reads ../configs/config.json relative to the cwd
    sys.path.insert(0, str(root / "utils"))
    try:
        spec = importlib.util.spec_from_file_location("old_ate_update", root / "utils" / "ATE_update.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(root / "utils"))
    return module


def sample_subgroups(df, attrs, min_size, n, rng):
    """n distinct random conjunctions of 1-3 predicates with at least min_size rows."""
    values = df[attrs].to_numpy()
    seen, out = set(), []
    for _ in range(n * 60):
        k = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
        cols = rng.choice(len(attrs), size=k, replace=False)
        row = values[rng.integers(len(df))]
        key = tuple(sorted((int(c), row[c]) for c in cols))
        if key in seen:
            continue
        seen.add(key)
        mask = np.ones(len(df), dtype=bool)
        for c, v in key:
            mask &= values[:, c] == v
        if mask.sum() >= min_size:
            out.append(({attrs[c]: v for c, v in key}, mask))
            if len(out) == n:
                break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    parser.add_argument("--n", type=int, default=500, help="subgroups sampled per rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(REPO / "calibration_results"))
    args = parser.parse_args()

    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][args.dataset]
    outcome, delta0 = ds["TARGET_COLUMN"], ds["DELTAS"][0]
    treat = config["TREATMENT_COL"]

    old = load_old_estimator(args.dataset)
    from ATE_update import estimate_cate, EstimationError, sanitize_column_name  # current estimator
    confounders = [sanitize_column_name(c) for c in ds["CONFOUNDERS"]]

    def new_cate(df):
        try:
            return estimate_cate(df, treat, outcome, confounders)
        except EstimationError:
            return np.nan

    full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=outcome)
    rules = [json.loads(line) for line in open(REPO / ds["RULES_FILE"])]
    rng = np.random.default_rng(args.seed)
    rows = []
    for ridx, rule in enumerate(rules, 1):
        t0 = time.time()
        df = prepare_rule_df(full_df, rule, treat, encode=(args.dataset != "acs"))
        all_old = old.calculate_ate_safe(df, treat, outcome, delta0)
        all_new = new_cate(df)
        attrs = [c for c in df.columns if c not in (treat, outcome)]
        subgroups = sample_subgroups(df, attrs, delta0, args.n, rng)
        for preds, mask in subgroups:
            sub = df[mask]
            rows.append(dict(dataset=args.dataset, rule=ridx, size=int(mask.sum()), predicates=str(preds),
                             all_old=all_old, all_new=all_new,
                             cate_old=old.calculate_ate_safe(sub, treat, outcome, delta0),
                             cate_new=new_cate(sub)))
        print(f"{args.dataset} rule {ridx}: {len(subgroups)} subgroups, all_old={all_old:.2f} "
              f"all_new={all_new:.2f} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out["dev_old"] = (out["cate_old"] - out["all_old"]).abs()
    out["dev_new"] = (out["cate_new"] - out["all_new"]).abs()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.dataset}_subgroup_deviations.csv"
    out.to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
