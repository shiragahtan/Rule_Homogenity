"""
Exact homogeneity check: enumerate EVERY subgroup (conjunction of attribute=value predicates,
one per attribute) with at least delta rows, estimate its effect with the previous and the
current CATE estimator, and report for each rule

    exact smallest epsilon = max over subgroups of |effect(subgroup) - effect(rule population)|

A rule is homogeneous for (delta, epsilon) exactly when that maximum is <= epsilon, so unlike the
randomized search (experiments/rw_min_epsilon.py, a lower bound) this gives the definitive answer.
Every subgroup's effect is saved, so any larger delta' (filter on `size`) or any epsilon can be
answered afterwards without recomputing (see summarize_exact below).

    python experiments/exact_min_epsilon.py --dataset german --rules 7
    python experiments/exact_min_epsilon.py --dataset so --workers 8
    python experiments/exact_min_epsilon.py --dataset acs --count-only

Subgroups are the same ones algorithms/bruteForce.py mines with FPGrowth (an itemset may not use
the same attribute twice), found here with a depth-first search on integer codes (see mine()).
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
sys.path.append(str(REPO / "experiments"))
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402
from calibrate_epsilon_delta import load_old_estimator  # noqa: E402

_G = {}


def mine(df, delta, exclude):
    """
    Every subgroup with >= delta rows as a list of (column, value) predicates, one per attribute at
    most: a depth-first search that extends a subgroup with one more attribute and keeps a value only
    while its support stays >= delta. It finds the same subgroups as FPGrowth over the one-hot table
    (what algorithms/bruteForce.py mines) but works on integer codes, which keeps the large datasets
    (hundreds of thousands of rows) tractable.
    """
    cols = [c for c in df.columns if c not in exclude]
    codes, uniques = [], []
    for c in cols:
        code, uniq = pd.factorize(df[c], sort=False)
        codes.append(code)
        uniques.append(uniq)
    out = []

    def extend(idx, start, preds):
        for j in range(start, len(cols)):
            counts = np.bincount(codes[j][idx], minlength=len(uniques[j]))
            for v in np.nonzero(counts >= delta)[0]:
                child = preds + [(cols[j], uniques[j][v])]
                out.append(child)
                extend(idx[codes[j][idx] == v], j + 1, child)

    extend(np.arange(len(df)), 0, [])
    return out


def _init(dataset, treat, outcome, delta, df, confounders):
    from ATE_update import estimate_cate, EstimationError
    _G.update(df=df, treat=treat, outcome=outcome, delta=delta, confounders=confounders,
              estimate_cate=estimate_cate, EstimationError=EstimationError,
              old=load_old_estimator(dataset))


def _estimate(masks_items):
    df, treat, outcome = _G["df"], _G["treat"], _G["outcome"]
    res = []
    for preds in masks_items:
        mask = np.ones(len(df), dtype=bool)
        for c, v in preds:
            mask &= (df[c] == v).to_numpy()
        sub = df[mask]
        try:
            new = _G["estimate_cate"](sub, treat, outcome, _G["confounders"])
        except _G["EstimationError"]:
            new = np.nan
        old = _G["old"].calculate_ate_safe(sub, treat, outcome, _G["delta"])
        res.append((" & ".join(f"{c}={v}" for c, v in preds), int(len(sub)), old, new))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    ap.add_argument("--rules", default="all", help="comma separated rule numbers (1-based) or all")
    ap.add_argument("--delta", type=int, default=None, help="default: configured delta")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--count-only", action="store_true", help="only count the subgroups to be estimated")
    ap.add_argument("--out", default=str(REPO / "calibration_results"))
    args = ap.parse_args()

    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][args.dataset]
    outcome, treat = ds["TARGET_COLUMN"], config["TREATMENT_COL"]
    delta = args.delta or ds["DELTAS"][0]
    eps = ds["EPSILONS"][0]
    rules = [json.loads(line) for line in open(REPO / ds["RULES_FILE"])]
    wanted = range(1, len(rules) + 1) if args.rules == "all" else [int(x) for x in args.rules.split(",")]
    full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=outcome)
    from ATE_update import sanitize_column_name
    confounders = [sanitize_column_name(c) for c in ds["CONFOUNDERS"]]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for r in wanted:
        df = prepare_rule_df(full_df, rules[r - 1], treat, encode=(args.dataset != "acs"))
        t0 = time.time()
        groups = mine(df, delta, exclude=[treat, outcome])
        print(f"{args.dataset} rule {r}: {len(df):,} rows, {len(groups):,} subgroups with >= {delta} rows "
              f"(mined in {time.time() - t0:.0f}s)", flush=True)
        if args.count_only:
            continue
        chunks = [groups[i:i + 200] for i in range(0, len(groups), 200)]
        t0 = time.time()
        with Pool(args.workers, initializer=_init, initargs=(args.dataset, treat, outcome, delta, df, confounders)) as pool:
            rows = [x for part in pool.imap_unordered(_estimate, chunks) for x in part]
        _init(args.dataset, treat, outcome, delta, df, confounders)
        all_old = _G["old"].calculate_ate_safe(df, treat, outcome, delta)
        all_new = _G["estimate_cate"](df, treat, outcome, confounders)
        out = pd.DataFrame(rows, columns=["subgroup", "size", "cate_old", "cate_new"])
        out["dev_old"] = (out.cate_old - all_old).abs()
        out["dev_new"] = (out.cate_new - all_new).abs()
        out.insert(0, "rule", r)
        out["all_old"], out["all_new"] = all_old, all_new
        path = Path(args.out) / "exact" / f"{args.dataset}_rule{r}_delta{delta}.csv"
        path.parent.mkdir(exist_ok=True)
        out.to_csv(path, index=False)
        print(f"  rule {r}: exact min epsilon old={out.dev_old.max():,.4g} new={out.dev_new.max():,.4g} "
              f"(article epsilon {eps:g}; {out.cate_new.isna().sum()} subgroups not estimable by the new code; "
              f"{time.time() - t0:.0f}s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
