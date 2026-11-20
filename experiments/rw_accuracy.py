"""
Accuracy and speed of RW against the exact answer.

For every rule of the dataset, RW is run at the configured (delta, epsilon) with several seeds and
each walk-extension strategy (algorithms/RW.py, RW_EXTENSION), and its verdict is compared with the
exact verdict (calibration_results/exact_summary_<dataset>.csv, every subgroup checked):

  * RW can only be wrong one way: it reports "homogeneous" when it missed the violating subgroups.
  * accuracy = share of (rule, seed) runs whose verdict equals the exact one;
    detection rate = share of runs on non-homogeneous rules where RW found a violation.

    python experiments/rw_accuracy.py --dataset german --seeds 5 --workers 8
    python experiments/rw_accuracy.py --dataset so --seeds 3 --extensions top8,weighted8 --workers 32

Output: calibration_results/rw_accuracy_<dataset>.csv (one row per rule, seed and strategy).
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
sys.path.append(str(REPO / "algorithms"))


def run_one(job):
    dataset, rule_idx, seed, extension, k_walks = job
    from preprocessing import clean_dataset, prepare_rule_df
    from ATE_update import calculate_ate_safe
    from RW import calc_utility_for_subgroups as rw
    from mlxtend.frequent_patterns import apriori
    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][dataset]
    outcome, treat = ds["TARGET_COLUMN"], config["TREATMENT_COL"]
    delta, eps = ds["DELTAS"][0], ds["EPSILONS"][0]
    rule = [json.loads(line) for line in open(REPO / ds["RULES_FILE"], encoding="utf-8") if line.strip()][rule_idx - 1]
    full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=outcome)
    df = prepare_rule_df(full_df, rule, treat, encode=(dataset != "acs"))
    overall = calculate_ate_safe(df, treat, outcome, delta)
    t0 = time.time()
    homogeneous, checked = rw(mode=0, algorithm=apriori, df=df, treatment_col=treat, tgtO=outcome, delta=delta,
                              epsilon=eps, utility_all=overall, k_walks=k_walks, size_stop=0.8,
                              rng=random.Random(seed), attribute_weights=ds.get("ATTRIBUTE_WEIGHTS", {}),
                              extension=extension)
    return dict(dataset=dataset, rule=rule_idx, seed=seed, extension=extension, k_walks=k_walks,
                rw_homogeneous=bool(homogeneous), subgroups_checked=checked, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--extensions", default="top8,weighted8,weighted_single")
    ap.add_argument("--k-walks", type=int, default=2000)
    ap.add_argument("--rules", default="all", help="comma separated rule numbers or all")
    ap.add_argument("--exact-summary", default=None, help="default: calibration_results/exact_summary_<dataset>.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ["HOMOGENEITY_DATASET"] = args.dataset  # this dataset's confounders in every worker

    config = json.load(open(REPO / "configs" / "config.json"))
    eps = config["DATASETS"][args.dataset]["EPSILONS"][0]
    exact = pd.read_csv(args.exact_summary or REPO / "calibration_results" / f"exact_summary_{args.dataset}.csv")
    exact = exact.set_index("rule")
    rules = list(exact.index) if args.rules == "all" else [int(x) for x in args.rules.split(",")]
    jobs = [(args.dataset, r, s, e, args.k_walks) for e in args.extensions.split(",") for r in rules for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        rows = []
        for row in pool.imap_unordered(run_one, jobs):
            row["exact_min_epsilon"] = exact.loc[row["rule"], "min_eps_new_final"]
            row["exact_homogeneous"] = bool(row["exact_min_epsilon"] <= eps)
            row["correct"] = row["rw_homogeneous"] == row["exact_homogeneous"]
            rows.append(row)
            print(f"{row['extension']:15s} rule {row['rule']:2d} seed {row['seed']}: RW {'H' if row['rw_homogeneous'] else 'N'} "
                  f"exact {'H' if row['exact_homogeneous'] else 'N'} {'ok' if row['correct'] else 'WRONG'} "
                  f"({row['subgroups_checked']} subgroups, {row['seconds']:.0f}s)", flush=True)
    res = pd.DataFrame(rows).sort_values(["extension", "rule", "seed"])
    out = args.out or REPO / "calibration_results" / f"rw_accuracy_{args.dataset}.csv"
    res.to_csv(out, index=False)
    print("\nper strategy:")
    for e, g in res.groupby("extension"):
        nonh = g[~g.exact_homogeneous]
        print(f"  {e:15s} accuracy {g.correct.mean():.0%}  detection on non-homogeneous {(~nonh.rw_homogeneous).mean():.0%}  "
              f"mean time {g.seconds.mean():.1f}s  mean subgroups {g.subgroups_checked.mean():.0f}")
    print("->", out)


if __name__ == "__main__":
    main()
