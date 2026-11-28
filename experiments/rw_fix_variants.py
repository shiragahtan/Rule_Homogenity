"""
Test fixes for RW missing non-homogeneous rules (found on Stack Overflow): each variant is run on every rule of a
dataset with several seeds and its verdict is compared with the exact verdict
(calibration_results/exact_summary_<dataset>.csv).

Variants (see the options at the top of algorithms/RW.py; all use the weighted8 extension):
  baseline     current RW: 2,000 walks, at most 5 added predicates, no revisit, no size prefilter
  depth10      walks may add up to 10 predicates
  walks10k     10,000 walks
  revisit      walks may pass through subgroups that earlier walks already checked
  prefilter    only predicates that keep both arms >= delta/10 rows are drawn
  pre+rev      prefilter + revisit
  pre+rev+d10  prefilter + revisit + depth10
  learned      Sean's proposal: weights learned during the check (start 1/n, grow with the effect change)
  learned+fix  learned + prefilter + revisit + depth10
  learned+pre  learned + prefilter
  frozen       instructor's version (Nov 24): weights learned once per dataset on other rules, normalized, the
               same in every walk (configs/rw_learned_weights_<ds>.json, experiments/learn_rw_weights.py)
               + prefilter + revisit + depth10
  frozen+adapt the frozen weights as the START, still adapted inside each check (not learned from scratch)
  adapt+pre    frozen+adapt weights with only the prefilter (no revisit, at most 5 added predicates)
  adapt+pre+d10  frozen+adapt weights with prefilter and depth10, no revisit

    python experiments/rw_fix_variants.py --dataset so --seeds 3 --workers 12
    python experiments/rw_fix_variants.py --dataset german --variants baseline,pre+rev+d10

Output: calibration_results/rw_fix_<dataset>.csv (one row per variant, rule and seed) and a summary per variant.
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

VARIANTS = {
    "baseline":    dict(),
    "depth10":     dict(max_depth=10),
    "walks10k":    dict(k_walks=10000),
    "revisit":     dict(revisit=True),
    "prefilter":   dict(arm_prefilter=True),
    "pre+rev":     dict(arm_prefilter=True, revisit=True),
    "pre+rev+d10": dict(arm_prefilter=True, revisit=True, max_depth=10),
    # Sean's proposal, tested on its own and with the fixes above
    "learned":     dict(learned_weights=True),
    "learned+fix": dict(learned_weights=True, arm_prefilter=True, revisit=True, max_depth=10),
    "learned+pre": dict(learned_weights=True, arm_prefilter=True),
    "frozen":      dict(frozen_weights=True, arm_prefilter=True, revisit=True, max_depth=10),
    "frozen+adapt": dict(frozen_weights=True, learned_weights=True, warm_start=True, arm_prefilter=True,
                         revisit=True, max_depth=10),
    # what revisit and the longer walks cost and bring (the frozen+adapt weights, with fewer walk options)
    "adapt+pre":     dict(frozen_weights=True, learned_weights=True, warm_start=True, arm_prefilter=True),
    "adapt+pre+d10": dict(frozen_weights=True, learned_weights=True, warm_start=True, arm_prefilter=True,
                          max_depth=10),
}


def run_one(job):
    dataset, rule_idx, seed, variant, weights_file = job
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
    opts = dict(VARIANTS[variant])
    k = opts.pop("k_walks", 2000)
    weights = ds.get("ATTRIBUTE_WEIGHTS", {})
    if opts.pop("frozen_weights", False):
        weights = json.load(open(weights_file))["weights"]
    t0 = time.time()
    homogeneous, checked = rw(mode=0, algorithm=apriori, df=df, treatment_col=treat, tgtO=outcome, delta=delta,
                              epsilon=eps, utility_all=overall, k_walks=k, size_stop=0.8,
                              rng=random.Random(seed), attribute_weights=weights,
                              extension="weighted8", **opts)
    return dict(dataset=dataset, variant=variant, rule=rule_idx, seed=seed,
                rw_homogeneous=bool(homogeneous), subgroups_checked=checked, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rules", default="all", help="comma separated rule numbers or all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default=None,
                    help="frozen weights file for the frozen variants (default configs/rw_learned_weights_<ds>.json)")
    args = ap.parse_args()
    os.environ["HOMOGENEITY_DATASET"] = args.dataset
    weights_file = str(Path(args.weights).resolve() if args.weights
                       else REPO / "configs" / f"rw_learned_weights_{args.dataset}.json")
    import contextlib, io
    config = json.load(open(REPO / "configs" / "config.json"))
    eps = config["DATASETS"][args.dataset]["EPSILONS"][0]
    exact = pd.read_csv(REPO / "calibration_results" / f"exact_summary_{args.dataset}.csv").set_index("rule")
    rules = list(exact.index) if args.rules == "all" else [int(x) for x in args.rules.split(",")]
    variants = args.variants.split(",")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    jobs = [(args.dataset, r, s, v, weights_file) for v in variants for s in range(args.seeds) for r in rules]
    out = Path(args.out or REPO / "calibration_results" / f"rw_fix_{args.dataset}.csv")
    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_one, jobs):
            row["exact_min_epsilon"] = exact.loc[row["rule"], "min_eps_new_final"]
            row["exact_homogeneous"] = bool(row["exact_min_epsilon"] <= eps)
            row["correct"] = row["rw_homogeneous"] == row["exact_homogeneous"]
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)  # keep partial results
            print(f"{row['variant']:12s} rule {row['rule']:2d} seed {row['seed']}: RW {'H' if row['rw_homogeneous'] else 'N'} "
                  f"exact {'H' if row['exact_homogeneous'] else 'N'} {'ok' if row['correct'] else 'WRONG'} "
                  f"({row['subgroups_checked']} subgroups, {row['seconds']:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    print("\nper variant:")
    for v in variants:
        g = res[res.variant == v]
        nonh = g[~g.exact_homogeneous]
        print(f"  {v:12s} accuracy {g.correct.mean():.0%}  found non-homogeneous {(~nonh.rw_homogeneous).mean():.0%}  "
              f"mean time {g.seconds.mean():.0f}s  mean subgroups {g.subgroups_checked.mean():.0f}")
    print("->", out)


if __name__ == "__main__":
    main()
