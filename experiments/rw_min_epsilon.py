"""
Smallest epsilon at which the RW algorithm reports a rule as homogeneous, per rule, for a chosen
code tree (so the previous and the current CATE estimator can be compared on the same rules).

    # current estimator
    python experiments/rw_min_epsilon.py --dataset acs --label new --out calibration_results
    # previous estimator: a checkout of the old code (utils/, algorithms/, configs/) whose
    # configs/config.json has CHOSEN_DATASET set to the same dataset
    python experiments/rw_min_epsilon.py --dataset acs --label old --repo /path/to/old_tree

Search: epsilon is doubled from --eps0 until RW reports "homogeneous", then bisected to a relative
precision of --tol. RW is randomized and only visits some subgroups, so the result is an
estimate (a lower bound of the exact smallest epsilon); with a fixed seed it is repeatable.
Datasets and rules are always read from THIS repository so both runs see identical data.
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

HERE = Path(__file__).resolve().parent.parent
sys.path.append(str(HERE / "utils"))
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402  (always from this repo)


def min_epsilon_for_rule(job):
    dataset, label, repo, ridx, delta, eps0, tol, seed, k_walks, min_arm = job
    repo = Path(repo)
    os.chdir(repo / "algorithms")  # the old estimator reads ../configs/config.json
    sys.path[:0] = [str(repo / "utils"), str(repo / "algorithms")]
    from mlxtend.frequent_patterns import apriori
    import ATE_update
    from ATE_update import calculate_ate_safe
    if min_arm is not None and hasattr(ATE_update, 'MIN_ARM_SIZE'):
        ATE_update.MIN_ARM_SIZE = min_arm  # not present in the previous estimator, which had its own fixed floor
    from RW import calc_utility_for_subgroups as rw

    config = json.load(open(HERE / "configs" / "config.json"))
    ds = config["DATASETS"][dataset]
    outcome, treat = ds["TARGET_COLUMN"], config["TREATMENT_COL"]
    rule = [json.loads(line) for line in open(HERE / ds["RULES_FILE"])][ridx - 1]
    full_df = clean_dataset(pd.read_csv(HERE / ds["FULL_DATASET_PATH"]), outcome_col=outcome)
    df = prepare_rule_df(full_df, rule, treat, encode=(dataset != "acs"))
    t0 = time.time()
    utility_all = calculate_ate_safe(df, treat, outcome, delta)
    row = dict(dataset=dataset, label=label, rule=ridx, utility_all=utility_all, delta=delta)
    if utility_all != utility_all:
        return dict(row, min_epsilon=float("nan"), rw_calls=0, seconds=0.0, note="overall effect not computable")

    calls = 0

    def homogeneous(eps):
        nonlocal calls
        calls += 1
        ok, _ = rw(mode=0, algorithm=apriori, df=df, treatment_col=treat, tgtO=outcome, delta=delta, epsilon=eps,
                   utility_all=utility_all, k_walks=k_walks, size_stop=0.8, rng=random.Random(seed),
                   attribute_weights=ds.get("ATTRIBUTE_WEIGHTS", {}))
        return ok

    hi = eps0
    while not homogeneous(hi):
        hi *= 2
        if hi > 1e12:
            return dict(row, min_epsilon=float("inf"), rw_calls=calls, seconds=time.time() - t0, note="no homogeneous epsilon")
    lo = 0.0 if hi == eps0 else hi / 2
    while hi - lo > tol * hi:
        mid = (lo + hi) / 2
        if homogeneous(mid):
            hi = mid
        else:
            lo = mid
    print(f"[{label}] {dataset} rule {ridx}: min epsilon ~ {hi:,.3f} (overall {utility_all:,.2f}, {calls} RW calls, {time.time() - t0:.0f}s)", flush=True)
    return dict(row, min_epsilon=hi, rw_calls=calls, seconds=time.time() - t0, note="")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    parser.add_argument("--label", required=True, help="name of this run, e.g. old / new")
    parser.add_argument("--repo", default=str(HERE), help="code tree providing utils/ and algorithms/")
    parser.add_argument("--rules", default="all", help="comma separated rule numbers (1-based) or all")
    parser.add_argument("--delta", type=int, default=None, help="minimum subgroup size (default: configured delta)")
    parser.add_argument("--min-arm", type=int, default=None, help="minimum rows per treatment arm (current estimator only)")
    parser.add_argument("--eps0", type=float, default=None, help="first epsilon tried (default: configured epsilon / 8)")
    parser.add_argument("--tol", type=float, default=0.05, help="relative precision of the bisection")
    parser.add_argument("--k-walks", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(HERE / "calibration_results"))
    args = parser.parse_args()

    config = json.load(open(HERE / "configs" / "config.json"))
    ds = config["DATASETS"][args.dataset]
    n_rules = sum(1 for _ in open(HERE / ds["RULES_FILE"]))
    rules = range(1, n_rules + 1) if args.rules == "all" else [int(x) for x in args.rules.split(",")]
    eps0 = args.eps0 or ds["EPSILONS"][0] / 8
    delta = args.delta or ds["DELTAS"][0]
    jobs = [(args.dataset, args.label, args.repo, r, delta, eps0, args.tol, args.seed, args.k_walks, args.min_arm) for r in rules]
    with Pool(args.workers) as pool:
        results = pool.map(min_epsilon_for_rule, jobs, chunksize=1)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    suffix = (f"_delta{delta}" if args.delta else "") + (f"_minarm{args.min_arm}" if args.min_arm else "")
    path = Path(args.out) / f"{args.dataset}_rw_min_epsilon_{args.label}{suffix}.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
