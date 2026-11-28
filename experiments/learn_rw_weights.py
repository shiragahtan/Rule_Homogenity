"""
Learn the RW attribute weights of a dataset ONCE, then freeze them (instructor's decision, Nov 24 2026):
one weight per attribute, normalized to sum to 1, the same weights in every random walk of every check on that
dataset (they are not learned again inside a check).

How:
1. Training rules: candidate rules of the dataset (calibration_results/candidates*/<ds>_candidate_rules_cov50.csv)
   that are NOT among the 10 chosen rules, drawn at random (--train-rules, --seed). The 10 rules we test on are
   never used for learning.
2. On each training rule, --walks learning walks (Sean's mechanism: weights start at 1/n and grow with the effect
   change, size prefilter, revisit, up to 10 added predicates). Walks do not stop at a violation. Every time an
   attribute is added to a walk, |change of the subgroup's effect| / epsilon is recorded.
3. Learning stops after the fixed budget (training rules x walks). Weight of an attribute =
   1/n + mean recorded change, normalized to sum to 1 (algorithms/RW.py weights_from_stats).

    python experiments/learn_rw_weights.py --dataset german --train-rules 10 --walks 500 --workers 8

Output: configs/rw_learned_weights_<ds>.json (weights + how they were learned). With --seed s != 0 the file gets
the suffix _seed<s> (used to check that the weights are stable).
"""
import argparse
import json
import os
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
sys.path.append(str(REPO / "algorithms"))

CANDIDATES = {
    "german": "calibration_results/candidates_newdata/german_candidate_rules_cov50.csv",
    "acs": "calibration_results/candidates/acs_candidate_rules_cov50.csv",
    "so": "calibration_results/candidates/so_candidate_rules_cov50.csv",
}


def _key(rule):
    (ca, cv), = rule["condition"].items()
    (ta, tv), = rule["treatment"].items()
    return (ca, str(cv), ta, str(tv))


def _value(v):
    """Candidate files store values as text; the data may hold numbers."""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def learn_one(job):
    dataset, rule, walks, seed = job
    from preprocessing import clean_dataset, prepare_rule_df
    from ATE_update import calculate_ate_safe
    from RW import calc_utility_for_subgroups as rw
    from mlxtend.frequent_patterns import apriori
    import contextlib, io
    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][dataset]
    outcome, treat = ds["TARGET_COLUMN"], config["TREATMENT_COL"]
    delta, eps = ds["DELTAS"][0], ds["EPSILONS"][0]
    full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=outcome)
    df = prepare_rule_df(full_df, rule, treat, encode=(dataset != "acs"))
    if df is None or len(df) < delta:
        return rule, None, 0, 0.0
    overall = calculate_ate_safe(df, treat, outcome, delta)
    stats = {}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        _, checked = rw(mode=0, algorithm=apriori, df=df, treatment_col=treat, tgtO=outcome, delta=delta,
                        epsilon=eps, utility_all=overall, k_walks=walks, size_stop=0.8, rng=random.Random(seed),
                        extension="weighted8", learned_weights=True, arm_prefilter=True, revisit=True,
                        max_depth=10, learn_stats=stats)
    return rule, stats, checked, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(CANDIDATES))
    ap.add_argument("--train-rules", type=int, default=10)
    ap.add_argument("--walks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--use-test-rules", action="store_true",
                    help="learn on the 10 test rules themselves instead of other candidates (biased; only to measure "
                         "how much that would change)")
    ap.add_argument("--out", default=None, help="output file (default configs/rw_learned_weights_<ds>[_seed<s>].json)")
    args = ap.parse_args()
    os.environ["HOMOGENEITY_DATASET"] = args.dataset
    from RW import weights_from_stats
    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][args.dataset]
    test_rules = [json.loads(l) for l in open(REPO / ds["RULES_FILE"], encoding="utf-8") if l.strip()]
    if args.use_test_rules:
        pool_rules = list(test_rules)
        print(f"{args.dataset}: learning on the {len(pool_rules)} test rules themselves", flush=True)
    else:
        chosen = {_key(r) for r in test_rules}
        cand = pd.read_csv(REPO / CANDIDATES[args.dataset])
        pool_rules = []
        for r in cand.itertuples():
            rule = {"condition": {r.condition_attr: _value(r.condition_val)},
                    "treatment": {r.treatment_attr: _value(r.treatment_val)}}
            if _key(rule) not in chosen:
                pool_rules.append(rule)
        print(f"{args.dataset}: {len(cand)} candidates, {len(pool_rules)} not among the 10 chosen rules", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(pool_rules)

    # draw until --train-rules usable rules (a candidate may be unusable for this data, e.g. too few rows)
    jobs = [(args.dataset, rule, args.walks, args.seed * 1000 + i) for i, rule in enumerate(pool_rules)]
    stats, used = {}, []
    with Pool(args.workers) as pool:
        for rule, st, checked, sec in pool.imap(learn_one, jobs):
            if len(used) >= args.train_rules:
                continue  # finish the jobs already started but ignore them, so the result is deterministic
            if st is None or checked == 0:  # no data for the rule, or no subgroup passes the size filter
                print(f"  skip (not usable): {rule}", flush=True)
                continue
            used.append(rule)
            for a, (s, c) in st.items():
                tot = stats.setdefault(a, [0.0, 0])
                tot[0] += s
                tot[1] += c
            print(f"  [{len(used)}/{args.train_rules}] {rule}: {checked} subgroups, {sec:.0f}s", flush=True)
            if len(used) >= args.train_rules:
                pool.terminate()
                break

    outcome = ds["TARGET_COLUMN"]
    full_cols = pd.read_csv(REPO / ds["FULL_DATASET_PATH"], nrows=1).columns
    attributes = sorted({re.sub(r'[,:\[\]\{\}"]', '_', c) for c in full_cols if c != outcome} | set(stats))
    weights = weights_from_stats(stats, attributes)
    out = Path(args.out) if args.out else \
        REPO / "configs" / f"rw_learned_weights_{args.dataset}{'' if args.seed == 0 else f'_seed{args.seed}'}.json"
    json.dump(dict(dataset=args.dataset, weights=weights, sum=sum(weights.values()),
                   how="1/n + mean |effect change|/epsilon per added attribute, normalized to sum 1",
                   train_rules=used, train_on_test_rules=args.use_test_rules, walks_per_rule=args.walks, seed=args.seed,
                   delta=ds["DELTAS"][0], epsilon=ds["EPSILONS"][0],
                   times_added={a: stats.get(a, [0, 0])[1] for a in weights}),
              open(out, "w"), indent=1, ensure_ascii=False)
    print("\nweights (sum %.6f):" % sum(weights.values()))
    for a, w in weights.items():
        print(f"  {w:.4f}  {a}  (added {stats.get(a, [0, 0])[1]} times)")
    print("->", out)


if __name__ == "__main__":
    main()
