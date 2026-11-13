"""
Select the 10 rules of a dataset by the procedure agreed with the instructor:

  * candidates: single-condition / single-treatment rules whose condition covers at least
    --min-coverage percent of the data, with a positive effect (current CATE) and at least 10% of
    the rule's population treated (experiments/generate_candidate_rules.py);
  * each candidate is homogeneous or not at the dataset's (delta, epsilon): its exact smallest
    epsilon (experiments/candidate_homogeneity.py --exact) is <= epsilon;
  * the 5 highest-effect homogeneous and the 5 highest-effect non-homogeneous candidates are the
    10 rules (homogeneous ones first).

    python experiments/select_rules.py --dataset german --min-coverage 70
    python experiments/select_rules.py --dataset acs --min-coverage 50 --epsilon 60000

Without --epsilon the configured epsilon is used; if fewer than --each candidates are homogeneous
at it, the smallest epsilon (rounded up to 2 significant digits) that makes --each homogeneous
candidates possible is reported, and the selection is made at that epsilon as well.
Writes calibration_results/selected_rules/<dataset>_cov<X>_eps<E>.json (rules file format) and .csv.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def round_up_2sig(x):
    if x <= 0:
        return x
    mag = 10 ** (math.floor(math.log10(x)) - 1)
    return math.ceil(x / mag) * mag


def select(pool, eps, each):
    homo = pool[pool.min_epsilon <= eps].sort_values("effect", ascending=False)
    non = pool[pool.min_epsilon > eps].sort_values("effect", ascending=False)
    return homo.head(each), non.head(each), len(homo), len(non)


def write(dataset, pool_sel, cov, eps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_cov{cov:g}_eps{eps:g}"
    sel = pool_sel.copy()
    sel["rule_number"] = range(1, len(sel) + 1)
    sel["homogeneous_at_epsilon"] = sel.min_epsilon <= eps
    sel.to_csv(out_dir / f"{tag}.csv", index=False)
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        for r in sel.itertuples():
            f.write(json.dumps({"condition": {r.condition_attr: _val(r.condition_val)}, "treatment": {r.treatment_attr: _val(r.treatment_val)},
                                "utility": float(r.effect), "coverage": float(r.coverage_pct)}, ensure_ascii=False) + "\n")
    return out_dir / f"{tag}.json"


def _val(v):
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v
    return int(v) if float(v).is_integer() else float(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    ap.add_argument("--min-coverage", type=float, required=True)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--each", type=int, default=5)
    ap.add_argument("--candidates", default=None, help="min-epsilon CSV (default: calibration_results/candidates_cov50/<dataset>_candidate_min_epsilon.csv)")
    ap.add_argument("--out", default=str(REPO / "calibration_results" / "selected_rules"))
    args = ap.parse_args()

    config = json.load(open(REPO / "configs" / "config.json", encoding="utf-8"))
    ds = config["DATASETS"][args.dataset]
    eps0 = ds["EPSILONS"][0]
    path = Path(args.candidates) if args.candidates else REPO / "calibration_results" / "candidates_cov50" / f"{args.dataset}_candidate_min_epsilon.csv"
    cand = pd.read_csv(path)
    pool = cand[cand.coverage_pct >= args.min_coverage].copy()
    eps = args.epsilon if args.epsilon is not None else eps0
    print(f"{args.dataset}: {len(pool)} eligible candidates (coverage >= {args.min_coverage:g}%, positive effect, >= 10% treated); delta={ds['DELTAS'][0]}")
    sel_h, sel_n, nh, nn = select(pool, eps, args.each)
    print(f"at epsilon {eps:g}: {nh} homogeneous, {nn} non-homogeneous candidates")
    results = []
    if nh >= args.each and nn >= args.each:
        results.append(eps)
    else:
        print(f"  -> {args.each} homogeneous + {args.each} non-homogeneous is NOT possible at epsilon {eps:g}")
        if len(pool) >= 2 * args.each and args.epsilon is None:
            need = pool.sort_values("min_epsilon").min_epsilon.iloc[args.each - 1]
            feasible = round_up_2sig(need)
            print(f"  -> {args.each}th smallest exact epsilon = {need:,.4g}; smallest feasible epsilon (2 significant digits) = {feasible:g}")
            results.append(feasible)
        # also write the best effort at the original epsilon
        best = pd.concat([sel_h, sel_n])
        print(f"  best effort at epsilon {eps:g}: {len(sel_h)} homogeneous + {len(sel_n)} non-homogeneous (rules: {len(best)})")
        if len(sel_h) + len(sel_n) > 0:
            print("  wrote", write(args.dataset, best.head(2 * args.each), args.min_coverage, eps, Path(args.out)))
    for e in results:
        h, n, nh2, nn2 = select(pool, e, args.each)
        if len(h) == args.each and len(n) == args.each:
            print("  wrote", write(args.dataset, pd.concat([h, n]), args.min_coverage, e, Path(args.out)))
            for r in pd.concat([h, n]).itertuples():
                print(f"    {'H' if r.min_epsilon <= e else 'N'}  {r.rule[:78]:78s} effect {r.effect:>10,.4g}  eps*={r.min_epsilon:>10,.4g}  cov {r.coverage_pct:.1f}%")
        else:
            print(f"  epsilon {e:g}: only {len(h)} homogeneous / {len(n)} non-homogeneous available")


if __name__ == "__main__":
    main()
