"""
Summaries of the exact per-subgroup effects written by experiments/exact_min_epsilon.py.

For every rule it reports, under the previous estimator and under the current one with its
reliability filter, the exact smallest epsilon at the configured delta and the resulting verdict.
The filter of the current estimator (utils/ATE_update.py) is applied here to the unfiltered
per-subgroup results: both treatment arms need at least MIN_ARM_FRACTION of the subgroup's rows.
It also reports how many subgroups the filter removes (a rule whose subgroups are all removed is
"vacuously" homogeneous and is flagged), and how the smallest epsilon moves when delta is raised.

    python experiments/analyze_exact.py --exact-dir <dir with exact/*.csv> --dataset german --out calibration_results/exact_summary_german.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DELTA_GRID = {"german": [50, 75, 100, 150, 200, 300], "so": [100, 200, 500, 1000, 2000, 5000],
              "acs": [100, 200, 500, 1000, 2000, 5000, 10000]}


def apply_filters(f, delta, fraction):
    """Subgroups of f (unfiltered exact results) that the current estimator would keep at `delta`."""
    keep = (f["size"] >= delta) & f["cate_new"].notna()
    keep &= f[["treated", "control"]].min(axis=1) >= fraction * f["size"]
    return f[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact-dir", required=True)
    ap.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    ap.add_argument("--delta", type=int, default=None, help="default: configured delta")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][args.dataset]
    delta0 = args.delta or ds["DELTAS"][0]
    eps = ds["EPSILONS"][0]
    fraction = config.get("MIN_ARM_FRACTION", 0.1)
    rows = []
    for r in range(1, 11):
        f = pd.read_csv(Path(args.exact_dir) / f"{args.dataset}_rule{r}_delta{delta0}.csv")
        all_old, all_new, p_all = f.all_old[0], f.all_new[0], f.p_all[0]
        f["dev_old"] = (f.cate_old - all_old).abs()
        f["dev_new"] = (f.cate_new - all_new).abs()
        at_delta = f[f["size"] >= delta0]
        raw = at_delta[at_delta.cate_new.notna()]
        final = apply_filters(f, delta0, fraction)
        top = final.loc[final.dev_new.idxmax()] if len(final) else None
        row = dict(dataset=args.dataset, rule=r, delta=delta0, epsilon=eps, subgroups=len(f),
                   effect_old=all_old, effect_new=all_new, p_value_overall=p_all,
                   min_eps_old=f.dev_old.max(), min_eps_new_unfiltered=raw.dev_new.max(),
                   min_eps_new_final=final.dev_new.max() if len(final) else np.nan,
                   not_estimable=int(f.cate_new.isna().sum()), removed_by_arm=len(raw) - len(final), kept=len(final),
                   vacuous=len(final) == 0,
                   violations_before_filter=int((raw.dev_new > eps).sum()), violations_after_filter=int((final.dev_new > eps).sum()))
        if top is not None:
            row.update(argmax_subgroup=top.subgroup, argmax_size=top["size"], argmax_treated=top.treated,
                       argmax_control=top.control, argmax_effect=top.cate_new)
        for g in DELTA_GRID[args.dataset]:
            sub = apply_filters(f, g, fraction)
            row[f"min_eps_old_delta{g}"] = f[f["size"] >= g].dev_old.max()
            row[f"min_eps_new_final_delta{g}"] = sub.dev_new.max() if len(sub) else np.nan
        rows.append(row)
        print(args.dataset, r, "done", flush=True)
    out = args.out or str(REPO / "calibration_results" / f"exact_summary_{args.dataset}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
