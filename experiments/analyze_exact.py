"""
Summaries of the exact per-subgroup effects written by experiments/exact_min_epsilon.py.

For every rule it reports, under the previous and the current estimator, the exact smallest epsilon
at the article's delta, the verdict at the article's epsilon, and how the verdict and the smallest
epsilon change when
  * delta is raised (a subgroup counts only if size >= delta'), and
  * a minimum number of treated AND control rows is required for an effect to count (MIN_ARM_SIZE).
The treated / control row counts of every subgroup are recomputed here (cheap, no regression).

    python experiments/analyze_exact.py --exact-dir <dir with exact/*.csv> --out calibration_results/exact_summary.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402

ARMS = [0, 10, 30, 100]
DELTA_GRID = {"german": [50, 75, 100, 150, 200, 300], "so": [1000, 2000, 3000, 5000, 8000, 12000],
              "acs": [1000, 2000, 5000, 10000, 20000, 50000]}


def arm_sizes(df, subgroups, treat, sep="="):
    """(treated, control) rows of each subgroup string 'col=value & col=value'."""
    cols = {c: df[c].astype(str).to_numpy() for c in df.columns}
    # values may themselves contain " & ", so split only in front of "<column>="
    splitter = re.compile(" & (?=(?:" + "|".join(re.escape(c) for c in df.columns) + ")" + re.escape(sep) + ")")
    t = df[treat].to_numpy() == 1
    out = []
    for s in subgroups:
        mask = np.ones(len(df), dtype=bool)
        for pred in splitter.split(s):
            c, v = pred.split(sep, 1)
            mask &= cols[c] == v
        out.append((int(t[mask].sum()), int((~t[mask]).sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact-dir", required=True)
    ap.add_argument("--out", default=str(REPO / "calibration_results" / "exact_summary.csv"))
    args = ap.parse_args()
    config = json.load(open(REPO / "configs" / "config.json"))
    treat = config["TREATMENT_COL"]
    rows = []
    for name, ds in config["DATASETS"].items():
        delta0, eps = ds["DELTAS"][0], ds["EPSILONS"][0]
        rules = [json.loads(l) for l in open(REPO / ds["RULES_FILE"])]
        full = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=ds["TARGET_COLUMN"])
        for r in range(1, len(rules) + 1):
            f = pd.read_csv(Path(args.exact_dir) / f"{name}_rule{r}_delta{delta0}.csv")
            df = prepare_rule_df(full, rules[r - 1], treat, encode=(name != "acs"))
            # the German files were written by an earlier version of exact_min_epsilon.py that joined "column|value"
            sep = "=" if "=" in f["subgroup"][0] else "|"
            arms = arm_sizes(df, f["subgroup"], treat, sep)
            f["treated"], f["control"] = zip(*arms)
            f["min_arm"] = f[["treated", "control"]].min(axis=1)
            row = dict(dataset=name, rule=r, delta=delta0, epsilon=eps, subgroups=len(f),
                       effect_old=f.all_old[0], effect_new=f.all_new[0],
                       min_eps_old=f.dev_old.max(), min_eps_new=f.dev_new.max(),
                       not_estimable_new=int(f.cate_new.isna().sum()))
            for k in ARMS[1:]:
                s = f[f.min_arm >= k]
                row[f"min_eps_new_arm{k}"] = s.dev_new.max()
                row[f"subgroups_arm{k}"] = len(s)
            for g in DELTA_GRID[name]:
                s = f[f["size"] >= g]
                row[f"min_eps_old_delta{g}"] = s.dev_old.max()
                row[f"min_eps_new_delta{g}"] = s.dev_new.max()
            top = f.loc[f.dev_new.idxmax()]
            row.update(argmax_new=top.subgroup, argmax_size=top["size"], argmax_treated=top.treated,
                       argmax_control=top.control, argmax_effect=top.cate_new)
            rows.append(row)
            print(name, r, "done", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
