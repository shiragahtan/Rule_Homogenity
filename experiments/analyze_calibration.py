"""
Turns the output of calibrate_epsilon_delta.py into the epsilon / delta evidence tables.

    python experiments/analyze_calibration.py --dir calibration_results [--json summary.json]

Per dataset it prints (and optionally saves as JSON):
 1. per rule: overall effect old/new, largest sampled deviation old/new, verdict at the
    configured epsilon under both estimators;
 2. the epsilon that gives the new estimator the same share of homogeneous rules as the old one
    had at the configured epsilon (quantile matching), and the median per-rule scale ratio;
 3. delta: for a grid of minimum subgroup sizes, the share of subgroups whose effect cannot be
    computed, the typical deviation, and the share of homogeneous rules at the configured epsilon.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def rule_max_dev(g, col, min_size):
    s = g[(g["size"] >= min_size)][col].dropna()
    return float(s.max()) if len(s) else np.nan


def homogeneous_share(df, col, eps, min_size, n_rules):
    flags = []
    for _, g in df.groupby("rule"):
        m = rule_max_dev(g, col, min_size)
        if np.isfinite(m):
            flags.append(m <= eps)
    return (100 * np.mean(flags)) if flags else np.nan


def stored_verdicts(name, ds, eps0, delta0):
    """{rule number: True if some sound algorithm found a violation in the stored results}."""
    path = REPO / "results" / f"{name}_homogeneity_results.xlsx"
    if not path.exists():
        return None
    res = pd.read_excel(path)
    res["algorithm"] = res["algorithm"].astype(str).str.strip()
    res = res[res["algorithm"].isin(["RW", "FPGrowth", "Greedy", "Random"]) & (res["delta"] == delta0)
              & np.isclose(res["epsilon"].astype(float), float(eps0))]
    verdicts = {}
    for i, line in enumerate(open(REPO / ds["RULES_FILE"]), 1):
        rule = json.loads(line)
        rows = res[(res["condition"].astype(str) == str(rule["condition"])) & (res["treatment"].astype(str) == str(rule["treatment"]))]
        if len(rows):
            ok = rows["homogeneity_status"].astype(str).str.lower().isin(["true", "1", "1.0"])
            verdicts[i] = bool((~ok).any())
    return verdicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(REPO / "calibration_results"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    config = json.load(open(REPO / "configs" / "config.json"))
    summary = {}
    for name, ds in config["DATASETS"].items():
        path = Path(args.dir) / f"{name}_subgroup_deviations.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        eps0, delta0 = ds["EPSILONS"][0], ds["DELTAS"][0]
        n_rules = df["rule"].nunique()
        print(f"\n===== {name}: epsilon={eps0}, delta={delta0}, {len(df)} sampled subgroups, {n_rules} rules")

        rules = []
        for r, g in df.groupby("rule"):
            m_old, m_new = rule_max_dev(g, "dev_old", delta0), rule_max_dev(g, "dev_new", delta0)
            rules.append(dict(rule=int(r), all_old=g["all_old"].iloc[0], all_new=g["all_new"].iloc[0],
                              max_dev_old=m_old, max_dev_new=m_new,
                              ratio=m_new / m_old if m_old else np.nan,
                              hom_old=bool(m_old <= eps0), hom_new=bool(m_new <= eps0),
                              skipped_old=100 * g["cate_old"].isna().mean(), skipped_new=100 * g["cate_new"].isna().mean()))
        rt = pd.DataFrame(rules)
        print(rt.round(2).to_string(index=False))

        share_old = 100 * rt["hom_old"].mean()
        share_new = 100 * rt["hom_new"].mean()
        # epsilon that gives the new estimator the old share of homogeneous rules
        eps_match = float(np.quantile(rt["max_dev_new"].dropna(), share_old / 100, method="higher")) if share_old > 0 else float("nan")
        both = df.dropna(subset=["dev_old", "dev_new"])
        corr = float(np.corrcoef(both["cate_old"], both["cate_new"])[0, 1]) if len(both) > 2 else float("nan")
        print(f"\nhomogeneous rules at eps={eps0}: old {share_old:.0f}% | new {share_new:.0f}%")
        print(f"epsilon giving the new estimator the old share ({share_old:.0f}%): {eps_match:,.3f}  "
              f"(x{eps_match / eps0:.2f} of the configured epsilon)")
        print(f"median per-rule ratio max_dev_new/max_dev_old: {rt['ratio'].median():.2f}")
        print(f"correlation of subgroup effects old vs new: {corr:.2f} over {len(both)} subgroups")

        # validation: does the sampled OLD verdict agree with the stored exhaustive/RW results?
        stored = stored_verdicts(name, ds, eps0, delta0)
        if stored is not None:
            rt["stored_violation_found"] = [stored.get(int(r)) for r in rt["rule"]]
            known = rt.dropna(subset=["stored_violation_found"])
            agree = int(((~known["hom_old"]) == known["stored_violation_found"].astype(bool)).sum())
            print(f"validation vs stored results (violation found by RW/FPGrowth/Greedy/Random at eps={eps0}, delta={delta0}): "
                  f"sampled old verdict agrees on {agree}/{len(known)} rules")
            print(rt[["rule", "hom_old", "hom_new", "stored_violation_found"]].to_string(index=False))

        sizes = np.sort(df["size"].unique())
        grid = sorted({int(delta0 * m) for m in (1, 2, 3, 5, 10, 20, 50)})
        print("\ndelta grid (minimum subgroup size):")
        print(f"{'delta':>8} {'subgroups':>9} {'skipped_new%':>12} {'skipped_old%':>12} {'median_dev_new':>14} {'median_dev_old':>14} {'hom%_old':>9} {'hom%_new':>9}")
        delta_rows = []
        for d in grid:
            sub = df[df["size"] >= d]
            if len(sub) < 20:
                continue
            row = dict(delta=d, subgroups=len(sub), skipped_new=100 * sub["cate_new"].isna().mean(),
                       skipped_old=100 * sub["cate_old"].isna().mean(),
                       median_dev_new=float(sub["dev_new"].median()), median_dev_old=float(sub["dev_old"].median()),
                       hom_old=homogeneous_share(df, "dev_old", eps0, d, n_rules),
                       hom_new=homogeneous_share(df, "dev_new", eps0, d, n_rules))
            delta_rows.append(row)
            print(f"{d:>8} {row['subgroups']:>9} {row['skipped_new']:>12.1f} {row['skipped_old']:>12.1f} "
                  f"{row['median_dev_new']:>14.3f} {row['median_dev_old']:>14.3f} {row['hom_old']:>9.0f} {row['hom_new']:>9.0f}")
        summary[name] = dict(epsilon=eps0, delta=delta0, rules=rules, share_old=share_old, share_new=share_new,
                             eps_match=eps_match, median_ratio=float(rt["ratio"].median()), corr=corr, delta_grid=delta_rows)
    if args.json:
        json.dump(summary, open(args.json, "w"), indent=1, default=float)
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
