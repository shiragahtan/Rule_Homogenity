"""
Compare two homogeneity result files of the same dataset (e.g. the results from the previous CATE
estimator vs the ones from the DoWhy estimator): how the share of homogeneous rules and the
per-rule verdicts change, per algorithm and epsilon.

    python utils/compare_homogeneity_results.py OLD.xlsx NEW.xlsx [--out comparison.csv]

Rows are matched on (algorithm, condition, treatment, delta, epsilon); when a run was repeated
the latest row is used. Timed-out runs are excluded. For precision/specificity of RW etc. against
the exhaustive baseline run graphs/homogeneity_calculate_metrics.py on each file separately.
"""
import argparse

import pandas as pd

KEY = ["algorithm", "condition", "treatment", "delta", "epsilon"]


def load(path):
    df = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path)
    df["algorithm"] = df["algorithm"].astype(str).str.strip()
    df["homogeneity_status"] = df["homogeneity_status"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    if "timed_out" in df.columns:
        df = df[~df["timed_out"].astype(str).str.lower().eq("yes")]
    return df.sort_values("date").drop_duplicates(KEY, keep="last") if "date" in df.columns else df.drop_duplicates(KEY, keep="last")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--out", default=None, help="write the summary table to this CSV")
    args = parser.parse_args()

    old, new = load(args.old), load(args.new)
    merged = old.merge(new, on=KEY, how="outer", suffixes=("_old", "_new"), indicator=True)
    both = merged[merged["_merge"] == "both"]

    rows = []
    for (algo, eps), g in merged.groupby(["algorithm", "epsilon"]):
        b = g[g["_merge"] == "both"]
        rows.append({
            "algorithm": algo, "epsilon": eps,
            "runs_old": int(g["homogeneity_status_old"].notna().sum()),
            "runs_new": int(g["homogeneity_status_new"].notna().sum()),
            "homogeneous_rate_old_%": round(100 * g["homogeneity_status_old"].dropna().astype(bool).mean(), 1),
            "homogeneous_rate_new_%": round(100 * g["homogeneity_status_new"].dropna().astype(bool).mean(), 1),
            "same_verdict_%": round(100 * (b["homogeneity_status_old"].astype(bool) == b["homogeneity_status_new"].astype(bool)).mean(), 1) if len(b) else float("nan"),
            "matched_runs": len(b),
        })
    summary = pd.DataFrame(rows).sort_values(["algorithm", "epsilon"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    only_old, only_new = (merged["_merge"] == "left_only").sum(), (merged["_merge"] == "right_only").sum()
    print(f"\nmatched runs: {len(both)} | only in old: {only_old} | only in new: {only_new}")
    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"summary -> {args.out}")


if __name__ == "__main__":
    main()
