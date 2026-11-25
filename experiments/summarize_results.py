"""
Summarize the finished experiment results into one JSON (for the instructor pages and the paper tables).

    python experiments/summarize_results.py --out results/summary.json

Reads results/<ds>_homogeneity_results.xlsx (main experiment), results/<ds>_algorithms_time.csv (AllSubgroups
runtimes), ablation_results/<ds>_ablation_summary.csv, calibration_results/exact_summary_<ds>.csv (exact verdicts),
calibration_results/rw_accuracy_<ds>_single.csv (RW single-draw). Missing files are skipped.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CONFIG = json.load(open(REPO / "configs" / "config.json"))
ALGO_ORDER = ["FPGrowth", "Greedy", "CausalForest", "WTE", "RW", "Random"]


def rules_of(ds):
    with open(REPO / CONFIG["DATASETS"][ds]["RULES_FILE"], encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def exact_verdicts(ds):
    p = REPO / "calibration_results" / f"exact_summary_{ds}.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p).set_index("rule")
    eps = CONFIG["DATASETS"][ds]["EPSILONS"][0]
    return {int(r): bool(e.loc[r, "min_eps_new_final"] <= eps) for r in e.index}


def rule_number(ds, cond, treat):
    for i, r in enumerate(rules_of(ds), start=1):
        if str(r["condition"]) == str(cond) and str(r["treatment"]) == str(treat):
            return i
    return None


def main_results(ds, exact):
    p = REPO / "results" / f"{ds}_homogeneity_results.xlsx"
    if not p.exists():
        return None
    x = pd.read_excel(p)
    x["rule"] = [rule_number(ds, c, t) for c, t in zip(x["condition"], x["treatment"])]
    x["verdict"] = x["homogeneity_status"].map({"Homogeneous": "H", "NOT Homogeneous": "N", "TIMEOUT": "T"})
    out = {"algorithms": [], "per_rule": {}}
    for a in [a for a in ALGO_ORDER if a in set(x["algorithm"])]:
        g = x[x["algorithm"] == a]
        decided = g[g["verdict"].isin(["H", "N"])]
        correct = sum((v == "H") == exact[r] for v, r in zip(decided["verdict"], decided["rule"])) if exact else None
        mask = [bool(exact) and not exact[r] for r in decided["rule"]]
        nonh = decided.loc[pd.Series(mask, index=decided.index, dtype=bool)]
        out["algorithms"].append(dict(
            algorithm=a, runs=int(len(g)), timeouts=int((g["verdict"] == "T").sum()),
            correct=int(correct) if correct is not None else None, decided=int(len(decided)),
            nonh_runs=int(len(nonh)), nonh_found=int((nonh["verdict"] == "N").sum()),
            mean_s=float(g["run_time_seconds"].mean()), median_s=float(g["run_time_seconds"].median()),
            mean_subgroups=float(pd.to_numeric(g["num_subgroups"], errors="coerce").mean())
            if g["num_subgroups"].notna().any() else None))
        out["per_rule"][a] = {int(r): "".join(gg["verdict"].astype(str)) for r, gg in g.groupby("rule")}
    out["exact"] = {int(k): ("H" if v else "N") for k, v in (exact or {}).items()}
    return out


def alltimes(ds):
    p = REPO / "results" / f"{ds}_algorithms_time.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    n_rules = len(rules_of(ds))
    rows = []
    for a in [a for a in ALGO_ORDER if a in set(t["algorithm"])]:
        g = t[t["algorithm"] == a]
        runs_per_rule = 3 if a in ("RW", "Random") else 1
        rows.append(dict(algorithm=a, runs=int(len(g)), expected=n_rules * runs_per_rule,
                         mean_s=float(g["run_time_seconds"].mean()), median_s=float(g["run_time_seconds"].median()),
                         max_s=float(g["run_time_seconds"].max()),
                         mean_subgroups=float(pd.to_numeric(g["num_subgroups"], errors="coerce").mean())))
    return rows


def ablation(ds):
    """From the merged raw results, or, while the parallel run is not finished, from the finished jobs."""
    raw = REPO / "ablation_results" / f"{ds}_ablation_raw_results.xlsx"
    jobs = sorted((REPO / "par_runs" / "jobs").glob(f"{ds}-ablation-*"))
    done = [d for d in jobs if (d / "DONE").exists()]
    if jobs and (len(done) < len(jobs) or not raw.exists()):
        frames = [pd.read_csv(d / "ablation_rows.csv") for d in done if (d / "ablation_rows.csv").exists()]
        if not frames:
            return None
        r = pd.concat(frames, ignore_index=True)
        source = f"{len(done)} of {len(jobs)} jobs finished"
    elif raw.exists():
        r = pd.read_excel(raw)
        source = "complete"
    else:
        return None
    out = {"source": source}
    for exp, col in (("varying_epsilon", "param_value"), ("varying_delta", "param_value")):
        g = r[r["experiment"] == exp]
        rows = []
        for v, gv in g.groupby(col):
            fp = gv[gv["algorithm"] == "FPGrowth"]
            rw = gv[gv["algorithm"] == "RW_Direct"]
            # agreement: RW run verdict equals the FPGrowth verdict of the same rule
            fp_by_rule = dict(zip(fp["rule_idx"], fp["is_homogeneous"]))
            agree = [bool(h) == bool(fp_by_rule[ri]) for ri, h in zip(rw["rule_idx"], rw["is_homogeneous"])
                     if ri in fp_by_rule]
            rows.append(dict(value=float(v), rules=int(fp["rule_idx"].nunique()),
                             fp_homogeneous=int(fp["is_homogeneous"].sum()),
                             rw_homogeneous_runs=int(rw["is_homogeneous"].sum()), rw_runs=int(len(rw)),
                             fp_mean_s=float(fp["runtime_seconds"].mean()) if len(fp) else None,
                             rw_mean_s=float(rw["runtime_seconds"].mean()) if len(rw) else None,
                             fp_mean_subgroups=float(fp["num_subgroups_checked"].mean()) if len(fp) else None,
                             rw_mean_subgroups=float(rw["num_subgroups_checked"].mean()) if len(rw) else None,
                             agreement=float(sum(agree) / len(agree)) if agree else None))
        out[exp] = rows
    return out


def rw_single(ds):
    p = REPO / "calibration_results" / f"rw_accuracy_{ds}_single.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p)
    nonh = r[~r["exact_homogeneous"]]
    return dict(runs=int(len(r)), correct=float(r["correct"].mean()),
                nonh_found=int((~nonh["rw_homogeneous"]).sum()), nonh_runs=int(len(nonh)),
                mean_s=float(r["seconds"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "results" / "summary.json"))
    args = ap.parse_args()
    summary = {}
    for ds in ("german", "acs", "so"):
        exact = exact_verdicts(ds)
        summary[ds] = dict(epsilon=CONFIG["DATASETS"][ds]["EPSILONS"][0], delta=CONFIG["DATASETS"][ds]["DELTAS"][0],
                           main=main_results(ds, exact), alltimes=alltimes(ds), ablation=ablation(ds),
                           rw_single=rw_single(ds))
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
