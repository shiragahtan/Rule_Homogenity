"""
Homogeneity of the candidate rules (experiments/generate_candidate_rules.py) at the dataset's
(delta, epsilon), to select the rule list of the article by an explicit, reproducible procedure:

  * candidates: single-condition / single-treatment rules whose condition covers more than the
    dataset's COVERAGE_THRESHOLD percent, with a positive effect (current CATE) and at least
    --min-treated-share of the rule's population treated (instructor: 10%);
  * every candidate is checked EXACTLY: all subgroups with >= delta rows are estimated with the
    current estimator (each treatment side needs >= delta/10 rows) and the rule is homogeneous
    when no subgroup deviates from the rule's overall effect by more than epsilon;
  * selection: the --each highest-effect homogeneous and the --each highest-effect
    non-homogeneous candidates (instructor: out of the 10 rules, 5 homogeneous and 5 not).

    python experiments/candidate_homogeneity.py --dataset german --workers 8
    python experiments/candidate_homogeneity.py --dataset so --workers 32 --each 5     # heavy: ~70 min per homogeneous candidate

A candidate is processed in descending order of effect; a rule found non-homogeneous is settled as
soon as one violating subgroup is seen (early exit), a homogeneous one needs every subgroup. The
run stops once --each rules of both kinds are found (use --all to evaluate every candidate).
Results are appended to <out>/<dataset>_candidate_homogeneity.csv, so an interrupted run resumes.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
sys.path.append(str(REPO / "experiments"))
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402
from exact_min_epsilon import mine  # noqa: E402

_G = {}


def _init(df, treat, outcome, confounders):
    from ATE_update import estimate_cate, EstimationError
    _G.update(df=df, treat=treat, outcome=outcome, confounders=confounders,
              estimate_cate=estimate_cate, EstimationError=EstimationError)


def _estimate(chunk):
    """(subgroup effect, treated rows, control rows) for every predicate list of the chunk."""
    df, treat, outcome = _G["df"], _G["treat"], _G["outcome"]
    out = []
    for preds in chunk:
        mask = np.ones(len(df), dtype=bool)
        for c, v in preds:
            mask &= (df[c] == v).to_numpy()
        sub = df[mask]
        treated = int(sub[treat].sum())
        try:
            eff = _G["estimate_cate"](sub, treat, outcome, _G["confounders"])
        except _G["EstimationError"]:
            eff = np.nan
        out.append((eff, treated, len(sub) - treated))
    return out


def coerce(full_df, col, val):
    """Value read from the candidate CSV (a string there) -> the type of the dataset column."""
    if pd.api.types.is_numeric_dtype(full_df[col]):
        v = float(val)
        return int(v) if v.is_integer() else v
    return str(val)


def check_rule(full_df, rule, treat, outcome, confounders, encode, delta, eps, workers, early_exit=True):
    """Exact homogeneity of one rule: (homogeneous, max deviation seen, subgroups checked, seconds)."""
    from ATE_update import estimate_cate
    t0 = time.time()
    df = prepare_rule_df(full_df, rule, treat, encode=encode)
    overall = estimate_cate(df, treat, outcome, confounders)
    groups = mine(df, delta, exclude=[treat, outcome])
    chunks = [groups[i:i + 200] for i in range(0, len(groups), 200)]
    max_dev, checked, violated = 0.0, 0, False
    with Pool(workers, initializer=_init, initargs=(df, treat, outcome, confounders)) as pool:
        for part in pool.imap_unordered(_estimate, chunks):
            for eff, treated, control in part:
                if np.isnan(eff) or min(treated, control) < delta / 10:
                    continue
                checked += 1
                max_dev = max(max_dev, abs(eff - overall))
            if max_dev > eps:
                violated = True
                if early_exit:
                    pool.terminate()
                    break
    return (not violated), max_dev, checked, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--each", type=int, default=5, help="homogeneous and non-homogeneous rules wanted")
    ap.add_argument("--exact", action="store_true", help="no early exit: the exact smallest epsilon of every candidate (slower; results go to *_candidate_min_epsilon.csv)")
    ap.add_argument("--all", action="store_true", help="evaluate every candidate instead of stopping at --each of both kinds")
    ap.add_argument("--min-treated-share", type=float, default=0.10)
    ap.add_argument("--min-coverage", type=float, default=None, help="only candidates whose condition covers at least this percent of the data")
    ap.add_argument("--candidates", default=None, help="candidate CSV (default: calibration_results/candidates/<dataset>_candidate_rules.csv)")
    ap.add_argument("--out", default=str(REPO / "calibration_results" / "candidates"))
    args = ap.parse_args()

    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][args.dataset]
    treat, outcome = config["TREATMENT_COL"], ds["TARGET_COLUMN"]
    delta, eps = ds["DELTAS"][0], ds["EPSILONS"][0]
    from ATE_update import sanitize_column_name
    confounders = [sanitize_column_name(c) for c in ds["CONFOUNDERS"]]
    full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=outcome)
    cand = pd.read_csv(args.candidates or REPO / "calibration_results" / "candidates" / f"{args.dataset}_candidate_rules.csv")
    cand = cand[(cand.utility_new > 0) & (cand.treated_share >= args.min_treated_share)]
    if args.min_coverage is not None:
        cand = cand[cand.coverage_pct >= args.min_coverage]
    cand = cand.sort_values("utility_new", ascending=False)
    out_path = Path(args.out) / f"{args.dataset}_candidate_{'min_epsilon' if args.exact else 'homogeneity'}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    key = lambda a, v, b, w: f"{a}={v} | {b}={w}"
    seen = set(done["rule"]) if len(done) else set()
    print(f"{args.dataset}: delta={delta} epsilon={eps:g}; {len(cand)} candidates (positive effect, treated share >= {args.min_treated_share:.0%}); {len(seen)} already done", flush=True)

    rows = done.to_dict("records") if len(done) else []
    for r in cand.itertuples():
        name = key(r.condition_attr, r.condition_val, r.treatment_attr, r.treatment_val)
        if name in seen:
            continue
        n_h = sum(1 for x in rows if x["homogeneous"])
        n_n = len(rows) - n_h
        if not args.all and n_h >= args.each and n_n >= args.each:
            break
        rule = {"condition": {r.condition_attr: coerce(full_df, r.condition_attr, r.condition_val)},
                "treatment": {r.treatment_attr: coerce(full_df, r.treatment_attr, r.treatment_val)}}
        homo, dev, checked, secs = check_rule(full_df, rule,
                                              treat, outcome, confounders, args.dataset != "acs", delta, eps, args.workers,
                                              early_exit=not args.exact)
        row = dict(rule=name, condition_attr=r.condition_attr, condition_val=rule["condition"][r.condition_attr],
                   treatment_attr=r.treatment_attr, treatment_val=rule["treatment"][r.treatment_attr],
                   coverage_pct=r.coverage_pct, treated_share=r.treated_share, effect=r.utility_new,
                   homogeneous=homo, max_deviation_seen=dev, subgroups_checked=checked, seconds=round(secs))
        if args.exact:
            row["min_epsilon"] = dev
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  {name[:90]:90s} effect {r.utility_new:.4g} -> {'HOMOGENEOUS' if homo else 'not homogeneous'} "
              f"(max deviation seen {dev:.4g} vs epsilon {eps:g}, {checked:,} subgroups, {secs:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    if len(res):
        print(f"\n{(res.homogeneous).sum()} homogeneous and {(~res.homogeneous).sum()} non-homogeneous candidates evaluated -> {out_path}")


if __name__ == "__main__":
    main()
