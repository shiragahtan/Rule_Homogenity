"""
Candidate prescription rules and replacement of rules whose effect turned negative.

Port of the original rule search (old repo, algorithms/generate_treatments.py) to the current
estimator. A rule is (condition = one value of an IMMUTABLE attribute, treatment = one value of a
MUTABLE attribute). As in the original search, a rule is a candidate when its condition covers
MORE than the dataset's COVERAGE_THRESHOLD percent of the data (config.json) and its effect is
positive; candidates are ranked by effect, highest first. Differences from the original: the
effect is computed with the current DoWhy estimator (the original used the previous estimator;
it is also computed here, as `utility_old`, to validate the port), and the rule's population needs
at least --min-arm rows in each treatment arm (the original function had a floor of 10 rows).

    python experiments/generate_candidate_rules.py --dataset german --workers 6
    python experiments/generate_candidate_rules.py --dataset german --replace-negative   # also proposes replacements

--replace-negative recomputes the effect of every rule in the dataset's rules file, and for each
rule whose effect is negative proposes the highest-effect candidate that is not already in the
file, covers more than the threshold, has a positive effect and a treated share of at least
--min-treated-share (default 0.1, so that the group-size filter leaves subgroups to test). The
proposal is written to <out>/<dataset>_proposed_rules.json (same format as the rules file, rule
positions kept); nothing is overwritten unless --write-rules is given.
"""
import argparse
import json
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
sys.path.append(str(REPO / "experiments"))
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402
from calibrate_epsilon_delta import load_old_estimator  # noqa: E402

_G = {}


def _py(v):
    """numpy scalar -> Python value; an integer-valued float (a column that pandas read as float
    because of missing values elsewhere) becomes an int, as in the rules files."""
    v = v.item() if hasattr(v, "item") else v
    return int(v) if isinstance(v, float) and v.is_integer() else v


def _init(dataset, full_df, treat, outcome, confounders, encode, min_arm, with_old):
    from ATE_update import estimate_cate, EstimationError
    _G.update(full_df=full_df, treat=treat, outcome=outcome, confounders=confounders, encode=encode,
              min_arm=min_arm, estimate_cate=estimate_cate, EstimationError=EstimationError,
              old=load_old_estimator(dataset) if with_old else None)


def effect_of(rule):
    """(rows, treated rows, new effect, old effect) of a rule's population; effects NaN when not computable."""
    treat, outcome = _G["treat"], _G["outcome"]
    df = prepare_rule_df(_G["full_df"], rule, treat, encode=_G["encode"])
    if df is None:
        return 0, 0, np.nan, np.nan
    n, treated = len(df), int(df[treat].sum())
    new = old = np.nan
    if min(treated, n - treated) >= _G["min_arm"]:
        try:
            new = _G["estimate_cate"](df, treat, outcome, _G["confounders"])
        except _G["EstimationError"]:
            pass
        if _G["old"] is not None:
            old = _G["old"].calculate_ate_safe(df, treat, outcome)  # no delta, as in the original search
    return n, treated, new, old


def _job(args):
    (c_attr, c_val), (t_attr, t_val), coverage = args
    n, treated, new, old = effect_of({"condition": {c_attr: c_val}, "treatment": {t_attr: t_val}})
    return dict(condition_attr=c_attr, condition_val=c_val, treatment_attr=t_attr, treatment_val=t_val,
                coverage_pct=coverage, rows=n, treated=treated, treated_share=treated / n if n else np.nan,
                utility_new=new, utility_old=old)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["so", "acs", "german"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-arm", type=int, default=10, help="rows needed in each treatment arm of a rule's population")
    ap.add_argument("--no-old", action="store_true", help="skip the previous estimator (faster, no validation)")
    ap.add_argument("--replace-negative", action="store_true")
    ap.add_argument("--min-treated-share", type=float, default=0.1)
    ap.add_argument("--write-rules", action="store_true", help="overwrite the dataset's rules file with the proposal")
    ap.add_argument("--out", default=str(REPO / "calibration_results" / "candidates"))
    args = ap.parse_args()

    config = json.load(open(REPO / "configs" / "config.json"))
    ds = config["DATASETS"][args.dataset]
    treat, outcome = config["TREATMENT_COL"], ds["TARGET_COLUMN"]
    threshold = ds.get("COVERAGE_THRESHOLD", 10)
    from ATE_update import sanitize_column_name
    confounders = [sanitize_column_name(c) for c in ds["CONFOUNDERS"]]
    full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=outcome)
    encode = args.dataset != "acs"
    total = len(full_df)

    conds = []
    for attr in ds["IMMUTABLE_ATTRIBUTES"]:
        if attr in full_df.columns:
            for val, cnt in full_df[attr].value_counts().items():
                if cnt / total * 100 > threshold:
                    conds.append(((attr, _py(val)), cnt / total * 100))
    treats = [(attr, _py(val)) for attr in ds["MUTABLE_ATTRIBUTES"] if attr in full_df.columns
              for val in full_df[attr].dropna().unique()]
    jobs = [(c, t, cov) for (c, cov), t in product(conds, treats)]
    print(f"{args.dataset}: {len(conds)} conditions with coverage > {threshold}%, {len(treats)} treatments -> {len(jobs)} candidate rules", flush=True)

    t0 = time.time()
    initargs = (args.dataset, full_df, treat, outcome, confounders, encode, args.min_arm, not args.no_old)
    with Pool(args.workers, initializer=_init, initargs=initargs) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    cand = pd.DataFrame(rows)

    rules_path = REPO / ds["RULES_FILE"]
    current = [json.loads(line) for line in open(rules_path)]
    key = lambda c, t: (json.dumps(c, sort_keys=True), json.dumps(t, sort_keys=True))
    in_file = {key(r["condition"], r["treatment"]): i + 1 for i, r in enumerate(current)}
    cand["current_rule"] = [in_file.get(key({a: _py(v)}, {b: _py(w)}), 0) for a, v, b, w in
                            zip(cand.condition_attr, cand.condition_val, cand.treatment_attr, cand.treatment_val)]
    cand = cand.sort_values("utility_new", ascending=False)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.dataset}_candidate_rules.csv"
    cand.to_csv(path, index=False)
    pos = cand[cand.utility_new > 0]
    print(f"{len(pos)} of {len(cand)} candidates have a positive effect ({time.time() - t0:.0f}s) -> {path}", flush=True)
    for r, i in sorted(in_file.items(), key=lambda x: x[1]):
        row = cand[cand.current_rule == i]
        if len(row):
            row = row.iloc[0]
            rank_new = int((cand.utility_new > row.utility_new).sum()) + 1
            print(f"  current rule {i}: effect new {row.utility_new:.4g} (rank {rank_new}/{len(cand)}), old {row.utility_old:.4g}, treated share {row.treated_share:.1%}")
        else:
            print(f"  current rule {i}: not among the candidates (its coverage is not above {threshold}%?)")

    if args.replace_negative:
        proposal, used = [], set()
        eligible = pos[(pos.current_rule == 0) & (pos.treated_share >= args.min_treated_share)]
        for i, r in enumerate(current, 1):
            row = cand[cand.current_rule == i]
            eff = _py(row.utility_new.iloc[0]) if len(row) else np.nan
            if not np.isnan(eff) and eff > 0:
                proposal.append(r)
                continue
            pick = eligible[~eligible.index.isin(used)].iloc[0]
            used.add(pick.name)
            print(f"  REPLACE rule {i} (effect {eff:.4g}) -> {pick.condition_attr}={pick.condition_val} / "
                  f"{pick.treatment_attr}={pick.treatment_val}: effect {pick.utility_new:.4g}, coverage {pick.coverage_pct:.1f}%, treated share {pick.treated_share:.1%}")
            proposal.append({"condition": {pick.condition_attr: _py(pick.condition_val)},
                             "treatment": {pick.treatment_attr: _py(pick.treatment_val)},
                             "utility": float(pick.utility_new), "coverage": float(pick.coverage_pct)})
        out_rules = Path(args.out) / f"{args.dataset}_proposed_rules.json"
        out_rules.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in proposal) + "\n", encoding="utf-8")
        print(f"proposal -> {out_rules}")
        if args.write_rules:
            rules_path.write_text(out_rules.read_text())
            print(f"rules file overwritten: {rules_path}")


if __name__ == "__main__":
    main()
