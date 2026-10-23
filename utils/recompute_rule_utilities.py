"""
Recompute the overall effect ("utility") of every rule with the current CATE estimator and compare
it with the value stored in the rules file (computed by the previous estimator).

Run from the repo root:
    python utils/recompute_rule_utilities.py                  # all datasets, report only
    python utils/recompute_rule_utilities.py --dataset so     # one dataset
    python utils/recompute_rule_utilities.py --update-rules   # also rewrite the rules files

The report (rule_utilities_check/<dataset>_rule_utilities.csv) has, per rule: size, treated count,
stored (old) utility, new utility, the raw difference in means (a sanity reference), and the new
utility as a multiple of the dataset's epsilon. With --update-rules each rules file is backed up
to <file>.bak and every rule gets utility = new value and utility_previous = old value;
`coverage` is untouched (it does not depend on the estimator).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
from ATE_update import estimate_cate, EstimationError, sanitize_column_name, TREATMENT_COL  # noqa: E402
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", help="so | acs | german | all (default)")
    parser.add_argument("--out", default=str(REPO / "rule_utilities_check"), help="report directory")
    parser.add_argument("--update-rules", action="store_true", help="rewrite the rules files (keeps a .bak)")
    args = parser.parse_args()

    config = json.load(open(REPO / "configs" / "config.json"))
    names = list(config["DATASETS"]) if args.dataset == "all" else [args.dataset]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        ds = config["DATASETS"][name]
        outcome = ds["TARGET_COLUMN"]
        confounders = [sanitize_column_name(c) for c in ds["CONFOUNDERS"]]
        epsilon = ds["EPSILONS"][0]
        rules_path = REPO / ds["RULES_FILE"]
        rules = [json.loads(line) for line in open(rules_path)]
        print(f"\n##### {name}: {len(rules)} rules, epsilon={epsilon}, confounders={confounders}")
        full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=outcome)

        rows, new_rules = [], []
        for i, rule in enumerate(rules):
            df = prepare_rule_df(full_df, rule, TREATMENT_COL, encode=(name != "acs"))
            old = rule.get("utility", np.nan)
            row = dict(dataset=name, rule=i, condition=rule["condition"], treatment=rule["treatment"],
                       old_utility=old)
            if df is None:
                row.update(status="rule matches nobody / nobody treated")
                new = np.nan
            else:
                treated = df[TREATMENT_COL] == 1
                row.update(n=len(df), treated=int(treated.sum()),
                           raw_diff_in_means=df.loc[treated, outcome].mean() - df.loc[~treated, outcome].mean())
                try:
                    new = estimate_cate(df, TREATMENT_COL, outcome, confounders)
                    row.update(status="ok")
                except EstimationError as e:
                    new = np.nan
                    row.update(status=f"NOT EVALUABLE ({e.reason})")
            row.update(new_utility=new, abs_change=abs(new - old),
                       new_over_epsilon=abs(new) / epsilon if np.isfinite(new) else np.nan)
            rows.append(row)
            print(f"rule {i:2d}  old={old:12.2f}  new={new:12.2f}  raw={row.get('raw_diff_in_means', np.nan):12.2f}  {row['status']}")
            new_rule = dict(rule)
            if np.isfinite(new):
                new_rule["utility_previous"] = old
                new_rule["utility"] = new
            new_rules.append(new_rule)

        report = out_dir / f"{name}_rule_utilities.csv"
        pd.DataFrame(rows).to_csv(report, index=False)
        print(f"report -> {report}")

        if args.update_rules:
            shutil.copy(rules_path, str(rules_path) + ".bak")
            with open(rules_path, "w") as f:
                for r in new_rules:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"rules file rewritten (backup: {rules_path}.bak)")


if __name__ == "__main__":
    main()
