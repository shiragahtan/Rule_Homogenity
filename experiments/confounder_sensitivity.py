"""
How much does the choice of confounders move each rule's overall effect (new DoWhy estimator)?

For every rule of every dataset: the effect with (a) no adjustment (plain difference in means),
(b) the configured CONFOUNDERS (currently the immutable attributes), (c) every other column.

    python experiments/confounder_sensitivity.py [--out calibration_results]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "utils"))
from ATE_update import estimate_cate, EstimationError, sanitize_column_name  # noqa: E402
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "calibration_results"))
    args = parser.parse_args()
    config = json.load(open(REPO / "configs" / "config.json"))
    treat = config["TREATMENT_COL"]
    rows = []
    for name, ds in config["DATASETS"].items():
        y = ds["TARGET_COLUMN"]
        configured = [sanitize_column_name(c) for c in ds["CONFOUNDERS"]]
        full_df = clean_dataset(pd.read_csv(REPO / ds["FULL_DATASET_PATH"]), outcome_col=y)
        for i, rule in enumerate(json.loads(l) for l in open(REPO / ds["RULES_FILE"])):
            df = prepare_rule_df(full_df, rule, treat, encode=(name != "acs"))
            everything = [c for c in df.columns if c not in (treat, y)]
            row = dict(dataset=name, rule=i + 1, condition=rule["condition"], treatment=rule["treatment"], n=len(df))
            for label, conf in (("no_confounders", []), ("configured_confounders", configured), ("all_other_columns", everything)):
                try:
                    row[label] = estimate_cate(df, treat, y, conf)
                except EstimationError:
                    row[label] = np.nan
            rows.append(row)
            print(name, i + 1, {k: round(v, 2) for k, v in row.items() if k in ("no_confounders", "configured_confounders", "all_other_columns")}, flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.out) / "confounder_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
