"""
Build the clean datasets used by every experiment. This is the only place where data is filtered:
the CATE code (utils/ATE_update.py) and the algorithms read the clean files as they are.

Rules (from the instructor):
  1. Every row with a missing value (empty cell or the sentinel "UNKNOWN") is removed.
  2. Every continuous attribute is cut into 10 equal-width bins: the range min..max is split into 10
     intervals of the same length (pandas.cut). Bins can hold very different numbers of rows, and a
     bin can be empty. Continuous attributes: German age and duration, ACS Age.
     German amount is kept binary as before (0 = below 4,000 DM, 1 = 4,000 DM or more), as the
     instructor asked. Outcome columns are not binned. Categorical attributes are kept as they are.

Outputs (the only dataset CSVs of the repository):
  datasets/german.csv          outcome credit_risk
  datasets/stackoverflow.csv   outcome ConvertedSalary
  datasets/acs.csv             outcome "Wages or salary income past 12 months"

Inputs (not in the repository; datasets/raw/ is git-ignored and filled automatically):
  * German: the original German Credit data (UCI Statlog, german.data, downloaded) gives the real
    numbers of duration (months), amount (DM) and age (years). Its 1000 rows are in the same order as
    the earlier file datasets/german_not_encoded.csv, which gives the other (categorical) columns;
    the script checks the order on 5 shared columns.
  * Stack Overflow and ACS: the earlier files datasets/stackoverflow_not_encoded.csv and
    datasets/acs_encoded.csv, taken from git history (commit RAW_COMMIT).

    python scripts/prepare_datasets.py
"""
import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "datasets"
RAW = OUT / "raw"
RAW_COMMIT = "2c18c55"  # last commit that has the earlier (unfiltered) dataset files
UCI_GERMAN = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
N_BINS = 10
AMOUNT_SPLIT = 4000  # German amount stays binary: the earlier file's split (0 = up to 3,990 DM, 1 = 4,006 DM and above)
MISSING = ["UNKNOWN"]


def raw_file(name):
    """An earlier dataset file, read from git history into datasets/raw/ the first time."""
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.run(["git", "-C", str(REPO), "show", f"{RAW_COMMIT}:datasets/{name}"],
                                        check=True, capture_output=True).stdout)
    return path


def german_numbers():
    """duration, amount, age (and 5 check columns) of the original German Credit data."""
    path = RAW / "german.data"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UCI_GERMAN) as r:
            path.write_bytes(zipfile.ZipFile(io.BytesIO(r.read())).read("german.data"))
    raw = pd.read_csv(path, sep=" ", header=None)
    return pd.DataFrame({"duration": raw[1], "amount": raw[4], "age": raw[12],
                         "installment_rate": raw[7], "present_residence": raw[10], "number_credits": raw[15],
                         "people_liable": raw[17], "credit_risk": (raw[20] == 1).astype(int)})


def drop_missing(df, name):
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    bad = df.isna().any(axis=1) | df.isin(MISSING).any(axis=1)
    print(f"{name}: {len(df)} rows, {bad.sum()} with a missing value removed -> {(~bad).sum()} rows")
    return df[~bad].reset_index(drop=True)


def equal_width_bins(values, as_codes=False):
    """10 bins of equal width over min..max of an integer attribute. Bin k holds the integers in
    (edge_k, edge_k+1] (the first bin also holds the minimum). Labels are the integer range of the bin,
    e.g. '19-24', or the bin numbers 1..10 (as_codes)."""
    bins, edges = pd.cut(values, N_BINS, labels=False, retbins=True)
    lo = [int(values.min())] + [int(np.floor(e)) + 1 for e in edges[1:-1]]
    hi = [int(np.floor(e)) for e in edges[1:-1]] + [int(values.max())]
    labels = {k: (k + 1 if as_codes else f"{lo[k]}-{hi[k]}") for k in range(N_BINS)}
    counts = bins.value_counts().reindex(range(N_BINS), fill_value=0)
    print(f"   {values.name}: " + "  ".join(f"[{lo[k]}-{hi[k]}]:{counts[k]}" + (f"(={k + 1})" if as_codes else "")
                                         for k in range(N_BINS)))
    return bins.map(labels)


def german():
    nums = german_numbers()
    df = pd.read_csv(raw_file("german_not_encoded.csv"))
    for c in ["installment_rate", "present_residence", "number_credits", "people_liable", "credit_risk"]:
        assert (df[c].astype(int).values == nums[c].values).all(), f"German rows not aligned on {c}"
    for c in ["duration", "amount", "age"]:  # real numbers replace the earlier coarse bins
        df[c] = nums[c]
    df = drop_missing(df, "german")
    for c in ["duration", "age"]:
        df[c] = equal_width_bins(df[c])
    df["amount"] = (df["amount"] >= AMOUNT_SPLIT).astype(int)  # binary, same split as the earlier file
    print(f"   amount: binary, 0 = below {AMOUNT_SPLIT:,} DM: {(df.amount == 0).sum()}, 1 = {AMOUNT_SPLIT:,} DM or more: {(df.amount == 1).sum()}")
    return df


def stackoverflow():
    # Every attribute is a survey answer given as a category or a range (e.g. "25 - 34 years old"):
    # there are no raw numbers to re-bin. Only missing values are removed.
    return drop_missing(pd.read_csv(raw_file("stackoverflow_not_encoded.csv")), "stackoverflow")


def acs():
    # All attributes are census codes (categories) except Age, which is binned. ACS stays numeric,
    # so the Age bins are numbered 1..10 in age order (ranges printed below and in datasets/README.md).
    df = drop_missing(pd.read_csv(raw_file("acs_encoded.csv")), "acs")
    df["Age"] = equal_width_bins(df["Age"], as_codes=True)
    return df


def main():
    for name, build in [("german", german), ("stackoverflow", stackoverflow), ("acs", acs)]:
        df = build()
        assert not df.isna().any().any() and not df.isin(MISSING).any().any()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"   -> {OUT / f'{name}.csv'} ({len(df)} rows, {df.shape[1]} columns)\n")


if __name__ == "__main__":
    main()
