"""
Build the clean datasets used by every experiment. This is the only place where data is filtered:
the CATE code (utils/ATE_update.py) and the algorithms read the clean files as they are.

Rules (from the instructor):
  1. Every row with a missing value (empty cell or the sentinel "UNKNOWN") is removed.
  2. Every numeric (non-categorical) attribute is cut into 10 equal-frequency bins (quantiles).
     When many rows share one value (e.g. German duration = 12 months) the quantile edges coincide
     and the column gets fewer than 10 bins; the script prints the bins it made.
     Outcome columns are not binned. Categorical attributes are kept as they are.

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

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "datasets"
RAW = OUT / "raw"
RAW_COMMIT = "2c18c55"  # last commit that has the earlier (unfiltered) dataset files
UCI_GERMAN = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
N_BINS = 10
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


def equal_frequency_bins(values, as_codes=False):
    """10 quantile bins. Labels 'min-max' of the values in the bin, or bin numbers 1..k (as_codes)."""
    bins = pd.qcut(values, N_BINS, labels=False, duplicates="drop")
    ranges = values.groupby(bins).agg(["min", "max", "size"])
    labels = {b: (b + 1 if as_codes else f"{lo:g}-{hi:g}") for b, (lo, hi, _) in ranges.iterrows()}
    print(f"   {values.name}: {len(ranges)} bins  " +
          "  ".join(f"[{lo:g}-{hi:g}]:{n}" + (f"(={b + 1})" if as_codes else "") for b, (lo, hi, n) in ranges.iterrows()))
    return bins.map(labels)


def german():
    nums = german_numbers()
    df = pd.read_csv(raw_file("german_not_encoded.csv"))
    for c in ["installment_rate", "present_residence", "number_credits", "people_liable", "credit_risk"]:
        assert (df[c].astype(int).values == nums[c].values).all(), f"German rows not aligned on {c}"
    for c in ["duration", "amount", "age"]:  # real numbers replace the earlier coarse bins
        df[c] = nums[c]
    df = drop_missing(df, "german")
    for c in ["duration", "amount", "age"]:
        df[c] = equal_frequency_bins(df[c])
    return df


def stackoverflow():
    # Every attribute is a survey answer given as a category or a range (e.g. "25 - 34 years old"):
    # there are no raw numbers to re-bin. Only missing values are removed.
    return drop_missing(pd.read_csv(raw_file("stackoverflow_not_encoded.csv")), "stackoverflow")


def acs():
    # All attributes are census codes (categories) except Age, which is binned. ACS stays numeric,
    # so the Age bins are numbered 1..10 in age order (ranges printed below and in datasets/README.md).
    df = drop_missing(pd.read_csv(raw_file("acs_encoded.csv")), "acs")
    df["Age"] = equal_frequency_bins(df["Age"], as_codes=True)
    return df


def main():
    for name, build in [("german", german), ("stackoverflow", stackoverflow), ("acs", acs)]:
        df = build()
        assert not df.isna().any().any() and not df.isin(MISSING).any().any()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"   -> {OUT / f'{name}.csv'} ({len(df)} rows, {df.shape[1]} columns)\n")


if __name__ == "__main__":
    main()
