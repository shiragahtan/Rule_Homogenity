import pandas as pd


def clean_dataset(df, outcome_col=None):
    """
    Prepare a raw dataset for the algorithms. Call it ONCE, right after reading the CSV and
    before any rule selection, subgroup search or sampling.

    The algorithms assume the input has no missing values (otherwise the ATE is undefined),
    so every row with a null in any column is removed here and never again downstream.

    1. Drop the "Unnamed: ..." index columns left by to_csv.
    2. Drop rows containing the literal "UNKNOWN" (the datasets' missing-value sentinel).
    3. Convert the outcome column to numeric. Done BEFORE the null removal so a value that
       cannot be parsed is removed here instead of becoming a NaN inside a subgroup later.
    4. Drop every row that contains a null.
    """
    n_start = len(df)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    df = df[~df.isin(["UNKNOWN"]).any(axis=1)]
    n_unknown = n_start - len(df)

    if outcome_col is not None:
        df = df.assign(**{outcome_col: pd.to_numeric(df[outcome_col], errors='coerce')})

    n_before_null = len(df)
    df = df.dropna().reset_index(drop=True)
    n_null = n_before_null - len(df)

    print(f"🧹 clean_dataset: {n_start:,} rows in -> {len(df):,} rows out "
          f"({n_unknown:,} removed for 'UNKNOWN', {n_null:,} removed for null/non-numeric outcome)")
    return df
