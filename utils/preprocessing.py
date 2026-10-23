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
          f"({n_unknown:,} removed for 'UNKNOWN', {n_null:,} removed for a null in any column, incl. an unparsable outcome)")
    return df


def prepare_rule_df(full_df, rule, treatment_col='TempTreatment', encode=True):
    """
    The population of one rule, prepared exactly as algorithms/main.py does it: rows matching the
    rule's condition, condition column dropped, binary `treatment_col` (1 where the treatment
    attribute equals the prescribed value), treatment attribute dropped, object columns
    integer-coded (skipped for ACS, encode=False), column names sanitized.
    Returns None when the rule matches nothing or nobody is treated.
    """
    import re
    (c_attr, c_val), = rule["condition"].items()
    (t_attr, t_val), = rule["treatment"].items()
    if c_attr not in full_df.columns or t_attr not in full_df.columns:
        return None
    sub_df = full_df[full_df[c_attr] == c_val].drop(columns=[c_attr]).copy()
    if sub_df.empty:
        return None
    sub_df[treatment_col] = (sub_df[t_attr] == t_val).astype(int)
    sub_df = sub_df.drop(columns=[t_attr])
    if sub_df[treatment_col].sum() == 0:
        return None
    if encode:
        for column in sub_df.select_dtypes(include=['object']).columns:
            codes = {val: idx + 1 for idx, val in enumerate(sub_df[column].unique())}
            sub_df[column] = sub_df[column].map(codes)
        for column in sub_df.select_dtypes(include=['bool']).columns:
            sub_df[column] = sub_df[column].astype(int)
    return sub_df.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
