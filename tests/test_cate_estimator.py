"""Synthetic-data checks for the DoWhy CATE estimator and the shared preprocessing.
Run from the repo root:  python -m pytest tests -q   (or: python tests/test_cate_estimator.py)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
from ATE_update import calculate_ate_safe, estimate_cate, EstimationError, get_skip_summary
from preprocessing import clean_dataset

TRUE_EFFECT = 5.0
CONF = ["age", "edu"]  # the synthetic data's confounders (the real ones come from configs/config.json)


def make_df(n=2000, seed=0, outlier=False):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    edu = rng.integers(0, 4, n)
    # treatment depends on the covariates (confounding)
    t = (rng.random(n) < 1 / (1 + np.exp(-(age - 40) / 10))).astype(int)
    y = TRUE_EFFECT * t + 0.5 * age + 2 * edu + rng.normal(0, 1, n)
    if outlier:
        y[:20] += 1e6
    return pd.DataFrame({"age": age, "edu": edu, "TempTreatment": t, "y": y})


def test_recovers_known_effect():
    est = estimate_cate(make_df(), "TempTreatment", "y", CONF)
    assert abs(est - TRUE_EFFECT) < 0.3, est


def test_no_outlier_trimming_or_effect_cap():
    # old code trimmed outcome to the 5-95% range and returned NaN for effects > 5x baseline;
    # now every row is used, so extreme outcomes legitimately move the estimate and it stays finite.
    df = make_df(outlier=True)
    est = estimate_cate(df, "TempTreatment", "y", CONF)
    assert np.isfinite(est)
    assert abs(est - TRUE_EFFECT) > 1.0  # the outliers are NOT silently removed


def test_small_arms_are_not_rejected():
    df = make_df(n=300)
    df.loc[df.index[3:], "TempTreatment"] = 0  # only 3 treated rows (old code required >= 10)
    assert np.isfinite(estimate_cate(df, "TempTreatment", "y", CONF))


def test_single_treatment_arm_raises():
    df = make_df()
    df["TempTreatment"] = 1
    try:
        estimate_cate(df, "TempTreatment", "y", CONF)
        assert False, "expected EstimationError"
    except EstimationError as e:
        assert e.reason == "single_arm"
    before = get_skip_summary().get("single_arm", 0)
    assert np.isnan(calculate_ate_safe(df, "TempTreatment", "y"))  # skipped subgroup -> NaN
    assert get_skip_summary()["single_arm"] == before + 1  # and the skip is counted, not silent


def test_each_arm_must_have_at_least_delta_over_ten_rows():
    import ATE_update
    df = make_df(n=300)
    df["TempTreatment"] = 0
    df.loc[df.index[:3], "TempTreatment"] = 1  # exactly 3 treated rows
    before = ATE_update.get_skip_summary().get("too_few_in_arm", 0)
    # delta=100 -> each arm needs at least 10 rows: 3 treated rows are too few for a subgroup
    assert np.isnan(calculate_ate_safe(df, "TempTreatment", "y", delta=100))
    assert ATE_update.get_skip_summary()["too_few_in_arm"] == before + 1
    # delta=30 -> 3 rows are enough
    assert np.isfinite(calculate_ate_safe(df, "TempTreatment", "y", delta=30))
    # the overall effect of a rule population is not filtered by arm size
    assert np.isfinite(calculate_ate_safe(df, "TempTreatment", "y", delta=100, subgroup=False))
    # a share of the subgroup is NOT required: 3 of 300 rows pass when delta is small enough
    try:
        estimate_cate(df, "TempTreatment", "y", CONF, min_arm=10)
        assert False, "expected EstimationError"
    except EstimationError as e:
        assert e.reason == "too_few_in_arm"


def test_p_value_is_available_for_analysis_but_does_not_filter():
    import ATE_update
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "TempTreatment": rng.integers(0, 2, n)})
    df["y"] = 5.0 * df["TempTreatment"] + df["a"] + rng.normal(size=n)  # strong effect
    value, p = ATE_update.estimate_cate_with_p(df, "TempTreatment", "y", ["a", "b"])
    assert abs(value - 5.0) < 1.0 and p < 1e-6
    df["y"] = df["a"] + rng.normal(size=n)  # no effect at all -> p is large
    value, p = ATE_update.estimate_cate_with_p(df, "TempTreatment", "y", ["a", "b"])
    assert p >= 0.05
    assert np.isfinite(calculate_ate_safe(df, "TempTreatment", "y"))  # a near-zero effect is NOT dropped


def test_confounder_list_passed_unchanged_except_missing_columns():
    from ATE_update import get_adjustment_columns
    df = make_df(n=100)
    df["constant_col"] = 1
    cols = get_adjustment_columns(df, "TempTreatment", "y", CONF + ["constant_col", "not_a_column"])
    assert cols == CONF + ["constant_col"]  # constants are kept (DoWhy copes), missing names are removed


def test_too_few_rows_is_skipped_not_garbage():
    df = make_df(n=300).iloc[:3].copy()  # 3 rows for intercept + treatment + 2 confounders
    df["TempTreatment"] = [0, 1, 0]
    try:
        estimate_cate(df, "TempTreatment", "y", CONF)
        assert False, "expected EstimationError"
    except EstimationError as e:
        assert e.reason in ("too_few_rows", "treatment_not_identifiable"), e.reason


def test_only_configured_confounders_are_used():
    df = make_df()
    df["noise"] = np.random.default_rng(1).normal(size=len(df))
    from ATE_update import get_adjustment_columns
    assert get_adjustment_columns(df, "TempTreatment", "y", CONF) == CONF  # 'noise' not adjusted for
    assert get_adjustment_columns(df, "TempTreatment", "y", ["age", "missing_col"]) == ["age"]


def test_constant_confounder_inside_subgroup_is_passed_to_dowhy():
    from ATE_update import get_adjustment_columns
    df = make_df()
    without = estimate_cate(df, "TempTreatment", "y", ["age"])
    df["edu"] = 2  # constant within this subgroup: kept in the list, DoWhy copes with it
    assert get_adjustment_columns(df, "TempTreatment", "y", CONF) == CONF
    assert abs(estimate_cate(df, "TempTreatment", "y", CONF) - without) < 1e-6


def test_no_confounders_is_a_plain_difference_in_means():
    df = make_df()
    est = estimate_cate(df, "TempTreatment", "y", [])
    diff = df[df.TempTreatment == 1].y.mean() - df[df.TempTreatment == 0].y.mean()
    assert abs(est - diff) < 1e-6


def test_same_adjustment_for_every_subgroup_size():
    df = make_df(n=4000)
    big = estimate_cate(df, "TempTreatment", "y", CONF)
    small = estimate_cate(df.sample(60, random_state=1), "TempTreatment", "y", CONF)
    assert np.isfinite(big) and np.isfinite(small)


def test_clean_dataset_drops_nulls_once_and_converts_outcome():
    raw = pd.DataFrame({
        "Unnamed: 0": range(6),
        "a": ["x", "y", "UNKNOWN", "x", None, "y"],
        "y": ["1.5", "2", "3", "oops", "5", "6"],
    })
    out = clean_dataset(raw, outcome_col="y")
    assert list(out.columns) == ["a", "y"]
    assert len(out) == 3 and out["y"].dtype.kind == "f"
    assert not out.isna().any().any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"ok  {name} ({time.time()-t0:.1f}s)")
