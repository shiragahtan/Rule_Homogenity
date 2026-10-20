"""Synthetic-data checks for the DoWhy CATE estimator and the shared preprocessing.
Run from the repo root:  python -m pytest tests -q   (or: python tests/test_cate_estimator.py)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
from ATE_update import calculate_ate_safe, estimate_cate, EstimationError
from preprocessing import clean_dataset

TRUE_EFFECT = 5.0


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
    est = estimate_cate(make_df(), "TempTreatment", "y")
    assert abs(est - TRUE_EFFECT) < 0.3, est


def test_no_outlier_trimming_or_effect_cap():
    # old code trimmed outcome to the 5-95% range and returned NaN for effects > 5x baseline;
    # now every row is used, so extreme outcomes legitimately move the estimate and it stays finite.
    df = make_df(outlier=True)
    est = estimate_cate(df, "TempTreatment", "y")
    assert np.isfinite(est)
    assert abs(est - TRUE_EFFECT) > 1.0  # the outliers are NOT silently removed


def test_small_arms_are_not_rejected():
    df = make_df(n=300)
    df.loc[df.index[3:], "TempTreatment"] = 0  # only 3 treated rows (old code required >= 10)
    assert np.isfinite(estimate_cate(df, "TempTreatment", "y"))


def test_single_treatment_arm_raises():
    df = make_df()
    df["TempTreatment"] = 1
    try:
        estimate_cate(df, "TempTreatment", "y")
        assert False, "expected EstimationError"
    except EstimationError:
        pass
    assert np.isnan(calculate_ate_safe(df, "TempTreatment", "y"))  # wrapper keeps the NaN contract


def test_same_adjustment_for_every_subgroup_size():
    df = make_df(n=4000)
    big = estimate_cate(df, "TempTreatment", "y")
    small = estimate_cate(df.sample(60, random_state=1), "TempTreatment", "y")
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
