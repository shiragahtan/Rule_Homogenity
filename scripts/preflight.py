"""
Pre-flight check before the long runs (run it on the machine that will do them):

    python scripts/preflight.py

Checks the Python packages, the dataset files, the config values (delta, epsilon, size filter),
that every rule of every rules file can be applied to its dataset (rows, treated rows, coverage),
and runs the estimator tests. Exits with status 1 if anything fails, so a bad setup is caught
before hours of computing are spent on it.
"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "utils"))
ok_all = True


def report(ok, text):
    global ok_all
    ok_all &= bool(ok)
    print(("  PASS  " if ok else "  FAIL  ") + text)


print(f"Python {sys.version.split()[0]} ({sys.executable})")
print("\n1. Packages")
for mod in ["pandas", "numpy", "scipy", "sklearn", "dowhy", "mlxtend", "econml", "xgboost", "lightgbm", "openpyxl", "matplotlib"]:
    try:
        m = importlib.import_module(mod)
        report(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        report(False, f"{mod}: {e}")

print("\n2. Config and datasets")
config = json.load(open(REPO / "configs" / "config.json", encoding="utf-8"))
print(f"  CHOSEN_DATASET = {config['CHOSEN_DATASET']}   MIN_ARM_DELTA_DIVISOR = {config.get('MIN_ARM_DELTA_DIVISOR')}   (no MIN_ARM_FRACTION / P_VALUE_THRESHOLD expected)")
report("MIN_ARM_FRACTION" not in config and "P_VALUE_THRESHOLD" not in config and config.get("MIN_ARM_DELTA_DIVISOR") == 10,
       "size filter = delta/10 per treatment side, no p-value filter")
for name, ds in config["DATASETS"].items():
    path = REPO / ds["FULL_DATASET_PATH"]
    report(path.exists(), f"{name}: dataset {ds['FULL_DATASET_PATH']}  delta={ds['DELTAS']} epsilon={ds['EPSILONS']} coverage threshold={ds.get('COVERAGE_THRESHOLD')}")

print("\n3. Rules (each must match rows and have treated rows)")
import pandas as pd  # noqa: E402
from preprocessing import clean_dataset, prepare_rule_df  # noqa: E402
for name, ds in config["DATASETS"].items():
    path = REPO / ds["FULL_DATASET_PATH"]
    if not path.exists():
        continue
    df = clean_dataset(pd.read_csv(path), outcome_col=ds["TARGET_COLUMN"])
    rules = [json.loads(line) for line in open(REPO / ds["RULES_FILE"], encoding="utf-8") if line.strip()]
    report(len(rules) == 10, f"{name}: {len(rules)} rules in {ds['RULES_FILE']}")
    for i, r in enumerate(rules, 1):
        sub = prepare_rule_df(df, r, config["TREATMENT_COL"], encode=(name != "acs"))
        good = sub is not None and 0 < sub[config["TREATMENT_COL"]].sum() < len(sub)
        (ca, cv), = r["condition"].items()
        (ta, tv), = r["treatment"].items()
        label = f"rule {i}: {ca}={cv} / {ta}={tv}"[:100]
        if not good:
            report(False, f"{name} {label}: empty population or no treated/control rows")
    print(f"  ({name}: all rules checked)")

print("\n4. Estimator tests")
res = subprocess.run([sys.executable, str(REPO / "tests" / "test_cate_estimator.py")], capture_output=True, text=True)
n_ok = sum(1 for line in res.stdout.splitlines() if line.startswith("ok"))
report(res.returncode == 0 and n_ok >= 13, f"{n_ok} tests passed")

print("\nRESULT:", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED - fix them before running")
sys.exit(0 if ok_all else 1)
