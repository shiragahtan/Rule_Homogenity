"""
LaTeX rows for the article's appendix rule tables (Tables 10 and 12: German Credit and ACS) from the
current rules files (condition, treatment, coverage, utility = overall effect with the current CATE)
and the prevalence computed by experiments/analyze_exact.py (calibration_results/prevalence_*.csv).

    python scripts/make_appendix_tables.py > calibration_results/appendix_tables_german_acs.tex

Stack Overflow (Table 11) is left out: its prevalence needs the exact run on the other machine, and
which of its two rule sets is the article's is still open.
"""
import csv
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
config = json.load(open(REPO / "configs" / "config.json", encoding="utf-8"))
PREV = {(r["dataset"], int(r["rule"])): float(r["prevalence_pct"]) for r in csv.DictReader(open(REPO / "calibration_results" / "prevalence_german_acs.csv"))}
BASE = "8eb4358"  # commit with the rules files before the six replacements


def tex(s):
    return re.sub(r"([&%$#_{}])", r"\\\1", str(s))


def original_rules(path):
    txt = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:{path}"], capture_output=True, text=True).stdout
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


for name, ds_key, table in [("GERMAN", "german", 10), ("ACS", "acs", 12)]:
    ds = config["DATASETS"][ds_key]
    rules = [json.loads(l) for l in open(REPO / ds["RULES_FILE"], encoding="utf-8") if l.strip()]
    old = original_rules(ds["RULES_FILE"])
    print(f"% Table {table}: Rules for {name} dataset (current rules, effects with the DoWhy CATE; delta={ds['DELTAS'][0]}, epsilon={ds['EPSILONS'][0]})")
    print("% Rule & Condition & Treatment & Coverage & Utility & Prevalence \\\\")
    for i, r in enumerate(rules, 1):
        (ca, cv), = r["condition"].items()
        (ta, tv), = r["treatment"].items()
        flag = "   % REPLACED (negative effect with the new CATE)" if (r["condition"], r["treatment"]) != (old[i - 1]["condition"], old[i - 1]["treatment"]) else ""
        print(f"{i} & {tex(ca)} = {tex(cv)} & {tex(ta)} $\\rightarrow$ {tex(tv)} & {r['coverage']:.2f}\\% & {r['utility']:.4f} & {PREV[(ds_key, i)]:.2f}\\% \\\\{flag}")
    print()
