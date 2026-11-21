---
name: rw-accuracy
description: Compare RW (random walk) walk-extension strategies (top8, weighted8, weighted_single) against the exact homogeneity verdicts of this project. Use when RW settings or weights change, or when asked how accurate or fast RW is.
---

# RW accuracy test

RW (`algorithms/RW.py`) is a lower bound: it can only be wrong by calling a non-homogeneous rule homogeneous
(it missed the violating subgroup).

Strategies (`RW_EXTENSION`, default `weighted8`, the instructor's choice):
- `weighted8`: draw 8 next predicates at random by attribute weight (Efraimidis–Spirakis keys u^(1/w)), take the
  one with the largest deviation.
- `top8`: the old code, only the 8 highest-weight predicates (low-weight attributes are never added).
- `weighted_single`: one predicate drawn by weight.

Attribute weights: `ATTRIBUTE_WEIGHTS` in `configs/config.json`, fixed numbers from an outside source (not
computed by us). Whether to compute them ourselves (e.g. Yarden's method) is open with the instructor.

## Run (on the lab machine, skill `lab-machine`)
Needs `calibration_results/exact_summary_<ds>.csv` for the CURRENT rule list (skill `select-rules`).
```
for d in german acs so; do python -u experiments/rw_accuracy.py --dataset $d --seeds 5 --extensions top8,weighted8 --workers 40 2>&1 | tee logs/rwacc_$d.log; done
```
Output: `calibration_results/rw_accuracy_<ds>.csv` (one row per rule × seed × strategy); the log ends with
"per strategy:" (accuracy, detection on non-homogeneous rules, mean time, mean subgroups).
Times: German ~1 min/rule, ACS ~5 min, SO 4–9 min.

## Results on the current data (Nov 2026, k = 2000 walks, 5 seeds)
| | top8 | weighted8 |
|---|---|---|
| German | 100% correct, 19 s | 100%, 42 s |
| ACS | 80% (misses rules 8, 9 every time), 5 min | 100%, 5 min |
| SO | 60% (misses all 4 non-homogeneous), 4 min | 60% (same), 9 min |

SO's violating subgroups have 5–9 predicates and exactly δ/10 = 10 treated rows, which walks almost never reach.
