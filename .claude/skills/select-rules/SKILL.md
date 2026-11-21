---
name: select-rules
description: Build the 10-rule list and choose epsilon for a dataset of this project (German, ACS, SO) - candidate generation, exact homogeneity check, epsilon choice, writing configs/*Chosen10Treatments.json. Use after a data change, a new coverage threshold, or when asked which rules or epsilon to use.
---

# Select the rules and epsilon

## Agreed procedure (approved by the instructor)
1. Candidates: single condition ⇒ single treatment, condition coverage ≥ 50%, positive effect (DoWhy CATE),
   at least 10% of the rule's population treated.
2. Exact check of every subgroup (≥ δ rows, each arm ≥ δ/10 rows) gives each candidate's exact smallest ε.
3. Homogeneous at ε ⇔ smallest ε ≤ ε. The list = highest-effect homogeneous + highest-effect non-homogeneous
   candidates, homogeneous first.
4. **ε choice:** both kinds must exist, **approximately half and half (6/4 or 4/6 is fine, not exactly 5/5)**.
   Take the ε that changes the Table 3 value the least (German 0.5, SO 520,000, ACS 17,500).

## Commands (on the lab machine, skill `lab-machine`)
```
python experiments/generate_candidate_rules.py --dataset <ds> --workers 32 --no-old --coverage-threshold 50 --out calibration_results/<dir>
python -u experiments/candidate_homogeneity.py --dataset <ds> --workers 32 --exact --all --min-coverage 50 \
    --candidates calibration_results/<dir>/<ds>_candidate_rules_cov50.csv --out calibration_results/<dir>
```
Output: `<ds>_candidate_min_epsilon.csv` (columns rule, condition_attr/val, treatment_attr/val, coverage_pct,
effect, min_epsilon, ...). The log ends with "N homogeneous and M non-homogeneous candidates evaluated".
SO is very heavy (~4M subgroups per rule); its exact run is `scripts/run_on_machine.sh exact_so 32`.

Then on the laptop: sort the min_epsilon values, pick ε by rule 4, and write the list.
`experiments/select_rules.py` does exactly 5 + 5; for a 6/4 or 4/6 split, write the rules file directly in the
same format (one JSON per line: `{"condition": {attr: val}, "treatment": {attr: val}, "utility": effect, "coverage": pct}`;
numbers as int when whole).

## Files to update
- `configs/GermanChosen10Treatments.json`, `configs/ACSChosen10Treatments.json`, `configs/Chosen10Treatments.json` (SO).
- `EPSILONS` in `configs/config.json`: edit only the number (keep the file's formatting; do not `json.dump` it).
- `calibration_results/exact_summary_<ds>.csv` with at least `rule` (1-based, file order) and
  `min_eps_new_final`: `experiments/rw_accuracy.py` reads it as the exact verdict.
- Copy the changed files to the machine before running anything there.

## Decisions on the current data (equal-width bins), Nov 2026
German ε 0.5 (5 H / 5 N, 57 candidates) | ACS ε 66,000 (4 H / 6 N; only 10 candidates, all used) |
SO ε 473,000 (6 H / 4 N, same 10 rules as before).
