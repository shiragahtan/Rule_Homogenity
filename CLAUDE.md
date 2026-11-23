# Rule homogeneity project: notes for Claude and for the team

Paper: "The Cost of Homogeneity in Prescription Causal Rules". Repo: `shiragahtan/Rule_Homogenity`.
**Working branch: `dowhy-cate-estimator`** (not `master`). Do not commit to `master`.

## What the project does
A prescription rule is "condition ⇒ treatment" (e.g. `Gender=Male ⇒ Exercise=3-4 times per week`). It is
**homogeneous** for (δ, ε) when every subgroup (a conjunction of attribute=value predicates with at least δ rows)
has an effect within ε of the rule's overall effect. The exact smallest ε of a rule = the largest deviation over
all subgroups. Datasets: German Credit, Stack Overflow (SO), ACS.

## Requirements from the instructor (Brit)
- Effect = DoWhy CATE (linear regression), `common_causes = IMMUTABLE_ATTRIBUTES` (config `CONFOUNDERS`).
- Size filter: in every subgroup, each treatment side needs at least δ/10 rows (`MIN_ARM_DELTA_DIVISOR = 10`).
  It applies to subgroups AND to the rule's overall effect (changed at Sean's request; a rule failing it is not
  evaluable). No p-value filter.
- Negative-effect rules are replaced. At least 10% of a rule's population must be treated.
- Each dataset's list of 10 rules: **5 homogeneous and 5 non-homogeneous**, chosen by a procedure we can explain
  (coverage at least X%, highest effect).

## Parameters: use Table 3 of the article (the text of the article differs; Table 3 is what `configs/config.json` uses)
German δ=50, ε=0.5 | SO δ=100, ε=520,000 | ACS δ=100, ε=17,500. RW default k=2000 walks; walk extension "weighted8" (8 predicates drawn by weight, then largest deviation; was "top8").

## Datasets
`datasets/german.csv`, `datasets/stackoverflow.csv`, `datasets/acs.csv`: the only datasets, already filtered by
`scripts/prepare_datasets.py` (no missing values; 10 equal-width bins for continuous attributes; German amount
stays binary at 4,000 DM). Details in
`datasets/README.md`. Filtering is done only there, not in the CATE code.

## Estimator and code map
- `utils/ATE_update.py`: the estimator (`calculate_ate_safe`, `estimate_cate`, `estimate_cate_with_p`).
- `utils/preprocessing.py`: `clean_dataset`, `prepare_rule_df` (binary treatment = attribute equals the prescribed
  value; text columns are integer-coded, ACS is already numeric).
- `algorithms/`: BruteForce, RW (random walks, attribute weights from `ATTRIBUTE_WEIGHTS` in the config; RW is a
  lower bound), greedy, random, causal forest. `algorithms/main.py` is the main runner.
- `experiments/exact_min_epsilon.py` + `analyze_exact.py`: exact per-subgroup effects and exact smallest ε.
- `experiments/generate_candidate_rules.py` → `candidate_homogeneity.py` → `select_rules.py`: the rule selection
  pipeline (candidates, exact homogeneity check, 5 + 5 choice). Outputs are in `calibration_results/`.
- `scripts/run_on_machine.sh <step>`: the steps to run on the lab machine (preflight, main, ablation,
  scalability, benchmarks, exact_so, candidates, collect). `scripts/preflight.py` checks the setup.
- `configs/config.json` and the rule files `configs/*Chosen10Treatments.json` (one JSON rule per line).

## Current state (Nov 2026; the meeting summary is `project_summary.html` in the parent folder)
- Rule lists rebuilt on the new data (skill `select-rules`), with ε as close to Table 3 as gives about half
  homogeneous: German ε 0.5 (5 H / 5 N), ACS ε 66,000 (4 / 6), SO ε 473,000 (6 / 4; same 10 rules as before).
  Candidates: `calibration_results/candidates_eqwidth/`; exact verdicts: `calibration_results/exact_summary_<ds>.csv`.
- RW comparison (`calibration_results/rw_accuracy_<ds>.csv`): top8 vs weighted8 correct German 100/100%,
  ACS 80/100%, SO 60/60%. On SO, RW misses the 4 non-homogeneous rules: their violating subgroups have 4-12
  predicates and almost all exactly δ/10 = 10 treated rows. With ≥ 15 treated rows per arm, 9 of the 10 SO rules
  would be homogeneous, so these violations are mostly small-sample noise.
- RW fixes (`experiments/rw_fix_variants.py`, options in `algorithms/RW.py`, all off by default): longer walks
  (`max_depth`), `revisit`, `arm_prefilter`, Sean's `learned_weights` (start 1/n, grow with the effect change).
  Suggested to the instructor: learned weights + arm prefilter (SO: finds 8 of 12 misses instead of 1; German and
  ACS stay 100% correct). Not the default until she approves; results in `calibration_results/rw_fix_*.csv`,
  `rw_learned*_*.csv`.
- Full experiments on the new data run on the lab machine in parallel (`experiments/parallel_runner.py`, skill
  `full-experiments`), because brute force now takes up to its 1-hour timeout per rule. `algorithms/main.py` ran
  only RW until Nov 2026 (hard-coded list); it now runs `ALGORITHM_NAMES` from the config.
- Ablation ε values changed (Nov 2026, user-approved) to span the chosen ε: ACS 20k-120k, SO 300k-650k.
- Open: how to compute the RW attribute weights (the config values come from an outside source).

## Working rules
- Run heavy experiments (exact SO, runtimes) on the lab machine in a tmux session, not locally.
- Ask before any outward action (push, force push, email). Never touch or pop the existing git stash
  `cate-dowhy-changes` on `master`.
- Explain things simply, in English, and keep answers short.
- Commit message trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Big result folders are git-ignored (`calibration_results/exact_so/`, `logs/`).
