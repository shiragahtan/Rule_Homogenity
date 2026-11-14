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
  It applies to subgroups only, not to the rule's overall effect. No p-value filter.
- Negative-effect rules are replaced. At least 10% of a rule's population must be treated.
- Each dataset's list of 10 rules: **5 homogeneous and 5 non-homogeneous**, chosen by a procedure we can explain
  (coverage at least X%, highest effect).

## Parameters: use Table 3 of the article (the text of the article differs; Table 3 is what `configs/config.json` uses)
German δ=50, ε=0.5 | SO δ=100, ε=520,000 | ACS δ=100, ε=17,500. RW default k=2000 walks.

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

## Current state (see `summary_for_brit_rule_selection.html` in the parent folder for the numbers)
- Exact verdicts with the current rule lists: German 3 homogeneous / 7 not, ACS 0 / 10, SO 7 / 3.
- Candidate search at coverage ≥ 50%: German gives 5 + 5 at ε=0.5; ACS needs ε≈88,000 (Table 3 gives 0
  homogeneous); SO needs ε between about 325,000 and 495,000 (at 520,000 only 4 non-homogeneous exist).
- The rule lists are NOT changed yet. Waiting for the instructor to decide: coverage threshold, per-dataset ε
  vs fewer than 5 homogeneous rules, and approval of the selection procedure.
- After the decision: recompute utilities (`utils/recompute_rule_utilities.py --update-rules`), update the
  configs, rerun all experiments on the machine (runtimes: RW vs brute force), update tables and figures.
- Open question about the paper: the RW attribute weights are fixed numbers in the config; the article says
  "based on correlation with the outcome" but gives no formula and the repo has no code that computes them.

## Working rules
- Run heavy experiments (exact SO, runtimes) on the lab machine in a tmux session, not locally.
- Ask before any outward action (push, force push, email). Never touch or pop the existing git stash
  `cate-dowhy-changes` on `master`.
- Explain things simply, in English, and keep answers short.
- Commit message trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Big result folders are git-ignored (`calibration_results/exact_so/`, `logs/`).
