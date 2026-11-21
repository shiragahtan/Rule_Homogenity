---
name: full-experiments
description: Run every experiment of the paper on the lab machine (main HomogeneityCheck, AllSubgroups runtimes, ablation, scalability, Problems 2-3 benchmarks) and collect the outputs for the figures. Use after the rule lists, epsilons or code change, or when asked to rerun "everything".
---

# Full experiments

All steps are in `scripts/run_on_machine.sh` (read its header). Run on the lab machine (skill `lab-machine`),
one step at a time: they measure runtimes, so nothing else may run in parallel, and `set_dataset.py` changes
`CHOSEN_DATASET` in the config.

## Before
1. Rule lists and ε are final (skill `select-rules`) and are on the machine (pushed and pulled, or copied).
2. `python scripts/preflight.py` must end with "RESULT: ALL CHECKS PASSED".
3. `DRY_RUN=1 scripts/run_on_machine.sh <step> <ds>` prints the commands without running them. Note that it
   still overwrites `logs/main_<ds>.log` and `logs/alltimes_<ds>.log` with an empty line.

## Queue (one detached tmux session, waits for the previous one)
```
for d in german acs so; do
  for s in main collect alltimes collect ablation scalability collect; do
    echo "$(date +%F_%T) begin $s $d" >> logs/full_progress.txt
    scripts/run_on_machine.sh $s $d; echo "$(date +%F_%T) end $s $d exit=$?" >> logs/full_progress.txt
  done
done
for d in so acs; do scripts/run_on_machine.sh benchmarks $d; done   # German has no benchmarks step
```
- `main.py` deletes `graphs/` outputs; the script backs them up to `graphs_backup_<time>/` first, and `collect`
  copies each dataset's outputs to `results/` (what the figure notebook reads). That is why collect runs after
  every step.
- Progress: `cat ~/Rule_Homogenity/logs/full_progress.txt`; errors: `grep -l Traceback logs/*.log`.
- The whole queue takes many hours. German's ablation alone takes about a day: brute force at δ = 1% of a
  rule's rows (≈ 5–9 rows) needs about 1 hour per rule. That is expected, not a bug: do not stop it.
- Ablation δ values are fractions of each rule's rows (0.01 = 1%), not percents.
- `ablation_study.py` saves each finished rule to `ablation_results/<ds>_ablation_partial.csv` and resumes from
  it if restarted; the file is deleted when the run completes.
- Scalability scripts append to `graphs/<ds>_scalability_*.csv`: move the old file away before a rerun.

## After
1. Copy `results/`, `ablation_results/` and the benchmark folders back to the laptop.
2. `jupyter nbconvert --to notebook --execute article_figures/generate_figures.ipynb --inplace`
3. Update the tables in the paper and the summary for the instructor (skill `instructor-update`).
4. Commit and push only with the user's approval.
