---
name: full-experiments
description: Run every experiment of the paper on the lab machine (main HomogeneityCheck, AllSubgroups runtimes, ablation, scalability, Problems 2-3 benchmarks, RW single-draw accuracy) and collect the outputs for the figures. Use after the rule lists, epsilons or code change, or when asked to rerun "everything".
---

# Full experiments

Run on the lab machine (skill `lab-machine`) with `experiments/parallel_runner.py` (read its header). It splits
every experiment into independent jobs (one rule; for scalability one data size or attribute count, one repeat and
one algorithm; for the ablation one tested value), runs them on many cores at once with exactly the settings of the
sequential scripts, and merges the outputs into the usual files. Each job is limited to one numpy/BLAS thread.

Why: brute force (FPGrowth) takes up to its 1-hour timeout per rule on the new data, so the sequential queue
(`scripts/run_on_machine.sh`) would take about 4 weeks; in parallel it takes 1-2 days.

## Before
1. Rule lists and ε are final (skill `select-rules`) and the code is on the machine (pushed and pulled, or copied).
2. `python scripts/preflight.py` must end with "RESULT: ALL CHECKS PASSED".
3. Nothing else heavy is running (`tmux ls`; `uptime`). The machine has 48 cores; use 36 slots.

## Run
```
cd ~/Rule_Homogenity && source .venv/bin/activate
python experiments/parallel_runner.py plan --datasets german,acs,so \
    --steps main,alltimes,rows,attrs,ablation,benchmarks,rwsingle
tmux new-session -d -s par scripts/run_parallel.sh   # run --slots 36, then merge, then "PAR DONE"
```
`scripts/run_parallel.sh` runs `run --slots 36` (`SLOTS=` to change), `status`, and `merge --datasets german,acs,so
--steps main,alltimes,rows,attrs,ablation`, then writes "PAR DONE" to `par_runs/progress.txt`.
- Progress: `python experiments/parallel_runner.py status`, `tail par_runs/progress.txt`.
- One job's output and log: `par_runs/jobs/<id>/` (`log.txt`, `DONE` or `FAILED`).
- `run` skips jobs with `DONE`, so after a crash or a fix just start it again. Delete a job's `DONE` to rerun it.
- `merge` merges only (dataset, step) groups whose jobs all finished; it backs up the files it replaces to
  `graphs_backup_par_<time>/` and also generates the scalability graphs and the ablation report.
- `--keep-existing-rows <ds>` keeps the rows already in `graphs/<ds>_scalability_rows.csv` (from an interrupted
  sequential run) and plans only the missing runs.
- The dataset comes from the environment (`HOMOGENEITY_DATASET`), not from `CHOSEN_DATASET`, so the three datasets
  run at the same time from one checkout. Do not run `scripts/set_dataset.py` or the sequential steps meanwhile.

## Notes
- Main experiment algorithms: `ALGORITHM_NAMES` in the config (FPGrowth, Random, Greedy, CausalForest, RW, WTE).
  The Random baseline runs inside each RW job, only for the main experiment (scalability compares RW and FPGrowth).
- Ablation ε values (in `experiments/ablation_study.py`) span each dataset's chosen ε: ACS 20k-120k, SO 300k-650k,
  German 0.1-0.9. δ values are fractions of each rule's rows (0.01 = 1%). The ablation's FPGrowth has no timeout.
- AllSubgroups (mode 1) writes no row for a run that hits the 1-hour timeout (check the job logs for "Timed out").
- An exception inside an algorithm is printed as "ERROR in the algorithm" and that run is recorded as not
  homogeneous (old behaviour of `algorithms/main.py`): grep the logs for it before trusting the results.
- The sequential way still works (`scripts/run_on_machine.sh <step> <ds>`, one step at a time, after
  `set_dataset.py`), e.g. for a single small rerun.

## After
1. Copy `results/`, `ablation_results/`, `problem_2_3_algorithms/benchmark_*` and
   `calibration_results/rw_accuracy_*_single.csv` back to the laptop.
2. `jupyter nbconvert --to notebook --execute article_figures/generate_figures.ipynb --inplace`
3. Update the tables in the paper and the summary for the instructor (skill `instructor-update`).
4. Commit and push only with the user's approval.
