#!/usr/bin/env bash
# Steps of the full experiments, to be run on the machine that measures the runtimes.
#
#   scripts/run_on_machine.sh preflight                 check packages, data, rules, tests
#   scripts/run_on_machine.sh main <german|so|acs>      HomogeneityCheck (mode 0) of every algorithm and rule
#   scripts/run_on_machine.sh alltimes <german|so|acs>  AllSubgroups (mode 1): runtime of enumerating every subgroup
#   scripts/run_on_machine.sh ablation <so|acs|german>  epsilon / delta ablation study
#   scripts/run_on_machine.sh scalability <so|acs|german>  rows and attributes scalability
#   scripts/run_on_machine.sh benchmarks <so|acs>       largest delta / smallest epsilon (Problems 2 and 3)
#   scripts/run_on_machine.sh exact_so [workers]        exact verdicts for Stack Overflow (heavy: ~4M subgroups per rule)
#   scripts/run_on_machine.sh collect <german|so|acs>   copy graphs/* outputs to results/ (what the figure notebook reads)
#
# DRY_RUN=1 prints the commands instead of running them. Output of every step goes to logs/<step>.log.
# Existing files in graphs/ are backed up to graphs_backup_<time>/ before main.py (which deletes them).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
mkdir -p logs

run() {  # run <logname> <command...>
  local name="$1"; shift
  echo ">> $*   (log: logs/${name}.log)"
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  "$@" 2>&1 | tee "logs/${name}.log"
}

pipe_mode() {  # main.py asks for the mode on stdin
  local mode="$1"; shift
  echo ">> echo $mode | $*"
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  echo "$mode" | "$@"
}

backup_graphs() {
  if ls graphs/*.xlsx graphs/*.csv >/dev/null 2>&1; then
    local b="graphs_backup_$(date +%Y%m%d_%H%M%S)"
    echo ">> backing up graphs/ outputs to $b/"
    [ "${DRY_RUN:-0}" = "1" ] || { mkdir -p "$b"; cp graphs/*.xlsx graphs/*.csv "$b"/ 2>/dev/null || true; }
  fi
}

step="${1:-}"; ds="${2:-}"
case "$step" in
  preflight)    run preflight "$PY" scripts/preflight.py ;;
  main)         backup_graphs; run set_dataset "$PY" scripts/set_dataset.py "$ds"
                pipe_mode 0 "$PY" algorithms/main.py 2>&1 | tee "logs/main_${ds}.log" ;;
  alltimes)     backup_graphs; run set_dataset "$PY" scripts/set_dataset.py "$ds"
                pipe_mode 1 "$PY" algorithms/main.py 2>&1 | tee "logs/alltimes_${ds}.log" ;;
  ablation)     run set_dataset "$PY" scripts/set_dataset.py "$ds"
                run "ablation_${ds}" "$PY" experiments/ablation_study.py ;;
  scalability)  run set_dataset "$PY" scripts/set_dataset.py "$ds"
                run "scalability_rows_${ds}" "$PY" experiments/scalability_rows.py
                run "scalability_attrs_${ds}" "$PY" experiments/scalability_attrs.py ;;
  benchmarks)   # output folders are the ones article_figures/generate_figures.ipynb reads
                case "$ds" in
                  so)  out=benchmark_two_phase_FIXED ;;
                  acs) out=benchmark_acs_full ;;
                  *)   echo "benchmarks: so or acs"; exit 1 ;;
                esac
                run "benchmarks_${ds}" "$PY" problem_2_3_algorithms/run_all_benchmarks.py --dataset "$ds" --output "$out" ;;
  exact_so)     w="${2:-6}"
                run exact_so "$PY" experiments/exact_min_epsilon.py --dataset so --workers "$w" --out calibration_results/exact_so
                run analyze_so "$PY" experiments/analyze_exact.py --exact-dir calibration_results/exact_so/exact --dataset so ;;
  collect)      echo ">> copying graphs/${ds}_* to results/"
                [ "${DRY_RUN:-0}" = "1" ] || cp -v graphs/${ds}_homogeneity_results.xlsx graphs/${ds}_scalability_rows.csv graphs/${ds}_scalability_attributes.csv graphs/${ds}_algorithms_time.csv results/ 2>/dev/null || true
                echo "Then regenerate the figures:  jupyter nbconvert --to notebook --execute article_figures/generate_figures.ipynb --inplace" ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
