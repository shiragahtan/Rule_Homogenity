#!/usr/bin/env bash
# Run every planned job of experiments/parallel_runner.py (36 at a time by default), then merge the outputs into
# graphs/, results/ and ablation_results/. Plan the jobs first (see the header of experiments/parallel_runner.py),
# then start this in tmux:   tmux new-session -d -s par scripts/run_parallel.sh
# Progress: par_runs/progress.txt (ends with "PAR DONE"); python experiments/parallel_runner.py status
cd "$(dirname "$0")/.." && source .venv/bin/activate
mkdir -p par_runs
python experiments/parallel_runner.py run --slots "${SLOTS:-36}" >> par_runs/runner.log 2>&1
python experiments/parallel_runner.py status >> par_runs/progress.txt 2>&1
python experiments/parallel_runner.py merge --datasets german,acs,so --steps main,alltimes,rows,attrs,ablation \
    >> par_runs/merge.log 2>&1
echo "$(date +%F_%T) PAR DONE" >> par_runs/progress.txt
