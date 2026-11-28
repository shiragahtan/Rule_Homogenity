"""
Problems 2 and 3 (largest delta / smallest epsilon) with one process per (rule, value) instead of one per rule.
run_all_benchmarks.py gives each rule one process that tries the values one after the other; on Stack Overflow one
value can take a day, so a rule takes about a week. The values are independent searches, so running every
(rule, value) pair at the same time gives the same results much sooner. The same worker functions and output files
as run_all_benchmarks.py are used.

    python problem_2_3_algorithms/run_benchmarks_parallel.py --dataset so --problems 3 \
        --output benchmark_two_phase_FIXED --workers 40
    python problem_2_3_algorithms/run_benchmarks_parallel.py --dataset so --problems 2,3 \
        --rules-file configs/Chosen10Treatments.json --output benchmark_so_current_rules --workers 40

Defaults are those of run_all_benchmarks.py (rules 1-5; SO: epsilons 10,000-60,000, deltas 500-3,000; ACS: its
own settings). Each finished task is appended to <output>/tasks_done.jsonl, so a restarted run skips finished tasks.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import run_all_benchmarks as R  # noqa: E402  (same workers and report code)


def ds_settings(dataset, rules_file):
    """The dataset settings of run_all_benchmarks.main(), with an optional other rule file."""
    if dataset == "acs":
        cfg = {"treatment_file": str(REPO / "configs" / "ACSChosen10Treatments.json"),
               "outcome_col": "Wages or salary income past 12 months",
               "base_dataset_path": str(REPO / "datasets" / "acs.csv"),
               "delta_min_pct": 5.0, "delta_max_pct": 20.0, "delta_percentages": [5, 10, 15, 20]}
        epsilons = "5000,10000,15000,17500,20000"
    else:
        cfg = {"base_dataset_path": str(REPO / "datasets" / "stackoverflow.csv")}
        epsilons = "10000,20000,30000,40000,50000,60000"
    if rules_file:
        cfg["treatment_file"] = str((REPO / rules_file).resolve())
    return cfg, epsilons


def run_task(task):
    problem, rule, value, a = task
    t0 = time.time()
    if problem == 2:
        rows = R._process_single_rule_delta((rule, a["rules"], [value], a["delta_min"], a["delta_max"],
                                             a["p2_dir"], a["cfg"]))
    else:  # value is an absolute delta, or (ACS) an index into delta_percentages
        cfg = dict(a["cfg"])
        deltas = a["deltas"]
        if cfg.get("delta_percentages"):
            cfg["delta_percentages"] = [value]
        else:
            deltas = [value]
        rows = R._process_single_rule_epsilon((rule, a["rules"], deltas, a["epsilon_0"], a["epsilon_max"],
                                               a["p3_dir"], cfg))
    return problem, rule, value, rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["so", "acs"])
    ap.add_argument("--problems", default="2,3")
    ap.add_argument("--output", required=True, help="folder under problem_2_3_algorithms/")
    ap.add_argument("--rules-file", default=None, help="rule file (default: the one run_all_benchmarks.py uses)")
    ap.add_argument("--rules", type=int, default=5)
    ap.add_argument("--epsilons", default=None)
    ap.add_argument("--deltas", default="500,1000,1500,2000,2500,3000")
    ap.add_argument("--delta-min", type=int, default=100)
    ap.add_argument("--delta-max", type=int, default=10000)
    ap.add_argument("--epsilon-0", type=float, default=None)
    ap.add_argument("--epsilon-max", type=float, default=1_000_000_000.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import pandas as pd
    cfg, default_eps = ds_settings(args.dataset, args.rules_file)
    out = HERE / args.output
    p2_dir, p3_dir = out / "problem2_largest_delta", out / "problem3_smallest_epsilon"
    p2_dir.mkdir(parents=True, exist_ok=True)
    p3_dir.mkdir(parents=True, exist_ok=True)
    epsilons = [float(x) for x in (args.epsilons or default_eps).split(",")]
    deltas = [int(x) for x in args.deltas.split(",")]
    shared = dict(rules=args.rules, delta_min=args.delta_min, delta_max=args.delta_max, p2_dir=str(p2_dir),
                  p3_dir=str(p3_dir), cfg=cfg, deltas=deltas, epsilon_0=args.epsilon_0,
                  epsilon_max=args.epsilon_max)
    problems = [int(p) for p in args.problems.split(",")]
    p3_values = cfg.get("delta_percentages") or deltas
    tasks = [(p, r, v, shared) for p in problems for r in range(1, args.rules + 1)
             for v in (epsilons if p == 2 else p3_values)]

    done_file = out / "tasks_done.jsonl"
    done = {}
    if done_file.exists():
        for line in done_file.read_text().splitlines():
            d = json.loads(line)
            done[(d["problem"], d["rule"], d["value"])] = d["rows"]
    todo = [t for t in tasks if (t[0], t[1], t[2]) not in done]
    print(f"{args.dataset}: {len(tasks)} tasks ({len(done)} already done), {args.workers} workers, rules from "
          f"{cfg.get('treatment_file') or HERE / 'Chosen10Treatments.json'} -> {out}", flush=True)
    with Pool(args.workers) as pool:
        for problem, rule, value, rows, sec in pool.imap_unordered(run_task, todo):
            done[(problem, rule, value)] = rows
            with open(done_file, "a") as f:
                f.write(json.dumps(dict(problem=problem, rule=rule, value=value, rows=rows, seconds=sec),
                                   default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            print(f"done: problem {problem} rule {rule} value {value} ({sec / 3600:.1f} h); "
                  f"{len(done)} of {len(tasks)}", flush=True)

    for p in problems:
        rows = [row for (pp, r, v), rs in sorted(done.items(), key=lambda kv: (kv[0][1], kv[0][2]))
                if pp == p for row in rs]
        df = pd.DataFrame(rows)
        if p == 2:
            summary = R.generate_summary_statistics(df)
            R.create_visualizations(df, str(p2_dir))
            R.generate_delta_html_report(df, summary, str(p2_dir))
        else:
            summary = R.gen_epsilon_summary(df)
            R.create_epsilon_viz(df, str(p3_dir))
            R.generate_epsilon_html_report(df, summary, str(p3_dir))
        print(f"problem {p}: {len(df)} rows written to {p2_dir if p == 2 else p3_dir}", flush=True)
    if set(problems) == {2, 3} or (p2_dir / "find_delta_benchmark_results.csv").exists():
        R.generate_combined_summary_single_html(str(out))
    print("ALL TASKS DONE", flush=True)


if __name__ == "__main__":
    main()
