"""
Run the paper's experiments on many cores at once.

Every experiment is split into independent jobs (one rule; for scalability one data size or attribute count, one
repeat and one algorithm; for the ablation one tested value). Each job runs in its own process with exactly the
settings of the sequential scripts (algorithms/main.py, experiments/scalability_rows.py, scalability_attrs.py,
ablation_study.py), and the outputs are then merged into the usual files. Only the wall-clock time changes.
Each job is limited to one numpy/BLAS thread, so jobs do not compete for cores.

Steps
  main        HomogeneityCheck of every algorithm (mode 0)      -> graphs/ and results/<ds>_homogeneity_results.xlsx
  alltimes    AllSubgroups runtimes (mode 1)                    -> <ds>_algorithms_time.csv
  rows        scalability in the number of rows (mode 2)       -> <ds>_scalability_rows.csv (+ graphs)
  attrs       scalability in the number of attributes (mode 3) -> <ds>_scalability_attributes.csv (+ graphs)
  ablation    epsilon / delta ablation                          -> ablation_results/<ds>_ablation_*
  benchmarks  Problems 2-3 (so, acs; run_all_benchmarks.py with 5 rules in parallel)
  rwsingle    RW accuracy with the weighted_single extension    -> calibration_results/rw_accuracy_<ds>_single.csv

Usage
  python experiments/parallel_runner.py plan --datasets acs --steps rows,attrs [--keep-existing-rows acs]
  python experiments/parallel_runner.py run --slots 36          # runs every unfinished job; can be restarted
  python experiments/parallel_runner.py status
  python experiments/parallel_runner.py merge --datasets acs --steps rows,attrs

Order: main results first, then runtimes, RW single-draw, ablation, benchmarks, and scalability (largest first).
Jobs live in par_runs/jobs/<job id>/ (job.json, outputs, log.txt, DONE or FAILED). `run` skips jobs with DONE.
--keep-existing-rows keeps the rows already in graphs/<ds>_scalability_rows.csv (from an interrupted sequential
run) and plans only the missing (data size, repeat, algorithm, rule) runs.
"""
import argparse
import datetime
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "algorithms", REPO / "utils", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PAR = REPO / "par_runs"
JOBS = PAR / "jobs"
CONFIG = json.load(open(REPO / "configs" / "config.json"))
STEPS = ["main", "alltimes", "rows", "attrs", "ablation", "benchmarks", "rwsingle"]
BENCH_OUT = {"so": "benchmark_two_phase_FIXED", "acs": "benchmark_acs_full"}  # folders read by the figure notebook
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
OUTPUT_NAME = {"main": "{ds}_homogeneity_results.xlsx", "alltimes": "{ds}_algorithms_time.csv",
               "rows": "{ds}_scalability_rows.csv", "attrs": "{ds}_scalability_attributes.csv"}
# settings of experiments/scalability_rows.py and scalability_attrs.py (checked against the scripts in `plan`)
SCAL_ALGOS = ["RW", "FPGrowth"]
REPEATS = 3
ROW_PERCENTAGES = [x / 10 for x in range(1, 11)]
NUM_RW_RUNS = 3  # rows written per RW job (algorithms/main.py NUM_RW_RUNS)
HEAVY = {"FPGrowth", "Greedy", "Apriori"}


def rules_of(ds):
    with open(REPO / CONFIG["DATASETS"][ds]["RULES_FILE"], encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def log(msg):
    PAR.mkdir(exist_ok=True)
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(PAR / "progress.txt", "a") as f:
        f.write(line + "\n")


def load_jobs():
    return [json.load(open(d / "job.json")) for d in sorted(JOBS.glob("*")) if (d / "job.json").exists()]


def is_done(job):
    return (JOBS / job["id"] / "DONE").exists()


# ----------------------------------------------------------------------------------------------------- plan
def _check_script_settings():
    """Fail loudly if the sequential scripts were changed and this runner no longer matches them."""
    import re
    rows = (REPO / "experiments" / "scalability_rows.py").read_text()
    attrs = (REPO / "experiments" / "scalability_attrs.py").read_text()
    main = (REPO / "algorithms" / "main.py").read_text()
    for text in (rows, attrs):
        assert 'SCALABILITY_ALGORITHMS = ["RW", "FPGrowth"]' in text and "REPEATS_PER_CONFIG = 3" in text
    assert "ROW_PERCENTAGES = [x / 10 for x in range(1, 11)]" in rows
    assert "seed = 42 + int(p * 100) + r" in rows
    assert "random.seed(42 + i + n_features + r)" in attrs
    assert re.search(r"^NUM_RW_RUNS = 3$", main, re.M)


def plan(datasets, steps, keep_rows, algos_only=None):
    _check_script_settings()
    import ablation_study as A  # only for its per-dataset values
    algos = [a for a in CONFIG["ALGORITHM_NAMES"] if a != "Random"]  # Random runs inside the RW job
    new = 0
    for ds in datasets:
        n = len(rules_of(ds))
        for step in steps:
            jobs = []
            if step in ("main", "alltimes"):
                for ai, a in enumerate(algos):
                    for i in range(n):
                        base = 120 if step == "main" else 110  # main results first, then the runtimes
                        jobs.append(dict(id=f"{ds}-{step}-{ai}{a}-r{i:02d}", algo=a, rule=i, order=[ai, i],
                                         slots=1, prio=base if a in HEAVY else base - 5))
            elif step == "rows":
                done = {}
                meta = {"base": None}
                if ds in keep_rows:
                    import pandas as pd
                    src = REPO / "graphs" / f"{ds}_scalability_rows.csv"
                    if src.exists():
                        base = PAR / f"base_{ds}_rows.csv"
                        PAR.mkdir(exist_ok=True)
                        if not base.exists():
                            shutil.copy2(src, base)
                        meta["base"] = str(base)
                        b = pd.read_csv(base)
                        for (p, a, c, t), g in b.groupby(["dataset_percentage", "algorithm", "chosen_condition",
                                                          "chosen_treatment"]):
                            done[(round(float(p), 3), a, c, t)] = len(g)
                (PAR / f"plan_{ds}_rows.json").write_text(json.dumps(meta))
                rules = rules_of(ds)
                for p in ROW_PERCENTAGES:
                    for r in range(REPEATS):
                        for ai, a in enumerate(SCAL_ALGOS):
                            per_job = NUM_RW_RUNS if a == "RW" else 1
                            for i in range(n):
                                key = (round(p, 3), a, str(rules[i]["condition"]), str(rules[i]["treatment"]))
                                if done.get(key, 0) >= (r + 1) * per_job:
                                    continue  # already in the kept rows
                                jobs.append(dict(id=f"{ds}-rows-p{int(round(p * 100)):03d}-r{r}-{a}-r{i:02d}", algo=a,
                                                 rule=i, p=p, r=r, order=[p, r, ai, i], slots=1,
                                                 prio=(60 + 10 * p) if a in HEAVY else 20))
            elif step == "attrs":
                max_attr = CONFIG["DATASETS"][ds].get("MAX_SCALABILITY_ATTRIBUTES", 10)
                for nf in range(0, max(0, max_attr - 3) + 1, 2):
                    for r in range(REPEATS):
                        for ai, a in enumerate(SCAL_ALGOS):
                            for i in range(n):
                                jobs.append(dict(id=f"{ds}-attrs-n{nf:02d}-r{r}-{a}-r{i:02d}", algo=a, rule=i, nf=nf,
                                                 r=r, order=[nf, r, ai, i], slots=1,
                                                 prio=(50 + nf) if a in HEAVY else 20))
            elif step == "ablation":
                prm = A._ABLATION_PARAMS[ds]
                for ei, (exp, values) in enumerate([("varying_epsilon", prm["EPSILON_VALUES"]),
                                                     ("varying_delta", A.DELTA_PERCENTAGES)]):
                    for vi, v in enumerate(values):
                        for i in range(n):
                            jobs.append(dict(id=f"{ds}-ablation-{ei}{vi}-r{i:02d}", experiment=exp, value=v, rule=i,
                                             order=[ei, vi, i], slots=1, prio=90))
            elif step == "benchmarks":
                if ds in BENCH_OUT:
                    jobs.append(dict(id=f"{ds}-benchmarks", order=[0], slots=5, prio=85))
            elif step == "rwsingle":
                jobs.append(dict(id=f"{ds}-rwsingle", order=[0], slots=6, prio=100))
            else:
                raise SystemExit(f"unknown step {step}")
            if algos_only:  # e.g. rerun only RW: drop the other algorithms' jobs, ablation jobs skip FPGrowth
                jobs = [j for j in jobs if "algo" not in j or j["algo"] in algos_only]
                for j in jobs:
                    if step == "ablation":
                        j["algos"] = list(algos_only)
            for j in jobs:
                j.update(dataset=ds, step=step)
                d = JOBS / j["id"]
                if (d / "job.json").exists():
                    continue
                d.mkdir(parents=True)
                (d / "job.json").write_text(json.dumps(j))
                new += 1
            print(f"{ds} {step}: {len(jobs)} jobs")
    print(f"{new} new jobs in {JOBS}")


# ------------------------------------------------------------------------------------------------------ run
def _command(job):
    ds, step = job["dataset"], job["step"]
    if step == "benchmarks":
        return [sys.executable, "-u", "problem_2_3_algorithms/run_all_benchmarks.py", "--dataset", ds,
                "--output", BENCH_OUT[ds], "--parallel", str(job["slots"])]
    if step == "rwsingle":
        return [sys.executable, "-u", "experiments/rw_accuracy.py", "--dataset", ds, "--seeds", "5",
                "--extensions", "weighted_single", "--workers", str(job["slots"]),
                "--out", f"calibration_results/rw_accuracy_{ds}_single.csv"]
    return [sys.executable, "-u", str(Path(__file__).resolve()), "worker", str(JOBS / job["id"])]


def _start(job):
    d = JOBS / job["id"]
    for f in d.iterdir():  # a restarted job starts clean (no duplicated rows)
        if f.name != "job.json":
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    if job["step"] == "benchmarks":  # keep the previous benchmark outputs instead of mixing old and new files
        out = REPO / "problem_2_3_algorithms" / BENCH_OUT[job["dataset"]]
        if out.exists():
            out.rename(out.with_name(out.name + f"_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    env = dict(os.environ, HOMOGENEITY_DATASET=job["dataset"], HOMOGENEITY_OUT_DIR=str(d), PYTHONUNBUFFERED="1")
    env.update({v: "1" for v in THREAD_VARS})
    # Python's hash randomization changes the order of some sets of strings between processes, which changed RW's
    # walks (not its verdicts) from run to run; a fixed hash seed makes every job reproducible.
    env["PYTHONHASHSEED"] = "0"
    logf = open(d / "log.txt", "w")
    return subprocess.Popen(_command(job), cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)


def run(slots, adopt=False, only=None):
    """Start jobs in priority order. Strict order: when the next job needs more free slots than there are (a
    multi-core job), nothing else starts until it fits, so multi-core jobs are not starved by one-core jobs.
    adopt=True: jobs whose worker process is still alive (from a stopped runner) are watched, not restarted;
    they are DONE when their log ends with "JOB OK"."""
    jobs = [j for j in load_jobs() if not only or j["id"].startswith(only)]
    live = _running_ids() if adopt else set()
    adopted = {j["id"]: j for j in jobs if j["id"] in live and not is_done(j)}
    pending = sorted([j for j in jobs if not is_done(j) and j["id"] not in adopted], key=lambda j: -j["prio"])
    log(f"RUN start: {len(pending)} of {len(jobs)} jobs to run, {len(adopted)} adopted, {slots} slots")
    running, finished, failed = [], 0, 0
    used = sum(j["slots"] for j in adopted.values())
    last_ps = 0.0
    while pending or running or adopted:
        while pending and used + pending[0]["slots"] <= slots:
            j = pending.pop(0)
            running.append((_start(j), j, time.time()))
            used += j["slots"]
        time.sleep(5)
        if adopted and time.time() - last_ps > 30:
            last_ps = time.time()
            live = _running_ids()
            for jid in [k for k in adopted if k not in live]:
                j = adopted.pop(jid)
                used -= j["slots"]
                d = JOBS / jid
                ok = (d / "log.txt").exists() and "JOB OK" in (d / "log.txt").read_text(errors="ignore")[-3000:]
                (d / ("DONE" if ok else "FAILED")).write_text("adopted\n")
                finished += ok
                failed += not ok
                if not ok:
                    log(f"FAILED {jid} (adopted, no JOB OK in its log)")
        for item in list(running):
            proc, j, t0 = item
            rc = proc.poll()
            if rc is None:
                continue
            running.remove(item)
            used -= j["slots"]
            d = JOBS / j["id"]
            secs = time.time() - t0
            if rc == 0:
                (d / "DONE").write_text(f"{secs:.0f}\n")
                finished += 1
            else:
                (d / "FAILED").write_text(f"exit {rc} after {secs:.0f}s\n")
                failed += 1
                log(f"FAILED {j['id']} (exit {rc}); see {d / 'log.txt'}")
            if (finished + failed) % 25 == 0 or rc != 0 or j["slots"] > 1:
                log(f"{finished} done, {failed} failed, {len(running)} running, {len(pending)} waiting "
                    f"(last: {j['id']} {secs:.0f}s)")
    log(f"RUN end: {finished} done, {failed} failed")
    return failed


# --------------------------------------------------------------------------------------------------- worker
def worker(job_dir):
    d = Path(job_dir)
    job = json.load(open(d / "job.json"))
    ds, step = job["dataset"], job["step"]
    assert os.environ.get("HOMOGENEITY_DATASET") == ds
    import pandas as pd
    from preprocessing import clean_dataset
    if step in ("main", "alltimes", "rows", "attrs"):
        from algorithms import main as M
        assert M.CHOSEN_DS == ds and Path(M.OUT_DIR) == d, (M.CHOSEN_DS, M.OUT_DIR)
        with open(M.RULES_FILE, "r") as f:
            rules = [json.loads(line) for line in f]
        tgt = M.TARGET_COLUMN_NAME
        full_df = clean_dataset(pd.read_csv(M.FULL_DATASET_PATH), outcome_col=tgt)
        i, algo = job["rule"], job["algo"]
        rule = rules[i]
        if step == "main":
            M.process_dataset_dynamic(i, rule, full_df, 0, algo, tgt)
        elif step == "alltimes":
            M.process_dataset_dynamic(i, rule, full_df, 1, algo, tgt)
        elif step == "rows":  # as experiments/scalability_rows.py
            p, r = job["p"], job["r"]
            seed = 42 + int(p * 100) + r
            sampled_df = full_df if p == 1.0 else full_df.sample(frac=p, random_state=seed)
            M.process_dataset_dynamic(i, rule, full_df, chosen_mode=2, chosen_algorithm_name=algo, tgtO=tgt,
                                      override_df=sampled_df, metric_value=p)
        else:  # attrs, as experiments/scalability_attrs.py
            n_features, r = job["nf"], job["r"]
            all_columns = [c for c in full_df.columns if c != tgt]
            cond_col = list(rule["condition"].keys())[0]
            treat_col = list(rule["treatment"].keys())[0]
            mandatory = {cond_col, treat_col, tgt}
            remaining_pool = [c for c in all_columns if c not in mandatory]
            count_to_pick = min(n_features, len(remaining_pool))
            random.seed(42 + i + n_features + r)
            selected_features = random.sample(remaining_pool, count_to_pick)
            final_cols = list(mandatory) + selected_features
            M.process_dataset_dynamic(i, rule, full_df, chosen_mode=3, chosen_algorithm_name=algo, tgtO=tgt,
                                      override_df=full_df[final_cols].copy(), metric_value=n_features)
    elif step == "ablation":  # one (experiment, value, rule) of experiments/ablation_study.py
        import ablation_study as A
        assert A.CHOSEN_DS == ds
        with open(A.RULES_FILE, "r") as f:
            rules = [json.loads(line) for line in f]
        full_df = clean_dataset(pd.read_csv(A.FULL_DATASET_PATH), outcome_col=A.TARGET_COLUMN_NAME)
        exp, v, i = job["experiment"], job["value"], job["rule"]
        eps, dpct = (v, A.FIXED_DELTA_PERCENTAGE) if exp == "varying_epsilon" else (A.FIXED_EPSILON, v)
        rows = []
        if "FPGrowth" in job.get("algos", ["FPGrowth"]):
            res = A.process_rule(rules[i], full_df, A.TARGET_COLUMN_NAME, None, eps, "FPGrowth", dpct)
            if res:
                res["experiment"] = exp
                rows.append(res)
        for run_number in range(A.NUM_RW_RUNS):
            res = A.process_rule(rules[i], full_df, A.TARGET_COLUMN_NAME, None, eps, "RW_Direct", dpct)
            if res:
                res["experiment"] = exp
                res["run_number"] = run_number + 1
                rows.append(res)
        for r_ in rows:
            r_["param_value"] = float(v)
            r_["rule_idx"] = i + 1
        if rows:
            pd.DataFrame(rows).to_csv(d / "ablation_rows.csv", index=False)
    else:
        raise SystemExit(f"step {step} has no worker")
    print("JOB OK", flush=True)


# ---------------------------------------------------------------------------------------------------- merge
def merge_one(ds, step, allow_partial):
    import pandas as pd
    jobs = sorted([j for j in load_jobs() if j["dataset"] == ds and j["step"] == step], key=lambda j: j["order"])
    if not jobs:
        print(f"{ds} {step}: no jobs")
        return
    missing = [j["id"] for j in jobs if not is_done(j)]
    if missing and not allow_partial:
        print(f"{ds} {step}: {len(missing)} of {len(jobs)} jobs not finished, not merged (first: {missing[0]})")
        return
    if step in ("benchmarks", "rwsingle"):
        print(f"{ds} {step}: written in place by the script")
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = REPO / f"graphs_backup_par_{stamp}"

    def backup(path):
        if path.exists():
            backup_dir.mkdir(exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)

    if step == "ablation":
        import ablation_study as A
        assert A.CHOSEN_DS == ds
        frames = [pd.read_csv(f) for j in jobs if (f := JOBS / j["id"] / "ablation_rows.csv").exists()]
        results_df = pd.concat(frames, ignore_index=True)
        rd = REPO / "ablation_results"
        rd.mkdir(exist_ok=True)
        for f in rd.glob(f"{ds}_ablation_*"):
            backup(f)
        results_df.to_excel(rd / f"{ds}_ablation_raw_results.xlsx", index=False)
        A.generate_visualizations(results_df, rd)
        print(f"{ds} ablation: {len(results_df)} rows -> {rd}")
        return
    name = OUTPUT_NAME[step].format(ds=ds)
    frames = []
    if step == "rows":
        meta_path = PAR / f"plan_{ds}_rows.json"
        base = json.loads(meta_path.read_text())["base"] if meta_path.exists() else None
        if base:
            frames.append(pd.read_csv(base))
    for j in jobs:
        f = JOBS / j["id"] / name
        if f.exists():
            frames.append(pd.read_excel(f) if name.endswith(".xlsx") else pd.read_csv(f))
    out = pd.concat(frames, ignore_index=True)
    graphs, results = REPO / "graphs", REPO / "results"
    graphs.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    target = graphs / name
    backup(target)
    if name.endswith(".xlsx"):
        out.to_excel(target, index=False)
    else:
        out.to_csv(target, index=False)
    shutil.copy2(target, results / name)
    print(f"{ds} {step}: {len(out)} rows -> {target} and results/")
    if step in ("rows", "attrs"):
        mod = __import__("scalability_rows" if step == "rows" else "scalability_attrs")
        mod.generate_graphs()


def merge(datasets, steps, allow_partial):
    for ds in datasets:
        for step in steps:
            env = dict(os.environ, HOMOGENEITY_DATASET=ds)
            env.pop("HOMOGENEITY_OUT_DIR", None)
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "merge-one", ds, step]
            if allow_partial:
                cmd.append("--allow-partial")
            subprocess.run(cmd, cwd=REPO, env=env, check=False)


def _running_ids():
    """Ids of the jobs that have a live process now (read from `ps`, so jobs of a stopped run count as waiting)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    ids = set()
    for line in out.splitlines():
        if " worker " in line and str(JOBS) in line:
            ids.add(Path(line.split(" worker ", 1)[1].strip()).name)
        for ds in BENCH_OUT:
            if "run_all_benchmarks.py" in line and f"--dataset {ds} " in line + " ":
                ids.add(f"{ds}-benchmarks")
        for ds in CONFIG["DATASETS"]:
            if "rw_accuracy.py" in line and f"--dataset {ds} " in line and "weighted_single" in line:
                ids.add(f"{ds}-rwsingle")
    return ids


def status():
    from collections import Counter
    c = Counter()
    running = _running_ids()
    for j in load_jobs():
        d = JOBS / j["id"]
        state = "done" if (d / "DONE").exists() else "failed" if (d / "FAILED").exists() else \
            "running" if j["id"] in running else "waiting"
        c[(j["dataset"], j["step"], state)] += 1
    keys = sorted({(ds, st) for ds, st, _ in c})
    print(f"{'dataset':8s} {'step':10s} {'done':>6s} {'running':>8s} {'waiting':>8s} {'failed':>7s}")
    for ds, st in keys:
        print(f"{ds:8s} {st:10s} {c[(ds, st, 'done')]:6d} {c[(ds, st, 'running')]:8d} {c[(ds, st, 'waiting')]:8d} "
              f"{c[(ds, st, 'failed')]:7d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--datasets", required=True)
    p.add_argument("--steps", required=True)
    p.add_argument("--keep-existing-rows", default="", help="datasets whose finished row-scalability runs are kept")
    p.add_argument("--algos", default="", help="plan only these algorithms (e.g. RW); ablation jobs skip FPGrowth")
    r = sub.add_parser("run")
    r.add_argument("--slots", type=int, default=36)
    r.add_argument("--adopt", action="store_true", help="watch, not restart, jobs still running from a stopped runner")
    r.add_argument("--only", default=None, help="run only the jobs whose id starts with this (e.g. so-ablation)")
    m = sub.add_parser("merge")
    m.add_argument("--datasets", required=True)
    m.add_argument("--steps", required=True)
    m.add_argument("--allow-partial", action="store_true")
    w = sub.add_parser("worker")
    w.add_argument("job_dir")
    mo = sub.add_parser("merge-one")
    mo.add_argument("dataset")
    mo.add_argument("step")
    mo.add_argument("--allow-partial", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.datasets.split(","), a.steps.split(","), [x for x in a.keep_existing_rows.split(",") if x],
             [x for x in a.algos.split(",") if x])
    elif a.cmd == "run":
        sys.exit(1 if run(a.slots, a.adopt, a.only) else 0)
    elif a.cmd == "merge":
        merge(a.datasets.split(","), a.steps.split(","), a.allow_partial)
    elif a.cmd == "merge-one":
        merge_one(a.dataset, a.step, a.allow_partial)
    elif a.cmd == "worker":
        worker(a.job_dir)
    else:
        status()


if __name__ == "__main__":
    main()
