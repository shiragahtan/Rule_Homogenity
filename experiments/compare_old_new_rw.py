"""Compare the old RW (run in ~/Rule_Homogenity) with the new RW (RW_OPTIONS on, run in a copy ~/rw_new) on the
lab machine: main vs exact verdicts, ablation / scalability vs the brute-force (FPGrowth) verdict of the same job.
Output: old_vs_new_rw.csv (copied to calibration_results/rw_old_vs_new_full_runs.csv)."""
import re
import pandas as pd, glob, os, json
OLD = os.path.expanduser("~/Rule_Homogenity/par_runs/jobs"); NEW = os.path.expanduser("~/rw_new/par_runs/jobs")
H = lambda s: str(s).strip() == "Homogeneous"
def rd(p): return pd.read_excel(p) if p.endswith("xlsx") else pd.read_csv(p)
def files(root, pat):
    for d in sorted(glob.glob(f"{root}/{pat}")):
        fs = [f for f in glob.glob(d + "/*") if f.endswith((".csv", ".xlsx"))]
        if fs and os.path.exists(d + "/DONE"):
            df = rd(fs[0])
            if "algorithm" in df and "ablation" not in d: df = df[df.algorithm.isin(["RW", "FPGrowth"])]
            yield os.path.basename(d), df
out = []
for ds in ["german", "acs", "so"]:
    ex = pd.read_csv(os.path.expanduser(f"~/Rule_Homogenity/calibration_results/exact_summary_{ds}.csv"))
    rules=[json.loads(l) for l in open(os.path.expanduser(f"~/rw_new/configs/{json.load(open(os.path.expanduser('~/rw_new/configs/config.json')))['DATASETS'][ds]['RULES_FILE'].split('/')[-1]}")) if l.strip()]
    if "rule_text" not in ex: ex["rule_text"]=[f"{list(r['condition'].items())[0][0]}={list(r['condition'].items())[0][1]} | {list(r['treatment'].items())[0][0]}={list(r['treatment'].items())[0][1]}" for r in rules]
    truth = {re.sub(r"\s","",t).replace(".0",""): h for t,h in zip(ex.rule_text, ex.min_eps_new_final <= ex.epsilon)}
    # MAIN vs exact
    for tag, root in [("old", OLD), ("new", NEW)]:
        ok = n = nh = nhf = 0; t = []
        for jid, df in files(root, f"{ds}-main-3RW-r*"):
            for s, sec, c, t_ in zip(df.homogeneity_status, df.run_time_seconds, df.condition, df.treatment):
                (ca,cv),=eval(c).items(); (ta,tv),=eval(t_).items(); tr = bool(truth[re.sub(r"\s","",f"{ca}={cv}|{ta}={tv}").replace(".0","")])
                if s not in ("Homogeneous", "NOT Homogeneous"): continue
                n += 1; ok += (H(s) == tr); t.append(sec)
                if not tr: nh += 1; nhf += (not H(s))
        out.append(dict(ds=ds, exp="main vs exact", rw=tag, n=n, correct=ok, pct=round(100*ok/max(n,1),1),
                        nonH=nh, nonH_found=nhf, fp_wrong=n-ok-(nh-nhf), med_sec=round(pd.Series(t).median(),1) if t else None))
    # ROWS / ATTRS vs FPGrowth of the same job
    for step in ["rows", "attrs"]:
        fp = {}
        for jid, df in files(OLD, f"{ds}-{step}-*-FPGrowth-r*"):
            s = [x for x in df.status if x in ("Homogeneous", "NOT Homogeneous")]
            if s: fp[jid.replace("FPGrowth", "RW")] = H(s[0])
        for tag, root in [("old", OLD), ("new", NEW)]:
            ok = n = nh = nhf = 0; t = []
            for jid, df in files(root, f"{ds}-{step}-*-RW-r*"):
                if jid not in fp: continue
                for s, sec in zip(df.status, df.run_time_seconds):
                    if s not in ("Homogeneous", "NOT Homogeneous"): continue
                    n += 1; ok += (H(s) == fp[jid]); t.append(sec)
                    if not fp[jid]: nh += 1; nhf += (not H(s))
            out.append(dict(ds=ds, exp=f"{step} vs FP", rw=tag, n=n, correct=ok, pct=round(100*ok/max(n,1),1) if n else None,
                            nonH=nh, nonH_found=nhf, fp_wrong=n-ok-(nh-nhf), med_sec=round(pd.Series(t).median(),1) if t else None))
    # ABLATION vs FPGrowth of the same job
    fp = {}; oldrw = {}
    for jid, df in files(OLD, f"{ds}-ablation-*"):
        f = df[df.algorithm == "FPGrowth"]
        if len(f) and str(f.is_homogeneous.iloc[0]) in ("True", "False"): fp[jid] = str(f.is_homogeneous.iloc[0]) == "True"
        oldrw[jid] = df[df.algorithm == "RW_Direct"]
    newrw = {jid: df[df.algorithm == "RW_Direct"] for jid, df in files(NEW, f"{ds}-ablation-*")}
    for tag, src in [("old", oldrw), ("new", newrw)]:
        ok = n = nh = nhf = 0; t = []
        for jid, df in src.items():
            if jid not in fp: continue
            for s, sec in zip(df.is_homogeneous, df.runtime_seconds):
                if str(s) not in ("True", "False"): continue
                h = str(s) == "True"; n += 1; ok += (h == fp[jid]); t.append(sec)
                if not fp[jid]: nh += 1; nhf += (not h)
        out.append(dict(ds=ds, exp="ablation vs FP", rw=tag, n=n, correct=ok, pct=round(100*ok/max(n,1),1) if n else None,
                        nonH=nh, nonH_found=nhf, fp_wrong=n-ok-(nh-nhf), med_sec=round(pd.Series(t).median(),1) if t else None))
r = pd.DataFrame(out); print(r.to_string()); r.to_csv(os.path.expanduser("~/rw_new/old_vs_new_rw.csv"), index=False)
