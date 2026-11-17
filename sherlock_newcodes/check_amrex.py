"""The Castro/Nyx trial (amrex_trial.sbatch): each run's state, steps, end-time errors (exact-solution tests) and energy
drift, beside the same test and grid in codes already in the suite.   python3 check_amrex.py RESULTS TASKS_FILE"""
import glob
import os
import sys
import tomllib

res, lines = sys.argv[1], [l for l in open(sys.argv[2]).read().split() if l]
S = os.path.expandvars("$SCRATCH")
OTHERS = (("cholla plmc", f"{S}/mhd-results-newcodes", "cholla-plmc"), ("athenapk matched", f"{S}/mhd-results-newcodes", "athenapk-matched"),
          ("athenak matched", f"{S}/mhd-results-sweep", "athenak-matched"))
bad = 0


def summary(rec):
    m = tomllib.loads(open(rec).read());fr = m.get("frames") or [{}]
    e = fr[-1].get("errors", {});out = f"{m.get('status', '?'):9s} steps {m.get('steps', '?')}"
    for q in ("density_relative_L1", "magnetic_relative_L1"):
        if e.get(q):out += f"  {q.split('_')[0]} {e[q]:.3e}"
    return m, out


for line in lines:
    code, variant, test, seed, grids, _ = line.split("|")
    for n in grids.split(","):
        name = f"{test}-{code}-{variant}-run-N{n}-s{seed}";rec = glob.glob(f"{res}/{name}/data/*_r1/run.toml")
        if not rec:print(f"{name:44s} NO RECORD");bad = 1;continue
        m, out = summary(rec[0])
        if m.get("status") != "complete":bad = 1;out += "  | " + str(m.get("error", ""))[:160].replace("\n", " ")
        print(f"{name:44s} {out}")
        for label, root, tag in OTHERS:
            o = glob.glob(f"{root}/{test}-{tag}-run-N{n}-s{seed}/data/*_r1/run.toml")
            if o:print(f"   {label:18s} {summary(o[0])[1]}")
print("TRIAL PASSED" if not bad else "TRIAL FAILED");sys.exit(bad)
