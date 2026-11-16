"""The single-precision pilot: every pilot run must be complete, and its end-time errors (exact-solution tests) within
25 % of the same run in double precision, or within FLOOR of it: a difference below 1e-5 is single precision's own
round-off (job 52355180: the field loop's density error, 1e-7 in fp64, is 3-8x larger in fp32; the builds are fine).
   python3 check_f32.py PILOT_RESULTS TASKS_FILE   (exit 1 if not)"""
import glob
import os
import sys
import tomllib

res, lines = sys.argv[1], [l for l in open(sys.argv[2]).read().split() if l]
roots = {"athenak": os.path.expandvars("$SCRATCH/mhd-results-sweep")}
bad = 0
FLOOR = 1e-5
for line in lines:
    code, variant, test, seed, grids, _ = line.split("|");base = code.removesuffix("f32")
    for n in grids.split(","):
        name = f"{test}-{code}-{variant}-run-N{n}-s{seed}"
        rec = glob.glob(f"{res}/{name}/data/*_r1/run.toml")
        if not rec:print(f"{name:52s} NO RECORD");bad = 1;continue
        m = tomllib.loads(open(rec[0]).read())
        ref = glob.glob(f"{roots.get(base, os.path.expandvars('$SCRATCH/mhd-results-newcodes'))}/{test}-{base}-{variant}-run-N{n}-s{seed}/data/*_r1/run.toml")
        r = tomllib.loads(open(ref[0]).read()) if ref else {}
        e = (m.get("frames") or [{}])[-1].get("errors", {});er = (r.get("frames") or [{}])[-1].get("errors", {})
        out = f"{name:52s} {m.get('status'):9s} steps {m.get('steps')} (fp64 {r.get('steps', '?')})"
        if m.get("status") != "complete":bad = 1;out += "  | " + str(m.get("error", ""))[:120].replace("\n", " ")
        for q in ("density_relative_L1", "magnetic_relative_L1"):
            if e.get(q) and er.get(q):
                ratio = e[q]/er[q];out += f"  {q.split('_')[0]} {e[q]:.4g} / fp64 {er[q]:.4g} = {ratio:.3f}"
                if not 0.8 <= ratio <= 1.25:
                    floor = abs(e[q] - er[q]) < FLOOR;bad |= not floor;out += " (round-off)" if floor else " (!)"
        print(out)
print("PILOT PASSED" if not bad else "PILOT FAILED");sys.exit(bad)
