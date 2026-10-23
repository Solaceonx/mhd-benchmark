"""The mini-ramses-ism runs and their split into GPU tasks (run on the Mac, where the sweep's records are).

    python3 sherlock_miniramses/make_tasks.py [--hours 6]

Writes sherlock_miniramses/runs.json: the runs of classic RAMSES in the sweep (the same tests, grids, seeds and
geometry, for both set-ups), plus the speed test throughput_ot3d at its power-of-two grids, which classic RAMSES did not
run; each with the GPU seconds it should take. The estimate: the steps classic RAMSES took on that run (the same
Courant factor), at 0.3 billion cell updates a second in 3D and 0.15 billion in 2D (RAMSES II in double precision on
an H100, sherlock_ramses2), plus 90 seconds for the start, the initial data and the measurements; a run classic RAMSES
did not finish is estimated from AthenaK's steps. Then tasks of about --hours each, the longest runs first onto the
least loaded task. miniramses.sbatch runs task k of tasks.json.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"scripts"))
import sweep_data as sd  # noqa: E402

RATE = {3: 0.3e9, 2: 0.15e9}
OVERHEAD = 90.0


def steps_of(sw, test, variant, n, seed):
    for label in (f"ramses_{variant}", "athenak_matched", "athenak_best"):
        m = sw.meta(test, label, n, seed)
        if m and m.get("steps") and m.get("status") == "complete":return int(m["steps"]), label
    for label in (f"ramses_{variant}", "athenak_matched"):
        m = sw.meta(test, label, n, seed)
        if m and m.get("steps") and m.get("time_reached"):
            return int(m["steps"]*float(sw.configs[test]["t_end"])/max(float(m["time_reached"]), 1e-12)), label + " (scaled)"
    return int(20*n), "guess"


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--hours", type=float, default=6.0);a = p.parse_args()
    sw = sd.Sweep();runs = []
    for r in sw.runs:
        if r["code"] != "ramses" or r["mode"] != "run":continue
        runs.append(dict(test=r["test"], variant=r["variant"], n=r["n"], seed=r["seed"], geometry=r["geometry"], repeats=1))
    c = sw.configs["throughput_ot3d"]
    for variant in ("matched", "best"):
        for n in c["grids"]:
            if n & (n - 1):continue                          # mini-ramses needs a power of two
            reps = int(c.get("repeats_by_grid", {}).get(str(n), c.get("repeats", 1)))
            runs.append(dict(test="throughput_ot3d", variant=variant, n=n, seed=42, geometry="cube", repeats=reps))
    for r in runs:
        ndim = 2 if r["geometry"] == "plane" else 3
        steps, source = steps_of(sw, r["test"], r["variant"], r["n"], r["seed"])
        r["est_seconds"] = round(r["repeats"]*(r["n"]**ndim*steps/RATE[ndim] + OVERHEAD), 1);r["steps_from"] = source
        r["dir"] = f'{r["test"]}-miniramses-{r["variant"]}-run-N{r["n"]}-s{r["seed"]}'
    total = sum(r["est_seconds"] for r in runs);k = max(1, round(total/3600/a.hours))
    load = [0.0]*k;tasks = [[] for _ in range(k)]
    for i in sorted(range(len(runs)), key=lambda i: -runs[i]["est_seconds"]):
        j = min(range(k), key=load.__getitem__);tasks[j].append(i);load[j] += runs[i]["est_seconds"]
    (HERE/"runs.json").write_text(json.dumps(dict(runs=runs, tasks=tasks), indent=1) + "\n")
    print(f"{len(runs)} runs, about {total/3600:.1f} GPU hours; {k} tasks of {min(load)/3600:.1f}-{max(load)/3600:.1f} h -> {HERE/'runs.json'}")


if __name__ == "__main__":
    main()
