"""The runs of the official RAMSES II rerun (ramses2.sbatch) and their split into GPU tasks (run on the Mac).

    python3 sherlock_miniramses/make_tasks_ramses2.py [--hours 4]

Writes runs_ramses2.json: every run of the mini-ramses-ism campaign (runs.json: the sweep's RAMSES runs plus the speed
test), for three set-ups (matched HLL, best HLLD, uct UCT-HLLD) in two precisions (code ramses2 = double, ramses2f32 =
single). Each run's GPU seconds are estimated from what the same run took in that campaign (miniramses_results/,
double precision; uct from best), times 1.3 for UCT-HLLD and 0.45 for single precision (the speed test ran about 3x
faster in single at 512³, less on small grids); a run the campaign has no record of keeps its planned estimate. Tasks of
about --hours each, the longest runs first onto the least loaded task.
"""
import argparse
import json
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/"miniramses_results"
UCT, F32 = 1.3, 0.45


def old_seconds(test, variant, n, seed):
    recs = sorted((OLD/f"{test}-miniramses-{variant}-run-N{n}-s{seed}"/"data").glob("miniramses_*_r*/run.toml"))
    return sum(float(tomllib.loads(r.read_text()).get("wall_seconds") or 0) for r in recs) or None


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--hours", type=float, default=4.0);a = p.parse_args()
    base = json.loads((HERE/"runs.json").read_text())["runs"];runs = []
    for code, fac in (("ramses2", 1.0), ("ramses2f32", F32)):
        for variant in ("matched", "best", "uct"):
            for r in base:
                if r["variant"] != ("best" if variant == "uct" else variant):continue
                old = old_seconds(r["test"], r["variant"], r["n"], r["seed"])
                est = (old if old else r["est_seconds"])*fac*(UCT if variant == "uct" else 1.0)
                runs.append(dict(test=r["test"], code=code, variant=variant, n=r["n"], seed=r["seed"], geometry=r["geometry"],
                                 repeats=r["repeats"], est_seconds=round(est, 1), from_record=bool(old),
                                 dir=f'{r["test"]}-{code}-{variant}-run-N{r["n"]}-s{r["seed"]}'))
    total = sum(r["est_seconds"] for r in runs);k = max(1, round(total/3600/a.hours))
    load = [0.0]*k;tasks = [[] for _ in range(k)]
    for i in sorted(range(len(runs)), key=lambda i: -runs[i]["est_seconds"]):
        j = min(range(k), key=load.__getitem__);tasks[j].append(i);load[j] += runs[i]["est_seconds"]
    (HERE/"runs_ramses2.json").write_text(json.dumps(dict(runs=runs, tasks=tasks), indent=1) + "\n")
    by = {}
    for r in runs:by[(r["code"], r["variant"])] = by.get((r["code"], r["variant"]), 0) + r["est_seconds"]/3600
    print(f"{len(runs)} runs, about {total/3600:.0f} GPU hours ({', '.join(f'{c} {v} {h:.0f}' for (c, v), h in by.items())}); "
          f"{k} tasks of {min(load)/3600:.1f}-{max(load)/3600:.1f} h -> {HERE/'runs_ramses2.json'}")


if __name__ == "__main__":
    main()
