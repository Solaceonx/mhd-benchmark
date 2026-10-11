"""Read the logs of the RAMSES II speed runs and print one line per run.

    python3 summarize.py RESULTS_DIR [--json FILE]

RESULTS_DIR/<variant>/<solver>_L<level>/run.log (written by speed.sbatch). Per run: cells, steps done, cell-updates
per second from the median of the per-step times the code prints ("Time elapsed since last coarse step"), skipping
the first SKIP steps; the seconds all steps took; the peak GPU memory sampled during the run (gpu_mem.txt, MiB); and,
for validation against the sweep's throughput_ot3d records, the kinetic, magnetic and thermal energy at t = 0.05.

The code does not shorten its last step to end at t = 0.05, so the energies are interpolated in time (a parabola
through the three steps nearest t = 0.05). It prints, at step k, the energies of the state at the time of the
"Fine step= k" line that follows; "ekin" in its "Main step" line is the total energy (kinetic + thermal + magnetic).
"""
import json
import re
import statistics
import sys
from pathlib import Path

SKIP = 5
T_END = 0.05
NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


def at(t, y, t0):
    """y at t0 from the parabola through the three points nearest t0 (linear with two)."""
    order = sorted(range(len(t)), key=lambda i: abs(t[i] - t0))[:3]
    if len(order) < 2:return None
    xs = [t[i] for i in order];ys = [y[i] for i in order];total = 0.0
    for i in range(len(xs)):
        w = 1.0
        for j in range(len(xs)):
            if i != j:w *= (t0 - xs[j])/(xs[i] - xs[j])
        total += w*ys[i]
    return total


def read(run):
    text = (run/"run.log").read_text(errors="replace") if (run/"run.log").is_file() else ""
    times = [float(x) for x in re.findall(r"Time elapsed since last coarse step:\s*([0-9.eE+-]+)", text)]
    grids = re.findall(r"Level\s+(\d+) has (\d+) grids", text)
    out = dict(run=f"{run.parent.name}/{run.name}", completed="Run completed" in text, steps=len(times))
    if grids:
        level, n = max((int(a), int(b)) for a, b in grids);out["n"] = 2**level;out["cells"] = 8*n
    mem = run/"gpu_mem.txt"
    if mem.is_file():
        v = [int(x) for x in re.findall(r"^\s*(\d+)", mem.read_text(), re.M)]
        if v:out["gpu_mem_mib"] = max(v)
    use = times[SKIP:] if len(times) > SKIP + 2 else times
    if use and "cells" in out:
        out["step_seconds"] = statistics.median(use);out["cell_updates_per_second"] = out["cells"]/out["step_seconds"]
    pairs = re.findall(r"Main step=\s*\d+\s+mcons=\s*" + NUM + r"\s+econs=\s*" + NUM + r"\s+epot=\s*" + NUM + r"\s+ekin=\s*" + NUM + r"\s+eint=\s*" + NUM
                       + r"\s+emag=\s*" + NUM + r"\s*\n\s*Fine step=\s*\d+\s+t=\s*" + NUM, text)
    if len(pairs) >= 2:
        t = [float(x[6]) for x in pairs];out["t_last"] = t[-1]
        out["mass_drift"] = max(abs(float(x[0])) for x in pairs);out["energy_drift"] = max(abs(float(x[1])) for x in pairs)
        if t[-1] >= T_END:
            total, thermal, magnetic = (at(t, [float(x[i]) for x in pairs], T_END) for i in (3, 4, 5))
            out.update(energy=total, thermal=thermal, magnetic=magnetic, kinetic=total - thermal - magnetic)
    if times:out["seconds"] = sum(times)
    if not out["completed"]:
        bad = [l.strip() for l in text.splitlines() if re.search(r"error|fail|abort|nan|cuda|segmentation|killed", l, re.I)]
        out["problem"] = (bad[-1] if bad else (text.strip().splitlines() or ["no output"])[-1])[:160]
    return out


def main(argv):
    target = None
    if "--json" in argv:
        i = argv.index("--json");target = argv[i + 1];argv = argv[:i] + argv[i + 2:]
    if len(argv) != 1:sys.exit(__doc__)
    runs = sorted(p.parent for p in Path(argv[0]).glob("*/*/run.nml") if not p.parent.name.endswith("_dump"))
    rows = [read(r) for r in runs]
    print(f"{'run':24s} {'N':>4s} {'steps':>5s} {'Mcell-upd/s':>11s} {'seconds':>8s} {'GPU MiB':>8s}  {'kinetic':>11s} {'magnetic':>11s} {'thermal':>11s}  {'|dM|,|dE| max':>15s}")
    for r in rows:
        if "cell_updates_per_second" in r and r["completed"]:
            e = " ".join(f"{r[k]:11.8f}" if k in r else f"{'-':>11s}" for k in ("kinetic", "magnetic", "thermal"))
            print(f"{r['run']:24s} {r['n']:4d} {r['steps']:5d} {r['cell_updates_per_second']/1e6:11.1f} {r.get('seconds', 0):8.2f} {r.get('gpu_mem_mib', 0):8d}  {e}  "
                  f"{r.get('mass_drift', 0):7.1e} {r.get('energy_drift', 0):7.1e}")
        else:
            print(f"{r['run']:24s} {r.get('n', 0):4d} {r['steps']:5d}   DID NOT FINISH: {r.get('problem', '')}")
    if target:Path(target).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
