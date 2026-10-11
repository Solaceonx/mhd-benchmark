#!/usr/bin/env python3
"""Table of the driven-turbulence runs:  python3 summarize.py RESULTS [--config config.toml] [--from T]

For every run directory (run.toml + series.csv) the time averages over time/T >= the config's steady_from_T, and the
steady-state check the averages rest on. Also written to RESULTS/summary.csv. Standard library only.

Steady: the mean rms Mach number of the second half of that window differs from the first half's by less than
STEADY (10 %). The two halves of a truly steady run differ by a few per cent by chance (the Mach number wanders by
"fluct" on a time scale of about T), so read the number, not only the verdict; the two seeds of the GPU runs show
how much two realisations of the same driving differ.

state: steady / NOT STEADY for a finished run; failed: the solver stopped (the reason is printed below the table);
measuring: RAMSES is done and its frames are being measured now, the line shows what is measured so far;
unmeasured: that measurement stopped; too short: fewer than four measured times in the window.

Columns
  Mach      rms Mach number, volume-weighted, sqrt(<v^2>)/cs with the sound speed of the gas at that time
  halves    its mean over the second half of the window against the first half
  fluct     its standard deviation over the window, relative to the mean
  M_mass    mass-weighted rms Mach number, sqrt(<rho v^2>/<rho>)/cs
  sig_s     standard deviation of s = ln(rho/<rho>), volume-weighted
  b         sqrt(exp(sig_s^2) - 1)/Mach: about 1/3 for solenoidal and 1 for compressive driving in the literature
  comp      <(div v)^2>/(<|curl v|^2> + <(div v)^2>), centred differences on the grid
  eps       energy injection rate = slope of the total energy against time; in a steady state also the dissipation rate
  C_eps     eps L/(<rho> v^3) with v the mass-weighted rms velocity
  Re_eff    v L/nu with nu = eps/(<rho |curl v|^2> + 4/3 <rho (div v)^2>): the viscosity that would dissipate eps
            through the velocity gradients the grid holds. It grows with resolution; at one resolution a larger value
            means sharper gradients for the same dissipation, i.e. a less diffusive scheme.
  cs_end    sound speed at the end (1 at the start): how isothermal the gas stayed
  rho_min   lowest density in the window; "floor" counts the largest number of cells at or below the density floor
            (1e-6 of the mean). A cell below the floor is measured at the floor, as both codes treat it when they form
            velocities; runs in which the density reached zero or less are listed under the table
  Mupd/s    million cell updates per second (GPU runs: the whole device; RAMSES: per MPI rank)
"""
import argparse, csv, math, sys, time, tomllib
from pathlib import Path

STEADY = 0.10


def mean(x):
    return sum(x)/len(x) if x else float("nan")


def slope(t, y):
    """Least-squares slope of y against t."""
    if len(t) < 2:return float("nan")
    tm, ym = mean(t), mean(y);den = sum((a - tm)**2 for a in t)
    return sum((a - tm)*(b - ym) for a, b in zip(t, y))/den if den > 0 else float("nan")


def summarize(run, start):
    meta = tomllib.loads((run/"run.toml").read_text())
    row = dict(run=run.name, code=meta.get("code", "?"), method=meta.get("method", "?"), N=meta.get("N", 0), mach_target=meta.get("mach_target", float("nan")),
               seed=meta.get("seed", 0), status=meta.get("status", "?"), a_rms=meta.get("a_rms", float("nan")))
    series = run/"series.csv"
    if meta.get("code") == "ramses" and not meta.get("measured"):
        # RAMSES has finished and its frames are being measured one by one (a line of series.csv each), unless
        # nothing was written for a while: then the measurement stopped, and running the job again repeats it.
        fresh = series.is_file() and time.time() - series.stat().st_mtime < 900
        row["status"] = "measuring" if fresh else "unmeasured"
    rows = list(csv.DictReader(series.open())) if series.is_file() else []
    col = lambda name, rs: [float(r[name]) for r in rs]
    row["reached_T"] = float(rows[-1]["time_over_T"]) if rows else 0.0
    # cells whose density the code let fall to zero or below (RAMSES at high Mach numbers): how many at most, and from when
    negative = [(float(r["time_over_T"]), int(float(r.get("negative_density_cells") or (float(r["rho_min"]) <= 0)))) for r in rows]
    row["negative_cells"] = max((n for _, n in negative), default=0)
    row["negative_from_T"] = next((t for t, n in negative if n > 0), float("nan"))
    speed = meta.get("cell_updates_per_second") or meta.get("perf", {}).get("cell_updates_per_second_per_rank")
    row["mupd_per_s"] = speed/1e6 if speed else float("nan")
    win = [r for r in rows if float(r["time_over_T"]) >= start - 1e-9]
    if len(win) < 4:
        row["verdict"] = "too short";return row
    m = col("mach_vol", win);half = len(m)//2;mm = mean(m)
    row.update(mach=mm, halves=mean(m[half:])/mean(m[:half]) - 1, fluct=math.sqrt(mean([(x - mm)**2 for x in m]))/mm,
               mach_mass=mean(col("mach_mass", win)), sigma_s=mean(col("sigma_s", win)), sigma_rho=mean(col("sigma_rho", win)),
               cs_end=float(win[-1]["cs"]), rho_min=min(col("rho_min", win)), rho_max=max(col("rho_max", win)),
               floor_cells=int(max(col("density_floor_cells", win))), points=len(win))
    row["b"] = math.sqrt(math.expm1(row["sigma_s"]**2))/mm if mm > 0 else float("nan")
    w2, d2 = mean(col("enstrophy", win)), mean(col("dilatation", win));row["comp"] = d2/(w2 + d2) if w2 + d2 > 0 else float("nan")
    eps = slope(col("time", win), col("energy", win));rho = mean(col("mass", win))
    v = mean([float(r["mach_mass"])*float(r["cs"]) for r in win])
    grad = mean(col("enstrophy_rho", win)) + 4.0/3.0*mean(col("dilatation_rho", win))
    row.update(eps=eps, c_eps=eps/(rho*v**3) if v > 0 else float("nan"), re_eff=v*grad/eps if eps > 0 else float("nan"))
    row["verdict"] = ("steady" if abs(row["halves"]) < STEADY else "NOT STEADY") if row["status"] == "complete" else row["status"]
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path);p.add_argument("--config", type=Path);p.add_argument("--from", dest="start", type=float)
    args = p.parse_args()
    start = args.start if args.start is not None else (tomllib.loads(args.config.read_text())["steady_from_T"] if args.config else 2.0)
    runs = sorted(d for d in args.results.iterdir() if d.is_dir() and (d/"run.toml").is_file()) if args.results.is_dir() else []
    if not runs:
        print(f"   no runs in {args.results} yet");return 0
    rows = [summarize(r, start) for r in runs]
    rows.sort(key=lambda r: (float(r["mach_target"]), r["N"], r["run"]))
    f = lambda v, fmt: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)
    print(f"averages over time/T >= {start:g}; steady: the two halves of that window within {100*STEADY:.0f} %")
    head = f"{'run':34s} {'state':>10s} {'to T':>5s} {'Mach':>6s} {'halves':>7s} {'fluct':>6s} {'M_mass':>6s} {'sig_s':>6s} {'b':>5s} {'comp':>5s} {'eps':>8s} {'C_eps':>6s} {'Re_eff':>7s} {'cs_end':>7s} {'rho_min':>8s} {'floor':>5s} {'Mupd/s':>7s}"
    print(head);last = None
    for r in rows:
        group = (r["mach_target"], r["N"])
        if last is not None and group != last:print()
        last = group
        print(f"{r['run']:34s} {r.get('verdict', r['status']):>10s} {f(r['reached_T'], '.2f'):>5s} {f(r.get('mach'), '.2f'):>6s} "
              f"{f(100*r['halves'] if 'halves' in r else None, '+.1f') + ('%' if 'halves' in r else ''):>7s} {f(100*r['fluct'] if 'fluct' in r else None, '.1f') + ('%' if 'fluct' in r else ''):>6s} "
              f"{f(r.get('mach_mass'), '.2f'):>6s} {f(r.get('sigma_s'), '.3f'):>6s} {f(r.get('b'), '.2f'):>5s} {f(r.get('comp'), '.2f'):>5s} {f(r.get('eps'), '.3g'):>8s} "
              f"{f(r.get('c_eps'), '.2f'):>6s} {f(r.get('re_eff'), '.0f'):>7s} {f(r.get('cs_end'), '.4f'):>7s} {f(r.get('rho_min'), '.1e'):>8s} {f(r.get('floor_cells'), 'd'):>5s} {f(r['mupd_per_s'], '.3g'):>7s}")
    keys = ["run", "code", "method", "N", "mach_target", "seed", "status", "verdict", "a_rms", "reached_T", "points", "mach", "halves", "fluct", "mach_mass", "sigma_s", "sigma_rho",
            "b", "comp", "eps", "c_eps", "re_eff", "cs_end", "rho_min", "rho_max", "floor_cells", "negative_cells", "negative_from_T", "mupd_per_s"]
    try:
        with (args.results/"summary.csv").open("w", newline="") as out:
            w = csv.DictWriter(out, fieldnames=keys);w.writeheader()
            for r in rows:w.writerow({k: r.get(k, "") for k in keys})
    except OSError as err:
        print(f"   (summary.csv not written: {err})")
    for r in rows:
        if r.get("negative_cells"):
            print(f"   {r['run']}: density at or below zero in up to {r['negative_cells']} cells, first at {r['negative_from_T']:.2f} T "
                  "(measured at the density floor, as the code treats them)")
    failed = [r for r in rows if r["status"] == "failed"]
    for r in failed:
        reason = tomllib.loads((args.results/r["run"]/"run.toml").read_text()).get("reason", "")
        print(f"   {r['run']} stopped at {r['reached_T']:.2f} T: {' '.join(reason.split())[:260]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
