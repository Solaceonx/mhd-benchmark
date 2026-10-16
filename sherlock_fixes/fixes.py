#!/usr/bin/env python3
"""Runs of the sweep-fixes branch of Tom Abel's kernels beside the sweep's (README.md).

    python3 fixes.py configs                     write configs/ from ../sherlock_sweep/configs (when MATRIX changes)
    python3 fixes.py plan [EXPORT]               list the runs; with the sweep's export (progress/sweep_status/
                                                 export_latest.json) also the GPU time each should take
    python3 fixes.py runs RESULTS [--trial] [--ok FILE]      the driver's run list (CONFIG OUTDIR N SEED METHODS)
    python3 fixes.py verdict RESULTS             judge the trial; writes RESULTS/trial_ok.txt; exit 1 if nothing ran
    python3 fixes.py summary RESULTS [SWEEP_RESULTS]         table of the runs, beside the sweep's original kernels
"""
import csv
import json
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/"sherlock_sweep"/"configs"
FIX = ["glm_fix", "glm_fix_hll16", "glm_fix_hlld", "glm_fix_ppm", "glm_fct16", "glm_fct32", "ct_fix"]
ORIGINAL = {"glm_fix": "glm", "glm_fix_hll16": "glm_hll16", "glm_fix_hlld": "glm_hlld", "glm_fix_ppm": "glm_ppm",
            "glm_fct16": "glm_hll16", "glm_fct32": "glm_hlld", "ct_fix": "ct"}     # what each is set beside
SAME_JOB = ["glm", "glm_hll16", "glm_ppm"]     # originals the branch still runs bit for bit (speed in the same job)
F16 = ["glm_fix_hll16", "glm_fix_ppm", "glm_fct16"]
TRIAL = FIX + ["glm_hll16"]
TRIAL_GRIDS = [32, 48]                          # 48: the 6-cell tile


def grid(ns, seeds=(42,)):
    return [(n, s) for n in ns for s in seeds]


# test, [(N, seed)], kernels. The Courant number and every other setting are the sweep's; one repeat.
MATRIX = [
    # convergence on smooth problems: Float16 storage and the CT prototype
    ("entropy", grid([32, 64, 128, 256]), F16 + ["glm_fct32", "ct_fix"]),
    ("entropy", grid([512]), ["glm_fix_hll16", "glm_fix_ppm"]),
    ("alfven", grid([32, 64, 128, 256]), F16 + ["glm_fct32", "ct_fix"]),
    # Float16 range and c_h on shortened steps (every Float16 kernel stopped on both magnetised blasts)
    ("blast_2d_b50", grid([64, 128, 256]), F16 + ["glm_fix"]),
    ("blast_2d_b100", grid([64, 128, 256]), F16 + ["glm_fix", "glm_fct32"]),
    ("blast_3d_b100", grid([64, 128]), F16 + ["glm_fix_hlld"]),
    # the HLLD guard
    ("current_sheet", grid([64, 128, 256]), ["glm_fix_hlld", "glm_fct16"]),
    ("gem", grid([128, 256]), ["glm_fix_hlld", "glm_fix_hll16", "glm_fct16"]),
    # Mach 9: Float16 range, HLLD
    ("decay_supersonic", grid([64, 128, 256]), F16 + ["glm_fix_hlld"]),
    # leakage into the field-free vortex, and the field loop: cell-centred CT
    ("kh_localized", grid([128, 256]), ["glm_fix_hll16", "glm_fct16", "glm_fct32", "ct_fix"]),
    ("field_loop", grid([64, 128, 256]), ["glm_fix_hll16", "glm_fct16", "glm_fct32", "ct_fix"]),
    # shocks everywhere, 2D and 3D: nothing may get worse
    ("orszag_tang", grid([128, 256]), FIX),
    ("orszag_tang_3d", grid([128, 256]), FIX),
    # speed, with the originals in the same job
    ("throughput_ot3d", grid([128, 256, 384, 512]), SAME_JOB + FIX),
]


def runs():
    """(config file, run directory, N, seed, kernels), cheapest first."""
    out = {}
    for test, cells, kernels in MATRIX:
        for n, seed in cells:
            key = (n, test, seed)
            if key in out:sys.exit(f"MATRIX names {test} N={n} seed {seed} twice")
            out[key] = (f"configs/{test}.toml", f"{test}-N{n}-s{seed}", n, seed, list(kernels))
    return [out[k] for k in sorted(out)]


def trial_runs():
    return [("configs/trial.toml", f"trial-N{n}", n, 42, list(TRIAL)) for n in TRIAL_GRIDS]


def setting(text, key, value):
    """Replace the one line `key = ...` of a config (value None: remove it)."""
    text, n = re.subn(rf"^{key} *=.*\n", "" if value is None else f"{key} = {value}\n", text, flags=re.M)
    if n != 1:sys.exit(f"expected exactly one line '{key} = ...', found {n}")
    return text


def quoted(ks):
    return "[" + ", ".join(f'"{k}"' for k in ks) + "]"


def write_configs():
    (HERE/"configs").mkdir(exist_ok=True)
    for old in (HERE/"configs").glob("*.toml"):old.unlink()
    made = []
    for test in dict.fromkeys(t for t, _, _ in MATRIX):
        mine = [(cells, k) for t, cells, k in MATRIX if t == test]
        ns = sorted({n for cells, _ in mine for n, _ in cells});seeds = sorted({s for cells, _ in mine for _, s in cells})
        kernels = list(dict.fromkeys(k for _, ks in mine for k in ks))
        text = (SOURCE/f"{test}.toml").read_text()
        text = setting(text, "grids", ns);text = setting(text, "seeds", seeds);text = setting(text, "methods", quoted(kernels))
        text = setting(text, "repeats", 1)
        if re.search(r"^repeats_by_grid *=", text, flags=re.M):text = setting(text, "repeats_by_grid", None)
        text = f"# sherlock_fixes: {test} with the sweep-fixes kernels (fixes.py configs). Everything else as in the sweep's\n# config, which follows.\n" + text
        (HERE/"configs"/f"{test}.toml").write_text(text);tomllib.loads(text);made.append(test)
    text = (SOURCE/"orszag_tang.toml").read_text()
    for key, value in (("t_end", 0.05), ("frames", 3), ("grids", TRIAL_GRIDS), ("repeats", 1), ("repeats_by_grid", None),
                       ("snapshot_frames_by_grid", None), ("methods", quoted(TRIAL))):
        text = setting(text, key, value)
    text = "# sherlock_fixes: the trial (fixes.py configs). Orszag-Tang to t = 0.05 with every new kernel.\n" + text
    (HERE/"configs"/"trial.toml").write_text(text);tomllib.loads(text)
    print(f"wrote {len(made) + 1} configs: trial, " + ", ".join(made))


def estimate(export):
    """GPU seconds of each planned run from the sweep's records of the original kernel it is set beside."""
    rec = json.loads(Path(export).read_text())["runs"]
    slower = {"glm_fct16": 1.25, "glm_fct32": 1.25, "ct_fix": 2.5, "glm_fix_hll16": 1.1, "glm_fix_ppm": 1.1}
    out = {}
    for config, rundir, n, seed, kernels in runs():
        test = Path(config).stem;prefix = f"{test}-gpu-N{n}-s{seed}-"
        parts = {p["part"]: p for r in rec if r["dir"].startswith(prefix) for p in r.get("parts", [])}
        steps = max([p.get("steps") or 0 for p in parts.values() if p.get("status") == "complete"] or [0])
        for k in kernels:
            p = parts.get(f"{ORIGINAL.get(k, k)}_r1") or {};rate = p.get("rate") or (parts.get("glm_hlld_r1") or {}).get("rate") or 0
            out[(rundir, k)] = 30 + (steps*n**3/(rate*1e6)*slower.get(k, 1.0) if steps and rate else float("nan"))
    return out


def tile(n):
    return 6 if n % 6 == 0 else 4


def cmd_runs(argv):
    results = Path(argv[0]);trial = "--trial" in argv
    ok = None
    if "--ok" in argv:ok = set(Path(argv[argv.index("--ok") + 1]).read_text().split())
    for config, rundir, n, seed, kernels in (trial_runs() if trial else runs()):
        if ok is not None:kernels = [k for k in kernels if k in ok]
        if kernels:print(config, results/rundir/"data", n, seed, ",".join(kernels))


def record(path):
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None


def cmd_verdict(argv):
    results = Path(argv[0]);ran = {k: 0 for k in TRIAL}
    for n in TRIAL_GRIDS:
        data = results/f"trial-N{n}"/"data"
        for k in TRIAL:
            r = record(data/f"{k}_r1"/"run.toml")
            if r and r.get("status") == "complete":
                ran[k] += 1;print(f"   N={n}: {k} ran {r.get('steps')} steps to t = {r.get('time_reached')}")
            else:
                why = (r or {}).get("error", (r or {}).get("status", "no record"))
                print(f"   N={n}: {k} did NOT run: {str(why).splitlines()[0][:200] if why else why}")
    ok = [k for k in TRIAL if ran[k] == len(TRIAL_GRIDS)]
    (results/"trial_ok.txt").write_text(" ".join(ok) + "\n")
    print(f"   kernels that ran on both grids: {' '.join(ok) or 'none'}")
    if not [k for k in ok if k in FIX]:sys.exit(1)


def measure(r):
    if r is None:return None
    done = r.get("status") == "complete";err = None
    frames = r.get("frames") or []
    if frames and isinstance(frames[-1].get("errors"), dict):
        e = frames[-1]["errors"];err = e.get("density_relative_L1") or e.get("magnetic_relative_L1") or None
    return {"done": done, "t": r.get("time_reached"), "steps": r.get("steps"), "seconds": r.get("solve_seconds"),
            "rate": r.get("million_cell_steps_per_second"), "err": err}


def cell(m):
    if m is None:return f"{'':>30}"
    if not m["done"]:return f"{'stopped at t = %.4g' % (m['t'] or 0):>30}"
    err = f"{m['err']:.2e}" if m["err"] else "       -"
    return f"{err:>9} {m['steps'] or 0:>7d} {m['seconds'] or 0:>8.1f}s {(m['rate'] or 0) / 1e3:>4.2f}"


def cmd_summary(argv):
    results = Path(argv[0]);sweep = Path(argv[1]) if len(argv) > 1 else None
    want = sum(len(k) for *_, k in runs());have = 0
    print("Each fixed kernel beside the original it changes. Columns: error at the end (density, else magnetic, relative")
    print("L1), steps, seconds in the solver, 10^9 cell updates per second. 'sweep' = the sweep's record of the original.")
    for test in dict.fromkeys(t for t, *_ in MATRIX):
        print(f"\n{test}\n{'kernel':<14}{'N':>5}  {'fixed':>30}   {'original':<9}{'':>21}")
        for config, rundir, n, seed, kernels in runs():
            if Path(config).stem != test:continue
            for k in kernels:
                m = measure(record(results/rundir/"data"/f"{k}_r1"/"run.toml"));have += m is not None
                o = ORIGINAL.get(k);om = None;src = ""
                if o and o in kernels:om = measure(record(results/rundir/"data"/f"{o}_r1"/"run.toml"));src = "same job"
                elif o and sweep:
                    found = [d for d in (f"{test}-gpu-N{n}-s{seed}-all", f"{test}-gpu-N{n}-s{seed}-{o}") if (sweep/d/"data"/f"{o}_r1"/"run.toml").is_file()]
                    om = measure(record(sweep/found[0]/"data"/f"{o}_r1"/"run.toml")) if found else None;src = "sweep"
                if m is None and om is None:continue
                print(f"{k:<14}{n:>5}  {cell(m):>30}   {(o or '') + ' (' + src + ')' if o else '':<22} {cell(om) if o else ''}")
    print(f"\n{have} of {want} runs recorded.")


def main(argv):
    if not argv:sys.exit(__doc__)
    command, rest = argv[0], argv[1:]
    if command == "configs":write_configs()
    elif command == "plan":
        est = estimate(rest[0]) if rest else {}
        total = 0.0
        for config, rundir, n, seed, kernels in runs():
            s = sum(est.get((rundir, k), 0.0) for k in kernels);total += s if s == s else 0
            print(f"{rundir:<32} {','.join(kernels)}" + (f"   ~{s/60:.0f} min" if est and s == s else ("   (no record to estimate from)" if est else "")))
        print(f"{len(runs())} run directories, {sum(len(k) for *_, k in runs())} runs" + (f", about {total/3600:.1f} GPU hours from the sweep's rates" if est else ""))
    elif command == "runs" and rest:cmd_runs(rest)
    elif command == "verdict" and rest:cmd_verdict(rest)
    elif command == "summary" and rest:cmd_summary(rest)
    else:sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
