#!/usr/bin/env python3
"""Runs of the sweep-fixes branch of Tom Abel's kernels beside the sweep's (README.md).

    python3 fixes.py configs                     write configs/ from ../sherlock_sweep/configs (when MATRIX changes)
    python3 fixes.py plan [EXPORT]               list the runs; with the sweep's export (progress/sweep_status/
                                                 export_latest.json) also the GPU time each should take
    python3 fixes.py runs RESULTS [--trial] [--ok FILE]      the driver's run list (CONFIG OUTDIR N SEED METHODS)
    python3 fixes.py verdict RESULTS             judge the trial; writes RESULTS/trial_ok.txt; exit 1 if nothing ran
    python3 fixes.py summary RESULTS [SWEEP_RESULTS]         table of the runs, beside the sweep's original kernels
    python3 fixes.py --plan verify control RESULTS SWEEP_RESULTS   originals on the branch against the sweep, digit by digit

--plan verify (default: fixes) selects the second set of runs, which checks the fixes rather than measuring them:
configs in configs_verify/, job mhd-verify (verify.sh), results $SCRATCH/mhd-results-verify.
--plan emfcheck: the CT prototype with its upwinded edge EMF on the tests that judge it (emfcheck.sh).
--plan rerun: the fixed kernels on every test of the sweep at every grid and seed, one run directory per kernel,
split into Slurm array tasks of similar length (rerun.sh; configs_rerun/estimates.json holds the expected seconds):
    python3 fixes.py --plan rerun configs EXPORT       write configs_rerun/ and the estimates (on the Mac)
    python3 fixes.py --plan rerun tasks KERNELS HOURS  how many array tasks of about HOURS each
    python3 fixes.py --plan rerun runs RESULTS --kernels KERNELS --task I --tasks K   the run list of task I
"""
import array
import csv
import json
import math
import statistics
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
    # the Float32 tile (sherlock_checks: as accurate as anchored Float16 and faster than any Float16 4-cell tile) beside
    # the Float16 kernels on one node; at 384 the Float16 kernels use their 6-cell tile, which a Float32 tile cannot
    ("throughput_f32", grid([128, 256, 512]), ["glm_hll32", "glm_hll16", "glm_fix_hll16", "glm_fct16"]),
    ("throughput_f32", grid([384]), ["glm_hll16", "glm_fix_hll16", "glm_fct16"]),
]
ALIAS = {"throughput_f32": "throughput_ot3d"}   # a second run directory set from the same sweep config
CONFIG_DIR = "configs"

# The checks of the fixes (job mhd-verify). Every setting is the sweep's; one repeat.
SHOCK = FIX                                      # every fixed kernel on the shock tests
VERIFY = [
    # 1. with the fixes switched off the branch is Tom's code: diagnostics.csv equal to the sweep's, digit by digit
    ("entropy", grid([128]), ["glm", "glm_hll16", "glm_ppm"]),
    ("orszag_tang", grid([128]), ["glm", "glm_hll16", "glm_ppm"]),
    # 2. the 6-cell tile (grids divisible by 6, the production default there): every convergence run so far was 4-cell
    ("entropy", grid([48, 96, 192, 384]), ["glm_hll16", "glm_fix_hll16", "glm_fix_ppm", "glm_fct16"]),
    ("alfven_oblique", grid([48, 96, 192, 384]), ["glm_hll16", "glm_fix_hll16", "glm_fix_ppm", "glm_fct16"]),
    # 3. the CT predictor and cell-centred CT in 2D: a wave oblique to the grid, and the in-plane field loop
    ("alfven_oblique", grid([32, 64, 128, 256]), ["ct_fix", "glm_fct16", "glm_fct32", "glm_fix_hll16", "glm_fix_ppm"]),
    ("field_loop_novz", grid([64, 128, 256]), ["ct_fix", "glm_fct16", "glm_fct32", "glm_fix_hll16"]),
    # 4. shocks: scored on the Mac against the AthenaK 4096^2 references (scripts/fix_accuracy.py)
    ("brio_wu", grid([128, 256]), SHOCK),
    ("brio_wu_rotated", grid([128, 256]), SHOCK),
    ("rotor", grid([128, 256]), SHOCK),
    # 5. robustness in 3D
    ("blast_3d_b50", grid([64, 128]), F16 + ["glm_fix_hlld"]),
    ("decay", grid([64, 128], seeds=(42, 43, 44)), ["glm_fix", "glm_fix_hll16", "glm_fix_hlld", "glm_fix_ppm", "glm_fct16"]),
    ("decay", grid([256]), ["glm_fix", "glm_fix_hll16", "glm_fix_hlld", "glm_fix_ppm", "glm_fct16"]),
]
CONTROL = [(t, n, k) for t, cells, ks in VERIFY[:2] for n, _ in cells for k in ks]
VTRIAL = FIX + ["glm", "glm_hll16", "glm_ppm"]

# The CT prototype with the upwinded edge EMF (2026-10-21) on the tests that judge an edge EMF, before its full rerun
EMFCHECK = [
    ("field_loop_novz", grid([64, 128, 256]), ["ct_fix"]),
    ("field_loop", grid([64, 128, 256]), ["ct_fix"]),
    ("alfven", grid([64, 128, 256]), ["ct_fix"]),
    ("alfven_oblique", grid([64, 128, 256]), ["ct_fix"]),
    ("entropy", grid([64, 128, 256]), ["ct_fix"]),
    ("kh_localized", grid([128, 256]), ["ct_fix"]),
    ("orszag_tang", grid([128, 256]), ["ct_fix"]),
    ("rotor", grid([128, 256]), ["ct_fix"]),
    ("brio_wu", grid([128, 256]), ["ct_fix"]),
]
# The full rerun: every sweep test (not the resume gates) at the sweep's grids and seeds, each fixed kernel
RERUN_GROUPS = {"glm": ["glm_fix_hll16", "glm_fct16", "glm_fix_ppm", "glm_fix_hlld"], "ct": ["ct_fix"]}
KEEP_REPEATS = {"throughput_ot3d"}             # timing tests keep the sweep's repeats; every other run once
SPLIT = False                                   # one run directory per kernel (the rerun)


def rerun_matrix():
    """From the shipped configs_rerun/ when it exists (so Sherlock plans exactly what was sent), else the sweep's."""
    out = [];src = HERE/"configs_rerun" if (HERE/"configs_rerun").is_dir() and any((HERE/"configs_rerun").glob("*.toml")) else SOURCE
    for f in sorted(src.glob("*.toml")):
        if f.stem.startswith("gate"):continue
        c = tomllib.loads(f.read_text())
        out.append((f.stem, grid(c["grids"], c["seeds"]), RERUN_GROUPS["glm"] + RERUN_GROUPS["ct"]))
    return out


def use_plan(name):
    global MATRIX, TRIAL, CONFIG_DIR, SPLIT
    if name == "verify":MATRIX, TRIAL, CONFIG_DIR = VERIFY, VTRIAL, "configs_verify"
    elif name == "emfcheck":MATRIX, TRIAL, CONFIG_DIR = EMFCHECK, ["ct_fix"], "configs_emfcheck"
    elif name == "rerun":MATRIX, TRIAL, CONFIG_DIR, SPLIT = rerun_matrix(), [], "configs_rerun", True
    elif name != "fixes":sys.exit(f"unknown plan {name}")


def runs():
    """(config file, run directory, N, seed, kernels), cheapest first."""
    out = {}
    for test, cells, kernels in MATRIX:
        for n, seed in cells:
            for ks in ([[k] for k in kernels] if SPLIT else [kernels]):
                key = (n, test, seed, ks[0] if SPLIT else "")
                if key in out:sys.exit(f"MATRIX names {test} N={n} seed {seed} twice")
                out[key] = (f"{CONFIG_DIR}/{test}.toml", f"{test}-N{n}-s{seed}" + (f"-{ks[0]}" if SPLIT else ""), n, seed, list(ks))
    return [out[k] for k in sorted(out)]


def trial_runs():
    return [(f"{CONFIG_DIR}/trial.toml", f"trial-N{n}", n, 42, list(TRIAL)) for n in TRIAL_GRIDS]


def setting(text, key, value):
    """Replace the one line `key = ...` of a config (value None: remove it)."""
    text, n = re.subn(rf"^{key} *=.*\n", "" if value is None else f"{key} = {value}\n", text, flags=re.M)
    if n != 1:sys.exit(f"expected exactly one line '{key} = ...', found {n}")
    return text


def quoted(ks):
    return "[" + ", ".join(f'"{k}"' for k in ks) + "]"


def write_configs():
    out = HERE/CONFIG_DIR;out.mkdir(exist_ok=True)
    for old in out.glob("*.toml"):old.unlink()
    made = []
    for test in dict.fromkeys(t for t, _, _ in MATRIX):
        mine = [(cells, k) for t, cells, k in MATRIX if t == test]
        ns = sorted({n for cells, _ in mine for n, _ in cells});seeds = sorted({s for cells, _ in mine for _, s in cells})
        kernels = list(dict.fromkeys(k for _, ks in mine for k in ks))
        text = (SOURCE/f"{ALIAS.get(test, test)}.toml").read_text()
        text = setting(text, "grids", ns);text = setting(text, "seeds", seeds);text = setting(text, "methods", quoted(kernels))
        if not (SPLIT and test in KEEP_REPEATS):
            text = setting(text, "repeats", 1)
            if re.search(r"^repeats_by_grid *=", text, flags=re.M):text = setting(text, "repeats_by_grid", None)
        text = f"# sherlock_fixes: {test} with the sweep-fixes kernels (fixes.py configs). Everything else as in the sweep's\n# config, which follows.\n" + text
        (out/f"{test}.toml").write_text(text);tomllib.loads(text);made.append(test)
    if not TRIAL:
        print(f"wrote {len(made)} configs: " + ", ".join(made));return
    text = (SOURCE/"orszag_tang.toml").read_text()
    for key, value in (("t_end", 0.05), ("frames", 3), ("grids", TRIAL_GRIDS), ("repeats", 1), ("repeats_by_grid", None),
                       ("snapshot_frames_by_grid", None), ("methods", quoted(TRIAL))):
        text = setting(text, key, value)
    text = "# sherlock_fixes: the trial (fixes.py configs). Orszag-Tang to t = 0.05 with every new kernel.\n" + text
    (out/"trial.toml").write_text(text);tomllib.loads(text)
    print(f"wrote {len(made) + 1} configs: trial, " + ", ".join(made))


def estimate(export):
    """GPU seconds of each planned run from the sweep's records of the original kernel it is set beside."""
    rec = json.loads(Path(export).read_text())["runs"]
    # speed of each fixed kernel against its original, measured on one GPU (throughput_ot3d, 2026-10-16/19); ct_fix
    # with 15% more for the reference-EMF kernel added on 2026-10-21
    slower = {"glm_fct16": 1.37, "glm_fct32": 1.05, "ct_fix": 3.8, "glm_fix_hll16": 1.42, "glm_fix_ppm": 1.16, "glm_fix_hlld": 1.17}
    out = {}
    for config, rundir, n, seed, kernels in runs():
        test = Path(config).stem;prefix = f"{test}-gpu-N{n}-s{seed}-"
        parts = {p["part"]: p for r in rec if r["dir"].startswith(prefix) for p in r.get("parts", [])}
        steps = max([p.get("steps") or 0 for p in parts.values() if p.get("status") == "complete"] or [0])
        for k in kernels:
            p = parts.get(f"{ORIGINAL.get(k, k)}_r1") or {};rate = p.get("rate") or (parts.get("glm_hlld_r1") or {}).get("rate") or 0
            reps = 1
            if SPLIT and test in KEEP_REPEATS:
                c = tomllib.loads((SOURCE/f"{test}.toml").read_text());reps = int(c.get("repeats_by_grid", {}).get(str(n), c.get("repeats", 1)))
            out[(rundir, k)] = 30 + reps*(steps*n**3/(rate*1e6)*slower.get(k, 1.0) if steps and rate else float("nan"))
    return out


def estimates():
    """{"rundir|kernel": seconds} written by `configs` for the rerun (600 s where the sweep had no record)."""
    try:
        return json.loads((HERE/CONFIG_DIR/"estimates.json").read_text())
    except OSError:
        return {}


def assign(entries, ntasks):
    """Longest run first onto the least loaded task: [[entry, ...] per task]."""
    est = estimates();cost = lambda e: est.get(f"{e[1]}|{e[4][0]}", 600.0)
    tasks = [[] for _ in range(ntasks)];load = [0.0]*ntasks
    for e in sorted(entries, key=cost, reverse=True):
        i = min(range(ntasks), key=lambda t: load[t]);tasks[i].append(e);load[i] += cost(e)
    return tasks, load


def tile(n):
    return 6 if n % 6 == 0 else 4


def chosen(argv):
    """The kernels named by --kernels (a group of RERUN_GROUPS or a comma list), or None."""
    if "--kernels" not in argv:return None
    v = argv[argv.index("--kernels") + 1]
    return RERUN_GROUPS.get(v, v.split(","))


def cmd_progress(argv):
    """Per kernel: runs complete, stopped, started, not started (the rerun prints this instead of the long table)."""
    results = Path(argv[0]);count = {}
    for config, rundir, n, seed, kernels in runs():
        for k in kernels:
            r = record(results/rundir/"data"/f"{k}_r1"/"run.toml");c = count.setdefault(k, [0, 0, 0, 0])
            if r is None:c[3] += 1
            elif r.get("status") == "complete":c[0] += 1
            elif r.get("status") in ("failed", "stopped", "error"):c[1] += 1
            else:c[2] += 1
    print(f"{'kernel':<15}{'complete':>9}{'stopped':>9}{'running':>9}{'waiting':>9}")
    for k, c in count.items():print(f"{k:<15}" + "".join(f"{x:>9}" for x in c))


def cmd_tasks(argv):
    kernels = RERUN_GROUPS.get(argv[0], argv[0].split(","));hours = float(argv[1])
    est = estimates();mine = [e for e in runs() if e[4][0] in kernels]
    total = sum(est.get(f"{e[1]}|{e[4][0]}", 600.0) for e in mine)
    longest = max([est.get(f"{e[1]}|{e[4][0]}", 600.0) for e in mine] or [0])
    print(max(1, math.ceil(total/(hours*3600))))
    print(f"{len(mine)} runs, about {total/3600:.1f} GPU hours, longest run {longest/3600:.1f} h", file=sys.stderr)


def cmd_runs(argv):
    results = Path(argv[0]);trial = "--trial" in argv
    ok = None
    if "--ok" in argv:ok = set(Path(argv[argv.index("--ok") + 1]).read_text().split())
    entries = trial_runs() if trial else runs()
    if chosen(argv) is not None:entries = [e for e in entries if all(k in chosen(argv) for k in e[4])]
    if "--task" in argv:
        tasks, _ = assign(entries, int(argv[argv.index("--tasks") + 1]));entries = sorted(tasks[int(argv[argv.index("--task") + 1])], key=lambda e: (e[2], e[1]))
    for config, rundir, n, seed, kernels in entries:
        if ok is not None:kernels = [k for k in kernels if k not in TRIAL or k in ok]     # the trial judges only its own
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


# What the error column shows: the density where the wave is in the density, the field where it is in the field (their
# density is uniform and its error is rounding), and for kh_localized the field left in the field-free vortex.
METRIC = {"entropy": "density_relative_L1", "alfven": "magnetic_relative_L1", "field_loop": "magnetic_relative_L1"}
CORE = (0.0, 0.75, 0.06)        # centre and radius of the disc inside the upper vortex of kh_localized (paper_tomida.py)


def core_median(path, n):
    """Median |B| in the disc CORE of a planar state file (Float32, n x n x 1 x 9, column-major)."""
    try:
        a = array.array("f");a.frombytes(path.read_bytes())
    except OSError:
        return None
    if len(a) != 9*n*n:return None
    out = []
    for j in range(n):
        y = (j + .5)/n
        for i in range(n):
            x = (i + .5)/n;dx = min(abs(x - CORE[0]), 1 - abs(x - CORE[0]))
            if dx*dx + (y - CORE[1])**2 < CORE[2]**2:
                k = i + n*j;out.append(math.sqrt(a[k + 5*n*n]**2 + a[k + 6*n*n]**2 + a[k + 7*n*n]**2))
    return statistics.median(out) if out else None


def measure(r, test="", path=None, n=0):
    if r is None:return None
    done = r.get("status") == "complete";err = None
    frames = r.get("frames") or []
    if test == "kh_localized" and done and path is not None:
        last = sorted(path.glob("state_*.bin"));err = core_median(last[-1], n) if last else None
    elif frames and isinstance(frames[-1].get("errors"), dict):
        e = frames[-1]["errors"]
        for key in ([METRIC[test]] if test in METRIC else ["magnetic_relative_L1", "density_relative_L1"]):
            if e.get(key) is not None:err = e[key];break
    why = str(r.get("error") or "").strip().splitlines()
    return {"done": done, "t": r.get("time_reached"), "steps": r.get("steps"), "seconds": r.get("solve_seconds"),
            "rate": r.get("million_cell_steps_per_second"), "err": err, "why": why[0][:110] if why else ""}


def cell(m):
    if m is None:return f"{'':>30}"
    if not m["done"]:return f"{'stopped at t = %.4g' % (m['t'] or 0):>30}"
    err = f"{m['err']:.2e}" if m["err"] is not None else "       -"
    return f"{err:>9} {m['steps'] or 0:>7d} {m['seconds'] or 0:>8.1f}s {(m['rate'] or 0) / 1e3:>4.2f}"


def cmd_summary(argv):
    results = Path(argv[0]);sweep = Path(argv[1]) if len(argv) > 1 else None
    want = sum(len(k) for *_, k in runs());have = 0
    print("Each fixed kernel beside the original it changes. Columns: error at the end (relative L1 of the density for")
    print("entropy, of B for alfven and field_loop, else of B if recorded; kh_localized: median |B| in the field-free")
    print("vortex, reference 4.1e-07), steps, seconds in the solver, 10^9 cell updates per second. 'sweep' = the sweep's")
    print("record of the original. Under a stopped run: its error line.")
    for test in dict.fromkeys(t for t, *_ in MATRIX):
        print(f"\n{test}\n{'kernel':<14}{'N':>5}  {'fixed':>30}   {'original':<9}{'':>21}")
        for config, rundir, n, seed, kernels in runs():
            if Path(config).stem != test:continue
            for k in kernels:
                d = results/rundir/"data"/f"{k}_r1";m = measure(record(d/"run.toml"), test, d, n);have += m is not None
                o = ORIGINAL.get(k);om = None;src = ""
                if o and o in kernels:
                    od = results/rundir/"data"/f"{o}_r1";om = measure(record(od/"run.toml"), test, od, n);src = "same job"
                elif o and sweep:
                    found = [d for d in (f"{test}-gpu-N{n}-s{seed}-all", f"{test}-gpu-N{n}-s{seed}-{o}") if (sweep/d/"data"/f"{o}_r1"/"run.toml").is_file()]
                    od = sweep/found[0]/"data"/f"{o}_r1" if found else None
                    om = measure(record(od/"run.toml"), test, od, n) if found else None;src = "sweep"
                if m is None and om is None:continue
                print(f"{k:<14}{n:>5}  {cell(m):>30}   {(o or '') + ' (' + src + ')' if o else '':<22} {cell(om) if o else ''}")
                for who, x in (("fixed", m), ("original", om)):
                    if x and not x["done"] and x["why"]:print(f"{'':<21}{who}: {x['why']}")
    print(f"\n{have} of {want} runs recorded.")


def table_rows(path):
    try:
        with open(path, newline="") as f:return list(csv.reader(f))
    except OSError:
        return None


def cmd_control(argv):
    """Each original kernel run on the branch beside the sweep's run of it: every diagnostics.csv cell must agree."""
    results, sweep = Path(argv[0]), Path(argv[1]);bad = 0
    for test, n, k in CONTROL:
        mine = results/f"{test}-N{n}-s42"/"data"/f"{k}_r1"/"diagnostics.csv"
        theirs = next((p for p in (sweep/f"{test}-gpu-N{n}-s42-{x}"/"data"/f"{k}_r1"/"diagnostics.csv" for x in ("all", k)) if p.is_file()), None)
        a, b = (table_rows(mine) if mine.is_file() else None), (table_rows(theirs) if theirs else None)
        if not a or not b:print(f"  {test:<12} N={n:<4} {k:<10} not comparable yet ({'no run here' if not a else 'no sweep record'})");bad += 1;continue
        cols = [c for c in a[0] if c in b[0]];ia = [a[0].index(c) for c in cols];ib = [b[0].index(c) for c in cols]
        diff = [(i, c) for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), 1) for c, x, y in zip(cols, [ra[j] for j in ia], [rb[j] for j in ib]) if x != y]
        if len(a) != len(b):diff.append((min(len(a), len(b)), "number of rows"))
        print(f"  {test:<12} N={n:<4} {k:<10} " + (f"IDENTICAL: {len(a) - 1} outputs x {len(cols)} columns" if not diff else
              f"DIFFERS in {len(diff)} cells, first at output {diff[0][0]}, column {diff[0][1]}"));bad += bool(diff)
    return bad


def main(argv):
    if "--plan" in argv:
        i = argv.index("--plan");use_plan(argv[i + 1]);argv = argv[:i] + argv[i + 2:]
    if not argv:sys.exit(__doc__)
    command, rest = argv[0], argv[1:]
    if command == "configs":
        write_configs()
        if SPLIT and rest:
            est = estimate(rest[0]);(HERE/CONFIG_DIR/"estimates.json").write_text(json.dumps({f"{d}|{k}": (v if v == v else 600.0) for (d, k), v in est.items()}, indent=0))
            print(f"estimates for {len(est)} runs: {sum(v for v in est.values() if v == v)/3600:.1f} GPU hours")
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
    elif command == "control" and len(rest) == 2:sys.exit(1 if cmd_control(rest) else 0)
    elif command == "tasks" and len(rest) == 2:cmd_tasks(rest)
    elif command == "progress" and rest:cmd_progress(rest)
    else:sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
