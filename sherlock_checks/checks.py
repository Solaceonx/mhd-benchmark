#!/usr/bin/env python3
"""Courant-number and single-precision checks of Tom Abel's kernels (README.md).

    python3 checks.py configs                    write configs/ from ../sherlock_sweep/configs (when MATRIX changes)
    python3 checks.py plan                       list the runs
    python3 checks.py runs RESULTS [--trial] [--ok FILE]      the driver's run list (CONFIG OUTDIR N SEED METHODS)
    python3 checks.py verdict RESULTS            judge the trial; writes RESULTS/trial_ok.txt; exit 1 if the control fails
    python3 checks.py summary RESULTS [SWEEP_RESULTS]         table of the runs, beside the sweep's at Courant 0.25
"""
import csv
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/"sherlock_sweep"/"configs"
COURANT = {"c025": 0.25, "c040": 0.4, "c070": 0.7}
NEW = ["glm_hll32", "glm_ppm32"]                         # the Float32-tile variants (run_suite.jl)
KERNELS = ["glm", "glm_hll16", "glm_hlld", "glm_hll32"]  # what runs at the other Courant numbers
BASELINE = ["glm", "glm_hll16", "glm_hlld", "glm_ppm"]   # the sweep ran these at 0.25
TRIAL = ["glm_hlld", "glm_hlld_x", "glm_hll32", "glm_ppm32"]
TRIAL_GRIDS = [32, 48]                                   # 48 is divisible by 6: the Float32 kernels use their 6-cell tile


def grid(ns, seeds=(42,)):
    return [(n, s) for n in ns for s in seeds]


def rows(test, ns, seeds=(42,), ppm=True):
    """At 0.25 only the new kernels (the sweep has the others); at 0.4 and 0.7 the three sweep kernels and glm_hll32."""
    return [(test, "c025", grid(ns, seeds), NEW if ppm else NEW[:1])] + [(test, c, grid(ns, seeds), KERNELS) for c in ("c040", "c070")]


# test, Courant tag, [(N, seed)], kernels. Planar tests cost N^3 cells a step (the kernels are 3D), so they stop at 256.
MATRIX = (rows("entropy", [32, 64, 128, 256]) + [("entropy", "c025", grid([512]), ["glm_hll32"])]
          + rows("alfven", [32, 64, 128, 256]) + rows("brio_wu", [32, 64, 128, 256]) + rows("orszag_tang", [32, 64, 128, 256])
          + rows("blast_2d_b100", [64, 128, 256])
          + rows("kh_localized", [128], ppm=False) + [("kh_localized", "c025", grid([256]), ["glm_hll32"])]
          + rows("decay_supersonic", [64, 128], seeds=(42, 43), ppm=False) + rows("decay_supersonic", [256], ppm=False)
          + rows("orszag_tang_3d", [128, 256], ppm=False)
          + [("orszag_tang_3d", "c040", grid([512]), ["glm", "glm_hll32"]), ("orszag_tang_3d", "c070", grid([512]), ["glm"])]
          + [("throughput_ot3d", "c025", grid([96, 128, 192, 256, 384, 512]), NEW)])


def runs():
    """(config file, run directory, N, seed, kernels), cheapest first."""
    out = {}
    for test, tag, cells, kernels in MATRIX:
        for n, seed in cells:
            key = (n, test, tag, seed)
            if key in out:sys.exit(f"MATRIX names {test} {tag} N={n} seed {seed} twice")
            out[key] = (f"configs/{test}_{tag}.toml", f"{test}-{tag}-N{n}-s{seed}", n, seed, list(kernels))
    return [out[k] for k in sorted(out)]


def trial_runs():
    return [("configs/trial.toml", f"trial-N{n}", n, 42, list(TRIAL)) for n in TRIAL_GRIDS]


def setting(text, key, value):
    """Replace the one line `key = ...` of a config (value None: remove it)."""
    text, n = re.subn(rf"^{key} *=.*\n", "" if value is None else f"{key} = {value}\n", text, flags=re.M)
    if n != 1:sys.exit(f"expected exactly one line '{key} = ...', found {n}")
    return text


def write_configs():
    made = []
    (HERE/"configs").mkdir(exist_ok=True)
    for old in (HERE/"configs").glob("*.toml"):old.unlink()
    for test, tag in sorted({(t, c) for t, c, _, _ in MATRIX}):
        mine = [(cells, k) for t, c, cells, k in MATRIX if (t, c) == (test, tag)]
        ns = sorted({n for cells, _ in mine for n, _ in cells});seeds = sorted({s for cells, _ in mine for _, s in cells})
        kernels = [k for k in dict.fromkeys(KERNELS + NEW) if any(k in ks for _, ks in mine)]
        text = (SOURCE/f"{test}.toml").read_text()
        text = setting(text, "cfl", COURANT[tag]);text = setting(text, "grids", ns);text = setting(text, "seeds", seeds)
        text = setting(text, "methods", "[" + ", ".join(f'"{k}"' for k in kernels) + "]");text = setting(text, "repeats", 1)
        if re.search(r"^repeats_by_grid *=", text, flags=re.M):text = setting(text, "repeats_by_grid", None)
        text = f"# sherlock_checks: {test} at Courant number {COURANT[tag]} (checks.py configs). Everything else as in the sweep's\n# config, which follows.\n" + text
        (HERE/"configs"/f"{test}_{tag}.toml").write_text(text);tomllib.loads(text);made.append(f"{test}_{tag}")
    # the trial: Orszag-Tang (every term of the scheme is active) for a few steps, on a 4-cell-tile and a 6-cell-tile grid
    text = (SOURCE/"orszag_tang.toml").read_text()
    for key, value in (("t_end", 0.05), ("frames", 3), ("grids", TRIAL_GRIDS), ("repeats", 1), ("repeats_by_grid", None),
                       ("snapshot_frames_by_grid", None), ("methods", "[" + ", ".join(f'"{k}"' for k in TRIAL) + "]")):
        text = setting(text, key, value)
    text = "# sherlock_checks: the trial (checks.py configs). Orszag-Tang to t = 0.05 with the new kernels and the control.\n" + text
    (HERE/"configs"/"trial.toml").write_text(text);tomllib.loads(text)
    print(f"wrote {len(made) + 1} configs: trial, " + ", ".join(made))


def tile(n):
    return 6 if n % 6 == 0 else 4


def cmd_runs(argv):
    results = Path(argv[0]);trial = "--trial" in argv
    ok = None
    if "--ok" in argv:
        ok = {tuple(l.split()) for l in Path(argv[argv.index("--ok") + 1]).read_text().splitlines() if l.strip()}
    for config, rundir, n, seed, kernels in (trial_runs() if trial else runs()):
        if ok is not None:kernels = [k for k in kernels if k not in NEW or (k, str(tile(n))) in ok]
        if kernels:print(config, results/rundir/"data", n, seed, ",".join(kernels))


def record(path):
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None


def table_rows(path):
    try:
        with open(path, newline="") as f:return list(csv.reader(f))
    except OSError:
        return None


def cmd_verdict(argv):
    results = Path(argv[0]);ok = [];bad = False
    for n in TRIAL_GRIDS:
        data = results/f"trial-N{n}"/"data"
        a, b = table_rows(data/"glm_hlld_r1"/"diagnostics.csv"), table_rows(data/"glm_hlld_x_r1"/"diagnostics.csv")
        if not a or not b or len(a) < 3:
            print(f"   N={n}: the control did not run (glm_hlld and glm_hlld_x must both finish)");bad = True
        elif a != b:
            where = next((f"row {i}, column {a[0][j]}: {x} against {y}" for i, (ra, rb) in enumerate(zip(a, b)) for j, (x, y) in enumerate(zip(ra, rb)) if x != y), "different lengths")
            print(f"   N={n}: CONTROL FAILS. The driver's own launch of the HLLD kernel does not reproduce step_plm! ({where})");bad = True
        else:
            print(f"   N={n}: control passes: the driver's launch reproduces step_plm! digit for digit ({len(a) - 1} rows, {len(a[0])} columns)")
        for k in NEW:
            r = record(data/f"{k}_r1"/"run.toml")
            if r and r.get("status") == "complete":
                ok.append(f"{k} {tile(n)}");print(f"   N={n}: {k} with its {tile(n)}-cell tile ran {r.get('steps')} steps to t = {r.get('time_reached')}")
            else:
                why = (r or {}).get("error", (r or {}).get("status", "no record"))
                print(f"   N={n}: {k} with its {tile(n)}-cell tile did NOT run: {str(why).splitlines()[0][:200] if why else why}")
    (results/"trial_ok.txt").write_text("\n".join(ok) + ("\n" if ok else ""))
    if bad:sys.exit(1)
    if not any(x.startswith("glm_hll32 4") for x in ok):
        print("   the Float32 production kernel did not run on a power-of-two grid: nothing to check");sys.exit(1)


def measure(r):
    """What the table shows of one run record."""
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
    found = {}      # (test, n, seed, kernel, tag) -> measure
    for config, rundir, n, seed, kernels in runs():
        test, tag = Path(config).stem.rsplit("_", 1)
        for k in kernels:found[(test, n, seed, k, tag)] = measure(record(results/rundir/"data"/f"{k}_r1"/"run.toml"))
        if sweep:
            for k in BASELINE:
                found.setdefault((test, n, seed, k, "c025"), measure(record(sweep/f"{test}-gpu-N{n}-s{seed}-all"/"data"/f"{k}_r1"/"run.toml")))
    have = sum(v is not None for key, v in found.items() if key[3] in NEW or key[4] != "c025")
    want = sum(len(k) for *_, k in runs())
    print(f"{have} of {want} runs recorded. Columns per Courant number: error at the end (density, else magnetic, relative L1), steps,")
    print("seconds in the solver, 10^9 cell updates per second. Courant 0.25 for glm, glm_hll16, glm_hlld, glm_ppm is the sweep's run.")
    for test in dict.fromkeys(t for t, *_ in MATRIX):
        print(f"\n{test}\n{'kernel':<11}{'N':>5}{'seed':>5} " + " ".join(f"{'Courant %.2f' % COURANT[c]:>34}" for c in COURANT))
        keys = sorted({(n, seed, k) for (t, n, seed, k, _) in found if t == test}, key=lambda x: ((BASELINE + NEW).index(x[2]) if x[2] in BASELINE + NEW else 9, x[0], x[1]))
        for n, seed, k in keys:
            cells = [found.get((test, n, seed, k, c)) for c in COURANT]
            if all(c is None for c in cells):continue
            print(f"{k:<11}{n:>5}{seed:>5} " + " ".join(f"{cell(c):>34}" for c in cells))


def main(argv):
    if not argv:sys.exit(__doc__)
    command, rest = argv[0], argv[1:]
    if command == "configs":write_configs()
    elif command == "plan":
        for config, rundir, n, seed, kernels in runs():print(f"{rundir:<40} {','.join(kernels)}")
        print(f"{len(runs())} run directories, {sum(len(k) for *_, k in runs())} runs")
    elif command == "runs" and rest:cmd_runs(rest)
    elif command == "verdict" and rest:cmd_verdict(rest)
    elif command == "summary" and rest:cmd_summary(rest)
    else:sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
