#!/usr/bin/env python3
"""The order x solver study (README.md): in each code that can change them one at a time, every pair of reconstruction
(PLM, PPM) and Riemann solver (HLL-family, HLLD), at one time integrator, on the work-precision tests.

    python3 order.py configs                    (Mac) write configs/ and configs/estimates.json from the sweep's records
    python3 order.py plan                       the runs and the hours of each set-up
    python3 order.py groups KIND HOURS          lines "RANKS TASKS": the array tasks of about HOURS for each rank count
    python3 order.py runs KIND RESULTS [--pilot] [--ranks R --task I --tasks K]     the run list of one task
    python3 order.py external CODE VARIANT CONFIG N OUTDIR ...      the sweep's codes/external.py with VARIANTS below
    python3 order.py progress RESULTS           per set-up: complete, stopped, started, not started
    python3 order.py verdict RESULTS            the pilot: exit 1 unless every pilot run is complete and finite

KIND: glm (Abel et al.'s kernels, GPU), athenak (GPU), athenapp (CPU). Run directories: RESULTS/<test>-N<n>-s42-<label>,
as the fixes rerun's, the run itself in data/<label>_r1. One repeat; every other setting is the sweep's.
"""
import json
import math
import os
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = Path(os.environ.get("SWEEP_BUNDLE", HERE.parent/"sherlock_sweep"))
SOURCE = HERE.parent/"sherlock_sweep"/"configs"
# The work-precision tests (scripts/work_precision.py) except the current sheet, which no set-up converges on.
TESTS = ("entropy", "alfven", "alfven_oblique", "field_loop_novz", "brio_wu", "brio_wu_rotated", "orszag_tang", "rotor",
         "blast_2d_b0", "blast_2d_b50", "kh_uniform", "gem", "field_loop_3d")
# The new set-ups. With the sweep's PPM/HLLD/RK3 ("best") each code then has all four pairs at RK3; PLM/HLLE/RK3 also
# tells the integrator's share against the sweep's PLM/HLLE ("matched", RK2 in AthenaK, VL2 in Athena++).
VARIANTS = {
    "athenak": {
        "plm_hlle_rk3": dict(reconstruct="plm", rsolver="hlle", integrator="rk3", cfl=0.3, precision="fp64"),
        "ppm_hlle": dict(reconstruct="ppm4", rsolver="hlle", integrator="rk3", cfl=0.3, precision="fp64"),
        "plm_hlld": dict(reconstruct="plm", rsolver="hlld", integrator="rk3", cfl=0.3, precision="fp64"),
    },
    "athenapp": {
        "plm_hlle_rk3": dict(xorder=2, integrator="rk3", cfl=0.3, flux="hlle"),
        "ppm_hlle": dict(xorder=3, integrator="rk3", cfl=0.3, flux="hlle"),
        "plm_hlld": dict(xorder=2, integrator="rk3", cfl=0.3, flux="hlld"),
    },
}
# Abel et al.'s kernels: with glm_fix_hlld (PLM/HLLD, from the fixes rerun) the four pairs at Float32, 4-cell tile.
GLM = ("glm_fix_hll32", "glm_fix_ppm32", "glm_fix_ppm_hlld")
LABELS = {"glm": GLM, **{code: tuple(f"{code}_{v}" for v in vs) for code, vs in VARIANTS.items()}}
# For the estimates: the sweep's set-up each new one is timed from, and how much longer it should take.
PROXY = {"athenak_plm_hlle_rk3": ("athenak_matched", 1.5), "athenak_ppm_hlle": ("athenak_best", 1.0),
         "athenak_plm_hlld": ("athenak_best", 0.9), "athenapp_plm_hlle_rk3": ("athenapp_matched", 1.5),
         "athenapp_ppm_hlle": ("athenapp_best", 1.0), "athenapp_plm_hlld": ("athenapp_best", 0.9),
         "glm_fix_hll32": ("glm_fix_hlld", 0.8), "glm_fix_ppm32": ("glm_fix_hlld", 1.6), "glm_fix_ppm_hlld": ("glm_fix_hlld", 2.0)}
PILOT = ("orszag_tang", (32, 64))


def config(test):
    return tomllib.loads(((HERE/"configs") if (HERE/"configs"/f"{test}.toml").is_file() else SOURCE).joinpath(f"{test}.toml").read_text())


def grids(test):
    return [int(n) for n in config(test)["grids"] if int(n) <= 512]


def runs(kind):
    """(test, label, n) of a kind, cheapest grid first."""
    return sorted(((t, l, n) for t in TESTS for n in grids(t) for l in LABELS[kind]), key=lambda r: (r[2], r[0], r[1]))


def rundir(test, label, n):
    return f"{test}-N{n}-s42-{label}"


def estimates():
    try:return json.loads((HERE/"configs"/"estimates.json").read_text())
    except OSError:return {}


def ranks(test, n):
    """Athena++: the sweep's rank count for this test and grid (estimates.json), else the usual one."""
    return int(estimates().get(f"ranks|{test}|{n}", {32: 1, 64: 2, 128: 4, 256: 8, 512: 16}[n]))


def cost(test, label, n):
    return float(estimates().get(f"{test}|{label}|{n}", 900.0))


def setting(text, key, value):
    text, k = re.subn(rf"^{key} *=.*\n", "" if value is None else f"{key} = {value}\n", text, flags=re.M)
    if k != 1:sys.exit(f"expected exactly one line '{key} = ...', found {k}")
    return text


def cmd_configs():
    """The GPU driver's configs (the sweep's, with the new kernels and one repeat), and the estimates from the records."""
    sys.path.insert(0, str(HERE.parent/"scripts"))
    import sweep_data as sd
    out = HERE/"configs";out.mkdir(exist_ok=True)
    for test in TESTS:
        text = (SOURCE/f"{test}.toml").read_text()
        text = setting(text, "methods", "[" + ", ".join(f'"{k}"' for k in GLM) + "]");text = setting(text, "repeats", 1)
        if re.search(r"^repeats_by_grid *=", text, flags=re.M):text = setting(text, "repeats_by_grid", None)
        if not re.search(r"^split_methods *=", text, flags=re.M):text = text.replace("\nrepeats = 1\n", "\nsplit_methods = true\nrepeats = 1\n")
        text = f"# sherlock_order: {test} with the order x solver kernels (order.py configs). Everything else as in the\n# sweep's config, which follows.\n" + text
        tomllib.loads(text);(out/f"{test}.toml").write_text(text)
    sw = sd.Sweep();est = {}
    for test in TESTS:
        for n in grids(test):
            for label, (proxy, f) in PROXY.items():
                m = sw.meta(test, proxy, n) or {}
                wall = m.get("wall_seconds") or ((m.get("evolution_wall_seconds") or 0) + (m.get("startup_including_warmup_seconds") or 0))
                if wall:est[f"{test}|{label}|{n}"] = round(60 + f*float(wall))
            run = sw.path(test, "athenapp_best", n, "run.toml")              # its launcher is in data/environment.toml
            env = run.parent.parent/"environment.toml" if run is not None else None
            r = re.search(r"-n (\d+)", tomllib.loads(env.read_text()).get("launcher", "")) if env is not None and env.is_file() else None
            if r:est[f"ranks|{test}|{n}"] = int(r[1])
    (out/"estimates.json").write_text(json.dumps(est, indent=0, sort_keys=True) + "\n")
    missing = [f"{t} {l} {n}" for k in LABELS for t, l, n in runs(k) if f"{t}|{l}|{n}" not in est]
    print(f"wrote {len(TESTS)} configs and {len(est)} estimates" + (f"; no record to time {len(missing)} runs from: " + ", ".join(missing) if missing else ""))


def cmd_plan():
    for kind in LABELS:
        for label in LABELS[kind]:
            mine = [r for r in runs(kind) if r[1] == label]
            sec = sum(cost(*r) for r in mine);core = sum(cost(*r)*ranks(r[0], r[2]) for r in mine) if kind == "athenapp" else 0
            print(f"{label:24s} {len(mine):3d} runs  {sec/3600:6.1f} {'h on the GPU' if kind != 'athenapp' else 'h wall'}"
                  + (f", {core/3600:6.0f} core-hours" if core else "") + f", longest {max(cost(*r) for r in mine)/3600:.1f} h")


def assign(entries, ntasks):
    """Longest run first onto the least loaded task."""
    tasks = [[] for _ in range(ntasks)];load = [0.0]*ntasks
    for e in sorted(entries, key=lambda e: cost(*e), reverse=True):
        i = min(range(ntasks), key=lambda t: load[t]);tasks[i].append(e);load[i] += cost(*e)
    return tasks


def mine(kind, argv):
    entries = runs(kind)
    if kind == "athenapp" and "--ranks" in argv:
        r = int(argv[argv.index("--ranks") + 1]);entries = [e for e in entries if ranks(e[0], e[2]) == r]
    return entries


def cmd_groups(argv):
    kind, hours = argv[0], float(argv[1])
    for r in (sorted({ranks(t, n) for t, _, n in runs(kind)}) if kind == "athenapp" else [1]):
        entries = mine(kind, ["--ranks", str(r)]);total = sum(cost(*e) for e in entries)
        print(r, max(1, math.ceil(total/(hours*3600))))


def cmd_runs(argv):
    kind, results = argv[0], Path(argv[1])
    entries = [e for e in runs(kind) if e[0] == PILOT[0] and e[2] in PILOT[1]] if "--pilot" in argv else mine(kind, argv)
    if "--task" in argv:entries = sorted(assign(entries, int(argv[argv.index("--tasks") + 1]))[int(argv[argv.index("--task") + 1])], key=lambda e: (e[2], e[0]))
    for test, label, n in entries:
        out = results/rundir(test, label, n)/"data"
        if kind == "glm":print(f"configs/{test}.toml", out, n, 42, label);continue
        # as in the sweep: AthenaK on the GPU runs the planar tests as z-invariant cubes, Athena++ on the plane
        geometry = "plane" if kind == "athenapp" and config(test).get("problem", test) not in ("orszag_tang_3d", "field_loop_3d", "decay") else "cube"
        print(test, label.split("_", 1)[1], n, ranks(test, n) if kind == "athenapp" else 1, geometry, out)


def cmd_external(argv):
    """codes/external.py of the sweep bundle, with this study's variants added (its method label is CODE_VARIANT)."""
    sys.path.insert(0, str(SWEEP/"codes"))
    import external
    for code, vs in VARIANTS.items():external.VARIANTS[code].update(vs)
    sys.argv = ["external.py"] + argv
    external.main()


def record(path):
    try:return tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError):return None


def cmd_progress(argv):
    results = Path(argv[0]);print(f"{'set-up':<24}{'complete':>9}{'stopped':>9}{'started':>9}{'waiting':>9}")
    for kind in LABELS:
        for label in LABELS[kind]:
            c = [0, 0, 0, 0]
            for t, l, n in runs(kind):
                if l != label:continue
                r = record(results/rundir(t, l, n)/"data"/f"{l}_r1"/"run.toml")
                c[3 if r is None else 0 if r.get("status") == "complete" else 1 if r.get("status") in ("failed", "stopped", "error") else 2] += 1
            print(f"{label:<24}" + "".join(f"{x:>9}" for x in c))


def cmd_verdict(argv):
    results = Path(argv[0]);kinds = argv[1].split(",") if len(argv) > 1 else list(LABELS);bad = []
    for kind in kinds:
        for t, l, n in runs(kind):
            if t != PILOT[0] or n not in PILOT[1]:continue
            r = record(results/rundir(t, l, n)/"data"/f"{l}_r1"/"run.toml")
            ok = r is not None and r.get("status") == "complete" and abs(float(r.get("time_reached", 0)) - float(config(t)["t_end"])) < 1e-6
            print(f"{'ok   ' if ok else 'WRONG'} {l} N={n}: {r.get('status') if r else 'no record'}" + ("" if ok or not r else f" ({str(r.get('error', '')).splitlines()[:1]})"))
            ok or bad.append(f"{l} {n}")
    print("PILOT OK" if not bad else "PILOT FAILED: " + ", ".join(bad));sys.exit(1 if bad else 0)


if __name__ == "__main__":
    cmds = dict(configs=lambda a: cmd_configs(), plan=lambda a: cmd_plan(), groups=cmd_groups, runs=cmd_runs,
                external=cmd_external, progress=cmd_progress, verdict=cmd_verdict)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])
