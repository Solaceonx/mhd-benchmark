#!/usr/bin/env python3
"""RAMSES II (github.com/ramses-organisation/ramses2, develop 04abdd9: its own turbulence driving, on the GPU) in the
driven-turbulence set-up of README.md, with E. Moseley's upwind constrained transport and HLLD faces (UCT-HLLD).

  python3 ramses2_turb.py run CONFIG RESULTS MACH N SEED --exe EXE
      one run in RESULTS/ramses2-uct-M<mach>-N<n>-s<seed>: RAMSES II writes an output at every frame time; each is
      converted to the sweep's state layout and measured by turb_measure.jl (the measurements of every other code),
      then deleted. The run directory is on the node's own disk ($L_SCRATCH_JOB) while it runs, and only the measured
      results are copied to RESULTS. A run already measured is kept; one that was stopped runs again from the start
      (the longest, Mach 50 at 256^3, takes minutes on an H100).
  python3 ramses2_turb.py plan CONFIG RESULTS
      every run of the configuration, one line each: "run MACH N SEED done|todo NAME".

The driving is the same as for the other codes (README.md): the 'parabolic' spectrum (1 < |k| < 3), solenoidal
(comp_frac = 0), memory time T = 1/(2 M), turb_Ndt steps per T, and the per-component rms acceleration a_rms of
forcing.toml (found once with classic RAMSES), used unchanged. Two differences from classic RAMSES, neither a patch:
- RAMSES II has no turb_exact_rms. Its force is normalised analytically (turb_init.f90: power, projection and
  Ornstein-Uhlenbeck norms), so the rms of each component is a_rms on average rather than at every step;
- its projection is the corrected one (ramses2 PR #57), as our ramsesfix-* builds.
Gas: the same gamma (1 + gamma_minus_one, rounded to Float32), uniform density 1 and pressure 1/gamma at rest, zero
magnetic field (the MHD solver with B = 0: hydrodynamics through the HLLD and UCT code paths). Nothing in RAMSES II is
patched except newdt_fine.f90, which makes the outputs land on their times (sherlock_miniramses/prepare_patch.py).

Exit status: 0 done (a run in which RAMSES II itself stopped is recorded as failed, with what it reached); 1 not
measured; 76 the job is being stopped (nothing recorded).
"""
import argparse, os, re, shlex, shutil, subprocess, sys, time, tomllib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP = Path(os.environ.get("SWEEP_BUNDLE", HERE.parent/"sherlock_sweep"))
sys.path.insert(0, str(SWEEP/"codes"));sys.path.insert(0, str(HERE.parent/"sherlock_miniramses"));sys.path.insert(0, str(HERE))
import external as X                          # noqa: E402  the sweep's bookkeeping, kill detection, toml writer
import miniramses_run as MR                   # noqa: E402  RAMSES II's output reader (Float32 primitive state per oct)
from ramses_turb import frame_times, gamma_of, julia_threads, mkey      # noqa: E402

# The sweep's UCT set-up of RAMSES II (miniramses_run.py: X.VARIANTS["ramses2"]["uct"]).
SETUP = dict(riemann="uct-hlld", riemann2d="hlld", slope_type=2, courant=0.8)


def name_of(mach, n, seed):
    return f"ramses2-uct-M{mkey(mach)}-N{n}-s{seed}"


def namelist(config, mach, n, a_rms, seed, times):
    level = int(round(np.log2(n)))
    if 2**level != n:raise SystemExit("RAMSES II requires N to be a power of two")
    gamma = gamma_of(config);T = 1.0/(2.0*float(mach))
    # RAMSES II ends a run when the time reaches the last output time, before the step that would write it
    # (miniramses_run.namelist): one more output time just past the end makes it write the last frame and stop.
    times = list(times) + [times[-1]*(1 + 1e-9) + 1e-12]
    tout = ",\n     ".join(",".join(repr(float(t)) for t in times[i:i + 4]) for i in range(0, len(times), 4))
    return f"""&RUN_PARAMS
hydro=.true.
ncontrol=1000
nsubcycle=1
nstepmax=100000000
/
&AMR_PARAMS
levelmin={level}
levelmax={level}
ngridmax={n**3//7 + 1000}
boxlen=1.0
/
&INIT_PARAMS
nregion=1
region_type(1)='square'
x_center=0.5
y_center=0.5
z_center=0.5
length_x=10.0
length_y=10.0
length_z=10.0
exp_region=10.0
d_region=1.0
u_region=0.0
v_region=0.0
w_region=0.0
p_region={1.0/gamma!r}
/
&OUTPUT_PARAMS
tout={tout}
/
&HYDRO_PARAMS
gamma={gamma!r}
courant_factor={SETUP['courant']}
slope_type={SETUP['slope_type']}
riemann='{SETUP['riemann']}'
riemann2d='{SETUP['riemann2d']}'
smallr=1e-6
smallc=1e-6
/
&TURB_PARAMS
turb=.true.
turb_seed={int(seed)}
comp_frac={float(config['comp_frac'])!r}
forcing_power_spectrum='{config['spectrum']}'
turb_T={T!r}
turb_Ndt={int(config['turb_ndt'])}
turb_rms={float(a_rms)!r}
turb_min_rho=1d-50
/
&REFINE_PARAMS
/
"""


def solve(config, rundir, mach, n, a_rms, seed, tend_T, exe, riemann=None):
    """Run RAMSES II in rundir from the start to tend_T. Returns (state, frames, perf): state "complete", "failed" or
    "killed"; frames [(index, output directory)] of the outputs that landed on a frame time."""
    for o in rundir.glob("output_*"):shutil.rmtree(o)
    T, targets = frame_times(config, mach, tend_T);tol = 1e-6*T/int(config["frames_per_T"])
    text = namelist(config, mach, n, a_rms, seed, targets)
    if riemann:text = text.replace(f"riemann='{SETUP['riemann']}'", f"riemann='{riemann}'")      # CPU test only
    (rundir/"turb.nml").write_text(text);log = rundir/"ramses2.log"
    start = time.perf_counter()
    with log.open("w") as f:
        try:code = subprocess.run([str(exe), "turb.nml"], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=rundir).returncode
        except FileNotFoundError as err:
            f.write(f"could not start {exe}: {err}\n");code = 127
    perf = dict(wall_seconds=time.perf_counter() - start);last = log.read_text(errors="replace")
    if code != 0 and X.was_killed(code, last):return "killed", [], perf
    outs = MR.outputs_of(rundir);frames = []
    for o in outs:
        t = MR.info(o)["time"];k = int(round(t/(T/int(config["frames_per_T"]))))
        if k < len(targets) and abs(t - targets[k]) <= tol:frames.append((k, o))
    m = re.findall(r"Total elapsed time:\s*([\d.eE+-]+)", last)
    if m:perf["code_seconds"] = float(m[-1])
    if outs:
        perf["steps"] = MR.info(outs[-1])["steps"]
        if m and float(m[-1]) > 0:perf["cell_updates_per_second"] = n**3*perf["steps"]/float(m[-1])
    reached = bool(frames) and frames[-1][0] == len(targets) - 1
    if code != 0 or not reached:
        t_last = MR.info(outs[-1])["time"] if outs else 0.0
        perf["reason"] = f"RAMSES II ended with exit code {code} at t = {t_last:.6g} ({t_last/T:.2f} T); the end of its log: " + last[-600:]
        return "failed", frames, perf
    return "complete", frames, perf


def run(args, config):
    mach, n, seed = args.mach, args.n, args.seed
    forcing = tomllib.loads((args.results/"forcing.toml").read_text())[f"mach_{mkey(mach)}"]
    tend_T = float(args.tend_T if args.tend_T is not None else config["t_end_T"]);T, targets = frame_times(config, mach, tend_T)
    name = args.name or name_of(mach, n, seed);final = args.results/name;meta_file = final/"run.toml"
    if meta_file.is_file() and tomllib.loads(meta_file.read_text()).get("measured"):
        print(f"kept      {name}");return 0
    work = Path(os.environ["L_SCRATCH_JOB"])/name if os.environ.get("L_SCRATCH_JOB") else final
    if work != final and final.exists():shutil.rmtree(final)              # an earlier attempt that was not measured
    work.mkdir(parents=True, exist_ok=True)
    state, frames, perf = solve(config, work, mach, n, forcing["a_rms"], seed, tend_T, args.exe, args.riemann)
    if state == "killed":
        print(f"killed    {name}: RAMSES II was stopped from outside; run again", flush=True);return X.KILLED
    gamma = gamma_of(config);rows = []
    for k, out in frames:
        t, steps, u = MR.read_output(out, n, gamma);file = f"tmp_{k:03d}.bin"
        np.asarray(u, dtype="<f4").ravel(order="F").tofile(work/file)
        rows.append(dict(frame=k, time=t, steps=steps, file=file));shutil.rmtree(out)
    for o in work.glob("output_*"):shutil.rmtree(o)
    riemann = args.riemann or SETUP["riemann"]
    meta = dict(code="ramses2", method="uct" if not args.riemann else args.riemann,
                solver=f"RAMSES II (ramses2 develop), MUSCL-Hancock, slope_type={SETUP['slope_type']} (MonCen), riemann='{riemann}', "
                       f"riemann2d='{SETUP['riemann2d']}', courant_factor={SETUP['courant']}, zero magnetic field",
                projection="RAMSES II's own (corrected, ramses2 PR #57); force normalised analytically (no turb_exact_rms)",
                N=n, mach_target=mach, a_rms=float(forcing["a_rms"]), turb_T=T, turb_ndt=int(config["turb_ndt"]), gamma=gamma,
                comp_frac=float(config["comp_frac"]), spectrum=config["spectrum"], seed=seed, cfl=SETUP["courant"], t_end=targets[-1], t_end_T=tend_T,
                device=os.environ.get("TURB_DEVICE", f"{os.uname().nodename}"), precision=os.environ.get("TURB_PRECISION", "double (NPRE=8); outputs Float32"),
                slices_every=int(config.get("slices_every", 4)), status=state, converted=True, measured=False, steps=rows[-1]["steps"] if rows else 0,
                perf={k: v for k, v in perf.items() if k != "reason"}, frames=rows)
    if state == "failed":meta["reason"] = perf["reason"];meta["failed_at_time_over_T"] = (rows[-1]["time"] if rows else 0.0)/T
    (work/"run.toml").write_text(X.toml_dump(meta))
    julia = os.environ.get("MHD_JULIA", "julia")
    rc = subprocess.run([julia, "--startup-file=no", f"--project={SWEEP/'env'}", f"--threads={julia_threads()}", str(HERE/"turb_measure.jl"), str(work)]).returncode
    meta = tomllib.loads((work/"run.toml").read_text())
    if work != final:
        shutil.copytree(work, final, dirs_exist_ok=True);shutil.rmtree(work)
    if rc != 0 or not meta.get("measured"):
        print(f"UNMEASURED {name}: turb_measure.jl ended with exit code {rc}", flush=True);return 1
    reached = meta["frames"][-1]["time"]/T if meta["frames"] else 0.0
    print(f"{meta['status']:9s} {name}: {len(meta['frames'])} frames to {reached:.2f} T, {meta['steps']} steps, "
          f"{perf.get('code_seconds', perf['wall_seconds']):.0f} s" + (f"; {meta['reason'][:300]}" if meta["status"] == "failed" else ""), flush=True)
    return 0


def plan(args, config):
    seeds = config.get("ramses2", {}).get("seeds", [1, 2])
    for m in config["machs"]:
        for n in config["grids"]:
            for s in seeds:
                f = args.results/name_of(m, n, s)/"run.toml"
                done = f.is_file() and tomllib.loads(f.read_text()).get("measured")
                print("run", m, n, s, "done" if done else "todo", name_of(m, n, s))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("run");q.add_argument("config", type=Path);q.add_argument("results", type=Path)
    q.add_argument("mach", type=lambda s: float(s) if "." in s else int(s));q.add_argument("n", type=int);q.add_argument("seed", type=int)
    q.add_argument("--exe", required=True, type=Path);q.add_argument("--tend-T", type=float, default=None)
    q.add_argument("--name", default=None, help="run directory name (the trial's)")
    q.add_argument("--riemann", default=None, help="another solver: the Mac's CPU test (UCT-HLLD is GPU only)")
    q = sub.add_parser("plan");q.add_argument("config", type=Path);q.add_argument("results", type=Path)
    args = p.parse_args();config = tomllib.loads(args.config.read_text());args.results.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":return plan(args, config)
    args.exe = args.exe.resolve()
    return run(args, config)


if __name__ == "__main__":
    sys.exit(main())
