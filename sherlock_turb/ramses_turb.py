#!/usr/bin/env python3
"""RAMSES (its hydro solver and its own turbulence driving, unpatched) in the driven-turbulence set-up of README.md.

  python3 ramses_turb.py calibrate CONFIG RESULTS --exe EXE --mpi "srun -n 16" [--projection fixed]
      finds, for each target Mach number of the config, the driving strength that gives it (variant "matched" on the
      config's calibration grid, with the projection the config names) and adds it to RESULTS/forcing.toml, which
      every code and grid then uses unchanged. Mach numbers already there are kept.
  python3 ramses_turb.py run CONFIG RESULTS VARIANT MACH N --exe EXE --mpi "srun -n 64" [--projection fixed]
      one run in RESULTS/ramses-VARIANT-M<mach>-N<n>-s<seed> (ramsesfix-... with --projection fixed, which must be
      given the executable built by "build_ramses.sh fixed"): RAMSES writes an output at every frame time, each is
      converted to the sweep's state layout and measured by turb_measure.jl (the measurements the GPU driver makes),
      then deleted. A run that was interrupted continues from its newest complete output.
  python3 ramses_turb.py plan CONFIG RESULTS
      what is left to do, one line each, for check.sh: the calibration, the GPU job, every RAMSES run with the ranks
      and time limit to ask for.

Exit status: 0 done (a run in which RAMSES itself stopped is recorded as failed, with what it reached, and is also 0);
76 RAMSES was killed from outside (the job is being stopped): nothing recorded, run it again to continue.
"""
import argparse, os, re, shlex, shutil, subprocess, sys, time, tomllib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP = Path(os.environ.get("SWEEP_BUNDLE", HERE.parent/"sherlock_sweep"))
sys.path.insert(0, str(SWEEP/"codes"))
import external as X           # the sweep's reader of Fortran records, its output bookkeeping and its kill detection

# "matched" is the scheme of Tom Abel's hydro kernel as far as RAMSES has it: minmod slopes (slope_type=1), HLL, and the
# same Courant number in the same rule (dt = courant dx / max over cells of the sum over directions of |v| + c).
# "best" is the more accurate choice the sweep also made for its MHD runs: the contact-resolving solver, MonCen slopes.
VARIANTS = {
    "matched": dict(riemann="hll", slope_type=1, courant=0.5),
    "best": dict(riemann="hllc", slope_type=2, courant=0.8),
}
SPECTRA = ("parabolic", "power_law")
# RAMSES's driving as released takes the direction of a wave vector from the absolute values of its components
# (find_unitk), so its Helmholtz projection is wrong for every mode whose components differ in sign: with comp_frac = 0
# a fifth of the force's power is compressive. "fixed" is the build with the three lines corrected (build_ramses.sh).
PROJECTIONS = {
    "released": ("ramses", "as released: find_unitk takes the direction of k from |kx|, |ky|, |kz|; 20 % of the force's power is compressive at comp_frac = 0"),
    "fixed": ("ramsesfix", "corrected: signed wave numbers in find_unitk (three lines, compiled in through PATCH); the force is solenoidal to rounding at comp_frac = 0"),
}


def mkey(mach):
    """A Mach number as it appears in names and in forcing.toml: 3 -> "3", 0.5 -> "0p5"."""
    return str(mach).replace(".", "p")


def gamma_of(config):
    """The gamma every code uses: the single-precision number nearest to 1 + gamma_minus_one."""
    return float(np.float32(1.0 + float(config["gamma_minus_one"])))


def frame_times(config, mach, tend_T):
    T = 1.0/(2.0*float(mach));per = int(config["frames_per_T"])
    return T, [i*T/per for i in range(int(round(tend_T*per)) + 1)]


def namelist(config, variant, mach, n, a_rms, seed, times, ranks=1, restart=0):
    level = int(round(np.log2(n)))
    if 2**level != n:raise SystemExit("RAMSES requires N to be a power of two")
    if config["spectrum"] not in SPECTRA:raise SystemExit(f"spectrum {config['spectrum']!r}: RAMSES has {SPECTRA}")
    s = VARIANTS[variant];gamma = gamma_of(config);T = 1.0/(2.0*float(mach))
    octs = n**3//7;ngridmax = min(octs + 1000, 4*octs//max(ranks, 1) + 50000)       # as the sweep sizes it
    tout = ",\n     ".join(",".join(repr(float(t)) for t in times[i:i + 4]) for i in range(0, len(times), 4))
    return f"""&RUN_PARAMS
hydro=.true.
nrestart={restart}
ncontrol=100
nsubcycle=1
nstepmax=100000000
/
&AMR_PARAMS
levelmin={level}
levelmax={level}
ngridmax={ngridmax}
boxlen=1.0
/
&INIT_PARAMS
nregion=1
region_type='square'
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
foutput=1000000
write_conservative=.true.
read_conservative={".true." if restart else ".false."}
exact_output_time=.true.
/
&HYDRO_PARAMS
gamma={gamma!r}
courant_factor={s['courant']}
slope_type={s['slope_type']}
riemann='{s['riemann']}'
smallr=1e-6
smallc=1e-6
/
&TURB_PARAMS
driven_turb=.true.
turb_evolving=.true.
turb_seed={int(seed)}
comp_frac={float(config['comp_frac'])!r}
forcing_power_spectrum='{config['spectrum']}'
turb_T={T!r}
turb_Ndt={int(config['turb_ndt'])}
turb_rms={float(a_rms)!r}
turb_exact_rms=.true.
turb_min_rho=1d-50
/
"""


def read_ramses_hydro(outdir, n):
    """Uniform-grid output of the RAMSES hydro solver, written with write_conservative -> (time, coarse steps,
    (N,N,N,9) state in the sweep's layout: rho, momentum, total energy, zero field). The sweep's read_ramses with the
    hydro solver's five variables (density, momentum x y z, total energy) in place of the MHD solver's eleven."""
    nout = outdir.name.split("_")[-1]
    info = (outdir/f"info_{nout}.txt").read_text()
    t = float(re.search(r"^time\s*=\s*(\S+)", info, re.M).group(1));ncpu = int(re.search(r"^ncpu\s*=\s*(\d+)", info, re.M).group(1))
    steps = int(re.search(r"^nstep_coarse\s*=\s*(\d+)", info, re.M).group(1))
    names = [l.split(",")[1].strip() for l in (outdir/"hydro_file_descriptor.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    if names != ["density", "momentum_x", "momentum_y", "momentum_z", "total_energy"]:
        raise RuntimeError(f"{outdir}: variables {names} are not the conservative hydro set")
    level = int(round(np.log2(n)));u = np.full((n, n, n, 9), np.nan)
    for icpu in range(1, ncpu + 1):
        a = X.FortranRecords(outdir/f"amr_{nout}.out{icpu:05d}");h = X.FortranRecords(outdir/f"hydro_{nout}.out{icpu:05d}")
        _, ndim = a.read("<i4")[0], a.read("<i4")[0];a.read("<i4");nlevelmax = a.read("<i4")[0]
        a.read("<i4");nboundary = a.read("<i4")[0];a.read("<i4");boxlen = a.read("<f8")[0]
        a.skip(11)                        # noutput, tout, aout, t, dtold, dtnew, nstep, einit, omega, aexp, mass_sph
        a.read("<i4");a.read("<i4");numbl = a.read("<i4").reshape(nlevelmax, ncpu);a.read("<i4")
        if nboundary:a.skip(3)
        a.skip(1);a.skip(1);a.skip(1);a.skip(3)   # free-memory info, ordering, bound_key, son/flag1/cpu_map
        h.read("<i4");nvar = h.read("<i4")[0];h.read("<i4");h.read("<i4");h.read("<i4");h.read("<f8")
        if ndim != 3 or nvar != 5:raise RuntimeError(f"{outdir}: ndim {ndim}, nvar {nvar}; expected 3 and 5")
        for ilevel in range(nlevelmax):
            for ibound in range(nboundary + ncpu):
                ncache = numbl[ilevel, ibound] if ibound < ncpu else 0
                h.read("<i4");h.read("<i4")
                if ncache == 0:continue
                a.skip(3);xg = np.stack([a.read("<f8") for _ in range(ndim)]);a.skip(1 + 2*ndim + 3*2**ndim)
                vals = np.stack([np.stack([h.read("<f8") for _ in range(nvar)]) for _ in range(2**ndim)])  # (ind, var, grid)
                if ibound != icpu - 1 or ilevel + 1 != level:continue
                dx = boxlen/2**level
                for ind in range(2**ndim):
                    off = np.array([(ind >> d) & 1 for d in range(ndim)])
                    ijk = np.floor((xg.T + (off - .5)*dx)/dx).astype(int)
                    for c in range(5):u[ijk[:, 0], ijk[:, 1], ijk[:, 2], c] = vals[ind][c]
    if np.isnan(u[..., :5]).any():raise RuntimeError(f"RAMSES output {outdir} does not cover the uniform level {level}")
    u[..., 5:] = 0.0
    return t, steps, u


def mach_of(u, gamma):
    """Volume-weighted rms Mach number with the sound speed of the gas as it is, as turb_measure.jl defines mach_vol.
    Used only to choose the driving strength; every reported number comes from turb_measure.jl."""
    rho = np.maximum(u[..., 0], 1e-6);m2 = (u[..., 1:4]**2).sum(axis=-1)      # at the floor where RAMSES left less, as measure.jl
    p = (gamma - 1.0)*(u[..., 4] - 0.5*m2/rho)
    return float(np.sqrt((m2/rho**2).mean()/(gamma*p.mean()/rho.mean())))


def julia_threads():
    forced = os.environ.get("TURB_MEASURE_THREADS")
    if forced:return int(forced)
    try:return max(1, min(16, len(os.sched_getaffinity(0))))
    except AttributeError:return max(1, min(4, os.cpu_count() or 1))


def solve(config, rundir, variant, mach, n, a_rms, seed, tend_T, exe, mpi):
    """Run RAMSES in rundir to tend_T, continuing from its newest complete output if there is one.
    Returns (state, frames, perf): state "complete", "failed" (RAMSES stopped by itself) or "killed";
    frames {frame index: output directory}."""
    rundir.mkdir(parents=True, exist_ok=True)
    T, targets = frame_times(config, mach, tend_T);tol = 1e-6*T/int(config["frames_per_T"])
    complete = X.ramses_complete_outputs(rundir);restart = X.ramses_index(complete[-1]) if complete else 0
    for o in rundir.glob("output_*"):
        if X.ramses_index(o) > restart:shutil.rmtree(o)                 # partial outputs of the attempt that was stopped
    finished = bool(complete) and abs(X.ramses_time(complete[-1]) - targets[-1]) <= tol
    ranks = X.mpi_ranks(mpi);log = rundir/"ramses.log";perf = {};code = 0
    steps_before = int(re.search(r"^nstep_coarse\s*=\s*(\d+)", (complete[-1]/f"info_{complete[-1].name.split('_')[-1]}.txt").read_text(), re.M).group(1)) if complete else 0
    if finished:print(f"   RAMSES finished in an earlier attempt (the output at t = {targets[-1]:g} is there)", flush=True)
    else:
        (rundir/"turb.nml").write_text(namelist(config, variant, mach, n, a_rms, seed, targets, ranks, restart))
        cmd = shlex.split(mpi or "") + [str(exe), "turb.nml"];start = time.perf_counter()
        with log.open("a" if restart else "w") as f:
            if restart:f.write(f"\n=== resumed from output {restart} (t = {X.ramses_time(complete[-1]):.6g}) ===\n");f.flush()
            try:code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=rundir).returncode
            except FileNotFoundError as err:
                f.write(f"could not start {cmd[0]}: {err}\n");code = 127
        perf["wall_seconds"] = time.perf_counter() - start
    text = log.read_text(errors="replace") if log.is_file() else "";last = text.rsplit("=== resumed from output", 1)[-1]
    if restart:perf.update(resumed=True, resumed_from=f"output_{restart:05d}")
    if code != 0 and X.was_killed(code, last):return "killed", {}, perf
    good = X.ramses_complete_outputs(rundir);frames = X.ramses_frames(good, targets, tol)
    reached = bool(good) and abs(X.ramses_time(good[-1]) - targets[-1]) <= tol
    m = re.findall(r"Total elapsed time:\s*([\d.eE+-]+)", last) or re.findall(r"^\s*([\d.]+)\s+100\.0\s+TOTAL", last, re.M)
    if m and good and not finished:
        steps = int(re.search(r"^nstep_coarse\s*=\s*(\d+)", (good[-1]/f"info_{good[-1].name.split('_')[-1]}.txt").read_text(), re.M).group(1))
        perf.update(code_seconds=float(m[-1]), steps_this_attempt=steps - steps_before, ranks=ranks)
        if float(m[-1]) > 0:perf["cell_updates_per_second_per_rank"] = n**3*(steps - steps_before)/float(m[-1])/ranks
    if code != 0 or not reached:
        perf["reason"] = f"RAMSES ended with exit code {code} at t = {X.ramses_time(good[-1]) if good else 0.0:.6g} ({(X.ramses_time(good[-1]) if good else 0.0)/T:.2f} T); the end of its log: " + last[-600:]
        return "failed", frames, perf
    return "complete", frames, perf


def run(args, config):
    variant, mach, n = args.variant, args.mach, args.n
    forcing = tomllib.loads((args.results/"forcing.toml").read_text())[f"mach_{mkey(mach)}"]
    seed = int(config["ramses"]["seed"]);tend_T = float(config["t_end_T"]);T, targets = frame_times(config, mach, tend_T)
    prefix, projection = PROJECTIONS[args.projection]
    name = f"{prefix}-{variant}-M{mkey(mach)}-N{n}-s{seed}";rundir = args.results/name;meta_file = rundir/"run.toml"
    if meta_file.is_file() and tomllib.loads(meta_file.read_text()).get("measured"):
        print(f"kept      {name}");return 0
    if not (meta_file.is_file() and tomllib.loads(meta_file.read_text()).get("converted")):
        state, frames, perf = solve(config, rundir, variant, mach, n, forcing["a_rms"], seed, tend_T, args.exe, args.mpi)
        if state == "killed":
            print(f"killed    {name}: RAMSES was stopped from outside; run again to continue", flush=True);return X.KILLED
        rows = []
        for k, out in sorted(frames.items()):
            t, steps, u = read_ramses_hydro(out, n);file = f"tmp_{k:03d}.bin"
            np.asarray(u, dtype="<f4").ravel(order="F").tofile(rundir/file)
            rows.append(dict(frame=k, time=t, steps=steps, file=file))
        s = VARIANTS[variant]
        meta = dict(code="ramses", method=variant, solver=f"RAMSES hydro solver, MUSCL, slope_type={s['slope_type']}, riemann={s['riemann']}, courant_factor={s['courant']}",
                    projection=projection,
                    N=n, mach_target=mach, a_rms=float(forcing["a_rms"]), turb_T=T, turb_ndt=int(config["turb_ndt"]), gamma=gamma_of(config),
                    comp_frac=float(config["comp_frac"]), spectrum=config["spectrum"], seed=seed, cfl=s["courant"], t_end=targets[-1], t_end_T=tend_T,
                    device=f"CPU {os.uname().nodename} x {X.mpi_ranks(args.mpi)} ranks", precision="double; frames stored as Float32",
                    slices_every=int(config.get("slices_every", 4)), status=state, converted=True, measured=False, steps=rows[-1]["steps"] if rows else 0,
                    perf={k: v for k, v in perf.items() if k != "reason"}, frames=rows)
        if state == "failed":meta["reason"] = perf["reason"];meta["failed_at_time_over_T"] = (rows[-1]["time"] if rows else 0.0)/T
        meta_file.write_text(X.toml_dump(meta))
        for o in rundir.glob("output_*"):shutil.rmtree(o)          # the frames are converted; the native output is not kept
    julia = os.environ.get("MHD_JULIA", "julia")
    rc = subprocess.run([julia, "--startup-file=no", f"--project={SWEEP/'env'}", f"--threads={julia_threads()}", str(HERE/"turb_measure.jl"), str(rundir)]).returncode
    meta = tomllib.loads(meta_file.read_text())
    if rc != 0 or not meta.get("measured"):
        print(f"UNMEASURED {name}: turb_measure.jl ended with exit code {rc}; the frames are kept, run again to measure", flush=True);return 1
    reached = meta["frames"][-1]["time"]/T if meta["frames"] else 0.0
    print(f"{meta['status']:9s} {name}: {len(meta['frames'])} frames to {reached:.2f} T, {meta['steps']} steps"
          + (f"; {meta['reason'][:300]}" if meta["status"] == "failed" else ""), flush=True)
    return 0


def tag_of(config_path):
    """What tells the jobs and trial directories of one configuration file from another's: "" for config.toml,
    "-highmach" for config_highmach.toml."""
    stem = Path(config_path).stem
    return "" if stem == "config" else "-" + stem.removeprefix("config_").replace("_", "-")


def calibrate(args, config):
    cal = config["calibration"];n = int(cal["n"]);tend_T = float(cal["t_end_T"]);start = float(cal["from_T"])
    projection = cal.get("projection", "released")
    if args.projection != projection:raise SystemExit(f"the configuration calibrates with the {projection} projection; this job was given --projection {args.projection}")
    tolerance = float(cal["tolerance"]);gamma = gamma_of(config);seed = int(config["ramses"]["seed"])
    file = args.results/"forcing.toml";forcing = tomllib.loads(file.read_text()) if file.is_file() else {}
    for mach in config["machs"]:
        key = f"mach_{mkey(mach)}"
        if key in forcing:
            print(f"kept      {key}: a_rms = {forcing[key]['a_rms']:.6g} gave Mach {forcing[key]['mach_measured']:.3f}");continue
        a = float(cal["guess"][mkey(mach)]);history = [];T = 1.0/(2.0*float(mach))
        for it in range(1, int(cal["max_iterations"]) + 1):
            rundir = args.results/"calibration"/f"M{mkey(mach)}-try{it}"
            if rundir.exists():shutil.rmtree(rundir)
            state, frames, perf = solve(config, rundir, "matched", mach, n, a, seed, tend_T, args.exe, args.mpi)
            if state == "killed":
                print("killed    the calibration was stopped from outside; run it again", flush=True);return X.KILLED
            series = []
            for k, out in sorted(frames.items()):
                t, _, u = read_ramses_hydro(out, n);series.append((t/T, mach_of(u, gamma)))
            (rundir/"mach.csv").write_text("time_over_T,mach_vol\n" + "".join(f"{t!r},{m!r}\n" for t, m in series))
            for o in rundir.glob("output_*"):shutil.rmtree(o)
            if state != "complete":
                print(f"FAILED    calibration of Mach {mach} with a_rms = {a:.6g}: {perf.get('reason', '')[:600]}", flush=True);return 1
            window = [m for t, m in series if t >= start - 1e-9];got = sum(window)/len(window)
            half = len(window)//2;drift = (sum(window[half:])/(len(window) - half))/(sum(window[:half])/half) - 1
            history.append(dict(a_rms=a, mach=got, drift=drift))
            print(f"   Mach {mach} try {it}: a_rms = {a:.6g} -> Mach {got:.3f} over {start:g}-{tend_T:g} T (second half {100*drift:+.1f}% against the first)", flush=True)
            if abs(got/float(mach) - 1) <= tolerance:break
            # How fast the Mach number follows the strength: Mach ~ a^slope, slope 1/2 if a ~ Mach^2; from the last
            # two tries once they are far enough apart to measure it.
            slope = 0.5
            if len(history) >= 2:
                (a1, m1), (a2, m2) = [(h["a_rms"], h["mach"]) for h in history[-2:]]
                if abs(np.log(a2/a1)) > 0.05 and m1 > 0 and m2 > 0:slope = float(np.clip(np.log(m2/m1)/np.log(a2/a1), 0.3, 1.0))
            a *= (float(mach)/got)**(1.0/slope)
        best = min(history, key=lambda h: abs(h["mach"]/float(mach) - 1))
        forcing[key] = dict(a_rms=best["a_rms"], mach_target=mach, mach_measured=best["mach"], converged=abs(best["mach"]/float(mach) - 1) <= tolerance,
                            turb_T=T, calibrated_with=f"RAMSES matched ({projection} projection), {n}^3, mean of mach_vol over {start:g}-{tend_T:g} T, seed {seed}",
                            projection=projection, gamma=gamma, tries=history)
        file.write_text(X.toml_dump(forcing))
        print(f"{'set' if forcing[key]['converged'] else 'NOT WITHIN TOLERANCE, set':9s} {key}: a_rms = {best['a_rms']:.6g} (Mach {best['mach']:.3f})", flush=True)
    return 0


RATE = 1.0e5        # cell updates per second and rank assumed for time limits: well below what RAMSES does


def shape(n, steps):
    """MPI ranks (the sweep's choice per grid), ranks per node and a time limit for a RAMSES run of `steps` steps."""
    ranks = {32: 4, 64: 16, 128: 32, 256: 64, 384: 96, 512: 128}.get(n, 64)
    hours = min(48, max(2, int(np.ceil(2.0*n**3*steps/(RATE*ranks)/3600.0 + 1.0))))
    return ranks, min(ranks, 16), f"{hours:02d}:00:00"


def steps_of(n, mach, tend_T, courant):
    """Rough number of steps: the fastest signal, summed over directions, taken as 6 Mach + 3 sound speeds."""
    return tend_T/(2.0*float(mach))*n*(6.0*float(mach) + 3.0)/courant


def plan(args, config):
    def state(d):
        f = args.results/d/"run.toml"
        return tomllib.loads(f.read_text()) if f.is_file() else {}
    forcing = tomllib.loads((args.results/"forcing.toml").read_text()) if (args.results/"forcing.toml").is_file() else {}
    cal = config["calibration"];tries = int(cal["max_iterations"])*len(config["machs"])
    ranks, per_node, limit = shape(int(cal["n"]), tries*max(steps_of(int(cal["n"]), m, float(cal["t_end_T"]), 0.5) for m in config["machs"]))
    done = all(f"mach_{mkey(m)}" in forcing for m in config["machs"]);tag = tag_of(args.config)
    print("calibrate", "done" if done else "todo", ranks, per_node, limit, cal.get("projection", "released"), f"turb-cal{tag}")
    # A kernel that stopped in its trial has no run directories (hydro16): it is not waited for.
    stopped = {m for m in config["gpu"]["methods"] if state(f"smoke{tag}/gpu-{m}").get("status") == "failed"}
    gpu = [f"gpu-{method}-M{mkey(m)}-N{n}-s{seed}" for m in config["machs"] for n in config["grids"] for method in config["gpu"]["methods"]
           for seed in config["gpu"]["seeds"] if method not in stopped]
    gpu += [f"gpu-{method}-M{mkey(m)}-N{n}-s{seed}" for m in config["machs"] for n in config["gpu"].get("check_grids", [])
            for method in config["gpu"].get("check_methods", []) for seed in config["gpu"]["seeds"]]
    print("gpu", "done" if all(state(d).get("status") in ("complete", "failed") for d in gpu) else "todo", f"turb-gpu{tag}")
    seed = int(config["ramses"]["seed"])
    for projection in config["ramses"].get("projections", ["released"]):
        for variant in config["ramses"]["variants"]:
            for m in config["machs"]:
                for n in config["grids"]:
                    name = f"{PROJECTIONS[projection][0]}-{variant}-M{mkey(m)}-N{n}-s{seed}"
                    ranks, per_node, limit = shape(n, steps_of(n, m, float(config["t_end_T"]), VARIANTS[variant]["courant"]))
                    print("run", variant, m, n, "done" if state(name).get("measured") else "todo", ranks, per_node, limit, name, projection)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "run", "plan"):
        q = sub.add_parser(name);q.add_argument("config", type=Path);q.add_argument("results", type=Path)
        if name == "run":
            q.add_argument("variant", choices=sorted(VARIANTS));q.add_argument("mach", type=lambda s: float(s) if "." in s else int(s));q.add_argument("n", type=int)
        if name != "plan":q.add_argument("--projection", choices=sorted(PROJECTIONS), default="released")
        if name != "plan":
            q.add_argument("--exe", required=True, type=Path);q.add_argument("--mpi", default="")
    args = p.parse_args();config = tomllib.loads(args.config.read_text());args.results.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":return plan(args, config)
    args.exe = args.exe.resolve()
    return calibrate(args, config) if args.command == "calibrate" else run(args, config)


if __name__ == "__main__":
    sys.exit(main())
