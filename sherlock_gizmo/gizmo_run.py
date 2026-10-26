"""GIZMO in the comparison suite, through the sweep's external-code driver.

    python3 gizmo_run.py gizmo mfm CONFIG.toml N OUTDIR --exe PATH --mpi "srun -n K" --geometry plane [--repeats 1] [--resume]

The same command line, run directory layout, records and measurements as RAMSES, AthenaK, Athena++ and AREPO in
sherlock_sweep/codes/external.py: this file adds only what is specific to GIZMO (initial conditions, parameter file,
reading its snapshots) and then calls that driver's main(). Nothing in the sweep bundle is changed. It follows
sherlock_arepo/arepo_run.py, since GIZMO and AREPO share GADGET's file formats and timestep scheme.

GIZMO (github.com/pfhopkins/gizmo): mesh-free finite-volume Godunov methods (Hopkins 2015), MHD with Dedner cleaning
and the constrained-gradient correction (Hopkins & Raives 2016; Hopkins 2016). One set-up:
  mfm   meshless finite mass: the particles move with the gas and keep their mass (GIZMO's default method)
GIZMO's other methods were tried and left out (build.sh says why): meshless finite volume and the regular grid fail
from the start at this commit, pressure-SPH does not conserve energy here. One particle per cell centre of the other codes' grid to start, cubic-spline kernel with DesNumNgb neighbours
(20 in 2D, 32 in 3D, GIZMO's test problems), one global timestep, double precision. The Courant factor is GIZMO's own
for MHD, 0.2: it can be set only in GIZMO's DEVELOPER_MODE, which this build leaves off.

Units: GIZMO works internally with mu_0 = 1, as we do; with UnitMagneticField_in_gauss = sqrt(4 pi) (its own default
for cgs code units) the initial conditions and the snapshots carry B in our units unchanged.

Output times: as AREPO, GIZMO's steps are power-of-two fractions of TimeMax - TimeBegin; TimeMax is a power of two
times the output interval, which makes every output time a synchronisation point, and the run is ended with GIZMO's
stop file once the last snapshot is written.

Snapshots to our (N, N, Nz, 9) cell state: each particle's mass, momentum, total energy and field times its volume
(mass / density) are deposited cloud-in-cell on the grid and divided by the cell volume, so the grid holds exactly the
particles' mass, momentum, energy and magnetic flux; a particle still at a cell centre lands wholly in its cell. psi = 0.
"""
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SWEEP = Path(os.environ.get("SWEEP_BUNDLE", Path(__file__).resolve().parents[1]/"sherlock_sweep"))
sys.path.insert(0, str(SWEEP/"codes"))
import external as X  # noqa: E402

COURANT = 0.2                         # GIZMO's own for MHD: CourantFac is a tag of DEVELOPER_MODE only (begrun.cc: 0.4, at most 0.2 with MAGNETIC)
UNIT_B = 3.5449077018110318           # sqrt(4 pi): GIZMO's code unit of B in Gauss for cgs code units, i.e. ours
X.VARIANTS["gizmo"] = {
    "mfm": dict(method="meshless finite mass", divb="Dedner + constrained gradient", courant=COURANT, precision="fp64"),
}


# ---------------------------------------------------------------- initial conditions
def read_state(prefix, n, nz):
    """The sweep's initial cell state: (n, n, nz, 9) float64 from the Float32 file (x fastest)."""
    return np.fromfile(prefix + ".state", dtype="<f4").reshape((n, n, nz, 9), order="F").astype(np.float64)


def cell_centres(n, nz):
    """(n*n*nz, 3) centres in the order of u.reshape(-1, 9): x slowest, z fastest; z = 0 for a plane."""
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(nz), indexing="ij")
    return np.column_stack([(i.ravel() + .5)/n, (j.ravel() + .5)/n, (k.ravel() + .5)/n if nz > 1 else np.zeros(i.size)])


def write_ic(u, path):
    """GIZMO initial conditions (HDF5) from our cell state: one particle per cell centre, the cell's mass."""
    import h5py
    n, _, nz, _ = u.shape;count = n*n*nz;volume = 1.0/(n*n*nz)                     # 2D: surface densities times areas
    f = u.reshape(count, 9);rho = f[:, 0];vel = f[:, 1:4]/rho[:, None];b = f[:, 5:8]
    thermal = f[:, 4] - .5*rho*(vel**2).sum(1) - .5*(b**2).sum(1)
    with h5py.File(path, "w") as ic:
        h = ic.create_group("Header");num = np.array([count, 0, 0, 0, 0, 0], dtype=np.int32)
        for key, value in (("NumPart_ThisFile", num), ("NumPart_Total", num), ("NumPart_Total_HighWord", np.zeros(6, np.int32)),
                           ("MassTable", np.zeros(6)), ("Time", 0.0), ("Redshift", 0.0), ("BoxSize", 1.0), ("NumFilesPerSnapshot", 1),
                           ("Omega0", 0.0), ("OmegaLambda", 0.0), ("HubbleParam", 1.0), ("Flag_Sfr", 0), ("Flag_Cooling", 0),
                           ("Flag_StellarAge", 0), ("Flag_Metals", 0), ("Flag_Feedback", 0), ("Flag_DoublePrecision", 1)):
            h.attrs.create(key, value)
        p = ic.create_group("PartType0")
        p.create_dataset("ParticleIDs", data=np.arange(1, count + 1, dtype=np.int64 if count > 2**31 - 2 else np.int32))
        p.create_dataset("Coordinates", data=cell_centres(n, nz))
        p.create_dataset("Masses", data=rho*volume)
        p.create_dataset("Velocities", data=vel)
        p.create_dataset("InternalEnergy", data=thermal/rho)
        p.create_dataset("MagneticField", data=b)


# ---------------------------------------------------------------- parameter file
def timeline(targets):
    """(interval, TimeMax): the outputs are every `interval`; TimeMax is the next power of two of them."""
    count = len(targets) - 1;interval = (targets[-1] - targets[0])/count
    return interval, targets[0] + interval*2**math.ceil(math.log2(count))


def parameters(settings, targets, memory_mb, dims):
    """GIZMO's tags for our times, units and memory. The tags of its DEVELOPER_MODE (Courant factor, integration accuracy,
    ...) are not set: GIZMO stops at a tag its build does not know, and those keep GIZMO's own values."""
    interval, time_max = timeline(targets)
    values = dict(
        InitCondFile="./IC", ICFormat=3, OutputDir="./output/", SnapshotFileBase="snapshot", SnapFormat=3,
        TimeBegin=repr(targets[0]), TimeMax=repr(time_max), TimeOfFirstSnapshot=repr(targets[0]), TimeBetSnapshot=repr(interval),
        BoxSize=1.0, ComovingIntegrationOn=0, MaxSizeTimestep=repr(interval),
        MinSizeTimestep=repr(interval*1e-10), DesNumNgb=20 if dims == 2 else 32, MaxMemSize=memory_mb, TimeLimitCPU=1.0e9,
        CpuTimeBetRestartFile=1.0e9,
        UnitLength_in_cm=1.0, UnitMass_in_g=1.0, UnitVelocity_in_cm_per_s=1.0, UnitMagneticField_in_gauss=repr(UNIT_B),
        GravityConstantInternal=0.0, InitGasTemp=0.0, MinGasTemp=0.0)
    return "% written by gizmo_run.py (abel_mhd comparison suite)\n" + "".join(f"{k:36s}{v}\n" for k, v in values.items())


# ---------------------------------------------------------------- snapshots
def snapshot_time(path):
    import h5py
    with h5py.File(path, "r") as s:return float(s["Header"].attrs["Time"])


def deposit(pos, ext, n, nz):
    """Cloud-in-cell deposit of per-particle extensive quantities `ext` (count, k) on the periodic grid: (n, n, nz, k)
    sums. A particle at a cell centre lands wholly in that cell; one between centres is shared by area (volume)."""
    dims = 3 if nz > 1 else 2
    x = np.mod(pos[:, :dims], 1.0)*n - 0.5;i0 = np.floor(x).astype(int);f = x - i0
    out = np.zeros((n, n, nz, ext.shape[1]))
    for corner in np.ndindex(*(2,)*dims):
        w = np.prod([f[:, d] if corner[d] else 1 - f[:, d] for d in range(dims)], axis=0)
        idx = [(i0[:, d] + corner[d]) % n for d in range(dims)] + ([] if dims == 3 else [np.zeros(len(w), dtype=int)])
        np.add.at(out, tuple(idx), w[:, None]*ext)
    return out


def read_snapshot(path, n, nz, record=None):
    """(time, (n, n, nz, 9) cell state) of a GIZMO snapshot; conserved variables, B in our units, psi = 0.
    Each particle's mass, momentum, total energy and field times volume are deposited cloud-in-cell and divided by the
    cell volume, so the grid holds exactly the particles' mass, momentum, energy and magnetic flux.
    record: a dict that receives GIZMO's own divergence of B for this snapshot, where the snapshot has it."""
    import h5py
    with h5py.File(path, "r") as s:
        t = float(s["Header"].attrs["Time"]);p = s["PartType0"]
        pos = p["Coordinates"][:].astype(np.float64);rho = p["Density"][:].astype(np.float64);vel = p["Velocities"][:].astype(np.float64)
        e = p["InternalEnergy"][:].astype(np.float64);b = p["MagneticField"][:].astype(np.float64);mass = p["Masses"][:].astype(np.float64)
        if record is not None and "DivergenceOfMagneticField" in p:
            vol = mass/rho;div = p["DivergenceOfMagneticField"][:].astype(np.float64)
            size = vol**(1.0/(3 if nz > 1 else 2));brms = math.sqrt(float((vol*(b**2).sum(1)).sum()/vol.sum()))
            record.update(time=t, particles=len(rho), div_rms=math.sqrt(float((vol*div**2).sum()/vol.sum())), b_rms=brms,
                          normalized_div_rms=math.sqrt(float((vol*(div*size)**2).sum()/vol.sum()))/brms if brms > 0 else 0.0,
                          smallest_cell=float(size.min()), largest_cell=float(size.max()))
    vol = mass/rho
    ext = np.column_stack([mass, mass[:, None]*vel, mass*(e + .5*(vel**2).sum(1)) + .5*vol*(b**2).sum(1), vol[:, None]*b])
    u = deposit(pos, ext, n, nz)*n**(3 if nz > 1 else 2)
    return t, np.concatenate([u, np.zeros(u.shape[:3] + (1,))], axis=-1)


def last_sync(log):
    """(step, time) of the last Sync-Point in GIZMO's log, from its end (the log of a long run is large); None if none."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2);f.seek(max(0, f.tell() - 65536));tail = f.read().decode(errors="replace")
    except OSError:return None
    m = re.findall(r"Sync-Point (\d+), Time: ([-+0-9.eE]+)", tail)
    return (int(m[-1][0]), float(m[-1][1])) if m else None


def steps_of(text):
    """[(time, step number)] from GIZMO's log ("Sync-Point 12, Time: 0.0375, ...")."""
    return [(float(t), int(k)) for k, t in re.findall(r"Sync-Point (\d+), Time: ([-+0-9.eE]+)", text)]


# ---------------------------------------------------------------- the runner
def run_gizmo(args, config, spec, settings, rundir, rep, outputs, resume=False):
    planar = args.geometry == "plane";n = args.n;nz = 1 if planar else n;dims = 2 if planar else 3
    targets = X.output_times(config) if outputs else [0.0, float(config["t_end"])]
    tol = 1e-6*float(config["t_end"])/(int(config["frames"]) - 1);prefix = spec.removesuffix(".spec")
    if not Path(prefix + ".state").is_file():                # the driver writes the cell state for RAMSES only
        seed = args.seed if args.seed is not None else config.get("seeds", [42])[0]
        X.write_spec(args.config, n, prefix, seed, state=True, plane=planar)
    out = rundir/"output";log = rundir/"gizmo.log"
    last = out/f"snapshot_{len(targets) - 1:03d}.hdf5"
    finished = resume and last.is_file() and abs(snapshot_time(last) - targets[-1]) <= tol     # only the conversion was interrupted
    # The sweep's runaway guard (a collapsing timestep), as for the other codes: max_steps_per_n * N steps.
    cap = X.step_cap(config, n);reached = last_sync(log) if resume and log.is_file() else None
    capped = cap > 0 and reached is not None and reached[0] >= cap        # an earlier attempt already went past it
    if capped:print(f"GIZMO went past the step limit ({cap}) in an earlier attempt (step {reached[0]}, t = {reached[1]:g}): "
                    "its outputs are converted, it is not run again", flush=True)
    start = time.perf_counter();code = 0;stopped = False
    if not finished and not capped:
        # A requeued attempt starts this run again; GIZMO's own restart files are not used.
        for old in (out, rundir/"restartfiles"):shutil.rmtree(old, ignore_errors=True)
        out.mkdir(parents=True)
        write_ic(read_state(prefix, n, nz), rundir/"IC.hdf5")
        memory = int(os.environ.get("GIZMO_MEMORY_MB_PER_RANK", 3000))
        (rundir/"params.txt").write_text(parameters(settings, targets, memory, dims))
        cmd = shlex.split(args.mpi or "") + [args.exe, "params.txt"]
        with log.open("w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir)
            while proc.poll() is None:                           # end the run once the last snapshot is complete
                time.sleep(1.0)
                if not stopped and last.is_file():
                    try:
                        if abs(snapshot_time(last) - targets[-1]) <= tol:(out/"stop").touch();stopped = True
                    except (OSError, KeyError):pass              # still being written
                if not stopped and cap > 0:                      # or once it is past the step limit
                    now = last_sync(log)
                    if now and now[0] >= cap:(out/"stop").touch();stopped = capped = True
            code = proc.returncode
        print(f"GIZMO ended with exit code {code} on {X.mpi_ranks(args.mpi)} ranks" + (" after its stop file" if stopped else ""), flush=True)
    wall = time.perf_counter() - start;text = log.read_text(errors="replace") if log.is_file() else ""
    perf = dict(wall_seconds=wall)
    if finished:perf["converted_only"] = True
    sync = steps_of(text);snaps = sorted(out.glob("snapshot_*.hdf5")) if out.is_dir() else []
    times = {}
    for s in snaps:
        try:times[s] = snapshot_time(s)
        except (OSError, KeyError):pass
    keep = [s for s in snaps if s in times and any(abs(times[s] - t) <= tol for t in targets)]
    if sync:
        done = [k for t, k in sync if t <= targets[-1] + tol];perf["steps"] = max(done) if done else 0

    def frames():
        rows = []
        for s in keep:
            rec = {};t, u = read_snapshot(s, n, nz, record=rec)
            near = min(sync, key=lambda x: abs(x[0] - t))[1] if sync else 0
            if rec:rows.append(rec)
            yield t, near, u
        if rows:                                          # GIZMO's own measure of div B on its particles, per output
            keys = list(rows[0]);(rundir/"gizmo_divergence.csv").write_text(",".join(keys) + "\n" + "".join(",".join(repr(r[k]) for k in keys) + "\n" for r in rows))
    complete = bool(keep) and abs(times[keep[-1]] - targets[-1]) <= tol
    tail = text[-2000:]
    own = re.search(r"ENDRUN issued on task=\d+, function '([^']*)', file '([^']*)', line (\d+): error level (\d+)", text)
    if own and not complete:
        # GIZMO stopped itself (endrun aborts every rank through MPI_Abort). That is a result of the run, to be recorded as
        # failed; srun's report of the abort must not read like a job being stopped (as with AREPO, arepo_run.py).
        steps = steps_of(text)
        perf["gizmo_stopped"] = f"endrun in {own.group(1)} ({own.group(2)}:{own.group(3)}), error level {own.group(4)}"   # the name carries its ()
        if steps:perf.update(gizmo_stopped_at_time=steps[-1][0], gizmo_sync_points=steps[-1][1])
        before = text[:own.start()].strip().splitlines()[-3:]
        tail = "GIZMO stopped itself" + (f" at t = {steps[-1][0]:g} (sync point {steps[-1][1]})" if steps else "") + ": " \
            + perf["gizmo_stopped"] + ("; " + " | ".join(before) if before else "")
    if capped and not complete:
        at = last_sync(log);perf["gizmo_stopped"] = f"step limit {cap}"
        if at:perf.update(gizmo_stopped_at_time=at[1], gizmo_sync_points=at[0])
        # GIZMO's log ends with Slurm's report of the earlier attempt being stopped (time limit): that is not this
        # result, and the driver would take it for this job being killed and record nothing (job 49641025).
        kept = "\n".join(l for l in tail.splitlines() if not X.KILL_TEXT.search(l))
        tail = f"Step limit reached ({cap})" + (f" at t = {at[1]:g}, timestep collapsing" if at else "") + "\n" + kept
    rc = 0 if complete and (code == 0 or stopped) and not X.was_killed(code, tail) else (code or 1)
    return rc, tail, perf, frames()


def remove_native(outdir):
    """After a finished run: drop GIZMO's snapshots, restart files and initial conditions (converted; can be rebuilt);
    keep its log, parameter file and statistics."""
    for record in Path(outdir).glob("gizmo_*_r*/run.toml"):
        if 'status = "complete"' not in record.read_text():continue
        run = record.parent
        for f in list(run.glob("output/snapshot_*.hdf5")) + [run/"IC.hdf5"]:f.unlink(missing_ok=True)
        shutil.rmtree(run/"output"/"restartfiles", ignore_errors=True)


X.RUNNERS["gizmo"] = run_gizmo

if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    try:X.main()
    finally:
        if "--keep-native" not in sys.argv and len(positional) >= 5:remove_native(positional[4])
