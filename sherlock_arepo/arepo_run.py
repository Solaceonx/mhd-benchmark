"""AREPO in the comparison suite, through the sweep's external-code driver.

    python3 arepo_run.py arepo static|moving CONFIG.toml N OUTDIR --exe PATH --mpi "srun -n K" --geometry plane [--repeats 1] [--resume]

The same command line, run directory layout, records and measurements as RAMSES, AthenaK and Athena++ in
sherlock_sweep/codes/external.py: this file only adds what is specific to AREPO (initial conditions, parameter file,
reading its snapshots) and then calls that driver's main(). Nothing in the sweep bundle is changed.

AREPO (public release, arepo-code.org): finite volumes on a Voronoi mesh, second order, HLLD, Powell's eight-wave
scheme for the divergence of B. Two set-ups:
  static   the mesh is the Cartesian grid of the other codes and does not move (VORONOI_STATIC_MESH)
  moving   the same initial mesh, moving with the gas (the code's normal mode, with its mesh regularisation)

Initial data: the Float32 cell state the sweep writes for RAMSES (tools/write_spec.jl --state), as cell masses,
velocities, specific internal energies and magnetic field. AREPO's files hold B in Gaussian units (sqrt(4 pi) times ours).

Output times: AREPO writes a snapshot at the first synchronisation point at or after the requested time and its steps
are power-of-two fractions of TimeMax - TimeBegin. TimeMax is therefore set to a power of two times the output interval,
which makes every output time a synchronisation point (the snapshots land exactly), and the run is ended with AREPO's
stop file once the last snapshot is written.

Snapshots to our (N, N, Nz, 9) cell state: on the static mesh every mesh cell is a grid cell. On the moving mesh each
grid cell is sampled at SUB x SUB points (SUB^3 in 3D) and takes the mean of the Voronoi cells that hold them (the
nearest mesh-generating point), which is the volume average of AREPO's piecewise-constant state up to the sampling.

Also kept per run: AREPO's log, parameter file and text statistics, and arepo_divergence.csv, the divergence of B that
AREPO itself reports on its own mesh at every output. Its snapshots are removed once converted (--keep-native keeps them).
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

GAUSS = math.sqrt(4*math.pi)
SUB = 4                               # samples per grid cell and dimension when a moving mesh is put on the grid
COURANT = float(os.environ.get("AREPO_COURANT", 0.3))        # 0.3: AREPO's examples; the variable is for check runs
X.VARIANTS["arepo"] = {
    "static": dict(mesh="static", riemann="hlld", divb="powell", courant=COURANT, precision="fp64"),
    "moving": dict(mesh="moving", riemann="hlld", divb="powell", courant=COURANT, precision="fp64"),
}


# ---------------------------------------------------------------- initial conditions
def read_state(prefix, n, nz):
    """The sweep's initial cell state: (n, n, nz, 9) float64 from the Float32 file (x fastest)."""
    return np.fromfile(prefix + ".state", dtype="<f4").reshape((n, n, nz, 9), order="F").astype(np.float64)


def cell_centres(n, nz):
    """(n*n*nz, 3) centres in the order of u.reshape(-1, 9): x slowest, z fastest; z = 0 for a plane."""
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(nz), indexing="ij")
    pos = np.column_stack([(i.ravel() + .5)/n, (j.ravel() + .5)/n, (k.ravel() + .5)/n if nz > 1 else np.zeros(i.size)])
    return pos


def write_ic(u, path):
    """AREPO initial conditions (HDF5) from our cell state: one mesh-generating point per cell centre."""
    import h5py
    n, _, nz, _ = u.shape;count = n*n*nz;volume = 1.0/(n*n*nz)                     # 2D: masses are surface densities times areas
    f = u.reshape(count, 9);rho = f[:, 0];vel = f[:, 1:4]/rho[:, None];b = f[:, 5:8]
    thermal = f[:, 4] - .5*rho*(vel**2).sum(1) - .5*(b**2).sum(1)
    with h5py.File(path, "w") as ic:
        h = ic.create_group("Header");num = np.array([count, 0, 0, 0, 0, 0], dtype=np.int32)
        for key, value in (("NumPart_ThisFile", num), ("NumPart_Total", num), ("NumPart_Total_HighWord", np.zeros(6, np.int32)),
                           ("MassTable", np.zeros(6)), ("Time", 0.0), ("Redshift", 0.0), ("BoxSize", 1.0), ("NumFilesPerSnapshot", 1),
                           ("Omega0", 0.0), ("OmegaB", 0.0), ("OmegaLambda", 0.0), ("HubbleParam", 1.0), ("Flag_Sfr", 0), ("Flag_Cooling", 0),
                           ("Flag_StellarAge", 0), ("Flag_Metals", 0), ("Flag_Feedback", 0), ("Flag_DoublePrecision", 1)):
            h.attrs.create(key, value)
        p = ic.create_group("PartType0")
        p.create_dataset("ParticleIDs", data=np.arange(1, count + 1, dtype=np.int64 if count > 2**31 - 2 else np.int32))
        p.create_dataset("Coordinates", data=cell_centres(n, nz))
        p.create_dataset("Masses", data=rho*volume)
        p.create_dataset("Velocities", data=vel)
        p.create_dataset("InternalEnergy", data=thermal/rho)
        p.create_dataset("MagneticField", data=b*GAUSS)


# ---------------------------------------------------------------- parameter file
def timeline(targets):
    """(interval, TimeMax): the outputs are every `interval`; TimeMax is the next power of two of them."""
    count = len(targets) - 1;interval = (targets[-1] - targets[0])/count
    return interval, targets[0] + interval*2**math.ceil(math.log2(count))


def parameters(settings, targets, memory_mb):
    interval, time_max = timeline(targets)
    values = dict(
        InitCondFile="./IC", ICFormat=3, OutputDir="./output/", SnapshotFileBase="snap", SnapFormat=3, NumFilesPerSnapshot=1,
        NumFilesWrittenInParallel=1, ResubmitOn=0, ResubmitCommand="none", OutputListFilename="none", OutputListOn=0,
        CoolingOn=0, StarformationOn=0, Omega0=0.0, OmegaBaryon=0.0, OmegaLambda=0.0, HubbleParam=1.0, BoxSize=1.0,
        PeriodicBoundariesOn=1, ComovingIntegrationOn=0, MaxMemSize=memory_mb, TimeOfFirstSnapshot=repr(targets[0]),
        CpuTimeBetRestartFile=1.0e9, TimeLimitCPU=1.0e9, TimeBetStatistics=repr(interval), TimeBegin=repr(targets[0]),
        TimeMax=repr(time_max), TimeBetSnapshot=repr(interval), UnitVelocity_in_cm_per_s=1.0, UnitLength_in_cm=1.0, UnitMass_in_g=1.0,
        GravityConstantInternal=0.0, ErrTolIntAccuracy=0.1, ErrTolTheta=0.1, ErrTolForceAcc=0.1, MaxSizeTimestep=repr(interval),
        MinSizeTimestep=repr(interval*1e-10), CourantFac=settings["courant"], LimitUBelowThisDensity=0.0,
        LimitUBelowCertainDensityToThisValue=0.0, DesNumNgb=64, MultipleDomains=2, TopNodeFactor=4,
        ActivePartFracForNewDomainDecomp=0.5, MaxNumNgbDeviation=2, TypeOfTimestepCriterion=1, TypeOfOpeningCriterion=1, GasSoftFactor=0.01,
        InitGasTemp=0.0, MinGasTemp=0.0, MinEgySpec=0.0, MinimumDensityOnStartUp=0.0, CellShapingSpeed=0.5, CellMaxAngleFactor=2.25)
    for kind in range(6):
        values[f"SofteningComovingType{kind}"] = 0.1;values[f"SofteningMaxPhysType{kind}"] = 0.1;values[f"SofteningTypeOfPartType{kind}"] = 0
    return "% written by arepo_run.py (abel_mhd comparison suite); the cell-shaping tags are ignored by the static-mesh build\n" \
        + "".join(f"{k:40s}{v}\n" for k, v in values.items())


# ---------------------------------------------------------------- snapshots
def snapshot_time(path):
    import h5py
    with h5py.File(path, "r") as s:return float(s["Header"].attrs["Time"])


def nearest(points, generators, dims):
    """Index of the generator nearest to each point on the periodic unit box (the first `dims` coordinates)."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(np.mod(generators[:, :dims], 1.0), boxsize=1.0).query(np.mod(points[:, :dims], 1.0))[1]
    except ImportError:               # plain numpy: generators binned on a grid, candidates from the 5^dims bins around a point
        g = np.mod(generators[:, :dims], 1.0);m = max(1, int(round(len(g)**(1/dims))));key = np.floor(g*m).astype(int) % m
        flat = np.ravel_multi_index(key.T, (m,)*dims);order = np.argsort(flat, kind="stable");start = np.searchsorted(flat[order], np.arange(m**dims + 1))
        best = np.full(len(points), -1);dist = np.full(len(points), np.inf);cell = np.floor(np.mod(points[:, :dims], 1.0)*m).astype(int)
        for shift in np.ndindex(*(5,)*dims):
            nb = np.ravel_multi_index(((cell + np.array(shift) - 2) % m).T, (m,)*dims)
            for slot in range(int((start[1:] - start[:-1]).max())):
                idx = start[nb] + slot;ok = idx < start[nb + 1];cand = order[np.where(ok, idx, 0)]
                d = np.mod(points[:, :dims], 1.0) - g[cand];d -= np.round(d);d2 = (d**2).sum(1);better = ok & (d2 < dist)
                best[better] = cand[better];dist[better] = d2[better]
        return best


def read_snapshot(path, n, nz, moving, sub=SUB, record=None):
    """(time, (n, n, nz, 9) cell state) of an AREPO snapshot; conserved variables, B in our units, psi = 0.
    record: a dict that receives AREPO's own divergence of B for this snapshot (its MagneticFieldDivergence field)."""
    import h5py
    with h5py.File(path, "r") as s:
        t = float(s["Header"].attrs["Time"]);p = s["PartType0"]
        pos = p["Coordinates"][:].astype(np.float64);rho = p["Density"][:].astype(np.float64);vel = p["Velocities"][:].astype(np.float64)
        e = p["InternalEnergy"][:].astype(np.float64);b = p["MagneticField"][:].astype(np.float64)/GAUSS
        if record is not None and "MagneticFieldDivergence" in p and "Masses" in p:
            vol = p["Masses"][:].astype(np.float64)/rho;div = p["MagneticFieldDivergence"][:].astype(np.float64)/GAUSS
            size = vol**(1.0/(3 if nz > 1 else 2));brms = math.sqrt(float((vol*(b**2).sum(1)).sum()/vol.sum()))
            record.update(time=t, cells=len(rho), div_rms=math.sqrt(float((vol*div**2).sum()/vol.sum())), b_rms=brms,
                          normalized_div_rms=math.sqrt(float((vol*(div*size)**2).sum()/vol.sum()))/brms if brms > 0 else 0.0,
                          smallest_cell=float(size.min()), largest_cell=float(size.max()))
    cons = np.column_stack([rho, rho[:, None]*vel, rho*(e + .5*(vel**2).sum(1)) + .5*(b**2).sum(1), b, np.zeros_like(rho)])
    dims = 3 if nz > 1 else 2;shape = (n, n, nz)
    if not moving:
        idx = np.floor(np.mod(pos[:, :dims], 1.0)*n).astype(int);u = np.full(shape + (9,), np.nan)
        if dims == 2:u[idx[:, 0], idx[:, 1], 0] = cons
        else:u[idx[:, 0], idx[:, 1], idx[:, 2]] = cons
        if np.isnan(u).any() or len(pos) != n*n*nz:raise RuntimeError(f"{path}: the static mesh does not cover the {n}-cell grid one to one")
        return t, u
    off = (np.arange(sub) + .5)/sub;u = np.zeros(shape + (9,))
    centres = cell_centres(n, nz);lower = centres[:, :dims] - .5/n
    for shift in np.ndindex(*(sub,)*dims):
        pts = lower + np.array([off[a] for a in shift])/n
        u += cons[nearest(pts, pos, dims)].reshape(shape + (9,))
    return t, u/sub**dims


def sync_points(text):
    """[(time, step number)] from AREPO's log."""
    return [(float(t), int(k)) for k, t in re.findall(r"Sync-Point (\d+), Time: ([-+0-9.eE]+)", text)]


# ---------------------------------------------------------------- the runner
def run_arepo(args, config, spec, settings, rundir, rep, outputs, resume=False):
    planar = args.geometry == "plane";n = args.n;nz = 1 if planar else n;moving = settings["mesh"] == "moving"
    targets = X.output_times(config) if outputs else [0.0, float(config["t_end"])]
    tol = 1e-6*float(config["t_end"])/(int(config["frames"]) - 1);prefix = spec.removesuffix(".spec")
    if not Path(prefix + ".state").is_file():                # the driver writes the cell state for RAMSES only
        seed = args.seed if args.seed is not None else config.get("seeds", [42])[0]
        X.write_spec(args.config, n, prefix, seed, state=True, plane=planar)
    out = rundir/"output";log = rundir/"arepo.log"
    last = out/f"snap_{len(targets) - 1:03d}.hdf5"
    finished = resume and last.is_file() and abs(snapshot_time(last) - targets[-1]) <= tol     # only the conversion was interrupted
    start = time.perf_counter();code = 0;stopped = False
    if not finished:
        # A requeued attempt starts this run again: the planar runs are short, and AREPO's own restart files are not used.
        for old in (out, rundir/"restartfiles"):shutil.rmtree(old, ignore_errors=True)
        write_ic(read_state(prefix, n, nz), rundir/"IC.hdf5")
        ranks = X.mpi_ranks(args.mpi);memory = int(os.environ.get("AREPO_MEMORY_MB_PER_RANK", 3000))
        (rundir/"param.txt").write_text(parameters(settings, targets, memory))
        cmd = shlex.split(args.mpi or "") + [args.exe, "param.txt"]
        with log.open("w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir)
            while proc.poll() is None:                           # end the run once the last snapshot is complete
                time.sleep(1.0)
                if not stopped and last.is_file():
                    try:
                        if abs(snapshot_time(last) - targets[-1]) <= tol:(out/"stop").touch();stopped = True
                    except (OSError, KeyError):pass              # still being written
            code = proc.returncode
        print(f"AREPO ended with exit code {code} on {ranks} ranks" + (" after its stop file" if stopped else ""), flush=True)
    wall = time.perf_counter() - start;text = log.read_text(errors="replace") if log.is_file() else ""
    perf = dict(wall_seconds=wall)
    if finished:perf["converted_only"] = True
    sync = sync_points(text);snaps = sorted(out.glob("snap_*.hdf5")) if out.is_dir() else []
    times = {}
    for s in snaps:
        try:times[s] = snapshot_time(s)
        except (OSError, KeyError):pass
    keep = [s for s in snaps if s in times and any(abs(times[s] - t) <= tol for t in targets)]
    if sync:
        done = [k for t, k in sync if t <= targets[-1] + tol];perf["steps"] = max(done) if done else 0
        cpu = re.findall(r"^total\s+([0-9.]+)", (out/"cpu.txt").read_text(errors="replace"), re.M) if (out/"cpu.txt").is_file() else []
        if cpu:perf["arepo_cpu_txt_total_seconds"] = float(cpu[-1])

    def frames():
        rows = []
        for s in keep:
            rec = {};t, u = read_snapshot(s, n, nz, moving, record=rec)
            near = min(sync, key=lambda x: abs(x[0] - t))[1] if sync else 0
            if rec:rows.append(rec)
            yield t, near, u
        if rows:                                          # AREPO's own measure of div B on its own mesh, per output
            keys = list(rows[0]);(rundir/"arepo_divergence.csv").write_text(",".join(keys) + "\n" + "".join(",".join(repr(r[k]) for k in keys) + "\n" for r in rows))
    complete = bool(keep) and abs(times[keep[-1]] - targets[-1]) <= tol
    tail = text[-2000:]
    own = re.search(r"TERMINATE:[ *!]*(.*)", text)
    if own and not complete:
        # AREPO stopped itself (terminate() ends every rank through MPI_Abort). That is a result of the run, to be
        # recorded as failed. Handed on as it is, srun's report of the abort ("Job step aborted", "task 3: Killed")
        # reads like a job being stopped, and the sweep's driver would record nothing and ask for a requeue
        # (tasks 9, 11, 17, 21 of job 46813282).
        steps = re.findall(r"Sync-Point (\d+), Time: ([-+0-9.eE]+), (?:Redshift: [-+0-9.eE]+, )?Systemstep: ([-+0-9.eE]+)", text)
        reason = own.group(1).strip()[:300]
        perf["arepo_stopped"] = reason
        if steps:perf.update(arepo_stopped_at_time=float(steps[-1][1]), arepo_last_timestep=float(steps[-1][2]), arepo_sync_points=int(steps[-1][0]))
        tail = "AREPO stopped itself" + (f" at t = {float(steps[-1][1]):g} (sync point {steps[-1][0]}, last timestep {float(steps[-1][2]):.3g})" if steps else "") + ": " + reason
    rc = 0 if complete and (code == 0 or stopped) and not X.was_killed(code, tail) else (code or 1)
    return rc, tail, perf, frames()


def remove_native(outdir):
    """After a finished run: drop AREPO's snapshots, restart files and initial conditions (they are converted and can be
    rebuilt); keep its logs, parameter file and text statistics."""
    for record in Path(outdir).glob("arepo_*_r*/run.toml"):
        if 'status = "complete"' not in record.read_text():continue
        run = record.parent
        for f in list(run.glob("output/snap_*.hdf5")) + [run/"IC.hdf5"]:f.unlink(missing_ok=True)
        shutil.rmtree(run/"restartfiles", ignore_errors=True)


X.RUNNERS["arepo"] = run_arepo

if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    try:X.main()
    finally:
        if "--keep-native" not in sys.argv and len(positional) >= 5:remove_native(positional[4])
