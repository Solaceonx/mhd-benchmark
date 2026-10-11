"""Run external MHD codes on the comparison suite and convert their output to our format.

Each run produces the same layout as src/run_suite.jl (environment.toml, <method>_r<rep>/
run.toml, state_###.bin), after which tools/postprocess_external.jl writes diagnostics.csv,
errors and ref_###.bin with the same Julia functions the GPU driver uses.

  python3 codes/external.py CODE VARIANT CONFIG N OUTDIR [--repeats R] [--seed S]
      [--exe PATH] [--threads T] [--mpi "srun -n 8"] [--geometry cube|plane]

CODE: athenak | athenapp | ramses. VARIANT: a key of VARIANTS[CODE].
Repeat 1 writes every output; later repeats write only the final state (timing runs).
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
FIELDS = ("rho", "mx", "my", "mz", "E", "Bx", "By", "Bz", "psi")

# Solver settings. "matched" mirrors the GLM/CT GPU kernels as closely as each code allows
# (second-order PLM, HLL-family flux, two-stage time integration); "best" is each code's
# recommended accurate configuration; "reference" is for the high-resolution truth runs.
VARIANTS = {
    # AthenaK runs in double precision, its supported configuration: its single-precision build
    # fails to compile at the pinned commit (27 type errors across unrelated modules). GPU cost
    # comparisons with the Float32 GLM/CT kernels must state this precision difference.
    "athenak": {
        "matched": dict(reconstruct="plm", rsolver="hlle", integrator="rk2", cfl=0.3, precision="fp64"),
        "best": dict(reconstruct="ppm4", rsolver="hlld", integrator="rk3", cfl=0.3, precision="fp64"),
        "reference": dict(reconstruct="ppm4", rsolver="hlld", integrator="rk3", cfl=0.3, precision="fp64"),
    },
    "athenapp": {
        "reference": dict(xorder=3, integrator="rk3", cfl=0.3),
        "matched": dict(xorder=2, integrator="vl2", cfl=0.3),
    },
    "ramses": {
        "matched": dict(riemann="hll", riemann2d="hll", slope_type=2, courant=0.5),
        "best": dict(riemann="hlld", riemann2d="hlld", slope_type=2, courant=0.8),
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    config = tomllib.loads(Path(path).read_text())
    config.setdefault("problem", config["name"])
    return config


def planar(config):
    p = config["problem"]
    if p == "blast":return config.get("dim", 3) == 2
    return p not in ("orszag_tang_3d", "field_loop_3d", "decay")


def gamma_of(config):
    g = config.get("gamma", {"brio_wu": "2", "brio_wu_rotated": "2", "rotor": "7/5", "blast": "7/5"}.get(config["problem"], "5/3"))
    if isinstance(g, str) and "/" in g:
        a, b = g.split("/");return float(a)/float(b)
    return float(g)


def output_times(config):
    # Plain floats: NumPy 2 formats np.float64 as "np.float64(...)", which codes cannot parse.
    return [float(t) for t in np.linspace(0.0, float(config["t_end"]), int(config["frames"]))]


def mpi_ranks(command):
    """Ranks requested by a launcher string, falling back to the Slurm allocation."""
    tokens=shlex.split(command or "")
    for i,token in enumerate(tokens):
        if token in ("-n","--ntasks") and i+1<len(tokens):return int(tokens[i+1])
        if token.startswith("--ntasks="):return int(token.split("=",1)[1])
    return int(os.environ.get("SLURM_NTASKS","1"))


def athenapp_block(n, dims, ranks, maximum=64, minimum=8):
    """Largest divisor no bigger than maximum that provides a block per MPI rank."""
    block=min(n,maximum)
    while block>=minimum and (n%block or (n//block)**dims<ranks):block//=2
    if block<minimum:
        raise ValueError(f"Athena++ grid N={n} has fewer {minimum}-cell blocks than {ranks} MPI ranks")
    return block


def write_spec(config_path, n, prefix, seed, state=False):
    cmd = [os.environ.get("MHD_JULIA", "julia"), "--startup-file=no", str(BUNDLE/"tools/write_spec.jl"),
           str(config_path), str(n), str(prefix), "--seed", str(seed)] + (["--state"] if state else [])
    subprocess.run(cmd, check=True)


def block_average(u, r):
    """Conservative r^d block average of an (N,N,Nz,9) state; planar (Nz=1) stays planar."""
    n, _, nz, c = u.shape;rz = r if nz > 1 else 1
    return u.reshape(n//r, r, n//r, r, nz//rz, rz, c).mean(axis=(1, 3, 5))


def write_state(path, u, full):
    """u: (N,N,Nz,9) float array, Julia/Fortran order with x fastest."""
    state = u if full else u[:, :, 0, :]
    np.asarray(state, dtype="<f4").ravel(order="F").tofile(path)


# ------------------------------------------------------------------------------ AthenaK
def athenak_input(config, n, spec, settings, geometry, outputs, basename):
    nz = 1 if geometry == "plane" else n
    t_end = float(config["t_end"]);dt_out = t_end/(int(config["frames"]) - 1)
    lines = [
        "<job>", f"basename = {basename}",
        "<mesh>", "nghost = 3" if settings["reconstruct"] != "plm" else "nghost = 2",
    ]
    for d, nd in zip((1, 2, 3), (n, n, nz)):
        lines += [f"nx{d} = {nd}", f"x{d}min = 0.0", f"x{d}max = 1.0", f"ix{d}_bc = periodic", f"ox{d}_bc = periodic"]
    lines += ["<meshblock>", f"nx1 = {n}", f"nx2 = {n}", f"nx3 = {nz}",
        "<time>", "evolution = dynamic", f"integrator = {settings['integrator']}", f"cfl_number = {settings['cfl']}",
        "nlim = -1", f"tlim = {t_end!r}", "ndiag = 1000000",
        "<mhd>", "eos = ideal", f"reconstruct = {settings['reconstruct']}", f"rsolver = {settings['rsolver']}",
        f"gamma = {gamma_of(config)!r}", "dfloor = 1.0e-6", "pfloor = 1.0e-6",
        "<problem>", f"spec_file = {spec}",
        "<output1>", "file_type = bin", "variable = mhd_u_bcc", f"dt = {dt_out if outputs else t_end!r}", "single_file_per_rank = true"]
    return "\n".join(lines) + "\n"


def read_athenak(path, n):
    sys.path.insert(0, str(Path(os.environ.get("ATHENAK_DIR", BUNDLE.parent/"external/athenak"))/"vis/python"))
    # AthenaK's bin_convert imports h5py unconditionally, although read_binary
    # does not use it (h5py is needed only by write_athdf). Sherlock's lean
    # analysis environment intentionally has NumPy but not h5py, so provide a
    # placeholder when that optional writer dependency is absent.
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError:
        import types
        sys.modules["h5py"] = types.ModuleType("h5py")
    import bin_convert
    d = bin_convert.read_binary(str(path))
    nz = d["Nx3"];u = np.zeros((n, n, nz, 9))
    names = {"dens": 0, "mom1": 1, "mom2": 2, "mom3": 3, "ener": 4, "bcc1": 5, "bcc2": 6, "bcc3": 7}
    for var, data in d["mb_data"].items():
        if var not in names:continue
        for b, block in enumerate(data):
            i0, i1, j0, j1, k0, k1 = d["mb_index"][b]
            lx, ly, lz, _ = d["mb_logical"][b]
            ox, oy, oz = lx*d["nx1_mb"], ly*d["nx2_mb"], lz*d["nx3_mb"]
            u[ox:ox+block.shape[2], oy:oy+block.shape[1], oz:oz+block.shape[0], names[var]] = block.transpose(2, 1, 0)
    return d["time"], d["cycle"], u


def run_athenak(args, config, spec, settings, rundir, rep, outputs):
    deck = rundir/"athinput"
    deck.write_text(athenak_input(config, args.n, spec, settings, args.geometry, outputs, "abel"))
    log = rundir/"athenak.log"
    cmd = shlex.split(args.mpi or "") + [args.exe, "-i", str(deck), "-d", str(rundir)]
    start = time.perf_counter()
    with log.open("w") as f:proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir)
    wall = time.perf_counter() - start
    text = log.read_text()
    perf = dict(wall_seconds=wall)
    for key, pat in (("code_seconds", r"cpu time used\s*=\s*([\d.eE+-]+)"), ("zone_cycles_per_second", r"zone-cycles/cpu_second\s*=\s*([\d.eE+-]+)"),
                     ("steps", r"cycle=(\d+)")):
        m = re.findall(pat, text)
        if m:perf[key] = float(m[-1])
    files = sorted((rundir/"bin").rglob("abel.mhd_u_bcc.*.bin"))
    # Stream frames to the converter: a 384^3 reference frame is ~4 GiB as
    # float64. Retaining all 11 frames would exceed the external job's 64 GiB.
    frames = (read_athenak(f, args.n) for f in files)
    return proc.returncode, text[-2000:], perf, frames


# ------------------------------------------------------------------------------ Athena++
def athenapp_input(config, n, spec, settings, geometry, block):
    nz = 1 if geometry == "plane" else n
    lines = ["<job>", "problem_id = abel",
        "<output1>", "file_type = vtk", "variable = cons", "dt = 1e30",
        "<output2>", "file_type = rst", "dt = 1e30",
        "<time>", f"cfl_number = {settings['cfl']}", "nlim = -1", "tlim = 0", f"integrator = {settings['integrator']}",
        f"xorder = {settings['xorder']}", "ncycle_out = 1000000", "<mesh>"]
    for d, nd in zip((1, 2, 3), (n, n, nz)):
        lines += [f"nx{d} = {nd}", f"x{d}min = 0.0", f"x{d}max = 1.0", f"ix{d}_bc = periodic", f"ox{d}_bc = periodic"]
    lines += [f"num_threads = {os.environ.get('OMP_NUM_THREADS', '1')}", "<meshblock>", f"nx1 = {block}", f"nx2 = {block}",
        f"nx3 = {1 if nz == 1 else block}", "<hydro>", f"gamma = {gamma_of(config)!r}", "dfloor = 1.0e-6", "pfloor = 1.0e-6",
        "<problem>", f"spec_file = {spec}"]
    return "\n".join(lines) + "\n"


def read_athenapp(stage, n, nz):
    sys.path.insert(0, str(Path(os.environ.get("ATHENAPP_DIR", BUNDLE.parent/"external/athena"))/"vis/python"))
    import athena_read
    u = np.zeros((n, n, nz, 9));files = sorted(stage.glob("abel.*.vtk"))
    if not files:raise RuntimeError(f"No Athena++ VTK output in {stage}")
    for f in files:
        x, y, z, d = athena_read.vtk(str(f))
        i0 = int(round(x[0]*n));j0 = int(round(y[0]*n));k0 = int(round(z[0]*n)) if nz > 1 else 0
        rho = d["dens"];sz = rho.shape   # (z, y, x)
        sl = (slice(i0, i0 + sz[2]), slice(j0, j0 + sz[1]), slice(k0, k0 + sz[0]))
        u[sl + (0,)] = rho.transpose(2, 1, 0);u[sl + (4,)] = d["Etot"].transpose(2, 1, 0)
        for c in range(3):
            u[sl + (1 + c,)] = d["mom"][..., c].transpose(2, 1, 0);u[sl + (5 + c,)] = d["Bcc"][..., c].transpose(2, 1, 0)
    return u


def run_athenapp(args, config, spec, settings, rundir, rep, outputs):
    nz = 1 if args.geometry == "plane" else args.n;dims = 2 if nz == 1 else 3
    # Athena++ requires at least one MeshBlock per MPI rank. The previous code
    # compared against OpenMP threads, producing one block for a 32-rank job.
    block=athenapp_block(args.n,dims,mpi_ranks(args.mpi),int(os.environ.get("ATHENAPP_BLOCK","64")))
    deck = rundir/"athinput";deck.write_text(athenapp_input(config, args.n, spec, settings, args.geometry, block))
    targets = output_times(config)[1:] if outputs else [float(config["t_end"])]
    frames, perf, restart, rc, tail = [], dict(wall_seconds=0.0, code_seconds=0.0, steps=0.0), None, 0, ""
    # t = 0 state: a zero-length first segment writes the initial VTK and a restart file.
    for segment, target in enumerate([0.0] + targets):
        stage = rundir/f"segment_{segment:03d}";stage.mkdir()
        cmd = shlex.split(args.mpi or "") + [args.exe] + (["-i", str(deck)] if restart is None else ["-r", str(restart)]) + [f"time/tlim={target!r}"]
        start = time.perf_counter()
        with (stage/"run.log").open("w") as f:proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=stage)
        perf["wall_seconds"] += time.perf_counter() - start
        text = (stage/"run.log").read_text();tail = text[-2000:]
        if proc.returncode or "FATAL" in text or not list(stage.glob("abel.*.vtk")):rc = proc.returncode or 1;break
        m = re.findall(r"cpu time used\s*=\s*([\d.eE+-]+)", text);perf["code_seconds"] += float(m[-1]) if m else 0.0
        # The time Athena++ reports reaching, not the requested one: a silently ignored tlim
        # override must show up as a missing output, never as a correct-looking frame.
        final = re.findall(r"^time=([\d.eE+-]+) cycle=(\d+)", text, re.M)
        if not final:rc = 1;break
        restart = stage/"abel.final.rst"
        frames.append((float(final[-1][0]), int(final[-1][1]), read_athenapp(stage, args.n, nz)))
    if frames:perf["steps"] = frames[-1][1]
    return rc, tail, perf, frames

# ------------------------------------------------------------------------------ RAMSES
class FortranRecords:
    """Sequential unformatted Fortran records (4-byte markers), NumPy only."""
    def __init__(self, path):self.buf = Path(path).read_bytes();self.pos = 0
    def read(self, dtype):
        n = int(np.frombuffer(self.buf, "<i4", 1, self.pos)[0]);data = np.frombuffer(self.buf, dtype, n//np.dtype(dtype).itemsize, self.pos + 4)
        end = int(np.frombuffer(self.buf, "<i4", 1, self.pos + 4 + n)[0])
        if end != n:raise ValueError("Corrupt Fortran record")
        self.pos += n + 8;return data
    def skip(self, count=1):
        for _ in range(count):self.read("u1")


def read_ramses(outdir, n):
    """Uniform-grid (levelmin=levelmax) RAMSES MHD output -> (time, steps, (N,N,N,9) cell state)."""
    nout = outdir.name.split("_")[-1]
    info = (outdir/f"info_{nout}.txt").read_text()
    t = float(re.search(r"^time\s*=\s*(\S+)", info, re.M).group(1));ncpu = int(re.search(r"^ncpu\s*=\s*(\d+)", info, re.M).group(1))
    level = int(round(np.log2(n)));u = np.full((n, n, n, 9), np.nan);steps = int(re.search(r"^nstep_coarse\s*=\s*(\d+)", info, re.M).group(1))
    # Octs of the finest level: centre xg; child ind = 1 + ix + 2iy + 4iz at offsets -/+ dx/2.
    for icpu in range(1, ncpu + 1):
        a = FortranRecords(outdir/f"amr_{nout}.out{icpu:05d}");h = FortranRecords(outdir/f"hydro_{nout}.out{icpu:05d}")
        _, ndim = a.read("<i4")[0], a.read("<i4")[0];a.read("<i4");nlevelmax = a.read("<i4")[0]
        a.read("<i4");nboundary = a.read("<i4")[0];a.read("<i4");boxlen = a.read("<f8")[0]
        a.skip(11)                        # noutput, tout, aout, t, dtold, dtnew, nstep, einit, omega, aexp, mass_sph
        a.read("<i4");a.read("<i4");numbl = a.read("<i4").reshape(nlevelmax, ncpu);a.read("<i4")
        if nboundary:a.skip(3)
        a.skip(1);a.skip(1);a.skip(1);a.skip(3)   # free-memory info, ordering, bound_key, son/flag1/cpu_map
        h.read("<i4");nvar = h.read("<i4")[0];h.read("<i4");h.read("<i4");h.read("<i4");h.read("<f8")
        for ilevel in range(nlevelmax):
            for ibound in range(nboundary + ncpu):
                ncache = numbl[ilevel, ibound] if ibound < ncpu else 0
                hl, hc = h.read("<i4")[0], h.read("<i4")[0]
                if ncache == 0:continue
                a.skip(3);xg = np.stack([a.read("<f8") for _ in range(ndim)]);a.skip(1 + 2*ndim + 3*2**ndim)
                vals = np.stack([np.stack([h.read("<f8") for _ in range(nvar)]) for _ in range(2**ndim)])  # (ind, var, grid)
                if ibound != icpu - 1 or ilevel + 1 != level:continue
                dx = boxlen/2**level
                for ind in range(2**ndim):
                    off = np.array([(ind >> d) & 1 for d in range(3)])
                    ijk = np.floor((xg.T + (off - .5)*dx)/dx).astype(int)
                    v = vals[ind]
                    # write_conservative order: rho, m(3), B_left(3), B_right(3), E
                    u[ijk[:, 0], ijk[:, 1], ijk[:, 2], 0] = v[0]
                    for c in range(3):
                        u[ijk[:, 0], ijk[:, 1], ijk[:, 2], 1 + c] = v[1 + c]
                        u[ijk[:, 0], ijk[:, 1], ijk[:, 2], 5 + c] = .5*(v[4 + c] + v[7 + c])
                    u[ijk[:, 0], ijk[:, 1], ijk[:, 2], 4] = v[10]
    if np.isnan(u[..., :8]).any():raise RuntimeError(f"RAMSES output {outdir} does not cover the uniform level {level}")
    u[..., 8] = 0.0
    return t, steps, u


def ramses_namelist(config, n, settings, outputs):
    level = int(round(np.log2(n)))
    if 2**level != n:raise SystemExit("RAMSES requires N to be a power of two")
    times = output_times(config) if outputs else [0.0, float(config["t_end"])]
    tout = ",".join(repr(float(t)) for t in times)
    return f"""&RUN_PARAMS
hydro=.true.
nrestart=0
ncontrol=1000000
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
/
&OUTPUT_PARAMS
noutput={len(times)}
tout={tout}
write_conservative=.true.
exact_output_time=.true.
/
&HYDRO_PARAMS
gamma={gamma_of(config)!r}
courant_factor={settings['courant']}
slope_type={settings['slope_type']}
riemann='{settings['riemann']}'
riemann2d='{settings['riemann2d']}'
smallr=1e-6
smallc=1e-6
/
&REFINE_PARAMS
/
"""


def run_ramses(args, config, spec, settings, rundir, rep, outputs):
    if args.geometry != "cube":raise SystemExit("RAMSES runs use the full cube")
    nml = rundir/"abel.nml";nml.write_text(ramses_namelist(config, args.n, settings, outputs))
    env = dict(os.environ, ABEL_IC_PREFIX=spec.removesuffix(".spec"), ABEL_IC_N=str(args.n))
    cmd = shlex.split(args.mpi or "") + [args.exe, nml.name];log = rundir/"ramses.log"   # RAMSES truncates long paths
    start = time.perf_counter()
    with log.open("w") as f:proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env)
    wall = time.perf_counter() - start;text = log.read_text();perf = dict(wall_seconds=wall)
    m = re.findall(r"Total elapsed time:\s*([\d.eE+-]+)", text)
    if m:perf["code_seconds"] = float(m[-1])
    outputs = sorted(rundir.glob("output_*"))
    frames = (read_ramses(out, args.n) for out in outputs)
    rc = proc.returncode or (0 if outputs else 1)
    return rc, text[-2000:], perf, frames

# ------------------------------------------------------------------------------ driver
RUNNERS = {"athenak": run_athenak, "athenapp": run_athenapp, "ramses": run_ramses}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("code", choices=sorted(VARIANTS));p.add_argument("variant");p.add_argument("config", type=Path)
    p.add_argument("n", type=int);p.add_argument("outdir", type=Path)
    p.add_argument("--repeats", type=int, default=None);p.add_argument("--seed", type=int, default=None)
    p.add_argument("--exe", required=True);p.add_argument("--mpi", default="")
    p.add_argument("--geometry", choices=("cube", "plane"), default="cube")
    p.add_argument("--device", default=os.environ.get("MHD_DEVICE", "unknown"))
    p.add_argument("--device-kind", choices=("gpu", "cpu"), default=None,
                   help="what the cost is measured on; default gpu for AthenaK when nvidia-smi exists, else cpu")
    p.add_argument("--ic-only", action="store_true", help="evolve 1e-9 time units: checks initial data, not physics")
    p.add_argument("--keep-native", action="store_true", help="keep each code's own output files after conversion "
                   "(default: delete them; they are full resolution and would be packed into the returned archive)")
    p.add_argument("--restrict-to", default="", help="comma-separated comparison grids: save conservative "
                   "block averages (ref_N<M>_###.bin) instead of full-resolution states; for reference runs")
    args = p.parse_args()
    args.exe = str(Path(args.exe).resolve())
    if not Path(args.exe).is_file():sys.exit(f"Executable not found: {args.exe}")
    if args.variant not in VARIANTS[args.code]:sys.exit(f"variant must be one of {sorted(VARIANTS[args.code])}")
    if args.code not in RUNNERS:sys.exit(f"{args.code} runner not implemented yet")
    config = load_config(args.config);settings = VARIANTS[args.code][args.variant]
    restrict = [int(m) for m in args.restrict_to.split(",") if m]
    if any(args.n % m for m in restrict):sys.exit("--restrict-to grids must divide N")
    if args.ic_only:config.update(t_end=1e-9, frames=2)
    snapshot_frames = set(config.get("snapshot_frames", range(int(config["frames"]))))
    seed = args.seed if args.seed is not None else config.get("seeds", [42])[0]
    repeats = args.repeats or int(config.get("repeats", 1))
    if args.geometry == "plane" and not planar(config):sys.exit("--geometry plane only for planar problems")
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()):sys.exit(f"Output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out/"input.toml")
    spec = out/"ic";write_spec(args.config, args.n, spec, seed, state=args.code == "ramses")
    method = f"{args.code}_{args.variant}"
    kind = args.device_kind or ("gpu" if args.code == "athenak" and shutil.which("nvidia-smi") else "cpu")
    env = {"created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "device": args.device, "device_kind": kind, "smoke": False,
           "actual_N": args.n, "actual_seed": seed, "code": args.code, "variant": args.variant, "settings": settings,
           "geometry": args.geometry, "executable": str(Path(args.exe).resolve()), "executable_sha256": sha256(args.exe),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"), "launcher": args.mpi or "none",
           "spec_sha256": sha256(str(spec) + ".spec")}
    (out/"environment.toml").write_text(toml_dump(env))
    completed_native_dirs = []
    failed_runs = []
    for rep in range(1, repeats + 1):
        rundir = out/f"{method}_r{rep}";rundir.mkdir()
        meta = {"status": "running", "test": config["name"], "problem": config["problem"], "N": args.n, "seed": seed,
                "method": method, "repeat": rep, "gamma": gamma_of(config), "code": args.code, "variant": args.variant,
                "settings": settings, "geometry": args.geometry, "planar": planar(config),
                "saved_shape": [args.n, args.n, 9] if planar(config) else [args.n, args.n, args.n, 9],
                "snapshots_written": rep == 1, "frames": [], "precision": "fp64 evolution; Float32 snapshots",
                "t_end_requested": float(config["t_end"])}
        code_rc, tail, perf, frames = RUNNERS[args.code](args, config, str(spec) + ".spec", settings, rundir, rep, rep == 1)
        wanted = output_times(config) if rep == 1 else [float(config["t_end"])]
        tol = 1e-6*float(config["t_end"])/(int(config["frames"]) - 1)   # outputs land exactly; guard roundoff only
        targets = output_times(config)
        seen = set()
        for t, cycle, u in frames:
            idx = min(range(len(targets)), key=lambda i: abs(t-targets[i]))
            if abs(t-targets[idx]) > tol or (rep != 1 and idx != len(targets)-1):continue
            if idx in seen:raise RuntimeError(f"Duplicate output at frame {idx} for {method} repeat {rep}")
            seen.add(idx)
            stamp = {"frame": idx, "time": t, "steps": int(cycle)}
            if restrict:
                stamp["restricted"] = {}
                for m in restrict:
                    name = f"ref_N{m}_{idx:03d}.bin";write_state(rundir/name, block_average(u, args.n//m), not planar(config))
                    stamp["restricted"][str(m)] = name
            elif rep == 1 and idx not in snapshot_frames:
                pass                     # config limits snapshots (large 3D runs); diagnostics still use this frame
            elif rep == 1:
                name = f"state_{idx:03d}.bin";write_state(rundir/name, u, not planar(config));stamp["file"] = name
            else:
                name = f"final_{idx:03d}.bin";write_state(rundir/name, u, not planar(config));stamp["final_file"] = name
            meta["frames"].append(stamp)
        missing = [t for t in wanted if not any(abs(f["time"] - t) <= tol for f in meta["frames"])]
        meta.update(perf);meta["steps"] = int(perf.get("steps", meta["frames"][-1]["steps"] if meta["frames"] else 0))
        meta["time_reached"] = meta["frames"][-1]["time"] if meta["frames"] else 0.0
        meta["solve_seconds"] = perf.get("code_seconds", perf["wall_seconds"])
        cells = args.n**2*(args.n if args.geometry == "cube" else 1)
        if meta["solve_seconds"] > 0:meta["million_cell_steps_per_second"] = cells*meta["steps"]/meta["solve_seconds"]/1e6
        meta["timing_scope"] = "Code-reported evolution-loop time (includes its own output in repeat 1); wall_seconds includes startup"
        if code_rc != 0 or missing:
            meta["status"] = "failed";meta["error"] = f"exit code {code_rc}; missing output times {missing}\n{tail}"
        else:meta["status"] = "complete"
        (rundir/"run.toml").write_text(toml_dump(meta))
        if meta["status"] == "complete":completed_native_dirs.append(rundir)
        else:failed_runs.append(str(rundir))
        print(f"{method} N={args.n} repeat {rep}: {meta['status']} in {perf['wall_seconds']:.2f} s", flush=True)
    subprocess.run([os.environ.get("MHD_JULIA", "julia"), "--startup-file=no", str(BUNDLE/"tools/postprocess_external.jl"), str(out)], check=True)
    if not args.keep_native:
        for rundir in completed_native_dirs:remove_native_output(rundir)
    if failed_runs:raise RuntimeError("External solver failed in: " + ", ".join(failed_runs))


def remove_native_output(rundir):
    """Drop code-native snapshots/restarts once converted; keep logs, decks and our own files."""
    for path in list(rundir.glob("bin")) + list(rundir.glob("output_*")):
        if path.is_dir():shutil.rmtree(path)
        elif path.is_file():path.unlink()
    for pattern in ("segment_*/*.vtk", "segment_*/*.rst"):
        for path in rundir.glob(pattern):path.unlink()


def toml_dump(d):
    """Minimal TOML writer for flat/nested dicts and lists of dicts."""
    def value(v):
        if isinstance(v, bool):return "true" if v else "false"
        if isinstance(v, (int, np.integer)):return str(int(v))
        if isinstance(v, (float, np.floating)):return repr(float(v)) if np.isfinite(v) else ('nan' if np.isnan(v) else ('inf' if v > 0 else '-inf'))
        if isinstance(v, str):return json.dumps(v)
        if isinstance(v, (list, tuple)):return "[" + ", ".join(value(x) for x in v) + "]"
        if isinstance(v, dict):return "{" + ", ".join(f"{json.dumps(str(k))} = {value(x)}" for k, x in v.items()) + "}"
        raise TypeError(type(v))
    lines, tables, arrays = [], [], []
    for k, v in d.items():
        if isinstance(v, dict):tables.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):arrays.append((k, v))
        else:lines.append(f"{k} = {value(v)}")
    for k, v in tables:
        lines.append(f"\n[{k}]");lines += [f"{a} = {value(b)}" for a, b in v.items()]
    for k, v in arrays:
        for item in v:
            lines.append(f"\n[[{k}]]");lines += [f"{a} = {value(b)}" for a, b in item.items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
