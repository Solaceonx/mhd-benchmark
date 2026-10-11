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
import atexit
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
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
    # Athena++ fixes its Riemann solver at build time: codes/build.sh makes bin/athena_hlle and bin/athena_hlld.
    "athenapp": {
        "matched": dict(xorder=2, integrator="vl2", cfl=0.3, flux="hlle"),
        "best": dict(xorder=3, integrator="rk3", cfl=0.3, flux="hlld"),
        "reference": dict(xorder=3, integrator="rk3", cfl=0.3, flux="hlld"),
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


def step_cap(config, n):
    """Runaway guard (dt collapse): max_steps_per_n * N steps, or none (-1). External codes take fewer steps
    than the GPU driver (per-direction CFL), so the GPU driver's cap is generous for them."""
    per = int(config.get("max_steps_per_n", 0))
    return per*n if per > 0 else -1


def write_spec(config_path, n, prefix, seed, state=False, plane=False):
    cmd = [os.environ.get("MHD_JULIA", "julia"), "--startup-file=no", str(BUNDLE/"tools/write_spec.jl"),
           str(config_path), str(n), str(prefix), "--seed", str(seed)] + (["--state"] if state else []) + (["--plane"] if plane else [])
    subprocess.run(cmd, check=True)


def block_average(u, r):
    """Conservative r^d block average of an (N,N,Nz,9) state; planar (Nz=1) stays planar."""
    n, _, nz, c = u.shape;rz = r if nz > 1 else 1
    return u.reshape(n//r, r, n//r, r, nz//rz, rz, c).mean(axis=(1, 3, 5))


def write_state(path, u, full, plane=0):
    """u: (N,N,Nz,9) float array, Julia/Fortran order with x fastest. Not full: the z index `plane` only."""
    state = u if full else u[:, :, plane, :]
    np.asarray(state, dtype="<f4").ravel(order="F").tofile(path)


# ------------------------------------------------------------------------------ GPU memory
class GpuMemorySampler:
    """Peak device memory while a code runs, from nvidia-smi (device-wide used MiB).

    Per-process attribution is not used: under CUDA MPS (loaded on Marlowe) the MPS server,
    not the application, owns the allocation. The job holds its GPU exclusively, so the rise
    above the pre-launch baseline is the code's footprint including its CUDA context.
    """
    def __init__(self, interval=0.2):
        self.interval, self.peak, self.baseline, self.samples, self.thread = interval, None, None, 0, None
        self.enabled = shutil.which("nvidia-smi") is not None
        self.stop = threading.Event()

    def read(self):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        values = [float(v) for v in out.stdout.split() if v.strip()]
        if not values:return None
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # Several GPUs listed without device cgroups: use the one this job was given.
        if len(values) > 1 and visible.isdigit() and int(visible) < len(values):return values[int(visible)]
        return values[0] if len(values) == 1 else None   # ambiguous: record nothing rather than guess

    def _loop(self):
        while not self.stop.is_set():
            try:
                v = self.read()
                if v is not None:self.peak = v if self.peak is None else max(self.peak, v);self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop.wait(self.interval)

    def __enter__(self):
        if self.enabled:
            try:self.baseline = self.read()
            except (OSError, subprocess.SubprocessError, ValueError):self.baseline = None
            self.thread = threading.Thread(target=self._loop, daemon=True);self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:self.thread.join(timeout=15)

    def record(self, perf):
        if self.peak is None:return
        mib = 2**20
        perf["device_memory_peak_bytes"] = int(self.peak*mib)
        if self.baseline is not None:perf["device_memory_above_baseline_bytes"] = int((self.peak - self.baseline)*mib)
        perf["device_memory_samples"] = self.samples
        perf["device_memory_method"] = f"nvidia-smi memory.used every {self.interval} s (device-wide; peak and rise above pre-launch baseline)"


# ------------------------------------------------------------------------------ AthenaK
def athenak_input(config, n, spec, settings, geometry, outputs, basename, restart_every=0):
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
        f"nlim = {step_cap(config, n)}", f"tlim = {t_end!r}", "ndiag = 1000000",
        "<mhd>", "eos = ideal", f"reconstruct = {settings['reconstruct']}", f"rsolver = {settings['rsolver']}",
        f"gamma = {gamma_of(config)!r}", "dfloor = 1.0e-6", "pfloor = 1.0e-6",
        "<problem>", f"spec_file = {spec}",
        "<output1>", "file_type = bin", "variable = mhd_u_bcc", f"dt = {dt_out if outputs else t_end!r}", "single_file_per_rank = true"]
    if restart_every > 0:
        # Restart points on output times (the abel_suite patch lands steps on every output block's times).
        lines += ["<output2>", "file_type = rst", f"dt = {restart_every*dt_out!r}"]
    return "\n".join(lines) + "\n"


class RestartPruner:
    """Delete all but the newest `keep` restart files while a code runs (each is a full state)."""
    def __init__(self, pattern_dir, glob, keep=2, interval=30):
        self.dir, self.glob, self.keep, self.interval = pattern_dir, glob, keep, interval
        self.stop = threading.Event();self.thread = None

    def prune(self):
        files = sorted(self.dir.glob(self.glob), key=lambda f: f.stat().st_mtime)
        for f in files[:-self.keep]:
            try:f.unlink()
            except OSError:pass

    def _loop(self):
        while not self.stop.wait(self.interval):self.prune()

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True);self.thread.start();return self

    def __exit__(self, *exc):
        self.stop.set();self.thread.join(timeout=15);self.prune()


def latest_complete(files):
    """Newest file among those of full size: a restart cut short by preemption is smaller."""
    files = [f for f in files if f.stat().st_size > 0]
    if not files:return None
    size = max(f.stat().st_size for f in files)
    full = [f for f in files if f.stat().st_size == size]
    return max(full, key=lambda f: f.name)


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


def run_athenak(args, config, spec, settings, rundir, rep, outputs, resume=False):
    deck = rundir/"athinput";log = rundir/"athenak.log"
    every = args.restart_every if args.restart_every is not None else int(config.get("restart_every", 0))
    restart = latest_complete(sorted((rundir/"rst").glob("abel.*.rst"))) if resume else None
    # Output files of a finished run are numbered 0 .. expected-1 (repeat 1: every output; later repeats: start and end).
    expected = len(output_times(config)) if outputs else 2
    numbered = lambda: {int(f.name.split(".")[-2]): f for f in (rundir/"bin").rglob("abel.mhd_u_bcc.*.bin")}
    # An earlier attempt can have finished the solver and been stopped while its output was converted. Its last
    # restart file is then at the end time: restarting from it would only write the final output once more under the
    # next number, and the converter would find that output twice. The run is done: convert what is there.
    # (AthenaK prints "Terminating on time limit" only after it has written every output, the final ones included.)
    finished = False
    if restart is not None and log.is_file() and "Terminating on time limit" in log.read_text(errors="replace"):
        have = numbered()
        finished = all(i in have and have[i].stat().st_size > 0 for i in range(expected))
    if finished:
        print(f"AthenaK finished in an earlier attempt ({expected} outputs present): converting them", flush=True)
        cmd = None
    elif restart is None:
        # Fresh start (also after an interruption before the first restart point): drop partial output.
        for path in (rundir/"bin", rundir/"rst"):shutil.rmtree(path, ignore_errors=True)
        deck.write_text(athenak_input(config, args.n, spec, settings, args.geometry, outputs, "abel", every))
        cmd = shlex.split(args.mpi or "") + [args.exe, "-i", str(deck), "-d", str(rundir)]
    else:
        # AthenaK keeps each output block's last_time in its parameter text with 6 significant digits; after a restart
        # every later output would be offset by that truncation (0.0141421 for 0.01414213562373095). Restart file j
        # was written at output j*every, so give the exact time back on the command line (applied after the restart's
        # parameters are loaded).
        j = int(re.search(r"\.(\d+)\.rst$", restart.name).group(1));times = output_times(config)
        exact = times[j*every] if every > 0 and j*every < len(times) else None
        fix = [] if exact is None else ([f"output1/last_time={exact!r}"] if outputs else []) + [f"output2/last_time={exact!r}"]
        cmd = shlex.split(args.mpi or "") + [args.exe, "-r", str(restart), "-d", str(rundir)] + fix
    start = time.perf_counter();code = 0
    if cmd is not None:
        with log.open("a" if restart else "w") as f, GpuMemorySampler() as memory, RestartPruner(rundir/"rst", "abel.*.rst"):
            if restart:f.write(f"\n=== resumed from {restart.name} ===\n");f.flush()
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir).returncode
    wall = time.perf_counter() - start
    text = log.read_text()
    perf = dict(wall_seconds=wall, restart_every_outputs=every)
    if restart:perf.update(resumed=True, resumed_from=restart.name)
    if cmd is None:perf["converted_only"] = True
    else:memory.record(perf)
    for key, pat in (("code_seconds", r"cpu time used\s*=\s*([\d.eE+-]+)"), ("zone_cycles_per_second", r"zone-cycles/cpu_second\s*=\s*([\d.eE+-]+)"),
                     ("steps", r"cycle=(\d+)")):
        m = re.findall(pat, text)
        if m:perf[key] = float(m[-1])
    # A finished run converted later: its own outputs only (retries of the old driver wrote the final one again).
    files = [f for i, f in sorted(numbered().items()) if not finished or i < expected]
    # Stream frames to the converter: a 384^3 reference frame is ~4 GiB as
    # float64. Retaining all 11 frames would exceed the external job's 64 GiB.
    frames = (read_athenak(f, args.n) for f in files)
    return code, text[-2000:], perf, frames


# ------------------------------------------------------------------------------ Athena++
def athenapp_input(config, n, spec, settings, geometry, block):
    nz = 1 if geometry == "plane" else n
    lines = ["<job>", "problem_id = abel",
        "<output1>", "file_type = vtk", "variable = cons", "dt = 1e30",
        "<output2>", "file_type = rst", "dt = 1e30",
        "<time>", f"cfl_number = {settings['cfl']}", f"nlim = {step_cap(config, n)}", "tlim = 0", f"integrator = {settings['integrator']}",
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


def athenapp_segment_done(stage, target):
    """(time, cycle) reached by a finished segment, or None if it did not finish."""
    log = stage/"run.log"
    if not log.is_file() or not (stage/"abel.final.rst").is_file() or not list(stage.glob("abel.*.vtk")):return None
    text = log.read_text()
    # The time Athena++ reports reaching, not the requested one: a silently ignored tlim
    # override must show up as a missing output, never as a correct-looking frame.
    final = re.findall(r"^time=([\d.eE+-]+) cycle=(\d+)", text, re.M)
    if "FATAL" in text or not final:return None
    # Athena++ prints that time with 6 significant digits (e.g. 0.0141421 for 0.01414213562373095), and it lands
    # on tlim exactly; a printed time that agrees with the target to that precision is the target.
    t = float(final[-1][0])
    if abs(t - target) <= 1e-5*abs(target):t = target
    return t, int(final[-1][1])


def run_athenapp(args, config, spec, settings, rundir, rep, outputs, resume=False):
    nz = 1 if args.geometry == "plane" else args.n;dims = 2 if nz == 1 else 3
    # Athena++ requires at least one MeshBlock per MPI rank. The previous code
    # compared against OpenMP threads, producing one block for a 32-rank job.
    block=athenapp_block(args.n,dims,mpi_ranks(args.mpi),int(os.environ.get("ATHENAPP_BLOCK","64")))
    deck = rundir/"athinput";deck.write_text(athenapp_input(config, args.n, spec, settings, args.geometry, block))
    targets = output_times(config)[1:] if outputs else [float(config["t_end"])]
    done, perf, restart, rc, tail = [], dict(wall_seconds=0.0, code_seconds=0.0, steps=0.0), None, 0, ""
    # t = 0 state: a zero-length first segment writes the initial VTK and a restart file. Each segment
    # restarts from the previous one, so a resumed job reuses every finished segment.
    for segment, target in enumerate([0.0] + targets):
        stage = rundir/f"segment_{segment:03d}"
        reached = athenapp_segment_done(stage, target) if resume else None
        if reached is not None:
            perf["resumed"] = True;perf["segments_reused"] = perf.get("segments_reused", 0) + 1
        else:
            shutil.rmtree(stage, ignore_errors=True);stage.mkdir()
            cmd = shlex.split(args.mpi or "") + [args.exe] + (["-i", str(deck)] if restart is None else ["-r", str(restart)]) + [f"time/tlim={target!r}"]
            start = time.perf_counter()
            with (stage/"run.log").open("w") as f:proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=stage)
            perf["wall_seconds"] += time.perf_counter() - start
            reached = athenapp_segment_done(stage, target)
            if proc.returncode or reached is None:
                rc = proc.returncode or 1;tail = (stage/"run.log").read_text()[-2000:];break
        text = (stage/"run.log").read_text();tail = text[-2000:]
        m = re.findall(r"cpu time used\s*=\s*([\d.eE+-]+)", text);perf["code_seconds"] += float(m[-1]) if m else 0.0
        restart = stage/"abel.final.rst";done.append((reached[0], reached[1], stage))
    if done:perf["steps"] = done[-1][1]
    frames = ((t, cycle, read_athenapp(stage, args.n, nz)) for t, cycle, stage in done)   # one frame in memory at a time
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
    """Uniform-grid (levelmin=levelmax) RAMSES MHD output -> (time, steps, (N,N,Nz,9) cell state); Nz = 1 for NDIM=2."""
    nout = outdir.name.split("_")[-1]
    info = (outdir/f"info_{nout}.txt").read_text()
    t = float(re.search(r"^time\s*=\s*(\S+)", info, re.M).group(1));ncpu = int(re.search(r"^ncpu\s*=\s*(\d+)", info, re.M).group(1))
    ndim_info = int(re.search(r"^ndim\s*=\s*(\d+)", info, re.M).group(1))
    level = int(round(np.log2(n)));u = np.full((n, n, n if ndim_info == 3 else 1, 9), np.nan)
    steps = int(re.search(r"^nstep_coarse\s*=\s*(\d+)", info, re.M).group(1))
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
                    off = np.array([(ind >> d) & 1 for d in range(ndim)])
                    ijk = np.floor((xg.T + (off - .5)*dx)/dx).astype(int)
                    if ndim == 2:ijk = np.column_stack([ijk, np.zeros(len(ijk), int)])
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


def ramses_namelist(config, n, settings, outputs, ndim=3, ranks=1, restart=0, checkpoint=0):
    level = int(round(np.log2(n)))
    if 2**level != n:raise SystemExit("RAMSES requires N to be a power of two")
    # ngridmax is per MPI rank: the octs of all levels (n^d/(2^d - 1)) shared over the ranks, plus the ghost octs each
    # rank keeps of its neighbours (large for small domains: 1.5x the average ran out at 128^3 on 32 ranks). Four
    # times the average plus 50,000, never more than the whole tree (at 512^3 on 128 ranks: ~650k octs, ~1.3 GB).
    octs = n**ndim//(2**ndim - 1);ngridmax = min(octs + 1000, 4*octs//max(ranks, 1) + 50000)
    times = output_times(config) if outputs else [0.0, float(config["t_end"])]
    tout = ",".join(repr(float(t)) for t in times)
    return f"""&RUN_PARAMS
hydro=.true.
nrestart={restart}
ncontrol=1000000
nsubcycle=1
nstepmax={step_cap(config, n) if step_cap(config, n) > 0 else 100000000}
/
&AMR_PARAMS
levelmin={level}
levelmax={level}
ngridmax={ngridmax}
boxlen=1.0
/
&INIT_PARAMS
/
&OUTPUT_PARAMS
noutput={len(times)}
tout={tout}
foutput={checkpoint if checkpoint > 0 else 1000000}
write_conservative=.true.
read_conservative={".true." if restart else ".false."}
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


def ramses_files(outdir):
    """The per-rank amr and hydro files of an output, or None when its info file is missing or unreadable."""
    nout = outdir.name.split("_")[-1];info = outdir/f"info_{nout}.txt"
    if not info.is_file():return None
    m = re.search(r"^ncpu\s*=\s*(\d+)", info.read_text(), re.M)
    if not m:return None
    return [outdir/f"{kind}_{nout}.out{i:05d}" for kind in ("amr", "hydro") for i in range(1, int(m.group(1)) + 1)]


def ramses_output_complete(outdir):
    """An output directory with its info file and a nonempty amr and hydro file for every rank."""
    files = ramses_files(outdir)
    return bool(files) and all(f.is_file() and f.stat().st_size > 0 for f in files)


def ramses_complete_outputs(rundir):
    """The outputs RAMSES finished writing, in order. Besides having every file, an output must hold at least as much
    data as the one before it: on a uniform grid every output is the same size, so a smaller one was cut short while
    it was written (the job was stopped). The comparison is with the preceding output only, so that it can never
    reject more than the one output it is meant to catch."""
    good, before = [], None
    for o in sorted(rundir.glob("output_*"), key=ramses_index):
        try:
            if not ramses_output_complete(o):continue
            total = sum(f.stat().st_size for f in ramses_files(o))
        except OSError:continue                      # RAMSES is creating it
        if before is None or total >= before:good.append(o)
        before = total
    return good


def ramses_time(outdir):
    """Simulation time of an output, from its info file."""
    nout = outdir.name.split("_")[-1]
    return float(re.search(r"^time\s*=\s*(\S+)", (outdir/f"info_{nout}.txt").read_text(), re.M).group(1))


def ramses_index(outdir):
    return int(outdir.name.split("_")[-1])


# Restart points for interrupted RAMSES runs. A preempted job resumes from its newest output; with outputs only at the
# frames, hours apart at 512^3, an attempt that is preempted before the next frame is lost. A resumed attempt therefore
# also writes an output every `foutput` coarse steps, about CHECKPOINT_SECONDS apart, and keeps only the newest one.
# foutput does not change the timestep, so the solution is the one an undisturbed run gives. First attempts write
# none: their timing stays comparable with the runs that were never interrupted.
CHECKPOINT_SECONDS = 1200
CHECKPOINT_RATE = 6e4            # cell-steps per second and rank: the low end of what RAMSES did in this sweep


def ramses_checkpoint_steps(n, ndim, ranks):
    forced = os.environ.get("MHD_RAMSES_CHECKPOINT_STEPS")       # tests, or an operator's choice (0: none)
    if forced is not None:return max(0, int(forced))
    return max(20, int(CHECKPOINT_SECONDS*CHECKPOINT_RATE*max(ranks, 1)/n**ndim))


def ramses_frames(outs, targets, tol):
    """{target index: output} for the outputs that sit on a frame time; the lowest-numbered output wins (a restart
    from an output can write the same state once more)."""
    frames = {}
    for o in sorted(outs, key=ramses_index):
        t = ramses_time(o);k = min(range(len(targets)), key=lambda i: abs(t - targets[i]))
        if abs(t - targets[k]) <= tol:frames.setdefault(k, o)
    return frames


def ramses_prune(rundir, targets, tol, settle=180.0, keep_newest=True):
    """Delete restart points that are no longer needed: complete outputs that are not frames and are older than the
    newest complete output. An output counts only once it is settled: a later one exists, or nothing in it changed
    for `settle` seconds (RAMSES may still be writing the newest). Returns the names removed."""
    outs = sorted(rundir.glob("output_*"), key=ramses_index);done = ramses_complete_outputs(rundir)
    try:
        if done and done[-1] == outs[-1] and time.time() - max(f.stat().st_mtime for f in done[-1].iterdir()) < settle:done.pop()
    except OSError:return []
    if not done:return []
    frames = set(ramses_frames(done, targets, tol).values());newest = done[-1];removed = []
    for o in done:
        if o in frames or (keep_newest and o == newest):continue
        shutil.rmtree(o, ignore_errors=True);removed.append(o.name)
    return removed


class RamsesPruner:
    """Runs ramses_prune every `interval` seconds while RAMSES writes restart points."""
    def __init__(self, rundir, targets, tol, interval=60.0):
        self.args = (rundir, targets, tol);self.interval = interval;self.stop = threading.Event();self.thread = None
    def _loop(self):
        while not self.stop.wait(self.interval):
            try:ramses_prune(*self.args)
            except Exception as err:print(f"restart-point pruning: {err!r}", flush=True)      # never fail the run for housekeeping
    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True);self.thread.start();return self
    def __exit__(self, *exc):
        self.stop.set();self.thread.join()


RAMSES_READ_ERROR = re.compile(r"read_block_direct|unformatted_read|file structure has been corrupted|End of file|past end of record", re.I)


def run_ramses(args, config, spec, settings, rundir, rep, outputs, resume=False):
    ndim = 2 if args.geometry == "plane" else 3
    targets = output_times(config) if outputs else [0.0, float(config["t_end"])]
    tol = 1e-6*float(config["t_end"])/(int(config["frames"]) - 1)
    # Resume from the newest complete output (RAMSES restarts from any output); later partial ones are dropped.
    restart = 0;complete = []
    if resume:
        complete = ramses_complete_outputs(rundir)
        if complete:restart = ramses_index(complete[-1])
        for o in rundir.glob("output_*"):
            if ramses_index(o) > restart:shutil.rmtree(o)
    else:
        for o in rundir.glob("output_*"):shutil.rmtree(o)
    # The end time is on disk: an earlier attempt finished the solver and was stopped while its output was converted.
    finished = bool(complete) and abs(ramses_time(complete[-1]) - targets[-1]) <= tol
    ranks = mpi_ranks(args.mpi);checkpoint = ramses_checkpoint_steps(args.n, ndim, ranks) if resume and not finished else 0
    nml = rundir/"abel.nml";nml.write_text(ramses_namelist(config, args.n, settings, outputs, ndim, ranks, restart, checkpoint))
    env = dict(os.environ, ABEL_IC_PREFIX=spec.removesuffix(".spec"), ABEL_IC_N=str(args.n))
    cmd = shlex.split(args.mpi or "") + [args.exe, nml.name];log = rundir/"ramses.log"   # RAMSES truncates long paths
    start = time.perf_counter();code = 0
    if finished:print(f"RAMSES finished in an earlier attempt (the output at t = {targets[-1]:g} is present): converting", flush=True)
    else:
        with log.open("a" if restart else "w") as f:
            if restart:f.write(f"\n=== resumed from output {restart} (t = {ramses_time(complete[-1]):.6g}) ===\n");f.flush()
            if checkpoint:
                print(f"RAMSES writes a restart point every {checkpoint} steps (this attempt resumes an interrupted run)", flush=True)
                with RamsesPruner(rundir, targets, tol):
                    code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env).returncode
            else:code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env).returncode
    wall = time.perf_counter() - start;text = log.read_text() if log.is_file() else "";perf = dict(wall_seconds=wall)
    if restart:perf.update(resumed=True, resumed_from=f"output_{restart:05d}")
    if checkpoint:perf["restart_point_every_steps"] = checkpoint
    # RAMSES could not read the output it resumed from (an earlier stop cut a file short without making it smaller, or
    # damaged it): that output is of no use. Remove it, so that the next attempt resumes from the one before.
    if code != 0 and restart and RAMSES_READ_ERROR.search(text.rsplit("=== resumed from output", 1)[-1]) \
            and not any(ramses_index(o) > restart for o in ramses_complete_outputs(rundir)):
        print(f"RAMSES could not read output {restart}: removing it; the next attempt resumes from the output before it", flush=True)
        shutil.rmtree(rundir/f"output_{restart:05d}", ignore_errors=True);perf["interrupted"] = True
    if finished:perf["converted_only"] = True
    m = re.findall(r"Total elapsed time:\s*([\d.eE+-]+)", text)
    if m:perf["code_seconds"] = float(m[-1])
    # Only the outputs on frame times are results; the restart points in between are dropped once the solver is done.
    good = ramses_complete_outputs(rundir)
    if code == 0 and good and abs(ramses_time(good[-1]) - targets[-1]) <= tol:
        ramses_prune(rundir, targets, tol, settle=0.0, keep_newest=False);good = ramses_complete_outputs(rundir)
    frame_outputs = [o for _, o in sorted(ramses_frames(good, targets, tol).items())]
    frames = (read_ramses(out, args.n) for out in frame_outputs)
    rc = code or (0 if frame_outputs else 1)
    return rc, text[-2000:], perf, frames

# ------------------------------------------------------------------------------ a code that was killed is not a result
# When Slurm stops a job (preemption on owner nodes, a requeue, a cancel, the time limit) the code's processes are
# killed first, and this driver can live long enough to see their exit status. That is not the solver failing: nothing
# must be recorded for it, or the requeued job would find a "failed" run, keep it, and never resume. (Patch 10: eight
# 512^3 RAMSES runs ended that way.) Out-of-memory kills stay failures: running them again would end the same way.
KILL_CODES = (129, 130, 137, 143, -1, -2, -9, -15)                 # 128 + SIGHUP, SIGINT, SIGKILL, SIGTERM; or the bare signal
KILL_TEXT = re.compile(r"srun: error: \S+: tasks? [\d,\-]+: (Killed|Terminated)|srun: Force Terminated|srun: Job step aborted|"
                       r"CANCELLED AT|DUE TO (PREEMPTION|TIME LIMIT|NODE FAILURE)")
OOM_TEXT = re.compile(r"oom[-_ ]kill|out of memory", re.I)


def was_killed(code, text):
    """Whether a code's exit status and the end of its log say that it was killed from outside."""
    if OOM_TEXT.search(text or ""):return False
    return code in KILL_CODES or (code != 0 and bool(KILL_TEXT.search(text or "")))


def recorded_kill(meta):
    """Whether a part recorded as failed (by the driver before patch 10) was in fact killed."""
    err = meta.get("error", "");m = re.match(r"exit code (-?\d+);", err)
    return meta.get("status") == "failed" and bool(m) and was_killed(int(m.group(1)), err)


# ------------------------------------------------------------------------------ one job per run directory
IN_USE = 75      # exit status: the run directory is in use by another running job; nothing was touched
KILLED = 76      # exit status: the code was killed from outside; the run is left as it is, to be continued
# squeue states of a job that is not executing (waiting to start, or requeued and waiting).
WAITING = ("PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "SPECIAL_EXIT", "CONFIGURING")


def job_is_running(job):
    """Whether Slurm reports this job as executing now. Anything else (waiting, gone, squeue unavailable) is no."""
    for attempt in range(3):
        try:
            p = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T"], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):return False
        states = p.stdout.split()
        if p.returncode != 0:
            if "Invalid job id" in p.stderr or attempt == 2:return False     # gone from Slurm; or squeue keeps failing
            time.sleep(10);continue
        if not states or all(s in WAITING for s in states):return False
        if "COMPLETING" in states and attempt < 2:time.sleep(20);continue    # being stopped: look again
        return "COMPLETING" not in states
    return False


def claim_run(out):
    """Mark the run directory of OUT (OUT.lock beside it names the Slurm job) as worked on by this job. Returns the
    id of another job that holds it and is running, else None. Two jobs must never work in one run directory: each
    would resume from, overwrite and delete the other's files. The guard never stops a run through its own failure:
    outside Slurm, or if anything here goes wrong, it returns None."""
    me = os.environ.get("SLURM_JOB_ID")
    if not me:return None
    lock = out.parent/(out.name + ".lock")
    try:
        owner = (lock.read_text().split() or [""])[0] if lock.is_file() else ""
        if owner and owner != me and job_is_running(owner):return owner
        tmp = lock.with_name(f"{lock.name}.{me}.{os.getpid()}")
        tmp.write_text(f"{me} {socket.gethostname()} {datetime.datetime.now(datetime.UTC).isoformat()}\n");os.replace(tmp, lock)
        time.sleep(1)                                   # a job starting at the same moment may have written its own
        owner = (lock.read_text().split() or [""])[0]
        if owner and owner != me and job_is_running(owner):return owner
    except Exception as e:                              # noqa: BLE001
        print(f"run guard skipped ({e!r})", flush=True)
    return None


def release_run(out):
    try:
        lock = out.parent/(out.name + ".lock")
        if lock.is_file() and (lock.read_text().split() or [""])[0] == os.environ.get("SLURM_JOB_ID"):lock.unlink()
    except Exception:                                   # noqa: BLE001
        pass


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
    p.add_argument("--resume", action="store_true", help="continue an interrupted job in OUTDIR (same config): finished "
                   "repeats are kept and an interrupted one continues from the code's last restart point")
    p.add_argument("--restart-every", type=int, default=None, help="AthenaK: restart file every K outputs "
                   "(default: config restart_every, else none)")
    args = p.parse_args()
    args.exe = str(Path(args.exe).resolve())
    if not Path(args.exe).is_file():sys.exit(f"Executable not found: {args.exe}")
    if args.variant not in VARIANTS[args.code]:sys.exit(f"variant must be one of {sorted(VARIANTS[args.code])}")
    if args.code not in RUNNERS:sys.exit(f"{args.code} runner not implemented yet")
    config = load_config(args.config);settings = VARIANTS[args.code][args.variant]
    restrict = [int(m) for m in args.restrict_to.split(",") if m]
    if any(args.n % m for m in restrict):sys.exit("--restrict-to grids must divide N")
    if args.ic_only:config.update(t_end=1e-9, frames=2)
    snapshot_frames = set(config.get("snapshot_frames_by_grid", {}).get(str(args.n), config.get("snapshot_frames", range(int(config["frames"])))))
    # Movie runs of 3D problems save only the z midplane at every output (slice_output = true).
    slice_out = bool(config.get("slice_output", False)) and not planar(config)
    full_out = not planar(config) and not slice_out
    seed = args.seed if args.seed is not None else config.get("seeds", [42])[0]
    repeats = args.repeats or int(config.get("repeats_by_grid", {}).get(str(args.n), config.get("repeats", 1)))
    if args.geometry == "plane" and not planar(config):sys.exit("--geometry plane only for planar problems")
    out = args.outdir.resolve()
    holder = claim_run(out)
    if holder:
        print(f"SKIPPED: {out} is in use by running job {holder}; nothing was touched", flush=True);sys.exit(IN_USE)
    atexit.register(release_run, out)
    resuming = args.resume and (out/"input.toml").is_file()
    if resuming:
        if (out/"input.toml").read_bytes() != Path(args.config).read_bytes():sys.exit(f"{out} holds a different config; refusing to resume")
        print(f"RESUMING in {out}", flush=True)
    elif out.exists() and any(out.iterdir()):
        if not args.resume:sys.exit(f"Output directory must be new or empty: {out}")
        # An earlier attempt was stopped while it wrote the initial data: input.toml is written last, before any run
        # starts, so nothing in here is a result. Start over (refusing would leave the run stuck for good).
        print(f"RESTARTING in {out}: an earlier attempt stopped before its initial data was complete", flush=True)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = out/"ic"
    if not (resuming and Path(str(spec) + ".spec").is_file() and (args.code != "ramses" or Path(str(spec) + ".state").is_file())):
        write_spec(args.config, args.n, spec, seed, state=args.code == "ramses", plane=args.geometry == "plane")
    shutil.copy(args.config, out/"input.toml")     # last: a directory with input.toml has its initial data
    # Reference runs are labelled "reference" whatever settings they use (decay_supersonic: matched).
    method = f"{args.code}_reference" if restrict else f"{args.code}_{args.variant}"
    kind = args.device_kind or ("gpu" if args.code == "athenak" and shutil.which("nvidia-smi") else "cpu")
    env = {"created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "device": args.device, "device_kind": kind, "smoke": False,
           "actual_N": args.n, "actual_seed": seed, "code": args.code, "variant": args.variant, "settings": settings,
           "geometry": args.geometry, "executable": str(Path(args.exe).resolve()), "executable_sha256": sha256(args.exe),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"), "launcher": args.mpi or "none",
           "spec_sha256": sha256(str(spec) + ".spec")}
    envfile = "environment_resumed_" + os.environ.get("SLURM_RESTART_COUNT", "x") + ".toml" if resuming else "environment.toml"
    (out/envfile).write_text(toml_dump(env))
    completed_native_dirs = []
    failed_runs = []
    for rep in range(1, repeats + 1):
        rundir = out/f"{method}_r{rep}"
        if rep > 1 and str(out/f"{method}_r1") in failed_runs:
            # Later repeats only time a run that works (they keep the final state alone).
            print(f"{method} N={args.n}: repeat 1 failed, so repeats 2-{repeats} (timing) are not run", flush=True);break
        if resuming and (rundir/"run.toml").is_file():
            old = tomllib.loads((rundir/"run.toml").read_text());status = old.get("status")
            if recorded_kill(old):
                print(f"{method} N={args.n} repeat {rep}: recorded as failed, but the code had been killed ({old['error'].split(';')[0]}): "
                      "continuing it", flush=True)
            elif status in ("complete", "failed"):          # finished in an earlier attempt
                (completed_native_dirs if status == "complete" else failed_runs).append(rundir if status == "complete" else str(rundir))
                continue
        resume_native = resuming and rundir.is_dir()
        rundir.mkdir(exist_ok=True)
        meta = {"status": "running", "test": config["name"], "problem": config["problem"], "N": args.n, "seed": seed,
                "method": method, "repeat": rep, "gamma": gamma_of(config), "code": args.code, "variant": args.variant,
                "settings": settings, "geometry": args.geometry, "planar": planar(config),
                "reference_settings": args.variant if restrict else "",
                "saved_shape": [args.n, args.n, 9] if planar(config) or slice_out else [args.n, args.n, args.n, 9],
                "snapshots_written": rep == 1, "frames": [], "precision": "fp64 evolution; Float32 snapshots",
                "t_end_requested": float(config["t_end"])}
        if slice_out:meta["slice"] = "z midplane"
        code_rc, tail, perf, frames = RUNNERS[args.code](args, config, str(spec) + ".spec", settings, rundir, rep, rep == 1, resume_native)
        if was_killed(code_rc, tail) or perf.get("interrupted"):
            print(f"{method} N={args.n} repeat {rep}: the code did not end by itself (exit code {code_rc}: killed, or its restart file "
                  "was unreadable): the job is being stopped, lost a node or a file system. Nothing is recorded; the next attempt "
                  "continues from the last good output.", flush=True)
            sys.exit(KILLED)
        meta["timing_valid"] = not perf.get("resumed", False)
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
                    name = f"ref_N{m}_{idx:03d}.bin";write_state(rundir/name, block_average(u, args.n//m), full_out, (args.n//m)//2 if slice_out else 0)
                    stamp["restricted"][str(m)] = name
                if not slice_out:
                    # The reference's own diagnostics and spectra: measured by postprocess_external.jl, then deleted.
                    name = f"tmp_{idx:03d}.bin";write_state(rundir/name, u, full_out);stamp["file"] = name;stamp["temporary"] = True
            elif rep == 1 and idx not in snapshot_frames and not slice_out:
                # Not kept: tools/postprocess_external.jl measures it (diagnostics, spectra, slices) and deletes it.
                name = f"tmp_{idx:03d}.bin";write_state(rundir/name, u, full_out);stamp["file"] = name;stamp["temporary"] = True
            elif rep == 1 and idx not in snapshot_frames:
                pass
            elif rep == 1:
                name = f"state_{idx:03d}.bin";write_state(rundir/name, u, full_out, args.n//2 if slice_out else 0);stamp["file"] = name
            else:
                name = f"final_{idx:03d}.bin";write_state(rundir/name, u, full_out, args.n//2 if slice_out else 0);stamp["final_file"] = name
            meta["frames"].append(stamp)
        missing = [t for t in wanted if not any(abs(f["time"] - t) <= tol for f in meta["frames"])]
        meta.update(perf);meta["steps"] = int(perf.get("steps", meta["frames"][-1]["steps"] if meta["frames"] else 0))
        meta["time_reached"] = meta["frames"][-1]["time"] if meta["frames"] else 0.0
        meta["solve_seconds"] = perf.get("code_seconds", perf["wall_seconds"])
        meta.update(slice_frames=[], restrict_to=[], spectra_every=0)     # tools/postprocess_external.jl fills these
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
    # Measurements on the GPU when this job has one (AthenaK) and the grid is large (>= 256^3 cells); small grids
    # are quicker on the CPU than loading CUDA. FFTW/threads on the node's CPUs otherwise.
    threads = os.environ.get("SLURM_CPUS_ON_NODE", os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    big = args.n**2*(args.n if args.geometry == "cube" else 1) >= 256**3
    penv = dict(os.environ, MHD_POSTPROCESS_GPU="1" if kind == "gpu" and big else "0")
    subprocess.run([os.environ.get("MHD_JULIA", "julia"), "--startup-file=no", f"--project={BUNDLE/'env'}", f"--threads={threads}",
                    str(BUNDLE/"tools/postprocess_external.jl"), str(out)], check=True, env=penv)
    if not args.keep_native:
        for rundir in completed_native_dirs:remove_native_output(rundir)
    if failed_runs:raise RuntimeError("External solver failed in: " + ", ".join(failed_runs))


def remove_native_output(rundir):
    """Drop code-native snapshots/restarts once converted; keep logs, decks and our own files."""
    for path in list(rundir.glob("bin")) + list(rundir.glob("rst")) + list(rundir.glob("output_*")):
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
