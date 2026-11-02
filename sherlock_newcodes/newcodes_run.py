"""AthenaPK, Cholla, gPLUTO, Enzo and Phantom in the sweep's external-code driver.

    python3 newcodes_run.py CODE VARIANT CONFIG.toml N OUTDIR --exe PATH --mpi "srun -n K" --geometry plane|cube [--repeats 1] [--resume]

The driver of the deployed sweep bundle (sherlock_sweep/codes/external.py, imported, not changed) does everything
that is the same for every code: the run directory, the spec file (tools/write_spec.jl), run.toml, the frames and
the measurements. This file adds, per code, its input, its command and a reader of its own output that gives the
suite's state (N, N, Nz, 9: rho, momentum, total energy, cell-centred B, psi) at every output time:

  athenapk   GLM divergence cleaning on the GPU (Kokkos); one HDF5 file per output (Parthenon, blocks assembled)
  cholla     constrained transport on the GPU (CUDA), Van Leer integrator, HLLD; planar tests on an N x N x 8 slab
             (Cholla's MHD is 3D only), which stays z-invariant; one HDF5 file per output
  pluto      gPLUTO on the GPU (OpenACC): constrained transport or GLM, by build; planar tests in 2D; .dbl outputs
  enzo       Enzo on CPUs (MPI): Dedner cleaning (HydroMethod 4) or constrained transport (6); planar tests in 2D;
             HDF5 grids per output (hierarchy file read for their places)
  phantom    SPH on CPUs (OpenMP): equal-mass particles on lattices pasted to the suite's density (setup_abel.f90 reads them),
             each dump's particles (analysis_abel.f90) deposited on the grid as GIZMO's; planar tests on an 8-layer slab

Every code starts from the suite's state through abel_problems.hpp (its own patch). Each code's executable is the
build of the variant (build.sh): --exe names it. Native outputs are removed once converted (--keep-native keeps them).
"""
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP = Path(os.environ.get("SWEEP_BUNDLE", HERE.parent/"sherlock_sweep"))
sys.path.insert(0, str(SWEEP/"codes"))
import external as X  # noqa: E402

SLAB = 8            # Cholla: z cells of the slab a planar test runs on
X.VARIANTS["athenapk"] = {
    "matched": dict(reconstruction="plm", riemann="hlle", integrator="vl2", cfl=0.3, divb="GLM (Dedner), c_h from the step", precision="fp64"),
    "best": dict(reconstruction="ppm", riemann="hlld", integrator="rk3", cfl=0.3, divb="GLM (Dedner), c_h from the step", precision="fp64"),
}
X.VARIANTS["cholla"] = {
    "plmc": dict(build="abel_plmc", reconstruction="PLMC (characteristic)", riemann="hlld", integrator="Van Leer", cfl=0.3, divb="CT", precision="fp64"),
    "ppmc": dict(build="abel_ppmc", reconstruction="PPMC (characteristic)", riemann="hlld", integrator="Van Leer", cfl=0.3, divb="CT", precision="fp64"),
}
X.VARIANTS["pluto"] = {
    "ct_matched": dict(reconstruction="LINEAR", limiter="VANLEER_LIM", time_stepping="RK2", solver="hll", divb="CONSTRAINED_TRANSPORT", cfl=0.3),
    "ct_best": dict(reconstruction="PARABOLIC", limiter="DEFAULT", time_stepping="RK3", solver="hlld", divb="CONSTRAINED_TRANSPORT", cfl=0.3),
    "glm_matched": dict(reconstruction="LINEAR", limiter="VANLEER_LIM", time_stepping="RK2", solver="hll", divb="DIV_CLEANING", cfl=0.3),
    "glm_best": dict(reconstruction="PARABOLIC", limiter="DEFAULT", time_stepping="RK3", solver="hlld", divb="DIV_CLEANING", cfl=0.3),
}
X.VARIANTS["enzo"] = {
    "dedner": dict(HydroMethod=4, RiemannSolver=1, ReconstructionMethod=0, cfl=0.3, divb="Dedner (Wang & Abel 2009), HLL, PLM, RK2"),
    "ct": dict(HydroMethod=6, RiemannSolver=6, ReconstructionMethod=0, MHD_CT_Method=2, cfl=0.3, divb="CT (Collins et al. 2010), HLLD, PLM"),
}


def targets_of(config, outputs):
    return X.output_times(config) if outputs else [0.0, float(config["t_end"])]


def interval(config, outputs):
    t_end = float(config["t_end"])
    return t_end/(int(config["frames"]) - 1) if outputs else t_end


def run_code(cmd, rundir, log, env=None, gpu=True):
    (rundir/"command.txt").write_text(" ".join(shlex.quote(str(c)) for c in cmd) + "\n")
    start = time.perf_counter();perf = {}
    with log.open("w") as f:
        if gpu:
            with X.GpuMemorySampler() as memory:
                code = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env).returncode
            memory.record(perf)
        else:
            code = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env).returncode
    perf["wall_seconds"] = time.perf_counter() - start
    return code, perf


def spec_env(spec):
    return dict(os.environ, ABEL_SPEC=str(Path(spec).resolve()))


def conserved(rho, v, p, b, gamma, psi=None):
    """(N, N, Nz, 9) from primitives: rho, rho v, p/(gamma-1) + rho v^2/2 + B^2/2, B, psi."""
    u = np.zeros(rho.shape + (9,))
    u[..., 0] = rho
    for d in range(3):u[..., 1 + d] = rho*v[d];u[..., 5 + d] = b[d]
    u[..., 4] = p/(gamma - 1) + 0.5*rho*sum(x*x for x in v) + 0.5*sum(x*x for x in b)
    if psi is not None:u[..., 8] = psi
    return u


def unique(items, key, config, outputs):
    """Frames at distinct times (a code may write its last output twice: on the cadence and at the end)."""
    tol = 1e-6*interval(config, outputs);out = []
    for item in items:
        if not out or abs(key(item) - key(out[-1])) > tol:out.append(item)
    return out


def finish(code, text, perf, found, targets, config, outputs):
    """Run status: complete when every output was written and the last is at the end time."""
    tol = 1e-6*interval(config, outputs)
    complete = len(found) == len(targets) and abs(found[-1][0] - targets[-1]) <= max(tol, 1e-4*interval(config, outputs))
    if found:perf["steps"] = found[-1][1]
    rc = 0 if complete and code == 0 else (code or 1)
    return rc, text[-2000:]


# ------------------------------------------------------------------------------ AthenaPK
def athenapk_input(config, n, spec, settings, geometry, outputs):
    nz = 1 if geometry == "plane" else n
    # Blocks of 128: Kokkos' team scratch on the GPU, one line of cells per team, does not fit 256 ("could not find a valid execution configuration")
    block = min(n, 128)
    t_end = float(config["t_end"])
    lines = ["<job>", "problem_id = abel_suite", "<problem/abel>", f"spec_file = {Path(spec).resolve()}",
             "<parthenon/job>", "problem_id = abel",
             "<parthenon/mesh>", "refinement = none", f"nghost = {3 if settings['reconstruction'] == 'ppm' else 2}"]
    for d, nd in zip((1, 2, 3), (n, n, nz)):
        lines += [f"nx{d} = {nd}", f"x{d}min = 0.0", f"x{d}max = 1.0", f"ix{d}_bc = periodic", f"ox{d}_bc = periodic"]
    lines += ["<parthenon/meshblock>", f"nx1 = {block}", f"nx2 = {block}", f"nx3 = {min(nz, block)}",
              "<parthenon/time>", f"integrator = {settings['integrator']}", f"cfl = {settings['cfl']}", f"tlim = {t_end!r}",
              f"nlim = {X.step_cap(config, n)}", "perf_cycle_offset = 2", "ncycle_out = 1000",
              "<hydro>", "fluid = glmmhd", "eos = adiabatic", f"riemann = {settings['riemann']}",
              f"reconstruction = {settings['reconstruction']}", f"gamma = {X.gamma_of(config)!r}", "dfloor = 1.0e-6", "pfloor = 1.0e-6",
              "<parthenon/output0>", "file_type = hdf5", "id = cons", "variables = cons", f"dt = {interval(config, outputs)!r}"]
    return "\n".join(lines) + "\n"


def read_athenapk(path, n, nz):
    import h5py
    with h5py.File(path, "r") as f:
        info = f["Info"].attrs;t = float(info["Time"]);cycle = int(info["NCycle"])
        size = [int(s) for s in info["MeshBlockSize"]];loc = np.array(f["LogicalLocations"])
        data = np.array(f["cons"])
    nb = loc.shape[0];u = np.zeros((n, n, nz, 9))
    # (blocks, components, k, j, i) or (blocks, k, j, i, components)
    if data.ndim == 4:data = data[:, None] if data.shape[1] == size[2] else data[..., None]
    comp_first = data.shape[1] == 9 and data.shape[-1] != 9
    for b in range(nb):
        blk = data[b] if comp_first else np.moveaxis(data[b], -1, 0)
        ox, oy, oz = loc[b, 0]*size[0], loc[b, 1]*size[1], loc[b, 2]*size[2]
        u[ox:ox + size[0], oy:oy + size[1], oz:oz + size[2], :] = np.moveaxis(blk[:9], 0, -1).transpose(2, 1, 0, 3)
    return t, cycle, u


def run_athenapk(args, config, spec, settings, rundir, rep, outputs, resume=False):
    nz = 1 if args.geometry == "plane" else args.n
    for f in rundir.glob("abel.cons.*.phdf"):f.unlink()
    (rundir/"athenapk.in").write_text(athenapk_input(config, args.n, spec, settings, args.geometry, outputs))
    cmd = shlex.split(args.mpi or "") + [args.exe, "-i", "athenapk.in"]
    code, perf = run_code(cmd, rundir, rundir/"athenapk.log")
    text = (rundir/"athenapk.log").read_text(errors="replace")
    m = re.findall(r"zone-cycles/wallsecond\s*=\s*([\d.eE+-]+)", text)
    if m:perf["zone_cycles_per_second"] = float(m[-1])
    import h5py
    files = sorted(rundir.glob("abel.cons.*.phdf"));found = []
    for f in files:
        with h5py.File(f, "r") as h:found.append((float(h["Info"].attrs["Time"]), int(h["Info"].attrs["NCycle"])))
    rc, tail = finish(code, text, perf, found, targets_of(config, outputs), config, outputs)
    return rc, tail, perf, (read_athenapk(f, args.n, nz) for f in files)


# ------------------------------------------------------------------------------ Cholla
def cholla_input(config, n, settings, geometry, outputs):
    nz = SLAB if geometry == "plane" else n
    lines = [f"nx={n}", f"ny={n}", f"nz={nz}", f"tout={float(config['t_end'])!r}", f"outstep={interval(config, outputs)!r}",
             f"gamma={X.gamma_of(config)!r}", "init=Abel", "xmin=0.0", "ymin=0.0", "zmin=0.0",
             "xlen=1.0", "ylen=1.0", f"zlen={nz/n!r}"] + [f"{a}{s}_bcnd=1" for a in "xyz" for s in "lu"] + ["outdir=./output/"]
    return "\n".join(lines) + "\n"


def read_cholla(paths, n):
    """One output: the file of each rank (<index>.h5.<rank>), its part of the grid at "offset"."""
    import h5py
    u = None
    for path in paths:
        with h5py.File(path, "r") as f:
            a = f.attrs;t = float(np.ravel(a["t"])[0]);steps = int(np.ravel(a["n_step"])[0]) if "n_step" in a else 0
            g = lambda k: np.array(f[k], dtype=float)
            rho = g("density");o = [int(v) for v in np.ravel(a["offset"])] if "offset" in a else [0, 0, 0]
            if u is None:u = np.zeros((n, n, int(np.ravel(a["dims"])[2]) if "dims" in a else rho.shape[2], 9))
            lx, ly, lz = rho.shape;part = u[o[0]:o[0] + lx, o[1]:o[1] + ly, o[2]:o[2] + lz]
            part[..., 0] = rho;part[..., 1] = g("momentum_x");part[..., 2] = g("momentum_y");part[..., 3] = g("momentum_z");part[..., 4] = g("Energy")
            bx, by, bz = g("magnetic_x"), g("magnetic_y"), g("magnetic_z")      # face fields: one more along their own axis, lower face first
            part[..., 5] = 0.5*(bx[:-1] + bx[1:]);part[..., 6] = 0.5*(by[:, :-1] + by[:, 1:]);part[..., 7] = 0.5*(bz[:, :, :-1] + bz[:, :, 1:])
    return t, steps, u


# Cholla's GPU memory, ghost cells included: 13.06e9 bytes at 256^3 (264^3 cells with ghosts) on one H100, so 512^3 needs
# about 100 GB and runs on two (srun -n 2: one rank and one GPU each, cudaSetDevice(rank % GPUs))
CHOLLA_BYTES_PER_CELL = 710


def gpu_bytes():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=60)
        return float(q.stdout.split()[0])*2**20
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def run_cholla(args, config, spec, settings, rundir, rep, outputs, resume=False):
    out = rundir/"output";shutil.rmtree(out, ignore_errors=True);out.mkdir(parents=True)
    ranks = int(m.group(1)) if (m := re.search(r"-n\s*(\d+)", args.mpi or "")) else 1
    nz = SLAB if args.geometry == "plane" else args.n
    need, have = CHOLLA_BYTES_PER_CELL*(args.n + 8)**2*(nz + 8)/ranks, gpu_bytes()
    if have and need > 0.95*have:
        # Cholla does not check its allocations: it would start and stop on "an illegal memory access"
        return 1, (f"does not fit: Cholla needs about {need/1e9:.0f} GB a GPU at N={args.n} on {ranks} rank(s), "
                   f"and the GPU has {have/1e9:.0f} GB"), {"wall_seconds": 0.0, "ranks": ranks}, iter(())
    (rundir/"cholla.txt").write_text(cholla_input(config, args.n, settings, args.geometry, outputs))
    cmd = shlex.split(args.mpi or "") + [args.exe, "cholla.txt"]
    code, perf = run_code(cmd, rundir, rundir/"cholla.log", env=spec_env(spec));perf["ranks"] = ranks
    text = (rundir/"cholla.log").read_text(errors="replace")
    # Cholla's own stops (its div B limit of 1e-9, a GPU error) end in chexit(-1), an MPI abort, and srun reports the
    # rank killed: a result, not a lost job
    stop = max(text.find("The magnetic divergence "), text.find("GPU_Error_Check: Failed"))
    if stop >= 0:code, text = code or 1, text[:text.find("\n", stop) + 1]
    import h5py
    files = sorted(out.rglob("*.h5.0"), key=lambda f: int(f.name.split(".")[0]));found = []
    for f in files:
        with h5py.File(f, "r") as h:
            found.append((float(np.ravel(h.attrs["t"])[0]), int(np.ravel(h.attrs["n_step"])[0]) if "n_step" in h.attrs else 0))
    pairs = unique(list(zip(found, files)), lambda p: p[0][0], config, outputs);found = [p[0] for p in pairs];files = [p[1] for p in pairs]
    files = [sorted(f.parent.glob(f.name[:-1] + "*"), key=lambda p: int(p.name.split(".")[-1])) for f in files]
    rc, tail = finish(code, text, perf, found, targets_of(config, outputs), config, outputs)
    plane = args.geometry == "plane"
    def frames():
        for f in files:
            t, s, u = read_cholla(f, args.n)
            yield t, s, (u[:, :, :1, :] if plane else u)
    return rc, tail, perf, frames()


# ------------------------------------------------------------------------------ gPLUTO
def pluto_input(config, n, settings, geometry, outputs):
    nz = 1 if geometry == "plane" else n
    t_end = float(config["t_end"])
    lines = ["[Grid]", f"X1-grid    0.0  {n}  1.0", f"X2-grid    0.0  {n}  1.0", f"X3-grid    0.0  {nz}  1.0",
             "[Time]", f"CFL              {settings['cfl']}", "CFL_max_var      1.1", f"tstop            {t_end!r}", "first_dt         1.e-6",
             "[Solver]", f"Solver        {settings['solver']}",
             "[Boundary]"] + [f"X{d}-{s}        periodic" for d in (1, 2, 3) for s in ("beg", "end")] + [
             "[Static Grid Output]", "uservar    0", f"dbl        {interval(config, outputs)!r}  -1   single_file",
             "flt       -1.0  -1   single_file", "vtk       -1.0  -1   single_file", "dbl.h5    -1.0  -1", "flt.h5    -1.0  -1",
             "tab       -1.0  -1", "ppm       -1.0  -1", "png       -1.0  -1", "log              1000", "analysis  -1.0  -1",
             "[Particles]", "Nparticles          -1     1", "particles_dbl        -1.0  -1", "particles_flt       -1.0  -1",
             "particles_vtk       -1.0  -1", "particles_tab       -1.0  -1", "[Parameters]"]
    return "\n".join(lines) + "\n"


def read_pluto(rundir, index, n, nz, names, gamma):
    raw = np.fromfile(rundir/f"data.{index:04d}.dbl", dtype="<f8")
    cells = n*n*nz;field = {}
    for q, name in enumerate(names):                    # cell-centred variables come first; staggered ones after
        if name.endswith("s") and name.startswith("Bx"):break
        field[name] = raw[q*cells:(q + 1)*cells].reshape((n, n, nz), order="F")
    v = [field["vx1"], field["vx2"], field.get("vx3", 0*field["rho"])]
    b = [field["Bx1"], field["Bx2"], field.get("Bx3", 0*field["rho"])]
    return conserved(field["rho"], v, field["prs"], b, gamma, field.get("psi_glm"))


def run_pluto(args, config, spec, settings, rundir, rep, outputs, resume=False):
    nz = 1 if args.geometry == "plane" else args.n
    for f in list(rundir.glob("data.*.dbl")) + [rundir/"dbl.out"]:f.unlink(missing_ok=True)
    (rundir/"pluto.ini").write_text(pluto_input(config, args.n, settings, args.geometry, outputs))
    # --exe names the 2D build of the set-up (<set-up>-2d/pluto); the 3D tests run its sibling <set-up>-3d/pluto
    exe = Path(args.exe).resolve();want = f"-{2 if args.geometry == 'plane' else 3}d"
    if exe.parent.name[-3:] in ("-2d", "-3d"):exe = exe.parent.parent/(exe.parent.name[:-3] + want)/exe.name
    cmd = shlex.split(args.mpi or "") + [exe, "-i", "pluto.ini"]
    code, perf = run_code(cmd, rundir, rundir/"pluto.log", env=spec_env(spec))
    perf["executable_used"] = str(exe)
    text = (rundir/"pluto.log").read_text(errors="replace")
    rows = [l.split() for l in (rundir/"dbl.out").read_text().splitlines() if l.strip()] if (rundir/"dbl.out").is_file() else []
    # dbl.out prints the time to 7 digits (%12.6e), coarser than the driver's frame tolerance (1e-6 of the output interval):
    # the steps land on output times, so a time within the printed precision of one is that time
    targets = targets_of(config, outputs)
    for r in rows:
        near = min(targets, key=lambda s: abs(s - float(r[1])))
        if abs(near - float(r[1])) <= 6e-7*abs(near):r[1] = repr(near)
    rows = unique(rows, lambda r: float(r[1]), config, outputs)
    found = [(float(r[1]), int(r[3])) for r in rows]
    rc, tail = finish(code, text, perf, found, targets_of(config, outputs), config, outputs)
    gamma = X.gamma_of(config)
    return rc, tail, perf, ((float(r[1]), int(r[3]), read_pluto(rundir, int(r[0]), args.n, nz, r[6:], gamma)) for r in rows)


# ------------------------------------------------------------------------------ Enzo
def enzo_input(config, n, settings, geometry, outputs):
    rank = 2 if geometry == "plane" else 3
    lines = ["ProblemType = 777", f"TopGridRank = {rank}", "TopGridDimensions = " + " ".join([str(n)]*rank),
             "DomainLeftEdge = " + " ".join(["0.0"]*rank), "DomainRightEdge = " + " ".join(["1.0"]*rank),
             "LeftFaceBoundaryCondition = " + " ".join(["3"]*rank), "RightFaceBoundaryCondition = " + " ".join(["3"]*rank),
             f"StopTime = {float(config['t_end'])!r}", f"dtDataDump = {interval(config, outputs)!r}", "DataDumpName = data",
             f"StopCycle = {X.step_cap(config, n) if X.step_cap(config, n) > 0 else 1000000000}",
             f"HydroMethod = {settings['HydroMethod']}", f"RiemannSolver = {settings['RiemannSolver']}",
             f"ReconstructionMethod = {settings['ReconstructionMethod']}", f"Gamma = {X.gamma_of(config)!r}",
             f"CourantSafetyNumber = {settings['cfl']}", "DualEnergyFormalism = 0", "StaticHierarchy = 1", "MaximumRefinementLevel = 0",
             "WriteExternalAccel = 0", "OutputTemperature = 0"]
    if "MHD_CT_Method" in settings:lines.append(f"MHD_CT_Method = {settings['MHD_CT_Method']}")
    return "\n".join(lines) + "\n"


def enzo_dumps(rundir):
    return sorted((d for d in rundir.glob("DD*") if (d/f"data{d.name[2:]}.hierarchy").is_file()), key=lambda d: int(d.name[2:]))


def enzo_meta(dump):
    text = (dump/f"data{dump.name[2:]}").read_text(errors="replace")
    t = float(re.search(r"^InitialTime\s*=\s*([^\s]+)", text, re.M).group(1))
    c = int(re.search(r"^InitialCycleNumber\s*=\s*(\d+)", text, re.M).group(1))
    return t, c


def read_enzo(dump, n, rank, settings):
    import h5py
    nz = n if rank == 3 else 1
    t, c = enzo_meta(dump);u = np.zeros((n, n, nz, 9))
    text = (dump/f"data{dump.name[2:]}.hierarchy").read_text()
    for block in re.split(r"\n(?=Grid = )", text):
        m = re.search(r"^Grid = (\d+)", block, re.M)
        if not m:continue
        g = int(m.group(1))
        start = [int(x) for x in re.search(r"GridStartIndex\s*=\s*(.+)", block).group(1).split()]
        end = [int(x) for x in re.search(r"GridEndIndex\s*=\s*(.+)", block).group(1).split()]
        left = [float(x) for x in re.search(r"GridLeftEdge\s*=\s*(.+)", block).group(1).split()]
        fname = re.search(r"BaryonFileName\s*=\s*(.+)", block).group(1).strip()
        o = [int(round(x*n)) for x in left] + [0]*(3 - rank);size = [e - s + 1 for s, e in zip(start, end)] + [1]*(3 - rank)
        with h5py.File(dump/Path(fname).name, "r") as f:
            grp = f[f"Grid{g:08d}"]
            arr = lambda k: np.array(grp[k], dtype=float).reshape(size[::-1]).transpose()     # (nz, ny, nx) on disk
            rho = arr("Density");vel = [arr("x-velocity"), arr("y-velocity"), arr("z-velocity")]
            b = [arr("Bx"), arr("By"), arr("Bz")];te = arr("TotalEnergy")
            psi = arr("Phi") if "Phi" in grp else 0*rho
        sl = (slice(o[0], o[0] + size[0]), slice(o[1], o[1] + size[1]), slice(o[2], o[2] + size[2]))
        u[sl + (0,)] = rho
        for d in range(3):u[sl + (1 + d,)] = rho*vel[d];u[sl + (5 + d,)] = b[d]
        u[sl + (4,)] = rho*te;u[sl + (8,)] = psi
    return t, c, u


def run_enzo(args, config, spec, settings, rundir, rep, outputs, resume=False):
    rank = 2 if args.geometry == "plane" else 3
    for d in rundir.glob("DD*"):shutil.rmtree(d, ignore_errors=True)
    (rundir/"enzo.par").write_text(enzo_input(config, args.n, settings, args.geometry, outputs))
    cmd = shlex.split(args.mpi or "") + [args.exe, "-d", "enzo.par"]
    code, perf = run_code(cmd, rundir, rundir/"enzo.log", env=spec_env(spec), gpu=False)
    text = (rundir/"enzo.log").read_text(errors="replace")
    # Enzo's own stop (ERROR_MESSAGE: NaN pressure in mhd_dt.F, say) aborts MPI and srun reports the ranks killed: a result, not a lost job
    stop = text.find("=== ENZO ERROR ===")
    if stop >= 0:code, text = 1, text[:text.find("\n", stop) + 1]
    dumps = unique(enzo_dumps(rundir), lambda d: enzo_meta(d)[0], config, outputs);found = [enzo_meta(d) for d in dumps]
    rc, tail = finish(code, text, perf, found, targets_of(config, outputs), config, outputs)
    return rc, tail, perf, (read_enzo(d, args.n, rank, settings) for d in dumps)


# ------------------------------------------------------------------------------ Phantom
LAYERS = 8          # Phantom (3D only): particle layers of the periodic slab a planar test runs on
HFACT = 1.2         # Phantom's hfact_default: h = hfact (m / rho)^(1/3)


def phantom_particles(u, planar):
    """Equal-mass particles for the suite's (n, n, nz, 9) state, as Phantom's own shock and rotor set-ups place them:
    lattices of different spacing pasted together. Each cell takes the near-cubic lattice whose density is closest to
    its own (k = round(n (rho / mean rho)^(1/3)) points per unit length in x and y, round(LAYERS k / n) layers in the
    slab: as many particles as cells) and keeps that lattice's points inside it, so a uniform state is one lattice of a
    particle per cell, and a
    density is matched to steps of about 3/n (10 % at 32^2, 0.6 % at 512^2) with the lattice locally isotropic. The
    particle mass makes the total mass exact. Values of the cell each particle lies in. Planar: the n x n plane times
    a z-periodic slab of LAYERS/n. 3D: the uniform-density tests only (none of the suite's 3D tests is otherwise)."""
    n = u.shape[0]
    if planar:
        rho = u[:, :, 0, 0].astype(float);thick = LAYERS/n
        level = np.maximum(np.rint(n*np.cbrt(rho/rho.mean())).astype(int), 1)
        pts = []
        for k in np.unique(level):
            g = (np.arange(k) + 0.5)/k;c = np.minimum((g*n).astype(int), n - 1)
            a, b = np.nonzero(level[np.ix_(c, c)] == k)
            if not len(a):continue
            nzk = max(int(round(LAYERS*k/n)), 1);zl = (np.arange(nzk) + 0.5)*thick/nzk
            pts.append(np.column_stack([np.repeat(g[a], nzk), np.repeat(g[b], nzk), np.tile(zl, len(a))]))
        pos = np.concatenate(pts)
        cell = np.minimum((pos[:, 0]*n).astype(int), n - 1)*n + np.minimum((pos[:, 1]*n).astype(int), n - 1)
        state = u[:, :, 0, :].reshape(n*n, 9).astype(float)[cell];mass = rho.sum()/n**2*thick/len(pos)
    else:
        rho = u[..., 0].astype(float)
        if rho.max() - rho.min() > 1e-12*rho.max():raise SystemExit("Phantom's 3D set-up here needs a uniform density")
        g = (np.arange(n) + 0.5)/n;X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1);thick = 1.0
        state = u.reshape(n**3, 9).astype(float);mass = rho.mean()/len(pos)
    r = state[:, 0];v = state[:, 1:4]/r[:, None];b = state[:, 5:8]
    e = (state[:, 4] - 0.5*(state[:, 1:4]**2).sum(1)/r - 0.5*(b**2).sum(1))/r
    h = HFACT*(mass/r)**(1/3)
    return pos, h, v, e, b, mass, thick


def write_particles(path, pos, h, v, e, b, mass, thick, gamma, t_max, dt_max):
    with open(path, "wb") as f:
        np.array([len(pos)], dtype="<i8").tofile(f)
        np.array([mass, 0.0, 1.0, 0.0, 1.0, 0.0, thick, gamma, t_max, dt_max], dtype="<f8").tofile(f)
        np.concatenate([pos[:, 0], pos[:, 1], pos[:, 2], h, v[:, 0], v[:, 1], v[:, 2], e, b[:, 0], b[:, 1], b[:, 2]]).astype("<f8").tofile(f)


def read_phantom(path, n, planar, thick):
    """(time, (n, n, nz, 9)) of a dump's particles (analysis_abel.f90): mass, momentum, total energy and field times
    volume of each particle deposited on the grid as GIZMO's (gizmo_run.deposit; the kernel's support is 2h) and
    divided by the cell volume. psi = 0."""
    sys.path.insert(0, str(SWEEP.parent/"sherlock_gizmo"))
    import gizmo_run
    with open(path, "rb") as f:
        npart = int(np.fromfile(f, dtype="<i8", count=1)[0]);t, m = np.fromfile(f, dtype="<f8", count=2)
        a = np.fromfile(f, dtype="<f8", count=12*npart).reshape(12, npart)
    pos = a[0:3].T;h = np.abs(a[3]);rho = a[4];vel = a[5:8].T;e = a[8];b = a[9:12].T;mass = np.full(npart, m)
    vol = mass/rho;nz = 1 if planar else n
    ext = np.column_stack([mass, mass[:, None]*vel, mass*(e + .5*(vel**2).sum(1)) + .5*vol*(b**2).sum(1), vol[:, None]*b])
    u = gizmo_run.deposit(pos if not planar else pos[:, :2], ext, n, nz, 2*h)*(n**3 if not planar else n**2/thick)
    return float(t), np.concatenate([u, np.zeros(u.shape[:3] + (1,))], axis=-1)


def run_phantom(args, config, spec, settings, rundir, rep, outputs, resume=False):
    planar = args.geometry == "plane";n = args.n
    exe = Path(args.exe).resolve();bindir = exe.parent              # --exe: bin/phantom; phantomsetup and phantomanalysis beside it
    for f in list(rundir.glob("abel_0*")) + [rundir/"abel.in", rundir/"abel.particles"]:f.unlink(missing_ok=True)
    prefix = str(spec)[:-len(".spec")] + "_state"
    X.write_spec(args.config, n, prefix, args.seed if args.seed is not None else int(config.get("seeds", [42])[0]), state=True, plane=planar)
    u0 = np.fromfile(prefix + ".state", dtype="<f4").reshape((n, n, 1 if planar else n, 9), order="F")
    pos, h, v, e, b, mass, thick = phantom_particles(u0, planar)
    t_end = float(config["t_end"]);dt_out = interval(config, outputs)
    write_particles(rundir/"abel.particles", pos, h, v, e, b, mass, thick, X.gamma_of(config), t_end, dt_out)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", "8")))
    with (rundir/"phantomsetup.log").open("w") as f:
        subprocess.run([str(bindir/"phantomsetup"), "abel"], stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env, input=b"\n"*20)
    # Phantom's default projection of div B at each dump interval takes 26 % of a smooth field's energy (Orszag-Tang, at every
    # grid, independent of N): its hyperbolic cleaning alone, as in Price et al. 2018
    infile = rundir/"abel.in";infile.write_text(re.sub(r"(use_divb_proj\s*=\s*)T", r"\1F", infile.read_text()))
    code, perf = run_code(shlex.split(args.mpi or "") + [exe, "abel.in"], rundir, rundir/"phantom.log", env=env, gpu=False)
    perf.update(particles=len(pos), particle_mass=mass, slab_layers=LAYERS if planar else 0)
    dumps = sorted(f for f in rundir.glob("abel_0*") if f.suffix == "" and f.name[6:].isdigit())
    if dumps:
        with (rundir/"phantomanalysis.log").open("w") as f:
            subprocess.run([str(bindir/"phantomanalysis")] + [d.name for d in dumps], stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env, input=b"\n"*20)
    files = [d.with_name(d.name + ".abel") for d in dumps if d.with_name(d.name + ".abel").is_file()]
    times = []
    for f in files:
        with open(f, "rb") as g:np.fromfile(g, dtype="<i8", count=1);times.append(float(np.fromfile(g, dtype="<f8", count=1)[0]))
    # tmax and dtmax are rounded in abel.in: the last dump can fall 1e-13 short of tmax, and Phantom then takes one more step and dumps past the end
    keep = [p for p in unique(list(zip(times, files)), lambda p: p[0], config, outputs) if p[0] <= t_end + 1e-6*dt_out]
    text = (rundir/"phantom.log").read_text(errors="replace")
    m = re.findall(r"Since code start:\s*(\d+) timesteps", text)
    found = [(t, int(m[-1]) if m else 0) for t, _ in keep]
    rc, tail = finish(code, text, perf, found, targets_of(config, outputs), config, outputs)
    return rc, tail, perf, ((t, found[-1][1] if found else 0, read_phantom(f, n, planar, thick)[1]) for t, f in keep)


def remove_native(outdir):
    """After a finished run: drop each code's own outputs (converted); keep logs, inputs and commands."""
    for record in Path(outdir).glob("*_r*/run.toml"):
        if 'status = "complete"' not in record.read_text():continue
        run = record.parent
        for f in list(run.glob("abel.cons.*.phdf")) + list(run.glob("data.*.dbl")) + list(run.glob("abel_0*")) + [run/"abel.particles"]:
            f.unlink(missing_ok=True)
        for d in [run/"output"] + list(run.glob("DD*")):shutil.rmtree(d, ignore_errors=True)


X.VARIANTS["phantom"] = {
    "sph": dict(method="SPH (cubic spline, hfact 1.2), MHD with constrained hyperbolic/parabolic cleaning; Phantom's default "
                "artificial viscosity and resistivity switches, individual timesteps; div B projection off; planar tests on a slab of 8 layers"),
}
X.RUNNERS.update(athenapk=run_athenapk, cholla=run_cholla, pluto=run_pluto, enzo=run_enzo, phantom=run_phantom)

if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    try:X.main()
    finally:
        if "--keep-native" not in sys.argv and len(positional) >= 5:remove_native(positional[4])
