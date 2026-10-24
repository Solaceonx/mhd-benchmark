"""mini-ramses-ism in the comparison suite, through the sweep's external-code driver.

    python3 miniramses_run.py miniramses matched|best CONFIG.toml N OUTDIR --exe PATH [--geometry plane] [--repeats 1]
        [--device-kind gpu] [--resume]

The same command line, run directory layout, records and measurements as RAMSES, AthenaK, Athena++ and AREPO in
sherlock_sweep/codes/external.py: this file adds only what is specific to mini-ramses (namelist, running it, reading
its outputs) and then calls that driver's main(). Nothing in the sweep bundle is changed.

mini-ramses-ism (bitbucket.org/ermoseley/mini-ramses-ism, branch develop): Eric Moseley's version of RAMSES II, the
GPU (CUDA Fortran) rewrite of RAMSES. Its GPU path is 3D only (gpu/gpu_utils.cuf: "CUDA GPU path requires NDIM=3"), so
the planar tests run in 3D on an N x N x N grid that is the same at every z, as AthenaK's runs in the sweep; their
records keep the z = 0 plane, as AthenaK's. Constrained transport on a uniform grid here (levelmin = levelmax = log2 N),
MUSCL-Hancock with the MonCen slope, two set-ups as for classic RAMSES in the sweep:
  matched   HLL, Courant 0.5
  best      HLLD, Courant 0.8
The build (build.sh) is patched (prepare_patch.py) only to read the sweep's initial data and to land on the output
times; Eric's density and pressure switches to LLF (switch_llf_dmin, switch_llf_pmin) are left off, as upstream.

Initial data: the Float32 cell state and lower-face fields the sweep writes for RAMSES (tools/write_spec.jl --state),
read by the patched condinit. Outputs: mini-ramses writes, per level, the primitive state of every cell in Float32
(density, velocity, thermal pressure, cell-centred B as the mean of the two faces); they are turned back into our
conservative (N, N, Nz, 9) state with E = p/(gamma - 1) + rho v^2/2 + B^2/2 and psi = 0.
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

SWEEP = Path(os.environ.get("SWEEP_BUNDLE", Path(__file__).resolve().parents[1]/"sherlock_sweep"))
sys.path.insert(0, str(SWEEP/"codes"))
import external as X  # noqa: E402

PRECISION = {"8": "fp64", "4": "fp32"}.get(os.environ.get("MINIRAMSES_NPRE", "8"), "fp64")
X.VARIANTS["miniramses"] = {
    "matched": dict(riemann="hll", riemann2d="hll", slope_type=2, courant=0.5, precision=PRECISION),
    "best": dict(riemann="hlld", riemann2d="hlld", slope_type=2, courant=0.8, precision=PRECISION),
}


def namelist(config, n, settings, outputs, ndim):
    level = int(round(np.log2(n)))
    if 2**level != n:raise SystemExit("mini-ramses requires N to be a power of two")
    times = X.output_times(config) if outputs else [0.0, float(config["t_end"])]
    # mini-ramses ends the run when the time reaches the last output time, before the step that would write that output
    # (amr/update_time.f90). One more output time just after t_end makes it write the output at t_end (exactly: the
    # patched newdt_fine lands on it), take one step of 1e-9 t_end, and stop there; that last output is never written.
    times = times + [times[-1]*(1 + 1e-9) + 1e-12]
    cap = X.step_cap(config, n)
    octs = n**ndim//(2**ndim - 1)                       # the octs of every level of the tree, one GPU, no MPI
    return f"""&RUN_PARAMS
hydro=.true.
ncontrol=1000
nsubcycle=1
nstepmax={cap + 1 if cap > 0 else 100000000}
/
&AMR_PARAMS
levelmin={level}
levelmax={level}
ngridmax={octs + 1000}
boxlen=1.0
/
&INIT_PARAMS
/
&OUTPUT_PARAMS
tout={",".join(repr(float(t)) for t in times)}
/
&HYDRO_PARAMS
gamma={X.gamma_of(config)!r}
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


def info(outdir):
    text = (outdir/"info.txt").read_text()
    get = lambda key, kind: kind(re.search(rf"^{key}\s*=\s*(\S+)", text, re.M).group(1))
    return dict(time=get("time", float), steps=get("nstep_coarse", int), ncpu=get("ncpu", int), ndim=get("ndim", int),
                levelmin=get("levelmin", int), levelmax=get("levelmax", int))


def read_output(outdir, n, gamma, planar=False):
    """A uniform-grid mini-ramses output -> (time, steps, (N, N, Nz, 9) conservative state); Nz = 1 in 2D, and for a
    planar test run in 3D (planar: only its z = 0 plane is kept; the others are copies of it). The layout
    is that of utils/py/miniramses.py (rd_amr, rd_hydro): per file a header, the oct count of each level, then per level
    the integer key of every oct (and its refinement map) and the Float32 primitive state of its 2^ndim cells."""
    i = info(outdir);ndim = i["ndim"];lmin, lmax = i["levelmin"], i["levelmax"];level = int(round(np.log2(n)))
    nz = n if ndim == 3 and not planar else 1;u = np.full((n, n, nz, 9), np.nan)
    for cpu in range(1, i["ncpu"] + 1):
        amr = (outdir/f"amr.{cpu:05d}").read_bytes();hyd = (outdir/f"hydro.{cpu:05d}").read_bytes()
        counts = np.frombuffer(amr, "<i4", lmax - lmin + 1, 12)
        nvar = int(np.frombuffer(hyd, "<i4", 1, 4)[0])
        if not np.array_equal(np.frombuffer(hyd, "<i4", lmax - lmin + 1, 16), counts):raise RuntimeError(f"{outdir}: amr and hydro files disagree")
        pa = 12 + 4*(lmax - lmin + 1);ph = 16 + 4*(lmax - lmin + 1)
        for il, nc in zip(range(lmin, lmax + 1), counts):
            nc = int(nc)
            keys = np.frombuffer(amr, "<i4", nc*(ndim + 1), pa).reshape(nc, ndim + 1)[:, :ndim]
            q = np.frombuffer(hyd, "<f4", nc*nvar*2**ndim, ph).reshape(nc, nvar, 2**ndim)
            pa += 4*nc*(ndim + 1);ph += 4*nc*nvar*2**ndim
            if il != level:continue
            if ndim == 3 and nz == 1:                             # only the octs of the z = 0 plane (1/(N/2) of them)
                bottom = keys[:, 2] == 0;keys, q = keys[bottom], q[bottom];nc = len(keys)
            q = q.astype(np.float64)
            for ind in range(2**ndim):
                ijk = 2*keys + np.array([(ind >> d) & 1 for d in range(ndim)])
                if ndim == 2:ijk = np.column_stack([ijk, np.zeros(nc, int)])
                sel = ijk[:, 2] < nz                                  # a planar test in 3D: the z = 0 plane
                ijk = ijk[sel];rho, v, p, b = q[sel, 0, ind], q[sel, 1:4, ind], q[sel, 4, ind], q[sel, 5:8, ind]
                cells = (ijk[:, 0], ijk[:, 1], ijk[:, 2])
                u[cells + (0,)] = rho
                for c in range(3):
                    u[cells + (1 + c,)] = rho*v[:, c];u[cells + (5 + c,)] = b[:, c]
                u[cells + (4,)] = p/(gamma - 1) + .5*rho*(v**2).sum(1) + .5*(b**2).sum(1)
    if np.isnan(u[..., :8]).any():raise RuntimeError(f"mini-ramses output {outdir} does not cover the uniform level {level}")
    u[..., 8] = 0.0
    return i["time"], i["steps"], u


def outputs_of(rundir):
    """Complete outputs (info.txt written), in order."""
    return sorted((o for o in rundir.glob("output_*") if (o/"info.txt").is_file()), key=lambda o: int(o.name.split("_")[-1]))


def run_miniramses(args, config, spec, settings, rundir, rep, outputs, resume=False):
    ndim = 2 if args.geometry == "plane" else 3;n = args.n
    targets = X.output_times(config) if outputs else [0.0, float(config["t_end"])]
    tol = 1e-6*float(config["t_end"])/(int(config["frames"]) - 1)
    prefix = spec.removesuffix(".spec");planar = X.planar(config)
    if not Path(prefix + ".state").is_file():                # the driver writes the cell state for RAMSES only
        seed = args.seed if args.seed is not None else config.get("seeds", [42])[0]
        # a planar test: the plane, also for a 3D run (the patched condinit repeats it through z)
        X.write_spec(args.config, n, prefix, seed, state=True, plane=planar)
    # A requeued attempt runs again from the start: mini-ramses's backups are not used (the runs are short on a GPU).
    for o in rundir.glob("output_*"):shutil.rmtree(o)
    nml = rundir/"abel.nml";nml.write_text(namelist(config, n, settings, outputs, ndim))
    env = dict(os.environ, ABEL_IC_PREFIX=prefix, ABEL_IC_N=str(n))
    cmd = shlex.split(args.mpi or "") + [args.exe, nml.name];log = rundir/"miniramses.log"
    start = time.perf_counter()
    with log.open("w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env).returncode
    wall = time.perf_counter() - start;text = log.read_text(errors="replace") if log.is_file() else ""
    perf = dict(wall_seconds=wall, precision=f"{PRECISION} evolution; Float32 snapshots")
    m = re.findall(r"Total elapsed time:\s*([\d.eE+-]+)", text)
    if m:perf["code_seconds"] = float(m[-1])
    outs = outputs_of(rundir);gamma = X.gamma_of(config)
    keep = [o for o in outs if any(abs(info(o)["time"] - t) <= tol for t in targets)]
    if keep:perf["steps"] = info(outs[-1])["steps"]
    frames = (read_output(o, n, gamma, planar) for o in keep)
    complete = bool(keep) and abs(info(keep[-1])["time"] - targets[-1]) <= tol
    rc = code if code else (0 if complete else 1)
    return rc, text[-2000:], perf, frames


def remove_native(outdir):
    """After a finished run: drop mini-ramses's own outputs (converted; the run can be made again)."""
    for record in Path(outdir).glob("miniramses_*_r*/run.toml"):
        if 'status = "complete"' not in record.read_text():continue
        for o in record.parent.glob("output_*"):shutil.rmtree(o)


X.RUNNERS["miniramses"] = run_miniramses

if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    try:X.main()
    finally:
        if "--keep-native" not in sys.argv and len(positional) >= 5:remove_native(positional[4])
