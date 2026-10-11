"""MHD-ALE (T. Abel's fork) in the sweep's external-code driver: the suite's planar tests in the periodic box.

    python3 mhdale_run.py mhdale ale CONFIG.toml N OUTDIR --exe PATH --mpi "srun -n K" --geometry plane [--repeats 1] [--resume]

The driver of the deployed sweep bundle (sherlock_sweep/codes/external.py, imported, not changed) does everything
that is the same for every code: the run directory, the spec file, run.toml, the frames and the measurements. This
file adds the code: the mesh (N/2 x N/2 elements; the velocity has degree 2, so N nodes per side), the command with
the options of options.sh, and the frames the patched code writes on the suite's grid (abel_dump.cpp; 8 x 8 sample points per cell: the mean of a
cell that a jump crosses is then off by at most a sixteenth of the jump).

Kept per run: the code's log (mhdale.log), its command, its own per-step records in output/ (divergence error of B,
smallest density, timestep, remaps, magnetic audit) and what each frame's sampling left out (abel_frame_NNN.txt).
The frames themselves are removed once converted (--keep-native keeps them). MHD-ALE has no restart: a run that is
interrupted starts again.

A run MHD-ALE ends itself (an MFEM abort such as "The time step crashed!", or a signal) is recorded as failed with
the time it reached; srun's report of such an abort must not be taken for the job being stopped.
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
sys.path.insert(0, str(SWEEP/"codes"));sys.path.insert(0, str(HERE))
import external as X  # noqa: E402
import make_mesh  # noqa: E402

OPT = dict(re.findall(r'^(MHDALE_[A-Z_]+)="([^"]*)"', (HERE/"options.sh").read_text(), re.M))
X.VARIANTS["mhdale"] = {
    "ale": dict(mesh="moving with the gas, remapped every 10 steps to the comoving initial mesh; periodic box",
                elements="N/2 per side: velocity degree 2, energy and density degree 1, vector potential degree 3",
                integrator="SSP RK3", courant=0.5, viscosity="artificial, on", divb="B = curl A in a Raviart-Thomas space",
                options=OPT["MHDALE_ORDERS"] + " " + OPT["MHDALE_PERIODIC_ALE"], precision="fp64"),
}
OWN_END = re.compile(r"MFEM abort: *(.*)|Verification failed: .*\n *--> *(.*)|(MPI_ABORT was invoked.*)|(Segmentation fault|Floating point exception|terminate called.*)")


def frame_info(path):
    return dict(l.split(None, 1) for l in path.read_text().splitlines() if l.strip())


def run_mhdale(args, config, spec, settings, rundir, rep, outputs, resume=False):
    if args.geometry != "plane":raise SystemExit("MHD-ALE runs the suite's planar tests only (--geometry plane)")
    n = args.n
    if n % 2:raise SystemExit("MHD-ALE needs an even N (N/2 elements per side)")
    targets = X.output_times(config) if outputs else [0.0, float(config["t_end"])]
    tol = 1e-6*float(config["t_end"])/(int(config["frames"]) - 1)
    out = rundir/"output";log = rundir/"mhdale.log";last = out/f"abel_frame_{len(targets) - 1:03d}.bin"
    finished = resume and last.is_file() and last.stat().st_size == n*n*9*8          # only the conversion was interrupted
    start = time.perf_counter();code = 0
    if not finished:
        shutil.rmtree(out, ignore_errors=True);out.mkdir(parents=True)
        make_mesh.main(n//2, rundir/"mesh.mesh")
        (rundir/"output_times.dat").write_text("".join(repr(float(t)) + "\n" for t in targets[1:]))
        cmd = shlex.split(args.mpi or "") + [str(args.exe), "-p", "100", "-dim", "2", "-m", "mesh.mesh", "-rs", "0"] \
            + shlex.split(OPT["MHDALE_ORDERS"]) + shlex.split(OPT["MHDALE_PERIODIC_ALE"]) \
            + ["-tf", repr(float(targets[-1])), "-ptf", "output_times.dat", "-od", "output"]
        (rundir/"command.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")
        env = dict(os.environ, ABEL_SPEC=str(Path(spec).resolve()), ABEL_BOX="periodic", ABEL_N=str(n), ABEL_SUB=os.environ.get("ABEL_SUB", "8"))
        for key in ("ABEL_ONLY_INITIAL", "ABEL_DUMP_LAST"):env.pop(key, None)
        with log.open("w") as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=rundir, env=env).returncode
        print(f"MHD-ALE ended with exit code {code} on {X.mpi_ranks(args.mpi)} ranks", flush=True)
    wall = time.perf_counter() - start;text = log.read_text(errors="replace") if log.is_file() else ""
    perf = dict(wall_seconds=wall)
    if finished:perf["converted_only"] = True
    found = []                                          # (frame, time, steps) of the frames that were written
    missing = empty = 0.0
    for k in range(len(targets)):
        txt = out/f"abel_frame_{k:03d}.txt";data = out/f"abel_frame_{k:03d}.bin"
        if not (txt.is_file() and data.is_file() and data.stat().st_size == n*n*9*8):break
        info = frame_info(txt);found.append((k, float(info["time"]), int(info["steps"])))
        missing = max(missing, float(info.get("samples_missing", 0)));empty = max(empty, float(info.get("cells_empty", 0)))
    steps = re.findall(r"^step (\d+),\s*t = ([0-9.eE+-]+),\s*dt = ([0-9.eE+-]+)", text, re.M)
    perf.update(sampling_missing_samples_max=missing, sampling_empty_cells_max=empty, remaps=text.count("-----REMAP DONE-----"),
                repeated_steps=text.count("Repeating step"))
    if steps:perf["steps"] = max(f[2] for f in found) if found else 0
    div = out/"B_divergence_error.dat"
    if div.is_file():
        v = [float(l.split()[1]) for l in div.read_text().splitlines() if len(l.split()) == 2]
        if v:perf.update(native_divergence_error_first=v[0], native_divergence_error_max=max(v))

    def frames():
        for k, t, s in found:
            u = np.fromfile(out/f"abel_frame_{k:03d}.bin", dtype="<f8").reshape((n, n, 9), order="F")
            yield t, s, u.reshape((n, n, 1, 9))
    complete = len(found) == len(targets) and abs(found[-1][1] - targets[-1]) <= tol and empty == 0
    tail = text[-2000:]
    own = OWN_END.search(text)
    if not complete and (own or empty):
        reason = "the sampling onto the grid left cells empty" if empty and len(found) == len(targets) else next(g for g in own.groups() if g).strip()[:300]
        perf["mhdale_stopped"] = reason
        if steps:perf.update(mhdale_stopped_at_time=float(steps[-1][1]), mhdale_last_timestep=float(steps[-1][2]), mhdale_steps_printed=len(steps))
        tail = "MHD-ALE stopped itself" + (f" at t = {float(steps[-1][1]):g} (step {steps[-1][0]}, last timestep {float(steps[-1][2]):.3g})" if steps else " before its first step") + ": " + reason
    rc = 0 if complete and code == 0 and not X.was_killed(code, tail) else (code or 1)
    return rc, tail, perf, frames()


def remove_native(outdir):
    """After a finished run: drop the frames (they are converted) and the mesh file; keep the logs and the code's own records."""
    for record in Path(outdir).glob("mhdale_*_r*/run.toml"):
        if 'status = "complete"' not in record.read_text():continue
        run = record.parent
        for f in list(run.glob("output/abel_frame_*.bin")) + [run/"mesh.mesh"]:f.unlink(missing_ok=True)


X.RUNNERS["mhdale"] = run_mhdale

if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    try:X.main()
    finally:
        if "--keep-native" not in sys.argv and len(positional) >= 5:remove_native(positional[4])
