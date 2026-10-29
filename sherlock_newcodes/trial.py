"""Trial of the new codes (trial.sbatch): starting states and one short run per set-up, before production.

    python3 trial.py CODE VARIANT EXE MPI OUTDIR

1. Starting states (--ic-only): Orszag-Tang, GEM and the oblique Alfven wave at 64^2 (planar) and the 3D field loop at
   32^3, against the suite's own Float32 state (tools/write_spec.jl --state): largest difference of each field over
   that field's largest value. OK below 1e-5 (Float32 snapshots: about 1e-7). Phantom (particles deposited on the
   grid): the mean difference over the field's mean size, OK below 0.2 (catches a misplaced axis or field).
2. Orszag-Tang at 64^2 to t = 0.5, against AthenaK's (sweep, PPM/HLLD) at the same grid: L1 difference of the density
   over its mean. A set-up of another code differs by a few per cent; a misplaced axis or field by order one. OK below 0.1.
Prints one line per check and exits 1 if any is not OK.
"""
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP = Path(os.environ["SWEEP_BUNDLE"]);RESULTS = Path(os.environ.get("MHD_SWEEP_RESULTS", Path(os.environ["SCRATCH"])/"mhd-results-sweep"))
NAMES = ("rho", "mx", "my", "mz", "E", "Bx", "By", "Bz", "psi")


def run(code, variant, exe, mpi, test, n, out, geometry, ic_only):
    cmd = [sys.executable, str(HERE/"newcodes_run.py"), code, variant, str(SWEEP/"configs"/f"{test}.toml"), str(n), str(out),
           "--exe", exe, "--mpi", mpi, "--geometry", geometry, "--repeats", "1", "--device", os.uname().nodename] + (["--ic-only"] if ic_only else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-1500:]


def state(path, n, planar):
    shape = (n, n, 9) if planar else (n, n, n, 9)
    return np.fromfile(path, dtype="<f4").reshape(shape, order="F")


def main(code, variant, exe, mpi, outdir):
    out = Path(outdir);out.mkdir(parents=True, exist_ok=True);ok = True
    for test, n in (("orszag_tang", 64), ("gem", 64), ("alfven_oblique", 64), ("field_loop_3d", 32)):
        cfg = tomllib.load(open(SWEEP/"configs"/f"{test}.toml", "rb"));planar = not test.endswith("_3d")
        rd = out/f"{test}-N{n}-ic";rc, tail = run(code, variant, exe, mpi, test, n, rd, "plane" if planar else "cube", True)
        mine = sorted(rd.rglob("state_000.bin"))
        if rc or not mine:
            ok = False;print(f"   {test:16s} N={n}: NOT OK, the run failed (exit {rc}): {tail.strip().splitlines()[-1] if tail.strip() else ''}");continue
        prefix = out/f"{test}-N{n}-julia"
        subprocess.run([os.environ.get("MHD_JULIA", "julia"), "--startup-file=no", str(SWEEP/"tools/write_spec.jl"), str(SWEEP/"configs"/f"{test}.toml"),
                        str(n), str(prefix), "--seed", "42", "--state"] + (["--plane"] if planar else []), check=True, capture_output=True)
        ref = np.fromfile(str(prefix) + ".state", dtype="<f4").reshape((n, n, 1 if planar else n, 9), order="F")
        ref = ref[:, :, 0, :] if planar else ref
        got = state(mine[0], n, planar);worst = []
        for q, name in enumerate(NAMES[:8]):
            if code == "phantom":         # particles deposited on the grid: the mean difference over the field's mean size
                scale = float(np.abs(ref[..., q]).mean());worst.append((float(np.abs(got[..., q] - ref[..., q]).mean())/scale if scale > 0 else 0.0, name))
            else:
                scale = max(float(np.abs(ref[..., q]).max()), 1e-30)
                worst.append((float(np.abs(got[..., q] - ref[..., q]).max())/scale if np.abs(ref[..., q]).max() > 0 else float(np.abs(got[..., q]).max()), name))
        w = max(worst);good = w[0] < (0.2 if code == "phantom" else 1e-5);ok &= good
        print(f"   {test:16s} N={n}: {'OK' if good else 'NOT OK'}  starting state, largest {'mean ' if code == 'phantom' else ''}difference {w[0]:.2e} ({w[1]})")
    rd = out/"orszag_tang-N64";rc, tail = run(code, variant, exe, mpi, "orszag_tang", 64, rd, "plane", False)
    mine = sorted(rd.rglob("state_020.bin"));theirs = sorted((RESULTS/"orszag_tang-athenak-best-run-N64-s42/data").glob("athenak_best_r1/state_020.bin"))
    if rc or not mine:
        ok = False;print(f"   orszag_tang      N=64: NOT OK, the run failed (exit {rc}): {tail.strip().splitlines()[-1] if tail.strip() else ''}")
    elif not theirs:print("   orszag_tang      N=64: ran; AthenaK's state at t = 0.5 not found for the comparison")
    else:
        a, b = state(mine[0], 64, True), state(theirs[0], 64, True)
        d = float(np.abs(a[..., 0] - b[..., 0]).mean()/np.abs(b[..., 0]).mean());good = d < 0.1;ok &= good
        rec = tomllib.loads(next(rd.rglob("run.toml")).read_text())
        print(f"   orszag_tang      N=64: {'OK' if good else 'NOT OK'}  t = 0.5 against AthenaK PPM/HLLD: density L1 {d:.3f}; "
              f"{rec.get('steps', 0)} steps in {rec.get('wall_seconds', 0):.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 6:sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
