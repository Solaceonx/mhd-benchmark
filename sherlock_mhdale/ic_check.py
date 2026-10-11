"""Compare a state MHD-ALE wrote on the suite's grid with the suite's own state: one line.

    python3 ic_check.py LABEL STATE_PREFIX N OUTPUT_DIR [FRAME]

STATE_PREFIX.state is the Float32 cell state of tools/write_spec.jl --state --plane (N x N x 1 x 9, x fastest);
OUTPUT_DIR/abel_frame_FRAME.bin is abel_dump.cpp's file (N x N x 9 doubles, the same layout). Per group of fields
(density; momentum x, y; total energy; B x, y): the mean absolute difference over the cells divided by the largest
absolute value of the suite's field ("L1/max"), and the ratio or difference of the box means. A unit, sign or axis
mistake shows as a difference of order 0.1 to 1; what the finite-element projection of a discontinuous start leaves
is of order 0.001 to 0.01 at 64^2. The verdict is a guide, read the numbers: OK below 0.02 with the means within 0.5 per
cent; BAD (the exit code that stops the trial) from 0.1 or means off by 5 per cent, or empty cells; CHECK in between. A test that looks the same with x and y exchanged cannot show an exchanged axis; Orszag-Tang does.
"""
import sys
from pathlib import Path

import numpy as np


def read_dump(outdir, n, frame=0):
    u = np.fromfile(Path(outdir)/f"abel_frame_{frame:03d}.bin", dtype="<f8")
    if u.size != n*n*9:raise SystemExit(f"{outdir}: frame {frame} has {u.size} numbers, not {n*n*9}")
    info = dict(l.split(None, 1) for l in (Path(outdir)/f"abel_frame_{frame:03d}.txt").read_text().splitlines() if l.strip())
    return u.reshape((n, n, 9), order="F"), {k: float(v) for k, v in info.items()}


def main(label, prefix, n, outdir, frame="0"):
    n = int(n);ref = np.fromfile(prefix + ".state", dtype="<f4").reshape((n, n, 1, 9), order="F")[:, :, 0, :].astype(float)
    u, info = read_dump(outdir, n, int(frame))
    out = [];worst = 0.0
    for name, idx in (("rho", [0]), ("mom", [1, 2]), ("E", [4]), ("B", [5, 6])):
        scale = max(float(np.abs(ref[..., idx]).max()), 1e-30);l1 = float(np.abs(u[..., idx] - ref[..., idx]).mean())/scale
        if np.abs(ref[..., idx]).max() < 1e-12:l1 = float(np.abs(u[..., idx]).mean())     # the suite's field is zero: absolute
        worst = max(worst, l1);out.append(f"{name} {l1:8.2e}")
    mass = u[..., 0].mean()/ref[..., 0].mean();energy = u[..., 4].mean()/ref[..., 4].mean()
    db = max(abs(u[..., 5].mean() - ref[..., 5].mean()), abs(u[..., 6].mean() - ref[..., 6].mean()))
    out_of_plane = max(float(np.abs(ref[..., 3]).max()), float(np.abs(ref[..., 7]).max()))
    bscale = max(float(np.abs(ref[..., 5:7]).max()), 1e-30)
    sound = info.get("cells_empty", 1) == 0 and out_of_plane == 0.0
    ok = sound and worst < 0.02 and abs(mass - 1) < 5e-3 and abs(energy - 1) < 5e-3 and db < 5e-3*bscale + 1e-12
    gross = not sound or worst >= 0.1 or abs(mass - 1) >= 0.05 or abs(energy - 1) >= 0.05 or db >= 0.05*bscale + 1e-12
    print(f"{label:34s} {'OK   ' if ok else 'BAD  ' if gross else 'CHECK'}  L1/max: {'  '.join(out)}   means: mass x{mass:.5f} energy x{energy:.5f} dB {db:.1e}   "
          f"samples/cell {info.get('samples_per_cell_min', -1):.0f}-{info.get('samples_per_cell_max', -1):.0f}, missing {info.get('samples_missing', -1):.0f}"
          + (f"   suite has out-of-plane parts ({out_of_plane:.3g})" if out_of_plane else ""))
    return 1 if gross else 0


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
