"""Spectra of the central-cube snapshots of a compare_cap run, computed where the snapshots are (the 512^3-equivalent long
windows are 3 GB each; the Mac keeps ~100 GB):   python3 cube_spectra.py RUN_DIR  -> RUN_DIR/cube_spectra.npz
As scripts/nested_figures.snapshot_spectra: for each snap_c_<t>.bin (Float32 (7, n, n, n): rho, v, B), the Hann-windowed,
mean-removed shell sums 1/2 sum|f^|^2 of v (3 components), B (3) and rho, in integer shells of the cube's own wave number.
Arrays: times (snapshots), E of shape (snapshots, 3 [v, B, rho], n//2 + 1)."""
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
files = sorted(run.glob("snap_c_*.bin"), key=lambda f: float(f.name[7:12]))


def shells(f):
    n = f.shape[-1]
    w1 = np.hanning(n + 2)[1:-1];w = w1[:, None, None]*w1[None, :, None]*w1[None, None, :]
    f = (f - f.mean(axis=(1, 2, 3), keepdims=True))*w
    F = np.fft.fftn(f, axes=(1, 2, 3))/n**3
    P = 0.5*(np.abs(F)**2).sum(0)/(w**2).mean()
    k = np.fft.fftfreq(n)*n
    kk = np.rint(np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + k[None, None, :]**2)).astype(int)
    return np.bincount(kk.ravel(), P.ravel(), minlength=n)[:n//2 + 1]


times, E = [], []
for f in files:
    a = np.fromfile(f, np.float32);n = round((a.size//7)**(1/3));a = a.reshape(7, n, n, n).astype(np.float64)
    E.append([shells(a[1:4]), shells(a[4:7]), shells(a[0:1])]);times.append(float(f.name[7:12]))
    print(f.name, n, flush=True)
np.savez(run/"cube_spectra.npz", times=np.array(times), E=np.array(E))
print("wrote", run/"cube_spectra.npz", len(times), "snapshots")
