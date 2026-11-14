"""Shell-summed power spectra of bigrun.jl's 3-D snapshots, computed where the snapshots are (they are too big to fetch).
    python3 spectra.py RUN_DIR          -> RUN_DIR/spectra.npz
Per snapshot snap_L<l>_<t>.bin (Float32 (n, n, n, 7) in Fortran order: ρ, v, B): ½ Σ|f̂|² in integer shells of the patch's own
wave number, for v (3 components), B (3) and ρ. Level 1 is the periodic box; a refined level is not periodic, so its mean
is removed and it is multiplied by a Hann window (power divided by the window's mean square), as scripts/nested_figures.py.
Arrays in the file: levels, times, n (cells per side of each level), frac (the level's width in box units, from run.toml),
and E_v, E_B, E_rho of shape (levels, times, n_max//2 + 1), NaN beyond a level's Nyquist."""
import re
import sys
import tomllib
from pathlib import Path

import numpy as np

run = Path(sys.argv[1]);meta = tomllib.loads((run/"run.toml").read_text())
L, root, nc = meta["levels"], meta["root"], meta["n"]
files = sorted(run.glob("snap_L*_*.bin"))
times = sorted({float(re.search(r"_(\d+\.\d+)\.bin", f.name).group(1)) for f in files})
ns = [root] + [nc]*(L - 1);half = max(ns)//2 + 1
out = {k: np.full((L, len(times), half), np.nan) for k in ("E_v", "E_B", "E_rho")}


def shells(f, window):
    n = f.shape[-1]
    if window:
        w1 = np.hanning(n + 2)[1:-1];w = w1[:, None, None]*w1[None, :, None]*w1[None, None, :]
        f = (f - f.mean(axis=(1, 2, 3), keepdims=True))*w
    F = np.fft.fftn(f, axes=(1, 2, 3))/n**3
    P = 0.5*(np.abs(F)**2).sum(0)/((w**2).mean() if window else 1)
    k = np.fft.fftfreq(n)*n
    kk = np.rint(np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + k[None, None, :]**2)).astype(int)
    return np.bincount(kk.ravel(), P.ravel(), minlength=n)[:n//2 + 1]


for l in range(L):
    n = ns[l]
    for it, t in enumerate(times):
        f = run/f"snap_L{l + 1}_{t:.3f}.bin"
        if not f.is_file():continue
        a = np.fromfile(f, np.float32).reshape(7, n, n, n).astype(np.float64)       # [field, z, y, x]
        for key, sl in (("E_v", slice(1, 4)), ("E_B", slice(4, 7)), ("E_rho", slice(0, 1))):
            out[key][l, it, :n//2 + 1] = shells(a[sl], l > 0)
    print(f"level {l + 1}: {n}^3, {len(times)} snapshots", flush=True)
frac = [1.0] + [nc/(root*2**l) for l in range(1, L)]
np.savez(run/"spectra.npz", levels=np.arange(1, L + 1), times=np.array(times), n=np.array(ns), frac=np.array(frac), **out)
print("wrote", run/"spectra.npz")
