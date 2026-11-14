"""Movie of the driven nest from movie.jl's mid-plane slices.
    python3 sherlock_nested/plot_movie.py nested_results/movie/nest64 [fps]
Writes <case dir>/<case>_movie.mp4: whole box (every level pasted in at the finest resolution, nest outlined) and the
finest level alone, ln ρ and log10 |B|. One figure, updated in place and piped to ffmpeg: light on the Mac."""
import os
import sys
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import plot_style as ps

D = sys.argv[1]; FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
# movie.jl writes movie.toml (every level n cells a side); bigrun.jl writes run.toml (root cells, then n per refined level)
big = not os.path.isfile(os.path.join(D, "movie.toml"))
meta = tomllib.load(open(os.path.join(D, "run.toml" if big else "movie.toml"), "rb"))
L = meta["levels"]; root = meta["root"] if big else meta["n"]
ns = [root] + [meta["n"]] * (L - 1)                    # cells a side of each level
F = root * 2 ** (L - 1)                                # finest cells across the box
off = np.cumsum([0] + [3 * m * m * 4 for m in ns])
per = 3 * 8 + int(off[-1])
size = os.path.getsize(os.path.join(D, "slices.bin")); nf = size // per
raw = np.memmap(os.path.join(D, "slices.bin"), np.uint8, "r")


def frame(k):
    b = raw[k * per:(k + 1) * per]
    t, mm, mv = np.frombuffer(b[:24], np.float64)
    a = [np.frombuffer(b[24 + off[l]:24 + off[l + 1]], np.float32).reshape(3, ns[l], ns[l]) for l in range(L)]   # [field, y, x]
    return t, mm, a


def composite(a, q):
    img = np.empty((F, F), np.float32)
    for l in range(L):
        r = 2 ** (L - 1 - l); s = ns[l] * r; o = (F - s) // 2
        img[o:o + s, o:o + s] = np.kron(a[l][q], np.ones((r, r), np.float32))
    return img


ps.apply_movie_style(11)
LIM = dict(rho=(-6, 3.5), b=(-2.2, 0.8))
fig, axs = plt.subplots(2, 2, figsize=(10, 9.2), layout="constrained")
t, mm, a = frame(0)
wf = ns[-1] / F                                       # width of the finest level in box units
lo, hi = 0.5 - wf / 2, 0.5 + wf / 2
ims = []
for row, (q, cmap, lim, lab) in enumerate(((0, ps.field_map("density"), LIM["rho"], r"$\ln\rho$"),
                                           (2, ps.field_map("magnetic"), LIM["b"], r"$\log_{10}|B|$"))):
    for col, (img, ext) in enumerate(((composite(a, q), (0, 1, 0, 1)), (a[L - 1][q], (lo, hi, lo, hi)))):
        ax = axs[row, col]; ps.frame(ax); ax.set_xticks([]); ax.set_yticks([])
        ims.append((ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest"), q, col))
    for l in range(1, L):
        h = ns[l] / (root * 2 ** l) / 2
        axs[row, 0].add_patch(Rectangle((0.5 - h, 0.5 - h), 2 * h, 2 * h, fill=False, ec="white", lw=.8, ls=(0, (4, 3))))
    fig.colorbar(ims[-1][0], ax=axs[row], shrink=.85, label=lab)
axs[0, 0].set_title(f"whole box: level 1 = {root}$^3$, levels 2–{L} pasted in")
axs[0, 1].set_title(f"level {L} alone ({F}$^3$-equivalent)")
NAME = f"{root}$^3$ + {L - 1} $\\times$ {ns[-1]}$^3$" if big else f"{L} $\\times$ {root}$^3$"
title = fig.suptitle("")
out = os.path.join(D, f"{meta['case']}_movie.mp4")
w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "24"])
matplotlib.rcParams["animation.ffmpeg_path"] = ps.ffmpeg_path()
with w.saving(fig, out, dpi=100):
    for k in range(nf):
        t, mm, a = frame(k)
        for im, q, col in ims:
            im.set_data(composite(a, q) if col == 0 else a[L - 1][q])
        title.set_text(f"Nest {NAME}, driven to Mach 10 (now {mm:.1f}), mid-plane $z = 1/2$,  $t = {t:.3f}$")
        w.grab_frame()
print(f"{nf} frames, t = 0–{t:.3f}, {nf / FPS:.0f} s at {FPS} fps -> {out}")
