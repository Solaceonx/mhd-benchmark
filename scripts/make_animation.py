import re
from pathlib import Path

from plot_style import apply_plot_style, field_map
apply_plot_style()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

here = Path(__file__).parent
framedir = here / "out" / "frames"
manifest = dict(
    line.strip().split("=") for line in (framedir / "manifest.txt").read_text().splitlines()
)
N = int(manifest["N"])
nframes = int(manifest["nframes"])
t_end = float(manifest["t_end"])

frame_files = sorted(framedir.glob("frame_*.bin"))
frames = [np.fromfile(f, dtype=np.float32).reshape(N, N).T for f in frame_files]
times = np.linspace(0, t_end, nframes)

vmin = min(f.min() for f in frames)
vmax = max(f.max() for f in frames)

fig, ax = plt.subplots(figsize=(6, 5.2))
im = ax.imshow(
    frames[0], origin="lower", cmap=field_map("density"), vmin=vmin, vmax=vmax, animated=True
)
cbar = fig.colorbar(im, ax=ax, label="density")
title = ax.set_title(f"Orszag-Tang density, t={times[0]:.3f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()


def update(i):
    im.set_data(frames[i])
    title.set_text(f"Orszag-Tang density, t={times[i]:.3f}")
    return im, title


anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
outpath = here / "out" / "ot_density.gif"
anim.save(outpath, writer=PillowWriter(fps=15))
print("Saved:", outpath)
