"""Render saved Julia density planes and measured diagnostics into MP4 and GIF."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
import matplotlib
matplotlib.use("Agg")
from plot_style import INK, MUTED, RULE, RUN, apply_movie_style, ffmpeg_path, field_map, frame
apply_movie_style()
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framedir", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    root = args.framedir.resolve()
    meta = dict(line.split("=", 1) for line in (root / "manifest.txt").read_text().splitlines())
    n, nf = int(meta["N"]), int(meta["nframes"])
    diagnostics = np.genfromtxt(root / "diagnostics.csv", delimiter=",", names=True)
    assert len(diagnostics) == nf
    fields = {}
    for name in ("slice", "column"):
        paths = sorted(root.glob(f"{name}_*.bin"))
        assert len(paths) == nf
        # Julia writes x as the fastest index; imshow expects rows=y, columns=x.
        fields[name] = np.stack([np.fromfile(p, dtype="<f4").reshape(n, n) for p in paths])
        assert np.isfinite(fields[name]).all() and fields[name].min() > 0

    # The notebook's look (scripts/plot_style.py): white ground, its inks and fonts, its map for density.
    fig = plt.figure(figsize=(12, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], height_ratios=[1, 0.13],
                          left=.055, right=.96, bottom=.09, top=.775, wspace=.30, hspace=.18)
    ax_slice, ax_column = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    sub = gs[0, 2].subgridspec(2, 1, hspace=.48)
    ax_energy, ax_mach = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    fig.text(.055, .925, "Supersonic turbulence, left to decay", size=19)
    fig.text(.055, .875, f"3D hydrodynamics  /  {n}³ cells  /  periodic box  /  no driving  /  γ = {meta['gamma']}",
             color=MUTED, size=11)
    stamp = fig.text(.055, .826, "", color=INK, size=12)
    images = []
    for ax, key, title in ((ax_slice, "slice", "Density in the middle plane"),
                            (ax_column, "column", "Density averaged along the view")):
        data = fields[key]
        # Fixed, symmetric logarithmic limits throughout; no per-frame autoscaling.
        bound = max(abs(np.log10(data.min())), abs(np.log10(data.max())))
        bound = np.ceil(bound * 2) / 2
        im = ax.imshow(data[0], origin="lower", extent=(0, 1, 0, 1), cmap=field_map("density"),
                       norm=LogNorm(10**-bound, 10**bound), interpolation="nearest")
        ax.set(title=title, xlabel="x / L", ylabel="y / L");frame(ax)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.08, pad=.18)
        cb.set_label("density / initial density   ·   logarithmic scale", fontsize=9);cb.outline.set_linewidth(.6)
        images.append(im)
    tau = diagnostics["t_cross"]
    energy = diagnostics["kinetic_energy"] / diagnostics["kinetic_energy"][0]
    mach = diagnostics["mach"]
    lines, dots = [], []
    # one run, one colour: the green of Tom Abel's kernels on every page
    for ax, series, label, color in ((ax_energy, energy, "Kinetic energy / initial", RUN),
                                    (ax_mach, mach, "RMS sonic Mach number", RUN)):
        ax.set_xlim(0, tau[-1]); ax.set_ylim(0, max(series)*1.08)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xlabel("initial box crossing times", fontsize=9)
        ax.grid(True, color=RULE, lw=.6); ax.set_axisbelow(True)
        line, = ax.plot([], [], color=color, lw=2)
        dot, = ax.plot([], [], "o", color=color, ms=5)
        lines.append(line); dots.append(dot)
    ax_mach.axhline(1, color=MUTED, ls=":", lw=1)
    fig.text(.055, .07, "Light regions are compressed gas. Dark regions are rarefied gas.", size=10.5)
    fig.text(.055, .035, "Heat stays in the gas (near-isothermal). This small demonstration includes numerical dissipation.",
             color=MUTED, size=9.5)

    def update(i):
        for im, key in zip(images, ("slice", "column")):
            im.set_data(fields[key][i])
        for line, dot, series in zip(lines, dots, (energy, mach)):
            line.set_data(tau[:i+1], series[:i+1]); dot.set_data([tau[i]], [series[i]])
        stamp.set_text(f"Time  {tau[i]:.2f} crossing times     ·     Mach  {mach[i]:.2f}     ·     Kinetic energy remaining  {100*energy[i]:.1f}%")
        return *images, *lines, *dots, stamp

    # Extra final frames give viewers time to read the outcome.
    sequence = list(range(nf)) + [nf-1] * (2*args.fps)
    anim = FuncAnimation(fig, update, frames=sequence, interval=1000/args.fps, blit=False)
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path()
    mp4 = root / "decaying_turbulence.mp4"
    anim.save(mp4, writer=FFMpegWriter(fps=args.fps, codec="libx264",
              extra_args=["-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    # A compact animated preview for clients that do not play local MP4 files.
    anim.save(root / "decaying_turbulence.gif", writer=PillowWriter(fps=args.fps), dpi=70)
    update(nf//2)
    fig.savefig(root / "preview.png", dpi=130)
    plt.close(fig)
    print(f"Saved {mp4}")


if __name__ == "__main__":
    main()
