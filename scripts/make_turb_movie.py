"""The movie of the driven-turbulence page: the density in the midplane at every Mach number, side by side.

  .venv-movie/bin/python scripts/make_turb_movie.py [turb_movie [OUTPUT.mp4]]

Reads the movie runs fetched from $SCRATCH/mhd-results-turb-movie (sherlock_turb/movie.sh): per run rho_####.bin, the
midplane density 20 times per driving time. One panel per Mach number, each with its own colour limits (fixed over the
run), in the map of the Turbulence family. Writes progress/media/tests/driven_turbulence-density.mp4.
"""
import os
import sys
import tomllib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_movie_style, family_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"progress"/"media"/"tests"/"driven_turbulence-density.mp4"
FLOOR = 1.0e-6


def main(argv):
    import imageio_ffmpeg
    data = Path(argv[0]) if argv else ROOT/"turb_movie";out = Path(argv[1]) if len(argv) > 1 else OUT;runs = []
    for d in sorted(data.glob("gpu-hydro-M*-N*-s*")):
        meta = tomllib.loads((d/"run.toml").read_text());files = sorted(d.glob("rho_*.bin"))
        if meta.get("status") == "complete" and files:runs.append((meta["mach_target"], meta, files))
    if not runs:sys.exit(f"no movie runs in {data}: fetch them first (sherlock_turb/README.md, section Movie)")
    runs.sort(key=lambda r: r[0]);count = min(len(f) for _, _, f in runs);per = runs[0][1]["movie_per_T"]
    load = lambda f, n: np.log10(np.maximum(np.fromfile(f, dtype="<f4").reshape((n, n), order="F").T, FLOOR))
    limits = []
    for mach, meta, files in runs:        # limits from a sample of the steady part of the run
        sample = np.concatenate([load(f, meta["N"]).ravel() for f in files[count//5::max(1, count//12)]])
        limits.append(np.percentile(sample, [0.3, 99.95]))
    apply_movie_style(11)
    # the figure is a multiple of 16 pixels each way, as the encoder wants
    fig, axes = plt.subplots(1, len(runs), figsize=(4.48*len(runs), 5.44), dpi=100, layout="constrained");axes = np.atleast_1d(axes);ims = []
    for ax, (mach, meta, files), (lo, hi) in zip(axes, runs, limits):
        ims.append(ax.imshow(load(files[0], meta["N"]), origin="lower", extent=(0, 1, 0, 1), cmap=family_map("Turbulence", "density"), vmin=lo, vmax=hi, interpolation="bilinear"))
        ax.set_title(f"Mach {mach}");ax.set_xticks([]);ax.set_yticks([])
        bar = fig.colorbar(ims[-1], ax=ax, orientation="horizontal", shrink=.86, pad=.02);bar.set_label(r"$\log_{10}\,\rho$")
    n = runs[0][1]["N"]
    stamp = lambda i: f"Driven turbulence, Tom Abel's Euler kernel, ${n}^3$: density in the midplane · time = {i/per:.2f} driving times"
    title = fig.suptitle(stamp(0));fig.canvas.draw();fig.set_layout_engine("none")      # the layout is fixed with the title in place
    w, h = fig.canvas.get_width_height();w16, h16 = w - w % 16, h - h % 16
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(out), (w16, h16), fps=20, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=16,
                                         output_params=["-crf", "18", "-movflags", "+faststart"])
    writer.send(None)
    try:
        for i in range(count):
            for im, (mach, meta, files) in zip(ims, runs):im.set_data(load(files[i], meta["N"]))
            title.set_text(stamp(i))
            fig.canvas.draw();rgb = np.asarray(fig.canvas.buffer_rgba())[:h16, :w16, :3].copy()
            for _ in range(20 if i in (0, count - 1) else 1):writer.send(rgb)
    finally:
        writer.close();plt.close(fig)
    print(f"wrote {out}: {count} frames, {len(runs)} Mach numbers ({', '.join(str(m) for m, _, _ in runs)})")


if __name__ == "__main__":
    main(sys.argv[1:])
