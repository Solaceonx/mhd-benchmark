"""Render dense driven-turbulence density slices fetched from Sherlock.

  .venv-movie/bin/python scripts/make_turb_movie.py [turb_movie [OUTPUT.mp4]]
  .venv-movie/bin/python scripts/make_turb_movie.py --mach 30 50

Writes a synchronized compilation and a separate 1440 x 1600 movie and poster
for each Mach number. Every evolving frame is a saved simulation snapshot;
only the first and last frames are held. No temporal interpolation is used.
"""
import argparse
import json
import os
import tomllib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_style import apply_movie_style, family_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "progress/media/tests/driven_turbulence-density.mp4"
FLOOR = 1.0e-6


def density(path, n):
    values = np.fromfile(path, dtype="<f4")
    if values.size != n*n or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{path}: expected {n*n} finite, positive density values")
    return np.log10(np.maximum(values.reshape((n, n), order="F").T, FLOOR))


def load_runs(data, machs):
    runs = []
    for directory in sorted(data.glob("gpu-hydro-M*-N256-s1")):
        meta_path = directory / "run.toml"
        if not meta_path.is_file():
            continue
        meta = tomllib.loads(meta_path.read_text())
        mach = int(meta["mach_target"])
        if machs and mach not in machs:
            continue
        if meta.get("status") != "complete":
            raise ValueError(f"{directory}: run is {meta.get('status')}, not complete")
        per = int(meta.get("movie_per_T", 0))
        if per <= 0:
            raise ValueError(f"{directory}: missing movie_per_T")
        count = round(float(meta["t_end_T"]) * per) + 1
        files = sorted(directory.glob("rho_*.bin"))
        expected = [directory / f"rho_{i:04d}.bin" for i in range(count)]
        if files != expected or meta.get("movie_frames") != count:
            raise ValueError(f"{directory}: need {count} consecutive frames and matching metadata; found {len(files)}")
        for path in files:
            density(path, int(meta["N"]))
        sample = np.concatenate([density(path, int(meta["N"])).ravel()
                                 for path in files[count//5::max(1, count//12)]])
        runs.append(dict(mach=mach, meta=meta, files=files,
                         limits=np.percentile(sample, [0.3, 99.95])))
    runs.sort(key=lambda run: run["mach"])
    if not runs:
        raise ValueError(f"No dense movie runs in {data}; fetch them first (sherlock_turb/README.md, Movie)")
    if machs and set(machs) != {run["mach"] for run in runs}:
        missing = sorted(set(machs)-{run["mach"] for run in runs})
        raise ValueError(f"Missing requested Mach numbers: {missing}")
    cadence = {(len(run["files"]), run["meta"]["movie_per_T"]) for run in runs}
    if len(cadence) != 1:
        raise ValueError("Runs must have the same snapshot count and cadence for synchronized playback")
    return runs


def render(runs, out, fps):
    import imageio_ffmpeg
    single = len(runs) == 1
    width, height = (1440, 1600) if single else (768*len(runs), 896)
    apply_movie_style(22 if single else 13)
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    images, clocks = [], []
    for index, run in enumerate(runs):
        left = (index + .045)/len(runs)
        panel_width = .91/len(runs)
        panel_height = panel_width*width/height
        ax = fig.add_axes([left, .13, panel_width, panel_height])
        lo, hi = run["limits"]
        images.append(ax.imshow(density(run["files"][0], run["meta"]["N"]),
                                origin="lower", cmap=family_map("Turbulence", "density"),
                                vmin=lo, vmax=hi, interpolation="bilinear"))
        ax.set_title(f"Mach {run['mach']} Turbulence", pad=16 if single else 10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        label_style = dict(color="white", transform=ax.transAxes, va="top",
                           bbox=dict(facecolor="#10222a", alpha=.72, edgecolor="none", pad=5))
        ax.text(.025, .975, r"Density $\rho$", **label_style)
        clocks.append(ax.text(.975, .975, "", ha="right", **label_style))
        cax = fig.add_axes([left+.12*panel_width, .065, .76*panel_width, .015])
        bar = fig.colorbar(images[-1], cax=cax, orientation="horizontal")
        bar.set_label(r"$\log_{10}(\rho/\rho_0)$", labelpad=5)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.stem+".partial.mp4")
    writer = imageio_ffmpeg.write_frames(str(temp), (width, height), fps=fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=16,
        output_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"])
    count = len(runs[0]["files"])
    per = runs[0]["meta"]["movie_per_T"]
    try:
        writer.send(None)
        for i in range(count):
            for image, clock, run in zip(images, clocks, runs):
                image.set_data(density(run["files"][i], run["meta"]["N"]))
                clock.set_text(rf"$t/T\simeq {i/per:.2f}$")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            for _ in range(fps if i in (0, count-1) else 1):
                writer.send(rgb)
            if i == count-1:
                fig.savefig(out.with_suffix(".png"), dpi=100)
        writer.close()
        temp.replace(out)
    except BaseException:
        writer.close()
        temp.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)
    description = dict(width=width, height=height, fps=fps, snapshots=count,
        encoded_frames=count+2*(fps-1), snapshots_per_T=per,
        nominal_end_T=(count-1)/per, time_label="nominal; output may be less than one solver step late",
        density_floor=FLOOR, temporal_interpolation=False,
        runs=[dict(source=run["files"][0].parent.name, mach=run["mach"], N=run["meta"]["N"],
                   seed=run["meta"]["seed"], solver=run["meta"]["solver"],
                   vendor_sha256=run["meta"].get("vendor_sha256"),
                   log10_density_limits=run["limits"].tolist()) for run in runs])
    out.with_suffix(".json").write_text(json.dumps(description, indent=2)+"\n")
    print(f"Wrote {out}: {count} snapshots, {width} × {height}, {fps} fps", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=ROOT/"turb_movie")
    parser.add_argument("out", nargs="?", type=Path, default=OUT)
    parser.add_argument("--mach", nargs="+", type=int)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)
    if args.fps < 1:
        parser.error("--fps must be positive")
    try:
        runs = load_runs(args.data, args.mach)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    for run in runs:
        render([run], args.out.parent/f"driven_M{run['mach']}-density.mp4", args.fps)
    if len(runs) > 1:
        render(runs, args.out, args.fps)


if __name__ == "__main__":
    main()
