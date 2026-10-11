"""Side-by-side movies (GLM | CT prototype | truth) from the dense-output movie runs.

  .venv-movie/bin/python scripts/render_suite_movies.py scripts/out/sherlock_downloads/extracted_movies

Reads extracted sherlock_movies archives: configs movie_<test>__glm and movie_<test>__ct (241 outputs;
2D at 384^2, 3D at 256^3 saving the z midplane) plus the truth panel, which is one source for every
frame (never mixed, so it cannot blink): an AthenaK high-resolution reference block-averaged to the
movie grid, else the exact solution, else an AthenaK run on the same grid (3D). Writes
progress/media/movies/<test>-<field>.mp4 and progress/movies.json; build_progress_site.py shows each
movie on the matching test of the GPU suite page. No simulations are run.

The same script makes previews from ordinary returns: --grid 128 --prefix "" --tests kh_uniform,...
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MHD_ANALYSIS_BUNDLE", "sherlock_movies")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import build_gpu_suite as suite

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "progress"
TITLES = {"density": "density ρ", "magnetic": "|B|", "vorticity": "vorticity ω_z", "current": "current density J_z"}
LABELS = {"glm": "GLM (advisor), tiled LLF", "ct": "CT prototype"}
GLM_CFL_LABEL = "GLM (advisor), tiled LLF · c_h of the CFL step at outputs"
SAME_GRID = {"athenak_best": "AthenaK PPM/HLLD/RK3", "athenak_matched": "AthenaK PLM/HLLE/RK2"}


def flux_function(u):
    """A_z with B_x = dA_z/dy, B_y = -dA_z/dx (periodic, zero mean): its contours are the in-plane field lines."""
    s = u[u.shape[0] // 2] if u.ndim == 4 else u
    n = s.shape[0];k = np.fft.fftfreq(n, d=1 / n);ky, kx = np.meshgrid(k, k, indexing="ij")
    bx, by = np.fft.fft2(s[..., 5]), np.fft.fft2(s[..., 6]);k2 = kx**2 + ky**2;k2[0, 0] = 1
    az = -1j * (ky * bx - kx * by) / (2 * np.pi * k2);az[0, 0] = 0
    return np.real(np.fft.ifft2(az))


def repeat1(slot):
    return next(((d, m) for d, m in slot["runs"] if m["repeat"] == 1), None) if slot else None


def frame_stamps(run):
    return {s["frame"]: s for s in run[1].get("frames", []) if "file" in s} if run else {}


def smooth(values, window):
    """Centred running mean that keeps the ends; used for adaptive colour limits so they cannot flicker."""
    v = np.asarray(values, float);half = window // 2
    return np.array([v[max(0, i - half):i + half + 1].mean() for i in range(len(v))])


def frame_limits(kind, planes, adaptive):
    arrays = [p for role in planes for p in planes[role] if p is not None]
    shared = suite.field_bounds(kind, arrays)
    count = len(next(iter(planes.values())))
    if not adaptive:return [shared] * count
    per = [suite.field_bounds(kind, [planes[r][i] for r in planes if planes[r][i] is not None]) for i in range(count)]
    window = max(3, count // 16)
    if kind == "log":
        lo, hi = smooth(np.log10([b[0] for b in per]), window), smooth(np.log10([b[1] for b in per]), window)
        return [[10**a, 10**b] for a, b in zip(lo, hi)]
    if kind == "symmetric":
        lim = smooth([b[1] for b in per], window);return [[-x, x] for x in lim]
    lo, hi = smooth([b[0] for b in per], window), smooth([b[1] for b in per], window)
    return [[a, b] for a, b in zip(lo, hi)]


def render(path, test, field, times, planes, labels, n, fps, adaptive, midplane, lines=None, clock=(1.0, "t")):
    import imageio_ffmpeg
    roles = list(planes)
    arrays = [p for role in roles for p in planes[role] if p is not None]
    # Vorticity sharpens by an order of magnitude as vortex sheets thin, so a whole-run limit washes out
    # the early roll-up; its (symmetric) limits always follow the run, smoothed so they do not flicker.
    adaptive = adaptive or field in ("vorticity", "current")
    kind = suite.scale_kind(field, arrays);limits = frame_limits(kind, planes, adaptive)
    norm = matplotlib.colors.LogNorm(*limits[0]) if kind == "log" else matplotlib.colors.Normalize(*limits[0])
    width = 6.4 * len(roles)
    fig, axes = plt.subplots(1, len(roles), figsize=(width, 7.2), dpi=100, layout="constrained")
    ims, notes = [], []
    for ax, role in zip(np.atleast_1d(axes), roles):
        first = next((p for p in planes[role] if p is not None), np.zeros((n, n)))
        ims.append(ax.imshow(first, origin="lower", extent=(0, 1, 0, 1), cmap=suite.FIELD_CMAPS[field], norm=norm,
                             interpolation="bilinear"))
        ax.set_title(labels[role], fontsize=13);ax.set_xticks([]);ax.set_yticks([])
        notes.append(ax.text(.5, .04, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=12, color="white",
                             bbox=dict(boxstyle="round,pad=.4", fc="#000000aa", ec="none")))
    bar = fig.colorbar(ims[0], ax=axes, shrink=.9, pad=.01)
    scale = " (log)" if kind == "log" else ""
    bar.set_label(TITLES[field] + scale + (" · limits follow the run" if adaptive else ""), fontsize=11);bar.ax.tick_params(labelsize=10)
    title = fig.suptitle("", fontsize=15)
    grid = (np.arange(n) + .5) / n;drawn = []   # field-line contours of the current frame
    where = " · midplane z = ½" if midplane else ""
    fig.canvas.draw()
    # Freeze the layout after the first draw: constrained layout re-solved on every frame makes panels jitter.
    fig.set_layout_engine("none")
    w, h = fig.canvas.get_width_height();w16, h16 = w - w % 16, h - h % 16
    writer = imageio_ffmpeg.write_frames(str(path), (w16, h16), fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                         macro_block_size=16, output_params=["-crf", "16", "-movflags", "+faststart"])
    writer.send(None)
    try:
        for i, t in enumerate(times):
            norm.vmin, norm.vmax = limits[i]
            for im, note, role in zip(ims, notes, roles):
                p = planes[role][i]
                if p is None:   # the run stopped before this output: hold its last frame, say so
                    last = next((j for j in range(i - 1, -1, -1) if planes[role][j] is not None), None)
                    if last is not None:
                        im.set_data(planes[role][last]);note.set_text(f"run stopped at t = {times[last]:.2f}");note.set_visible(True)
                    else:im.set_data(np.full((n, n), np.nan));note.set_visible(False)
                else:im.set_data(p);note.set_visible(False)
            for c in drawn:c.remove()
            drawn = []
            if lines:
                present = [lines[r][i] for r in roles if lines[r][i] is not None]
                if present:
                    lo, hi = min(a.min() for a in present), max(a.max() for a in present)
                    levels = np.linspace(lo, hi, 26)[1:-1]
                    for ax, role in zip(np.atleast_1d(axes), roles):
                        if lines[role][i] is not None:
                            # light lines: the map of the current is black where the current vanishes
                            drawn.append(ax.contour(grid, grid, lines[role][i], levels=levels, colors="#ffffff", linewidths=.6, alpha=.45))
            scale, clock_label = clock
            stamp = f"t = {t:.3f}" if scale == 1 else f"{'$\\Omega_i t$' if clock_label == 'Omega_i t' else clock_label} = {t * scale:.1f}"
            title.set_text(f"{test} · {TITLES[field]}{where} · {stamp}")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:h16, :w16, :3].copy()
            for _ in range(fps if i in (0, len(times) - 1) else 1):writer.send(rgb)   # hold first and last frame 1 s
    finally:
        writer.close();plt.close(fig)


def truth_source(group, refs, n, seed, glm_stamps):
    """(label, frame -> state loader) for the single truth source covering every movie frame, or (None, None)."""
    frames = sorted(glm_stamps)
    for test in sorted(group):   # AthenaK high-resolution reference, restricted to the movie grid
        ref = refs.get((test, seed, "athenak_reference"))
        if not ref:continue
        stamps = {s["frame"]: s for s in ref["meta"]["frames"] if str(n) in s.get("restricted", {})
                  and s["frame"] in glm_stamps and abs(s["time"] - glm_stamps[s["frame"]]["time"]) <= 1e-6 * max(1, s["time"])}
        missing = [f for f in frames if f not in stamps]
        # A dense reference may lack a stray output (AthenaK skipped t = 1.2375 once in 241); hold the
        # previous output for up to 2% of frames rather than dropping the truth panel.
        if stamps and len(missing) <= max(1, len(frames) // 50):
            meta = glm_stamps[frames[0]]["_meta"];dims = "²" if len(meta["saved_shape"]) == 3 else "³"
            label = f"AthenaK PPM/HLLD/RK3 {ref['meta']['N']}{dims} → {n}{dims}"
            held = {f: max((g for g in stamps if g <= f), default=min(stamps)) for f in frames}
            if missing:print(f"{test}: truth held over {len(missing)} missing output(s) {missing}")
            return label, lambda f, ref=ref, stamps=stamps, meta=meta, held=held: suite.returns.load_state(ref["directory"] / stamps[held[f]]["restricted"][str(n)], meta)
    if all("reference_file" in glm_stamps[f] for f in frames):
        return "exact solution", lambda f: suite.returns.load_state(glm_stamps[f]["_dir"] / glm_stamps[f]["reference_file"], glm_stamps[f]["_meta"])
    for method, name in SAME_GRID.items():   # AthenaK on the same grid (3D movies)
        for slots in group.values():
            run = repeat1(slots.get(method));stamps = frame_stamps(run)
            if run and all(f in stamps for f in frames):
                dims = "²" if run[1].get("planar", True) else "³"
                return f"{name} {n}{dims} (same grid)", lambda f, run=run, stamps=stamps: suite.returns.load_state(run[0] / stamps[f]["file"], run[1])
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("returns", type=Path, help="directory of extracted sherlock_movies archives")
    parser.add_argument("--grid", type=int, default=None, help="only this grid (default: every grid found)")
    parser.add_argument("--prefix", default="movie_", help="only tests whose name starts with this")
    parser.add_argument("--tests", default="", help="comma-separated movie names (after the prefix; default: all)")
    parser.add_argument("--out", type=Path, default=SITE / "media" / "movies")
    parser.add_argument("--manifest", type=Path, default=SITE / "movies.json")
    parser.add_argument("--fps", type=int, default=None, help="default: 24 for dense runs, slower for sparse previews")
    parser.add_argument("--min-frames", type=int, default=11, help="skip runs with fewer saved outputs (e.g. 3D snapshots)")
    parser.add_argument("--fields", default="", help="comma-separated fields to render (default: all of build_gpu_suite.FIELD_CMAPS)")
    args = parser.parse_args()
    results, refs = suite.discover(args.returns.resolve())
    groups = {}   # (base, n, seed) -> {test: slots}; movie_X__glm and movie_X__ct share base X
    for (test, n, seed, smoke), slots in results.items():
        if smoke or not test.startswith(args.prefix) or (args.grid and n != args.grid):continue
        base = test.removeprefix(args.prefix).split("__")[0]
        if args.tests and base not in args.tests.split(","):continue
        groups.setdefault((base, n, seed), {})[test] = slots
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    for (base, n, seed), group in sorted(groups.items()):
        runs = {m: repeat1(slots.get(m)) for slots in group.values() for m in ("glm", "ct") if slots.get(m)}
        stamps = {m: frame_stamps(r) for m, r in runs.items()}
        if "glm" not in stamps:print(f"{base}: no GLM run");continue
        frames = sorted(stamps["glm"])
        if len(frames) < args.min_frames:print(f"{base}: {len(frames)} saved outputs, skipped");continue
        glm_stamps = {f: {**stamps["glm"][f], "_dir": runs["glm"][0], "_meta": runs["glm"][1]} for f in frames}
        label, truth = truth_source(group, refs, n, seed, glm_stamps)
        roles = [m for m in ("glm", "ct") if m in runs] + (["truth"] if truth else [])
        labels = {**LABELS, "truth": label}
        if runs["glm"][1].get("parameters", {}).get("glm_ch_at_outputs") == "cfl":labels["glm"] = GLM_CFL_LABEL
        params = runs["glm"][1].get("parameters", {})
        adaptive = params.get("movie_scale") == "adaptive"
        midplane = len(runs["glm"][1]["saved_shape"]) == 3 and not runs["glm"][1].get("planar", True)
        planes = {r: {f: [] for f in [*suite.FIELD_CMAPS, "az"]} for r in roles};times = []
        for frame in frames:
            times.append(stamps["glm"][frame]["time"])
            for role in roles:
                if role != "truth" and frame not in stamps[role]:   # that run stopped early: rendered as held
                    for f in [*suite.FIELD_CMAPS, "az"]:planes[role][f].append(None)
                    continue
                u = truth(frame) if role == "truth" else suite.returns.load_state(runs[role][0] / stamps[role][frame]["file"], runs[role][1])
                ok = u is not None and np.isfinite(u).all()
                for f in suite.FIELD_CMAPS:planes[role][f].append(suite.field_of(u, f) if ok else None)
                planes[role]["az"].append(flux_function(u) if ok and not midplane else None)
        fps = args.fps or (24 if len(frames) >= 120 else max(6, round(len(frames) / 5)))
        entry = manifest[base] = {"fields": {}}
        for field in [f for f in suite.FIELD_CMAPS if not args.fields or f in args.fields.split(",")]:
            path = args.out / f"{base}-{field}.mp4"
            lines = {r: planes[r]["az"] for r in roles} if field == "current" and not midplane else None
            clock = (float(params.get("display_time_scale", 1.0)), params.get("display_time_label", "t"))
            render(path, base, field, times, {r: planes[r][field] for r in roles}, labels, n, fps, adaptive, midplane, lines, clock)
            entry["fields"][field] = os.path.relpath(path, SITE)
        entry.update(frames=len(frames), tEnd=times[-1], grid=n, truth=label or "no reference at every output",
                     panels=[labels[r] for r in roles], midplane=midplane, adaptive=adaptive, source=args.returns.name)
        print(f"movies {base}: {len(frames)} outputs at {n}, panels {', '.join(roles)}{', adaptive scale' if adaptive else ''}", flush=True)
    args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print("Manifest:", args.manifest)


if __name__ == "__main__":
    main()
