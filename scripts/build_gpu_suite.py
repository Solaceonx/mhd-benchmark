"""Multi-code comparison data for the progress site, read from downloaded Sherlock returns.

Never launches simulations. The test list comes from sherlock_mhd/configs; results come from
extracted archives of every code (GPU driver: GLM variants and CT; AthenaK; RAMSES) plus the
AthenaK reference and Athena++ cross-check runs. Missing runs are reported as pending.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tomllib

import matplotlib
from plot_style import apply_plot_style, field_map
apply_plot_style()
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "progress"
# Which bundle's configs describe the results. Sherlock and Marlowe differ for current_sheet and
# the three KH tests, so analyze each site's returns with its own bundle and never mix them:
#   MHD_ANALYSIS_BUNDLE=marlowe_mhd python scripts/build_progress_site.py --gpu-only --gpu-returns DIR
BUNDLE = ROOT / os.environ.get("MHD_ANALYSIS_BUNDLE", "sherlock_mhd")
DEFAULT_RETURNS = ROOT / "scripts/out/sherlock_downloads/extracted"
MAX_FRAMES = None  # Preserve every physical output written by the simulation.

# Hue encodes the code (validated with the dataviz palette checks, all pairs, light surface);
# line style and marker encode the variant within a code. References are neutral ink.
FAMILIES = {
    "glm": dict(label="GLM (Abel et al.)", color="#199e70", device="gpu"),
    "ct": dict(label="CT prototype", color="#eb6834", device="gpu"),
    "athenak": dict(label="AthenaK", color="#2a78d6", device="gpu"),
    "ramses": dict(label="RAMSES", color="#4a3aa7", device="cpu"),
}
REFERENCE_INK = "#52514e"
METHODS = {   # method -> (family, label, linestyle, marker); order is display order
    "glm_hll16": ("glm", "GLM f16/HLL (production)", "-", "o"),
    "glm_hlld": ("glm", "GLM HLLD", "--", "s"),
    "glm_ppm": ("glm", "GLM PPM", "-.", "D"),
    "glm_constch": ("glm", "GLM constant c_h", (0, (6, 2, 1, 2, 1, 2)), "v"),
    "glm": ("glm", "GLM tiled LLF (original path)", (0, (1, 1)), "x"),
    "ct": ("ct", "CT prototype", "-", "o"),
    "athenak_best": ("athenak", "AthenaK PPM/HLLD/RK3", "-", "o"),
    "athenak_matched": ("athenak", "AthenaK PLM/HLLE/RK2", "--", "s"),
    "ramses_best": ("ramses", "RAMSES HLLD", "-", "o"),
    "ramses_matched": ("ramses", "RAMSES HLL", "--", "s"),
}
REFERENCE_METHODS = ("athenak_reference", "athenapp_reference")

_spec = importlib.util.spec_from_file_location("analyze_returns", BUNDLE / "tools/analyze_returns.py")
returns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(returns)
_plan_spec = importlib.util.spec_from_file_location("plan", BUNDLE / "codes/plan.py")
plan = importlib.util.module_from_spec(_plan_spec)
_plan_spec.loader.exec_module(plan)


def method_info(m):
    fam, label, ls, marker = METHODS.get(m, ("ct", m, "-", "."))
    return dict(family=fam, label=label, linestyle=ls, marker=marker, color=FAMILIES[fam]["color"])


def catalog(overrides=()):
    """Tests from BUNDLE; an override bundle replaces (or adds) every test its configs define."""
    notes = json.loads((SITE / "gpu_tests.json").read_text())
    paths = {}
    for bundle in [BUNDLE] + [b for b, _ in overrides]:
        for path in sorted(p for p in (bundle / "configs").glob("*.toml") if not p.name.startswith((".", "movie_"))):
            paths[plan.load(path)["name"]] = (bundle, path)
    tests = []
    for name, (bundle, path) in paths.items():
        text = path.read_text();config = plan.load(path)
        comment = " ".join(line[1:].strip() for line in text.splitlines() if line.startswith("#"))
        problem = config["problem"]
        note = notes["tests"].get(name, {"family": "Other", "source": "", "divb": False, "focus": ""})
        gamma = config.get("gamma", {"brio_wu": "2", "brio_wu_rotated": "2", "rotor": "7/5", "blast": "7/5"}.get(problem, "5/3"))
        expected = list(config.get("methods", ["glm", "ct"]))   # run_suite.jl DEFAULT_METHODS
        for code, variants in plan.EXTERNAL_VARIANTS.items():
            if plan.tasks(config, code, "run"):expected += [f"{code}_{v}" for v in variants]
        tests.append(dict(name=name, problem=problem, family=note["family"], source=note["source"], divb=note["divb"],
            focus=note["focus"], adaptation=comment, gamma=str(gamma), tEnd=config["t_end"], frames=config["frames"],
            grids=config["grids"], seeds=config.get("seeds", [42]), planar=plan.planar(config),
            reference=config.get("exact_reference", "none"), referenceN=config.get("reference_n"),
            pressurePolicy=config.get("pressure_policy", "stop"), expectedMethods=expected, bundle=bundle.name,
            timeScale=float(config.get("display_time_scale", 1.0)), timeLabel=config.get("display_time_label", "t"),
            defaultField=note.get("defaultField", "density")))
    order = {f: i for i, f in enumerate(notes["families"])}
    tests.sort(key=lambda t: (order.get(t["family"], 99), t["name"]))
    return tests, notes["families"]


def site_link(path):
    path = Path(path).resolve()
    return os.path.relpath(path, SITE) if path.is_relative_to(ROOT) else None


def read_csv(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))


def number(value, default=np.nan):
    try:return float(value)
    except (TypeError, ValueError):return default


def discover(root):
    """Newest result per (test, N, seed, smoke, method) across all result groups, plus references."""
    results, refs = {}, {}
    if not root or not Path(root).exists():return results, refs
    for env_path in Path(root).rglob("environment.toml"):
        directory = env_path.parent;env = tomllib.loads(env_path.read_text())
        for run_path in sorted(directory.glob("*_r*/run.toml")):
            meta = tomllib.loads(run_path.read_text());m = meta["method"]
            if m in REFERENCE_METHODS:
                key = (meta["test"], int(meta.get("seed", 42)), m)
                if key not in refs or str(env.get("created_utc", "")) > str(refs[key]["env"].get("created_utc", "")):
                    refs[key] = dict(directory=run_path.parent, meta=meta, env=env)
                continue
            key = (meta["test"], int(meta["N"]), int(meta.get("seed", 42)), bool(env.get("smoke", False)))
            slot = results.setdefault(key, {}).setdefault(m, dict(env=env, directory=directory, runs=[]))
            if str(env.get("created_utc", "")) > str(slot["env"].get("created_utc", "")):
                slot.update(env=env, directory=directory, runs=[])
            if slot["directory"] == directory:slot["runs"].append((run_path.parent, meta))
        # An external-code job killed by Slurm (time limit, cancellation) leaves its current repeat
        # without run.toml. Record it as failed rather than letting the method silently disappear.
        if "code" in env and (directory / "input.toml").exists():
            test = tomllib.loads((directory / "input.toml").read_text())["name"]
            for rundir in sorted(directory.glob("*_r*")):
                if not rundir.is_dir() or (rundir / "run.toml").exists():continue
                method, _, rep = rundir.name.rpartition("_r")
                meta = dict(method=method, test=test, N=int(env["actual_N"]), seed=int(env.get("actual_seed", 42)),
                            repeat=int(rep), status="failed", error=killed_run_error(directory.parent, rundir))
                key = (test, meta["N"], meta["seed"], bool(env.get("smoke", False)))
                slot = results.setdefault(key, {}).setdefault(method, dict(env=env, directory=directory, runs=[]))
                if slot["directory"] == directory:slot["runs"].append((rundir, meta))
    return results, refs


def killed_run_error(job_dir, rundir):
    """Why a repeat has no run record: the job's NOTE.txt if present, plus how far the code's own log got."""
    note = (job_dir / "NOTE.txt").read_text().strip() if (job_dir / "NOTE.txt").exists() else ""
    stopped = "the job was killed before this repeat finished" + ("" if (job_dir / "exit_code.txt").exists() else " (no exit code: Slurm time limit or cancellation)")
    log = rundir / "athenak.log"
    if log.exists():
        import re
        rows = re.findall(r"cycle=(\d+) time=([\d.eE+-]+) dt=([\d.eE+-]+)", log.read_text(errors="replace"))
        if len(rows) >= 2:
            (c0, t0, dt0), (c1, t1, dt1) = rows[0], rows[-1]
            stopped += f"; AthenaK log: t={float(t1):.4g} after {int(c1):,} cycles, dt fell from {float(dt0):.2g} to {float(dt1):.2g}"
    return (note + " " if note else "") + stopped


# Fields drawn on the page. Vorticity shows shear-layer roll-up (e.g. Kelvin–Helmholtz), which density
# cannot when both streams have the same density.
# the notebook's colour maps (scripts/plot_style.py): the two signed fields sit side by side, so each has its own
FIELD_CMAPS = {"density": field_map("density"), "magnetic": field_map("magnetic"), "vorticity": field_map("signed"), "current": field_map("current")}
LOG_RATIO = 30.0   # draw a positive field on a log scale when max/min exceeds this


def field_of(u, field):
    if field in ("vorticity", "current"):   # z component of curl v or curl B on the midplane, periodic centred
        s = u[u.shape[0] // 2] if u.ndim == 4 else u
        n = s.shape[0]
        if field == "vorticity":ax, ay = s[..., 1] / s[..., 0], s[..., 2] / s[..., 0]
        else:ax, ay = s[..., 5], s[..., 6]
        return (np.roll(ay, -1, 1) - np.roll(ay, 1, 1)) * n / 2 - (np.roll(ax, -1, 0) - np.roll(ax, 1, 0)) * n / 2
    q = u[..., 0] if field == "density" else np.linalg.norm(u[..., 5:8], axis=-1)
    return returns.midplane(q)


def scale_kind(field, arrays):
    if field in ("vorticity", "current"):return "symmetric"
    lo = min((x.min() for x in arrays), default=0.);hi = max((x.max() for x in arrays), default=1.)
    return "log" if lo > 0 and hi / lo > LOG_RATIO else "linear"


def field_bounds(kind, arrays):
    if kind == "symmetric":   # clip the rarest 0.5% so isolated extremes do not wash out the structure
        lim = max((float(np.percentile(np.abs(x), 99.5)) for x in arrays), default=1.)
        lim = lim if lim > 0 else 1e-12
        return [-lim, lim]
    lo = min((float(x.min()) for x in arrays), default=0.);hi = max((float(x.max()) for x in arrays), default=1.)
    if kind == "linear" and hi - lo < 1e-12 * max(1, abs(hi)):lo, hi = lo - 1e-6, hi + 1e-6
    if kind == "log" and hi <= lo:hi = lo * 1.0001
    return [lo, hi]


def save_image(path, q, bounds, cmap, kind="linear"):
    lo, hi = bounds;q = np.flipud(q)
    x = (np.log10(np.maximum(q, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if kind == "log" else (q - lo) / (hi - lo)
    rgba = matplotlib.colormaps[cmap](np.clip(x, 0, 1))
    Image.fromarray((255 * rgba).astype("uint8")).save(path, optimize=True)


def choose_frames(frames, referenced):
    """Keep all physical outputs by default; optional caps retain reference outputs."""
    if MAX_FRAMES is None or len(frames) <= MAX_FRAMES:return frames
    picks = {frames[round(i * (len(frames) - 1) / (MAX_FRAMES - 1))] for i in range(MAX_FRAMES)}
    if len(referenced) < len(frames):picks |= set(referenced)
    return sorted(picks)


def restricted_ref(refs, test, seed, method, n, frame, time):
    ref = refs.get((test, seed, method))
    if not ref:return None, None
    stamp = next((f for f in ref["meta"]["frames"] if f["frame"] == frame and str(n) in f.get("restricted", {})), None)
    if not stamp or abs(stamp["time"] - time) > 1e-6*max(1.0, time):return None, None
    return ref, stamp


def reference_state(test, n, seed, frame, stamps, refs):
    """(state, label): exact/driver reference, else AthenaK restricted reference, else bundled Athena."""
    s0 = stamps[0];time = s0["time"]
    for s in stamps:
        if s.get("reference_file"):
            return returns.load_state(s["_dir"]/s["reference_file"], s["_meta"]), s.get("reference", "exact reference")
    ref, stamp = restricted_ref(refs, test, seed, "athenak_reference", n, frame, time)
    if ref:
        dims = "²" if len(s0["_meta"]["saved_shape"]) == 3 else "³"
        return (returns.load_state(ref["directory"]/stamp["restricted"][str(n)], s0["_meta"]),
                f"AthenaK PPM/HLLD/RK3 {ref['meta']['N']}{dims}, block-averaged to {n}{dims}")
    return returns.reference(s0["_meta"], time, s0["_dir"], {}, BUNDLE/"references")


def reference_sensitivity(test, n, seed, frame, time, meta):
    """Relative L1 difference of the Athena++ cross-check from the AthenaK reference."""
    a, sa = restricted_ref(REFS, test, seed, "athenak_reference", n, frame, time)
    b, sb = restricted_ref(REFS, test, seed, "athenapp_reference", n, frame, time)
    if not (a and b):return None
    ua = returns.load_state(a["directory"]/sa["restricted"][str(n)], meta)
    ub = returns.load_state(b["directory"]/sb["restricted"][str(n)], meta)
    return {f: returns.error_norms(ub[..., returns.FIELDS[f]], ua[..., returns.FIELDS[f]])["relative_L1"] for f in ("density", "magnetic")}


def method_summary(slot, method):
    runs = slot["runs"]
    complete = [m for _, m in runs if m["status"] == "complete"]
    first_dir, first = min(runs, key=lambda r: r[1]["repeat"])
    status = "complete" if len(complete) == len(runs) else "failed" if any(m["status"] == "failed" for _, m in runs) else "incomplete"
    error = next((m.get("error", "") for _, m in runs if m["status"] != "complete"), "")
    diag = read_csv(first_dir / "diagnostics.csv")
    column = lambda k: [number(r.get(k)) for r in diag]
    finite_max = lambda v: float(np.nanmax(v)) if v and np.isfinite(v).any() else None
    finite_min = lambda v: float(np.nanmin(v)) if v and np.isfinite(v).any() else None
    # Timing medians exclude repeat 1 when later repeats exist (repeat 1 includes snapshot output).
    timed = [m for m in complete if m["repeat"] > 1] or complete
    solve = np.array([m["solve_seconds"] for m in timed], float)
    rate = np.array([m.get("million_cell_steps_per_second", np.nan) for m in timed], float)
    fam = method_info(method)["family"];env = slot["env"]
    return dict(status=status, repeatsComplete=len(complete), repeatsTotal=len(runs),
        error=error.strip().splitlines()[0][:300] if error else "",
        steps=first.get("steps"), timeReached=first.get("time_reached"),
        solver=first.get("solver", ", ".join(f"{k}={v}" for k, v in first.get("settings", {}).items())),
        ctModule=first.get("ct_module"), allocatedBytes=first.get("allocated_array_bytes"),
        firstNonpositivePressure=first.get("first_nonpositive_pressure_time"),
        solveMedian=float(np.median(solve)) if len(solve) else None,
        solveIQR=[float(np.quantile(solve, .25)), float(np.quantile(solve, .75))] if len(solve) else None,
        rateMedian=float(np.nanmedian(rate)) if len(rate) and np.isfinite(rate).any() else None,
        timingRepeats=len(timed), device=str(env.get("device", "unknown")),
        deviceKind=env.get("device_kind", "gpu"),
        costUnit="GPU-seconds" if env.get("device_kind", "gpu") == "gpu" else f"CPU wall-seconds ({env.get('launcher', 'ranks')})",
        precision="fp64" if any(w in str(first.get("precision", "")).lower() for w in ("fp64", "double")) else "fp32",
        geometry=first.get("geometry", "cube"),
        maxEnergyDrift=finite_max(column("energy_drift")), maxMassDrift=finite_max(column("mass_drift")),
        minPressure=finite_min(column("pressure_min")), maxNonpositiveCells=finite_max(column("nonpositive_pressure_cells")),
        maxBz=finite_max(column("bz_max")),
        maxDivergence=finite_max(column("div_face_max" if method == "ct" else "normalized_div_center_rms")),
        divergenceKind="face ∇·B max (CT constraint)" if method == "ct" else "cell-centred ∇·B rms / (N·B_rms)",
        links=[[label, link] for label, path in (("Run record", first_dir / "run.toml"), ("Diagnostics · CSV", first_dir / "diagnostics.csv"),
            ("Driver log", slot["directory"].parent / "driver.log")) if path.exists() and (link := site_link(path))])


def style(plot, method, x, y):
    i = method_info(method)
    plot(x, y, linestyle=i["linestyle"], marker=i["marker"], ms=4, lw=1.6, color=i["color"], label=i["label"])


def shared_legend(fig, axes, ncol):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:handles.append(h);labels.append(l)
    if handles:fig.legend(handles, labels, loc="outside lower center", ncol=min(ncol, len(labels)), fontsize=8, frameon=False)


def history_plot(path, slots, test, label, frames):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), layout="constrained")
    panels = [("magnetic", "Magnetic energy", False), ("error", "Density L1 error vs reference", True),
              ("divergence", "Divergence (GLM cell-centred rms/NB · CT face max)", True), ("bz_max", "max |Bz|", True),
              ("pressure_min", "Minimum unfloored pressure", False), ("psi_rms", "GLM ψ rms", True)]
    for ax, (key, title, log) in zip(axes.flat, panels):
        for method in [m for m in METHODS if m in slots]:
            run = next(((d, m) for d, m in slots[method]["runs"] if m["repeat"] == 1), None)
            if not run:continue
            if key == "error":
                pts = [(f["time"], f["errors"][method]["density"]["relative_L1"]) for f in frames if method in f["errors"]]
                pts = [(a, b) for a, b in pts if b and np.isfinite(b) and b > 0]
                if pts:style(ax.plot, method, *zip(*pts))
                continue
            if key == "psi_rms" and not method.startswith("glm"):continue
            if key == "divergence" and method.startswith(("athenak", "ramses")):continue   # not saved by external codes
            rows = read_csv(run[0] / "diagnostics.csv");t = [number(r["time"]) for r in rows]
            col = ("div_face_max" if method == "ct" else "normalized_div_center_rms") if key == "divergence" else key
            y = np.array([number(r.get(col)) for r in rows], float)
            if log and np.isfinite(y).any() and np.nanmax(np.abs(y)) == 0:continue
            if log:y = np.where(y > 0, y, np.nan)
            i = method_info(method)
            if np.isfinite(y).any():ax.plot(t, y, linestyle=i["linestyle"], color=i["color"], lw=1.5, label=i["label"])
        if log and ax.lines:ax.set_yscale("log")
        if not ax.lines:ax.text(.5, .5, "no reference at saved outputs" if key == "error" else "not measured", ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_title(title, fontsize=10);ax.set_xlabel("time", fontsize=9);ax.grid(alpha=.15);ax.tick_params(labelsize=8)
    shared_legend(fig, axes, 5)
    fig.suptitle(f"{test} · {label} · repeat 1 of every method", fontsize=12)
    fig.savefig(path, dpi=110);plt.close(fig)


def scaling_plot(path, test, rows):
    """rows: (N, method, final relative L1 errors or None, median solve s, throughput, measured on a GPU)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    a, b, c = axes
    for method in METHODS:
        mine = sorted((r for r in rows if r[1] == method), key=lambda r: r[0])
        if not mine:continue
        pts = [(n, e["density"]) for n, _, e, _, _, _ in mine if e and e.get("density")]
        if pts:style(a.loglog, method, *zip(*pts))
        pts = sorted((s, e["density"]) for n, _, e, s, _, gpu in mine if gpu and e and e.get("density") and s)
        if pts:style(b.loglog, method, *zip(*pts))
        pts = [(n, r) for n, _, _, _, r, _ in mine if r]
        if pts:style(c.semilogx, method, *zip(*pts))
    errs = [r[2]["density"] for r in rows if r[2] and r[2].get("density")]
    if errs:
        ns = sorted({r[0] for r in rows if r[2]});top = max(errs)
        for order, ls in ((1, (0, (4, 3))), (2, (0, (1, 2)))):
            a.loglog(ns, [top*(ns[0]/n)**order for n in ns], linestyle=ls, color=REFERENCE_INK, lw=1, label=f"N^-{order} guide")
    for ax, text in ((a, "needs a reference at the final output"), (b, "needs GPU timing repeats and a reference"), (c, "needs completed timing repeats")):
        if not ax.lines:ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.grid(alpha=.15, which="both");ax.tick_params(labelsize=8)
    grids = sorted({r[0] for r in rows})
    for ax in (a, c):
        if ax.lines:
            ax.set_xticks(grids);ax.set_xticklabels([str(n) for n in grids]);ax.minorticks_off()
            ax.set_xlim(grids[0]/1.25, grids[-1]*1.25)
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set(title="Accuracy vs grid", xlabel="N", ylabel="final density relative L1")
    b.set(title="Accuracy vs cost · GPU codes (lower-left is better)", xlabel="median solve time [GPU-seconds]", ylabel="final density relative L1")
    c.set(title="Throughput", xlabel="N", ylabel="million cell-updates / s")
    shared_legend(fig, axes, 6)
    fig.suptitle(f"{test} · only GPU-measured runs enter the cost panel; CPU seconds are not comparable", fontsize=11)
    fig.savefig(path, dpi=110);plt.close(fig)


LEAKAGE_CAPTION = ("Field in the upper shear layer, which starts field-free: max and rms |B| over the rows within 0.1 of "
    "y = 3/4, from repeat 1 of each method. The dotted line is the initial value there (the tail of the lower layer's "
    "sech² band). Growth before the layers interact is non-local ∇·B-error leakage (Tomida et al. 2026 §III.2).")
DT_JUMP_CAPTION = ("Response to the timestep drop (dt × 1/40 for 100 steps, shaded), logged every step from the drop "
    "to 300 steps after it. GLM variants that tie c_h to 1/dt raise c_h 40-fold during the drop; constant-c_h GLM "
    "and CT should not react. ∇·B is the GLM cell-centred rms/(N·B_rms); CT's face divergence is at round-off.")


RECON_CAPTION = ("GEM reconnected flux: half the integral of |B_y| along each sheet (y = 1/4 and 3/4), in B0 d_i, "
    "against Omega_i t; it starts at 2 psi0 = 0.2. Solid lines: lower sheet; faint: upper sheet. Without a Hall term or "
    "explicit resistivity the rate is set by each code's numerical resistivity, so it differs between methods and grids; the "
    "AthenaK 2048² reference (lambda = 40 cells) is the least dissipative. Birn et al. (2001) found resistive MHD slow.")


def recon_flux(u):
    """(lower, upper) reconnected flux of a planar state in box units, as Problems.diagnostics (Julia)."""
    s = u[u.shape[0] // 2] if u.ndim == 4 else u
    n = s.shape[0];by = s[..., 6]
    row = lambda j: float(np.abs((by[j - 1] + by[j]) / 2).sum() / (2 * n))   # sheet on the face between rows j-1, j
    return row(n // 4), row(3 * n // 4)


def recon_flux_plot(path, slots, test, label, n, seed, t):
    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained");ts, fs = t["timeScale"], t["timeScale"]
    for method in [m for m in METHODS if m in slots]:
        run = next(((d, m) for d, m in slots[method]["runs"] if m["repeat"] == 1), None)
        pts = [(f["time"], recon_flux(returns.load_state(run[0] / f["file"], run[1]))) for f in (run[1].get("frames", []) if run else []) if "file" in f]
        if not pts:continue
        i = method_info(method);tt = [a * ts for a, _ in pts]
        ax.plot(tt, [b[0] * fs for _, b in pts], linestyle=i["linestyle"], color=i["color"], lw=1.8, label=i["label"])
        ax.plot(tt, [b[1] * fs for _, b in pts], linestyle=i["linestyle"], color=i["color"], lw=1, alpha=.35)
    ref = REFS.get((test, seed, "athenak_reference"))
    if ref:
        stamps = [f for f in ref["meta"]["frames"] if str(n) in f.get("restricted", {})]
        shape_meta = {"saved_shape": [n, n, 9]}
        pts = [(f["time"], recon_flux(returns.load_state(ref["directory"] / f["restricted"][str(n)], shape_meta))) for f in stamps]
        if pts:
            ax.plot([a * ts for a, _ in pts], [b[0] * fs for _, b in pts], color=REFERENCE_INK, lw=2.2, linestyle=(0, (1, 1.5)),
                    label=f"AthenaK PPM/HLLD/RK3 {ref['meta']['N']}² → {n}²")
    if not ax.lines:plt.close(fig);return False
    ax.set(xlabel=r"$\Omega_i t$" if t["timeLabel"] == "Omega_i t" else t["timeLabel"], ylabel=r"reconnected flux [$B_0\,d_i$]",
           title=f"{test} · {label} · reconnected flux")
    ax.grid(alpha=.15);ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=110);plt.close(fig);return True


def repeat1_csv(slot, name):
    run = next((d for d, m in slot["runs"] if m["repeat"] == 1), None)
    return read_csv(run / name) if run else []


def leakage_plot(path, slots, test, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained");drew = False;initial = None
    for method in [m for m in METHODS if m in slots]:
        rows = repeat1_csv(slots[method], "diagnostics.csv")
        if not rows or "leak_band_bmax" not in rows[0]:continue
        t = [number(r["time"]) for r in rows]
        for ax, col in zip(axes, ("leak_band_bmax", "leak_band_brms")):
            y = np.array([number(r[col]) for r in rows], float);y = np.where(y > 0, y, np.nan)
            if np.isfinite(y).any():
                i = method_info(method);ax.semilogy(t, y, linestyle=i["linestyle"], color=i["color"], lw=1.6, label=i["label"]);drew = True
        initial = initial or number(rows[0]["leak_band_bmax"])
    if not drew:plt.close(fig);return False
    if initial and np.isfinite(initial):axes[0].axhline(initial, color=REFERENCE_INK, lw=1, linestyle=(0, (1, 2)), label="initial value")
    for ax, title in zip(axes, ("max |B| in the upper layer", "rms |B| in the upper layer")):
        ax.set_title(title, fontsize=10);ax.set_xlabel("time", fontsize=9);ax.grid(alpha=.15, which="both");ax.tick_params(labelsize=8)
    shared_legend(fig, axes, 6);fig.suptitle(f"{test} · {label} · leakage into the field-free layer", fontsize=12)
    fig.savefig(path, dpi=110);plt.close(fig);return True


def dt_jump_plot(path, slots, test, label):
    panels = [("dt", "timestep dt", True), ("psi_max", "GLM max |ψ|", True),
              ("normalized_div_center_rms", "GLM ∇·B rms / (N·B_rms)", True), ("leak_band_bmax", "max |B| in the upper layer", True)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained");drew = False
    for method in [m for m in METHODS if m in slots]:
        rows = repeat1_csv(slots[method], "dense_diagnostics.csv")
        if not rows:continue
        s0 = number(rows[0]["step"]);x = [number(r["step"]) - s0 for r in rows]
        jump = [xi for xi, r in zip(x, rows) if r["in_jump"] == "true"]
        i = method_info(method)
        for ax, (col, _, log) in zip(axes.flat, panels):
            if col in ("psi_max", "normalized_div_center_rms") and not method.startswith("glm"):continue
            y = np.array([number(r.get(col)) for r in rows], float)
            if log:y = np.where(y > 0, y, np.nan)
            if np.isfinite(y).any():ax.plot(x, y, linestyle=i["linestyle"], color=i["color"], lw=1.6, label=i["label"]);drew = True
        if jump and not getattr(fig, "_shaded", False):
            for ax in axes.flat:ax.axvspan(min(jump), max(jump), color=REFERENCE_INK, alpha=.08, lw=0)
            fig._shaded = True
    if not drew:plt.close(fig);return False
    for ax, (_, title, log) in zip(axes.flat, panels):
        if log and ax.lines:ax.set_yscale("log")
        ax.set_title(title, fontsize=10);ax.set_xlabel("steps since the timestep drop", fontsize=9);ax.grid(alpha=.15, which="both");ax.tick_params(labelsize=8)
    shared_legend(fig, axes, 5);fig.suptitle(f"{test} · {label} · response to a sudden timestep drop", fontsize=12)
    fig.savefig(path, dpi=110);plt.close(fig);return True


def movies_manifest(test_names):
    """Side-by-side movies from scripts/render_suite_movies.py (dense movie runs, or previews from the runs)."""
    movies = json.loads((SITE / "movies.json").read_text()) if (SITE / "movies.json").exists() else {}
    return {k: v for k, v in movies.items() if k in test_names and all((SITE / f).exists() for f in v["fields"].values())}


REFS = {}


def build(returns_root=DEFAULT_RETURNS, media=SITE / "media", overrides=()):
    """overrides: (bundle_dir, returns_dir) pairs. Results for a test come only from the returns of
    the bundle that owns its config, so a corrected rerun never mixes with the original setup."""
    global REFS
    tests, families = catalog(overrides);by_name = {t["name"]: t for t in tests}
    out = media / "gpu"
    if out.exists():shutil.rmtree(out)
    out.mkdir(parents=True)
    results, REFS = {}, {}
    for bundle, root in [(BUNDLE, returns_root)] + list(overrides):
        found, refs = discover(root)
        results.update({k: v for k, v in found.items() if by_name.get(k[0], {}).get("bundle") == bundle.name})
        REFS.update({k: v for k, v in refs.items() if by_name.get(k[0], {}).get("bundle") == bundle.name})
    runs, matrix = {}, {t["name"]: {} for t in tests}
    for (test, n, seed, smoke), slots in sorted(results.items()):
        if test not in matrix:continue
        t = by_name[test];methods = [m for m in METHODS if m in slots] + sorted(m for m in slots if m not in METHODS)
        label = f"{n}³ smoke" if smoke else f"{n}³" + (f" · seed {seed}" if len(t["seeds"]) > 1 else "")
        key = f"{n}-{seed}-{'smoke' if smoke else 'full'}"
        folder = out / test / key;folder.mkdir(parents=True)
        stamps = {}   # frame -> repeat-1 stamps with snapshots, one per method
        for m in methods:
            run = next(((d, meta) for d, meta in slots[m]["runs"] if meta["repeat"] == 1), None)
            if not run:continue
            for s in run[1].get("frames", []):
                if "file" in s:stamps.setdefault(s["frame"], []).append({**s, "_dir": run[0], "_meta": run[1], "_method": m})
        meta0 = next(iter(stamps.values()))[0]["_meta"] if stamps else None
        refstate = {f: reference_state(test, n, seed, f, ss, REFS) for f, ss in stamps.items()}
        chosen = choose_frames(sorted(stamps), [f for f, (r, _) in refstate.items() if r is not None])
        states, frames = {m: [] for m in methods + ["truth"]}, []
        for frame in chosen:
            ss = stamps[frame];ref, reflabel = refstate[frame]
            frames.append(dict(frame=frame, time=ss[0]["time"], reference=reflabel if ref is not None else None, errors={},
                               sensitivity=reference_sensitivity(test, n, seed, frame, ss[0]["time"], meta0)))
            planes = lambda u: None if u is None or not np.isfinite(u).all() else {f: field_of(u, f) for f in FIELD_CMAPS}
            states["truth"].append(planes(ref))
            for m in methods:
                s = next((x for x in ss if x["_method"] == m), None)
                u = returns.load_state(s["_dir"]/s["file"], s["_meta"]) if s else None
                states[m].append(planes(u))
                if u is not None and ref is not None and np.isfinite(u).all():
                    frames[-1]["errors"][m] = {f: returns.error_norms(u[..., returns.FIELDS[f]], ref[..., returns.FIELDS[f]]) for f in ("density", "magnetic")}
                del u
        # Two colour scales per field: shared over every output (shows growth and decay) and per output
        # (each output's own range). Both are shared across methods and the reference.
        bounds, images, scales, bounds_out, images_out = {}, {}, {}, {}, {}
        for field, cmap in FIELD_CMAPS.items():
            arrays = [p[field] for role in states for p in states[role] if p is not None]
            kind = scales[field] = scale_kind(field, arrays)
            bounds[field] = field_bounds(kind, arrays)
            bounds_out[field] = [field_bounds(kind, [states[r][i][field] for r in states if states[r][i] is not None])
                                 for i in range(len(chosen))]
            images[field], images_out[field] = {}, {}
            for role, frames_u in states.items():
                images[field][role], images_out[field][role] = [], []
                for i, (frame, p) in enumerate(zip(chosen, frames_u)):
                    if p is None:images[field][role].append(None);images_out[field][role].append(None);continue
                    name = folder / f"{role}-{field}-{frame:03d}.png";save_image(name, p[field], bounds[field], cmap, kind)
                    images[field][role].append(os.path.relpath(name, SITE))
                    name = folder / f"{role}-{field}-{frame:03d}-o.png";save_image(name, p[field], bounds_out[field][i], cmap, kind)
                    images_out[field][role].append(os.path.relpath(name, SITE))
        history = folder / "history.png";history_plot(history, slots, test, label, frames)
        extras = []
        if t["problem"] == "kh_localized":
            path = folder / "leakage.png"
            if leakage_plot(path, slots, test, label):
                extras.append(dict(image=os.path.relpath(path, SITE), caption=LEAKAGE_CAPTION))
        if t["problem"] == "gem":
            path = folder / "reconnected-flux.png"
            if recon_flux_plot(path, slots, test, label, n, seed, t):
                extras.append(dict(image=os.path.relpath(path, SITE), caption=RECON_CAPTION))
        if any((d / "dense_diagnostics.csv").exists() for m in slots for d, meta in slots[m]["runs"] if meta["repeat"] == 1):
            path = folder / "dt-jump.png"
            if dt_jump_plot(path, slots, test, label):
                extras.append(dict(image=os.path.relpath(path, SITE), caption=DT_JUMP_CAPTION))
        summaries = {m: method_summary(slots[m], m) for m in methods}
        devices = {s["device"] for s in summaries.values()}
        runs.setdefault(test, {})[key] = dict(label=label, N=n, seed=seed, smoke=smoke, methods=methods,
            synthetic=any(any(w in d.upper() for w in ("FIXTURE", "SYNTHETIC", "TEST", "MAC")) for d in devices),
            frames=frames, bounds=bounds, images=images, boundsPerOutput=bounds_out, imagesPerOutput=images_out, scales=scales,
            summaries=summaries, history=os.path.relpath(history, SITE), extras=extras,
            planar=len(meta0["saved_shape"]) == 3 if meta0 else True)
        matrix[test][label] = {m: summaries[m]["status"] for m in methods}
        print(f"GPU suite: {test} {label} · {len(methods)} methods · {len(chosen)} frames", flush=True)
    scaling = {}
    for test, keys in runs.items():
        full = [r for r in keys.values() if not r["smoke"]]
        if not full:continue
        seed0 = min(r["seed"] for r in full);rows = []
        for r in full:
            if r["seed"] != seed0:continue
            last = r["frames"][-1] if r["frames"] else {"errors": {}}
            for m in r["methods"]:
                e = last["errors"].get(m);s = r["summaries"][m]
                rows.append((r["N"], m, {f: e[f]["relative_L1"] for f in e} if e else None, s.get("solveMedian"), s.get("rateMedian"),
                             s["deviceKind"] == "gpu"))
        path = out / test / "scaling.png";scaling_plot(path, test, rows);scaling[test] = os.path.relpath(path, SITE)
    legend = {m: dict(label=method_info(m)["label"], family=method_info(m)["family"], color=method_info(m)["color"]) for m in METHODS}
    movies = movies_manifest(by_name)
    return finite_json(dict(tests=tests, families=families, runs=runs, matrix=matrix, scaling=scaling, legend=legend,
        codeFamilies={k: dict(label=v["label"], color=v["color"], device=v["device"]) for k, v in FAMILIES.items()},
        references=sorted(f"{k[0]} · {k[2]} · N={v['meta']['N']}" for k, v in REFS.items()), movies=movies,
        source=site_link(returns_root) or str(returns_root), sourceExists=Path(returns_root).exists(), groupsFound=len(results)))


def finite_json(x):
    """JSON has no inf/NaN; show them as missing ('—'). A run that reached p = -inf is already marked failed."""
    if isinstance(x, float):return x if np.isfinite(x) else None
    if isinstance(x, dict):return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):return [finite_json(v) for v in x]
    return x
