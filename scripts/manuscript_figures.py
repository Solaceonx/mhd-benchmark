"""Figures for our own paper, drawn from the comprehensive sweep: drafts and, later, the final versions.

    python3 scripts/manuscript_figures.py [NAME ...] [--preview DIR] [--list]

Writes paper_figures/NAME.pdf, one file per figure and nothing else: that folder holds only figures meant for the
paper. --preview also writes PNG copies to DIR (for looking at them; not kept in paper_figures/).

Not to be confused with scripts/paper_figures.py, which redraws the figures of *other* papers for the progress site.
Both read sweep_results/ through scripts/sweep_data.py and share the colours and labels of scripts/paper_common.py.

Figures:
  cost_vs_error   error against GPU time on one H100: two smooth waves and the field loop (run.toml errors and timings)
  speed           cell-updates per second against grid size (throughput_ot3d) and time to solution of 3D Orszag-Tang
  robustness      every test, grid and code: finished, or how far the solver got
  convergence     L1 error against resolution on the three smooth waves
  leakage         Kelvin-Helmholtz with a confined field: maps, the field in the field-free vortex, the timestep change
  divergence      rotated shock tube (field along the normal) and field loop with v_z (out-of-plane field)
  field_loop      magnetic energy of the advected loop in 2D and 3D, with Gardiner & Stone's fits
  turbulence      Mach 9 decay at 512^3: energies against time and spectra at the end
"""
import logging
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as pc  # noqa: E402  (selects the Agg backend)
import sweep_data  # noqa: E402
import sweep_status  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from plot_style import apply_plot_style  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)      # its notes on the font files' dates, once per figure
OUT = sweep_data.ROOT/"paper_figures"
FULL, COLUMN = 7.1, 3.4                  # figure widths in inches: both columns, one column
INK, MUTED, RULE = pc.INK, pc.MUTED, pc.RULE
# The codes that ran on the H100, in the order of the legends.
ON_GPU = ("glm", "glm_hlld", "glm_hll16", "glm_ppm", "glm_constch", "ct", "athenak_matched", "athenak_best")


def style():
    apply_plot_style()
    matplotlib.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                                "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": .6, "lines.linewidth": 1.1,
                                "lines.markersize": 3.6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.minor.width": .4,
                                "ytick.minor.width": .4, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
                                "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                                "axes.edgecolor": INK, "pdf.fonttype": 42, "savefig.dpi": 300})


def ticks(ax):
    ax.tick_params(which="both", top=True, right=True)


# ---------------------------------------------------------------- records
def solve_seconds(sw, test, method, n, seed=42):
    """(seconds, approximate) of a finished run: the median solver time over its repeats with a valid timing. A run that
    was interrupted and resumed has no valid timing; the times of its parts, added up, are returned as approximate.
    (None, False) without a finished run."""
    reps = int(sw.configs[test].get("repeats_by_grid", {}).get(str(n), sw.configs[test].get("repeats", 1)))
    v = []
    for rep in range(1, reps + 1):
        m = sw.meta(test, method, n, seed, rep)
        if m and m.get("status") == "complete" and m.get("timing_valid") and m.get("solve_seconds"):v.append(float(m["solve_seconds"]))
    if v:return statistics.median(v), False
    m = sw.meta(test, method, n, seed) or {}
    if m.get("status") == "complete" and m.get("solve_seconds") and sw.went_bad(test, method, n, seed) is None:return float(m["solve_seconds"]), True
    return None, False


def final_error(sw, test, method, n, key):
    """The error a finished run recorded against the exact solution at its last output; None otherwise."""
    m = sw.meta(test, method, n)
    if not m or m.get("status") != "complete" or not m.get("frames"):return None
    e = m["frames"][-1].get("errors", {}).get(key)
    return float(e) if e is not None and np.isfinite(e) and e > 0 else None


def name(method):
    return "CT prototype" if method == "ct" else pc.label(method, True)


def legend_handles(methods):
    return [Line2D([], [], color=pc.color(m), linestyle=pc.STYLE[m][3], marker=pc.STYLE[m][4], label=name(m),
                   markerfacecolor="white" if pc.STYLE[m][4] not in ("x", "") else pc.color(m), markeredgewidth=.9) for m in methods]


# ---------------------------------------------------------------- cost against error
def cost_vs_error(sw):
    panels = (("entropy", "density_absolute_L1", "Entropy wave", r"$L_1$ error of $\rho$"),
              ("alfven_oblique", "magnetic_relative_L1", "Alfvén wave oblique to the grid", r"relative $L_1$ error of $\mathbf{B}$"),
              ("field_loop", "magnetic_relative_L1", "Field loop advection", r"relative $L_1$ error of $\mathbf{B}$"))
    # The constant-c_h kernel is the f16/HLL kernel with another cleaning speed; its points lie on that kernel's here.
    methods = [m for m in ON_GPU if m != "glm_constch"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.75), gridspec_kw=dict(wspace=.34, left=.075, right=.99, top=.9, bottom=.3))
    for ax, (test, key, title, ylabel) in zip(axes, panels):
        for m in methods:
            pts = [(*solve_seconds(sw, test, m, n), final_error(sw, test, m, n, key), n) for n in sw.grids(test, m)]
            pts = [p for p in pts if p[0] and p[2]]
            if not pts:continue
            kw = dict(color=pc.color(m), markeredgewidth=.9, markerfacecolor="white" if pc.STYLE[m][4] != "x" else pc.color(m),
                      zorder=3 if m.startswith("glm") else 2)
            ax.plot([p[0] for p in pts], [p[2] for p in pts], linestyle=pc.STYLE[m][3], marker="", **kw)
            for sec, approx, err, n in pts:                # a pale symbol: the time of a resumed run
                ax.plot([sec], [err], marker=pc.STYLE[m][4], linestyle="", alpha=.45 if approx else 1, **kw)
            if m == "ct":                                  # the top curve carries the grid sizes
                for sec, approx, err, n in pts:
                    ax.annotate(f"{n}", (sec, err), xytext=(0, 4.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_xscale("log");ax.set_yscale("log");ax.set_title(title, pad=4);ax.set_xlabel("GPU time (s)");ax.set_ylabel(ylabel, labelpad=2)
        lo, hi = ax.get_ylim();ax.set_ylim(lo, hi*2.2);ticks(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=8));ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs="auto", numticks=12))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=4, columnspacing=1.6, handlelength=2.6, bbox_to_anchor=(.53, 0))
    return fig


# ---------------------------------------------------------------- speed
F16 = ("glm_hll16", "glm_constch", "glm_ppm")     # kernels whose fastest tiles need a grid divisible by 6


def throughput(sw, method, n):
    """(median, lowest, highest) cell-updates per second over the repeats of the timing run; None without one."""
    reps = int(sw.configs["throughput_ot3d"].get("repeats_by_grid", {}).get(str(n), sw.configs["throughput_ot3d"].get("repeats", 1)))
    v = []
    for rep in range(1, reps + 1):
        m = sw.meta("throughput_ot3d", method, n, 42, rep)
        if m and m.get("status") == "complete" and m.get("timing_valid"):v.append(1e6*float(m["million_cell_steps_per_second"]))
    return (statistics.median(v), min(v), max(v)) if v else None


def clock(seconds):
    return f"{seconds/60:.0f} min" if seconds < 5400 else f"{seconds/3600:.1f} h"


def speed(sw):
    fig = plt.figure(figsize=(FULL, 3.1))
    left = fig.add_axes([.07, .14, .345, .78]);right = fig.add_axes([.735, .14, .255, .78])
    # (a) cell-updates per second against grid size, one H100. The f16 kernels are drawn twice: on the grids of the
    # tests (powers of two) and, with filled symbols, on grids divisible by 6.
    ends = {}
    for m in ON_GPU:
        pts = [(n, throughput(sw, m, n)) for n in sw.grids("throughput_ot3d", m)]
        pts = [(n, t) for n, t in pts if t]
        if not pts:continue
        sets = [([p for p in pts if p[0] % 6], False), ([p for p in pts if p[0] % 6 == 0], True)] if m in F16 else [(pts, False)]
        for part, filled in sets:
            if not part:continue
            n = np.array([p[0] for p in part], dtype=float);med, lo, hi = (np.array([p[1][i] for p in part]) for i in range(3))
            left.fill_between(n**3, lo, hi, color=pc.color(m), alpha=.18, lw=0)
            left.plot(n**3, med, color=pc.color(m), linestyle=pc.STYLE[m][3], marker=pc.STYLE[m][4], markeredgewidth=.9,
                      markerfacecolor=pc.color(m) if filled or pc.STYLE[m][4] == "x" else "white")
            if not filled:ends[m] = med[-1]
    left.set_xscale("log");left.set_yscale("log")
    grids = sw.grids("throughput_ot3d", "glm")
    left.set_xticks([g**3 for g in grids]);left.set_xticklabels([f"${g}^3$" for g in grids]);left.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    left.set_xlim(grids[0]**3/1.5, grids[-1]**3*1.5);left.set_ylim(3e7, 7e9)
    left.set_xlabel("grid");left.set_ylabel("cell-updates per second, one H100", labelpad=2);ticks(left);left.tick_params(axis="x", which="both", top=False)
    left.text(.035, .965, "filled symbols: $N$ divisible by 6", transform=left.transAxes, fontsize=7, color=MUTED, va="top")
    # labels at the right-hand ends, pushed apart where the curves meet
    groups = (("glm", "glm_hll16", "glm_constch"), ("glm_hlld",), ("ct",), ("glm_ppm",), ("athenak_matched",), ("athenak_best",))
    names = {("glm", "glm_hll16", "glm_constch"): "GLM LLF, f16/HLL, const. $c_h$"}
    last_y = None
    for g in sorted((g for g in groups if all(m in ends for m in g)), key=lambda g: -max(ends[m] for m in g)):
        y = float(np.mean([ends[m] for m in g]))
        if last_y is not None and y > last_y/1.19:y = last_y/1.19
        last_y = y
        left.annotate(names.get(g, name(g[0])), (1.0, y), xycoords=("axes fraction", "data"), xytext=(7, 0), textcoords="offset points",
                      va="center", ha="left", fontsize=7.5, annotation_clip=False)
        left.plot([1.012], [y], marker="s", ms=3.0, color=pc.color(g[0]), transform=left.get_yaxis_transform(), clip_on=False, zorder=5)
    left.set_title("(a) Speed", loc="left", pad=4)

    # (b) time to solution of the same problem
    test, n = "orszag_tang_3d", 512
    rows = [("One H100", ON_GPU), (f"{sweep_data.plan.estimate.cpu_ranks(n, 3)} CPU cores", ("ramses_matched", "ramses_best", "athenapp_matched", "athenapp_best"))]
    y = 0.0;yt, yl = [], []
    for head, methods in rows:
        right.text(-.02, y, head, transform=right.get_yaxis_transform(), fontsize=7.5, color=MUTED, va="center", ha="right", style="italic")
        y += 1.0
        for m in methods:
            meta = sw.meta(test, m, n) or {};state = sw.state_of(test, m, n);bad = sw.went_bad(test, m, n)
            yt.append(y);yl.append(name(m));right.axhline(y, color=RULE, lw=.5, zorder=0)
            sec, approx = solve_seconds(sw, test, m, n) if state == "done" and bad is None else (None, False)
            if sec:
                hollow = pc.STYLE[m][4] != "x"
                right.plot([sec/60], [y], marker=pc.STYLE[m][4], color=pc.color(m), ms=4.6, markeredgewidth=1.0, alpha=.45 if approx else 1,
                           markerfacecolor="white" if hollow else pc.color(m), zorder=3)
                right.annotate(("$\\approx$" if approx else "") + clock(sec), (sec/60, y), xytext=(6, 0), textcoords="offset points", va="center", ha="left", fontsize=7)
            else:
                why = (f"not finite after $t = {bad:.1f}$" if bad is not None else "being rerun" if state in ("redo", "open")
                       else f"stopped at $t = {float(meta.get('time_reached') or 0):.1f}$" if state == "stopped" else "no result")
                right.text(.04, y, why, transform=right.get_yaxis_transform(), fontsize=7, color=MUTED, va="center", ha="left")
            y += 1.0
        y += .5
    right.set_yticks(yt);right.set_yticklabels(yl);right.set_ylim(y - .9, -.7);right.set_xscale("log");right.set_xlim(10, 1500)
    right.set_xticks([10, 30, 60, 180, 600]);right.set_xticklabels(["10 min", "30 min", "1 h", "3 h", "10 h"])
    right.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());right.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):right.spines[side].set_visible(False)
    right.set_xlabel(f"time to $t = {float(sw.configs[test]['t_end']):g}$")
    right.text(-.62, 1.0, f"(b) 3D Orszag–Tang at ${n}^3$", transform=right.transAxes, va="bottom", ha="left", fontsize=8.5)
    return fig


# ---------------------------------------------------------------- robustness
# (group, ((test, label, indent), ...)); a test with several seeds gets one row per seed.
TESTS = (("Smooth waves", (("uniform", "Uniform state", 0), ("entropy", "Entropy wave", 0), ("alfven", "Alfvén wave", 0),
                           ("alfven_oblique", "Alfvén wave, oblique to the grid", 0))),
         ("Field advection", (("field_loop_novz", "Field loop", 0), ("field_loop", r"Field loop, $v_z \neq 0$", 0), ("field_loop_3d", "Field loop, 3D", 0))),
         ("Shocks", (("brio_wu", "Brio–Wu shock tube", 0), ("brio_wu_rotated", "Brio–Wu, rotated", 0), ("rotor", "Rotor", 0),
                     ("orszag_tang", "Orszag–Tang", 0), ("orszag_tang_3d", "Orszag–Tang, 3D", 0),
                     ("blast_2d_b0", r"Blast 2D, $B = 0$", 0), ("blast_2d_b50", r"Blast 2D, $B = 50$", 0), ("blast_2d_b100", r"Blast 2D, $B = 100$", 0),
                     ("blast_3d_b0", r"Blast 3D, $B = 0$", 0), ("blast_3d_b50", r"Blast 3D, $B = 50$", 0), ("blast_3d_b100", r"Blast 3D, $B = 100$", 0))),
         ("Shear and reconnection", (("kh_uniform", "Kelvin–Helmholtz, uniform field", 0), ("kh_localized", "Kelvin–Helmholtz, confined field", 0),
                                     ("kh_localized_dtjump", "with a sudden timestep change", 1), ("current_sheet", "Current sheet", 0),
                                     ("gem", "GEM reconnection", 0))),
         ("Decaying turbulence", (("decay", "Mach 2", 0), ("decay_supersonic", "Mach 9", 0))))
SLOTS = (32, 64, 128, 256, 512)
COLUMNS = (("glm", "GLM", (("glm", "LLF"), ("glm_hll16", "f16/\nHLL"), ("glm_hlld", "HLLD"), ("glm_ppm", "PPM"), ("glm_constch", "const.\n$c_h$"))),
           ("ct", "CT", (("ct", "proto-\ntype"),)),
           ("athenak", "AthenaK", (("athenak_matched", "PLM/\nHLLE"), ("athenak_best", "PPM/\nHLLD"))),
           ("ramses", "RAMSES", (("ramses_matched", "HLL"), ("ramses_best", "HLLD"))),
           ("athenapp", "Athena++", (("athenapp_matched", "PLM/\nHLLE"), ("athenapp_best", "PPM/\nHLLD"))),
           ("arepo", "AREPO", (("arepo_moving", "moving\nmesh"), ("arepo_static", "static\nmesh"))))      # planar tests only
MARK_W, MARK_H, PITCH = .82, 1.22, 1.62      # a mark and the distance between rows, in units of the distance between grids


def tint(colour, amount=.5):
    c = np.array(matplotlib.colors.to_rgb(colour));return tuple(1 - amount*(1 - c))


def mark(ax, x, y, state, frac, colour):
    """One run: finished (solid), failed (the solver stopped or its state stopped being finite; the filled part is the part
    of the run it reached), being rerun (hatched), taken out (cross)."""
    x0, y0 = x - MARK_W/2, y - MARK_H/2
    if state == "done":ax.add_patch(Rectangle((x0, y0), MARK_W, MARK_H, facecolor=tint(colour), edgecolor="none"))
    elif state == "stopped":
        ax.add_patch(Rectangle((x0, y0), MARK_W*min(1.0, max(0.0, frac or 0.0)), MARK_H, facecolor=INK, edgecolor="none"))
        ax.add_patch(Rectangle((x0, y0), MARK_W, MARK_H, facecolor="none", edgecolor=INK, lw=.45))
    elif state in ("redo", "open", "unknown"):ax.add_patch(Rectangle((x0, y0), MARK_W, MARK_H, facecolor="white", edgecolor=MUTED, lw=.3, hatch="//////"))
    elif state == "skipped":ax.plot([x], [y], marker="x", ms=2.8, color=MUTED, markeredgewidth=.6)


def robustness(sw):
    t_end = {name: float(c["t_end"]) for name, c in sw.configs.items()}
    sweep_status.mark_nonfinite(sw.status, sw.nonfinite_parts(), t_end)       # codes that ran on after their state turned to NaN
    # columns: five grids per method
    xs, x = {}, 0.0;spans = []
    for fam, title, variants in COLUMNS:
        start = x
        for m, _ in variants:xs[m] = x;x += len(SLOTS) + 1.1
        spans.append((fam, title, start - .5, x - 1.1 - .5));x += .9
    width = x - .9 - 1.1
    # rows: one per test and seed
    rows, y = [], 0.0
    for group, tests in TESTS:
        rows.append(("head", group, y));y += PITCH
        for test, label, indent in tests:
            seeds = sw.seeds(test)
            for s in seeds:
                rows.append(("run", (test, s, label if len(seeds) == 1 else f"{label}, seed {seeds.index(s) + 1}", indent), y));y += PITCH
        y += .55
    height = y - .55 - PITCH
    label_w = 24.5;top, bottom = 6.9, 6.0
    unit = FULL/(label_w + width + 1.0)
    fig = plt.figure(figsize=(FULL, unit*(height + top + bottom)));ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-label_w, width + 1.0);ax.set_ylim(height + bottom, -top);ax.axis("off")
    # column heads: the code, then its set-ups
    for fam, title, a, b in spans:
        ax.plot([a, b], [-4.35, -4.35], color=pc.FAMILY[fam], lw=1.6, solid_capstyle="butt")
        ax.text((a + b)/2, -4.7, title, ha="center", va="bottom", fontsize=8)
    for fam, title, variants in COLUMNS:
        for m, text in variants:ax.text(xs[m] + (len(SLOTS) - 1)/2, -1.25, text, ha="center", va="bottom", fontsize=6.8, linespacing=.95)
    done = {m: 0 for m in xs};planned = {m: 0 for m in xs}
    for kind, what, y in rows:
        if kind == "head":
            ax.text(-label_w + .3, y + .15, what, ha="left", va="center", fontsize=7.4, color=MUTED, style="italic");continue
        test, seed, label, indent = what
        ax.text(-label_w + 1.6 + 1.6*indent, y, label, ha="left", va="center", fontsize=7.4)
        for m in xs:
            for i, n in enumerate(SLOTS):
                row = sw.rows.get((test, m, n, seed))
                if row is None:continue
                planned[m] += 1;done[m] += row["state"] == "done"
                mark(ax, xs[m] + i, y, row["state"], row.get("frac"), pc.FAMILY[pc.STYLE[m][0]])
    yb = height + 2.0
    ax.plot([-label_w + .3, width], [yb - 1.1, yb - 1.1], color=INK, lw=.5)
    ax.text(-label_w + .3, yb, "Finished, of the runs planned", ha="left", va="center", fontsize=7.4)
    for m in xs:ax.text(xs[m] + (len(SLOTS) - 1)/2, yb, f"{done[m]}/{planned[m]}", ha="center", va="center", fontsize=7)
    # key
    yk = height + 4.4;x = -label_w + .7
    for i in range(len(SLOTS)):mark(ax, x + i, yk, "done", None, MUTED)
    ax.text(x + len(SLOTS) + .1, yk, r"$N = 32,\ 64,\ 128,\ 256,\ 512$", ha="left", va="center", fontsize=7)
    x += 26.5
    for state, frac, text, w in (("done", None, "finished", 9.5), ("stopped", .45, "solver failed; filled part: how far it got", 30.0),
                                 ("redo", None, "being rerun", 12.0), ("skipped", None, "taken out", 0.0)):
        mark(ax, x, yk, state, frac, MUTED);ax.text(x + 1.0, yk, text, ha="left", va="center", fontsize=7);x += w
    return fig, dict(done=done, planned=planned)


# ---------------------------------------------------------------- shared by the figures below
class Data:
    """What the functions of the Papers page (scripts/paper_*.py) expect of their `F` argument: the sweep, and its
    series and stop tests. Parts that are missing are not listed here; the panels simply leave them out."""
    series = pc.Figures.series;stopped = pc.Figures.stopped;why = pc.Figures.why

    def __init__(self, sw):
        self.sw = sw;self.notes = []

    def miss(self, *a, **k):
        pass


def marker_kw(m):
    return dict(marker=pc.STYLE[m][4], markeredgewidth=.9, markerfacecolor="white" if pc.STYLE[m][4] != "x" else pc.color(m))


def curve(ax, x, y, m, markers=True, **kw):
    style = dict(color=pc.color(m), linestyle=pc.STYLE[m][3]);style.update(marker_kw(m) if markers else {});style.update(kw)
    return ax.plot(x, y, **style)


def line_handles(methods, markers=True, extra=()):
    hs = [Line2D([], [], color=pc.color(m), linestyle=pc.STYLE[m][3], label=name(m), **(marker_kw(m) if markers else {})) for m in methods]
    return hs + list(extra)


def panel_letter(ax, text, **kw):
    ax.set_title(text, loc="left", pad=4, **kw)


# ---------------------------------------------------------------- convergence
def convergence(sw):
    tests = (("entropy", "density_absolute_L1", "Entropy wave", r"$L_1$ error of $\rho$"),
             ("alfven", "magnetic_relative_L1", "Alfvén wave along $x$", r"relative $L_1$ error of $\mathbf{B}$"),
             ("alfven_oblique", "magnetic_relative_L1", "Alfvén wave oblique to the grid", r"relative $L_1$ error of $\mathbf{B}$"))
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.25), sharex=True, gridspec_kw=dict(wspace=.36, hspace=.1, left=.075, right=.775, top=.94, bottom=.1))
    guide = dict(color=MUTED, lw=.6)
    for col, (test, key, title, ylabel) in enumerate(tests):
        for row, methods in enumerate((pc.GPU, pc.OTHER + pc.AREPO)):
            ax = axes[row, col];top = 0.0
            for m in methods:
                pts = [(n, final_error(sw, test, m, n, key)) for n in sw.grids(test, m)]
                pts = [(n, e) for n, e in pts if e]
                if not pts:continue
                ns, es = map(np.array, zip(*pts));curve(ax, ns, es, m)
                if es[0] > 1e-5:top = max(top, es[0])          # the guides start above the curves that converge
            ax.set_xscale("log");ax.set_yscale("log");ticks(ax)
            ax.set_xticks([32, 64, 128, 256, 512]);ax.set_xticklabels(["32", "64", "128", "256", "512"]);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if top:                                                    # slopes -1 and -2 from above the first point
                ns = np.array([32.0, 512.0])
                ax.plot(ns, 2.2*top*(32/ns), ls=(0, (1, 2)), **guide);ax.plot(ns, 2.2*top*(32/ns)**2, ls=(0, (4, 2)), **guide)
            if row == 0:ax.set_title(title, pad=4)
            else:ax.set_xlabel("$N$")
            ax.set_ylabel(ylabel, labelpad=2)
    slopes = [Line2D([], [], ls=(0, (1, 2)), label=r"$\propto N^{-1}$", **guide), Line2D([], [], ls=(0, (4, 2)), label=r"$\propto N^{-2}$", **guide)]
    for row, methods in enumerate((pc.GPU, pc.OTHER + pc.AREPO)):
        box = axes[row, -1].get_position()
        fig.legend(handles=line_handles(methods, extra=slopes), loc="center left", bbox_to_anchor=(.785, (box.y0 + box.y1)/2), handlelength=2.6)
    return fig


# ---------------------------------------------------------------- leakage into a field-free region
def leakage(sw):
    import paper_tomida as pt
    from matplotlib.colors import LogNorm
    F = Data(sw);test = "kh_localized";n = 512
    fig = plt.figure(figsize=(FULL, 4.55))
    # (a) maps of |B| at t = 3
    shown = (("athenak_reference", "Reference ($4096^2$)"), ("athenak_matched", "AthenaK PLM/HLLE"), ("glm", "GLM LLF"), ("glm_hlld", "GLM HLLD"), ("ct", "CT prototype"))
    lo, hi = pt.KH[test]["b"];w = .158;gap = .012;x0 = .055;y0 = .615;h = w*FULL/4.55;im = None
    for i, (m, title) in enumerate(shown):
        ax = fig.add_axes([x0 + i*(w + gap), y0, w, h]);ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_linewidth(.5)
        u = sw.ref_field(test, m, n, "end") if m in sweep_data.REFERENCES else sw.field(test, m, n, "end")
        ax.set_title(title, pad=3, fontsize=8)
        if u is None:
            ax.text(.5, .5, "not here", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=7);continue
        im = pc.show(ax, np.clip(pc.bmag(u), lo, hi), cmap=pt.B_MAP, norm=LogNorm(lo, hi))
        for cx in (pt.CORE[0], pt.CORE[0] + 1):                         # where the leak is measured (the disc wraps in x)
            ax.add_patch(plt.Circle((cx, pt.CORE[1]), pt.CORE[2], fill=False, ec=pc.ARROW_ON_MAGNETIC, lw=.8))
        ax.set_xlim(0, 1);ax.set_ylim(0, 1)
        if i == 0:ax.set_ylabel(f"$|\\mathbf{{B}}|$ at $t = 3$, ${n}^2$", labelpad=3)
    if im is not None:
        cax = fig.add_axes([x0 + 5*(w + gap) + .002, y0, .011, h]);cb = fig.colorbar(im, cax=cax);cb.outline.set_linewidth(.5)
        cb.ax.tick_params(labelsize=7, length=2, width=.5);cb.set_label(r"$|\mathbf{B}|$", labelpad=2)
    fig.text(.012, y0 + h + .028, "(a)", fontsize=8.5)
    # (b) the field inside the field-free vortex over the reference's, against resolution
    ax = fig.add_axes([.075, .1, .3, .4]);grids = (64, 128, 256, 512);vals, ref, first = pt.core_table(F, test, grids)
    ax.axhline(1, color=pc.REFERENCE_INK, lw=1.8, alpha=.5)
    for m in pc.ALL:
        pts = sorted((g, v/ref[g]) for g, v in vals.get(m, {}).items() if g in ref)
        if pts:curve(ax, *zip(*pts), m)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xticks(list(grids));ax.set_xticklabels([str(g) for g in grids]);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("$N$");ax.set_ylabel("field in the field-free vortex / reference", labelpad=2);ticks(ax);panel_letter(ax, "(b)")
    # (c) the cleaning variable through a sudden timestep change
    jt = "kh_localized_dtjump";nj = 256;zoom = (2.995, 3.06);cx = fig.add_axes([.475, .1, .25, .4]);t0 = float(sw.configs[jt]["dt_jump"][0])
    cx.axvline(t0, color=MUTED, lw=.6, ls=(0, (4, 3)))
    for m in pt.GLM:
        sr = pt.with_dense(F, jt, m, nj, "psi_max")
        if sr is None:continue
        t, v = sr;keep = (v > 0) & (t >= zoom[0]) & (t <= zoom[1])
        if keep.any():curve(cx, t[keep], v[keep], m, markers=False)
    cx.set_yscale("log");cx.set_xlim(*zoom);cx.set_xlabel("$t$");cx.set_ylabel(r"largest $|\psi|$, $%d^2$" % nj, labelpad=2);ticks(cx);panel_letter(cx, "(c)")
    cx.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.02))
    cx.annotate(r"$\Delta t \to \Delta t/40$", (t0, .93), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=7, color=MUTED, va="top")
    ref_h = Line2D([], [], color=pc.REFERENCE_INK, lw=1.8, alpha=.5, label="reference")
    fig.legend(handles=line_handles(pc.ALL, extra=[ref_h]), loc="center left", bbox_to_anchor=(.745, .3), handlelength=2.6, fontsize=7)
    return fig


# ---------------------------------------------------------------- two tests of the divergence constraint
def divergence(sw):
    import paper_stone as ps
    F = Data(sw);shock, ns = "brio_wu_rotated", 256;loop, nl = "field_loop", 128
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.0), sharey=True, gridspec_kw=dict(wspace=.07, left=.2, right=.985, top=.9, bottom=.16))
    gamma = ps.gamma_of(sw, shock);rows = list(pc.ALL);y = {m: i + (i >= len(pc.GPU))*.6 for i, m in enumerate(rows)}
    for ax in (a, b):
        for m in rows:ax.axhline(y[m], color=RULE, lw=.5, zorder=0)
    for m in rows:
        u = sw.field(shock, m, ns, "end")
        if u is not None:
            xi, q = ps.normal_profile(u, gamma);mean, lo, hi = q["bn"]
            a.plot([max(abs(lo.min() - .75), abs(hi.max() - .75))], [y[m]], color=pc.color(m), ms=4.6, ls="", **marker_kw(m))
        else:a.text(.03, y[m], F.why(shock, m, ns).replace("\n", " "), transform=a.get_yaxis_transform(), fontsize=6.5, color=MUTED, va="center")
        sr = F.series(loop, m, nl, "bz_rms")
        if sr is not None and len(sr[1]) and not F.stopped(loop, m, nl):
            b.plot([max(sr[1][-1]/ps.B0, 1e-16)], [y[m]], color=pc.color(m), ms=4.6, ls="", **marker_kw(m))
        else:b.text(.03, y[m], F.why(loop, m, nl).replace("\n", " "), transform=b.get_yaxis_transform(), fontsize=6.5, color=MUTED, va="center")
    for ax in (a, b):
        ax.set_xscale("log");ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):ax.spines[side].set_visible(False)
    a.set_yticks([y[m] for m in rows]);a.set_yticklabels([name(m) for m in rows]);a.set_ylim(max(y.values()) + .7, -.7)
    a.set_xlabel(r"largest $|B_\parallel - 0.75|$");b.set_xlabel(r"rms $B_z/B_0$ at $t = 2$")
    panel_letter(a, f"(a) Rotated shock tube, ${ns}^2$: field along the shock normal");panel_letter(b, f"(b) Field loop with $v_z \\neq 0$, ${nl}^2$: out-of-plane field")
    return fig


# ---------------------------------------------------------------- advection of a field loop
def field_loop(sw):
    import paper_stone as ps
    F = Data(sw)
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.85), gridspec_kw=dict(wspace=.26, left=.075, right=.745, top=.9, bottom=.15))
    fit = dict(color=pc.REFERENCE_INK, lw=1.0, ls=(0, (4, 2)))
    for ax, test, n, curve_fit, title in ((a, "field_loop_novz", 64, ps.FIT_2005, "(a) Loop in the plane, $64^2$"), (b, "field_loop_3d", 128, ps.FIT_2008[128], "(b) Inclined loop, $128^3$")):
        for m in pc.ALL:
            sr = F.series(test, m, n, "magnetic")
            if sr is None:continue
            t, e = sr;curve(ax, t, e/e[0], m, markers=False)
            if F.stopped(test, m, n):ax.plot(t[-1], e[-1]/e[0], marker="x", color=pc.color(m), ms=4, mew=1)
        t = np.linspace(0, float(sw.configs[test]["t_end"]), 200);ax.plot(t, ps.fit_curve(t, curve_fit), **fit)
        ax.set_xlabel("$t$");ax.set_ylabel("magnetic energy / initial", labelpad=2);ax.set_xlim(0, t[-1]);ticks(ax);panel_letter(ax, title)
    extra = [Line2D([], [], label="Gardiner & Stone (2005, 2008):\ntheir fits at these grids", **fit)]
    fig.legend(handles=line_handles(pc.ALL, markers=False, extra=extra), loc="center left", bbox_to_anchor=(.755, .53), handlelength=2.6, fontsize=7)
    return fig


# ---------------------------------------------------------------- decaying supersonic turbulence
def turbulence(sw):
    import paper_kritsuk as pk
    F = Data(sw);n = 512;frame = int(sw.frames(pk.TEST)) - 1;t_end = float(sw.configs[pk.TEST]["t_end"])
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.6), gridspec_kw=dict(wspace=.27, hspace=.3, left=.085, right=.745, top=.95, bottom=.1))
    drawn = []
    for m in pc.ALL:
        mo = pk.moments(F, m, n)
        if mo is None or F.stopped(pk.TEST, m, n) or mo["t"][-1] < t_end*(1 - 1e-6):continue      # runs that reach the end
        drawn.append(m)
        curve(axes[0, 0], mo["t"], mo["ek"], m, markers=False);curve(axes[0, 1], mo["t"], mo["em"], m, markers=False)
        sp = pk.spectrum(F, m, n, frame)
        if sp is None:continue
        k = sp["k"];keep = k <= n/2
        curve(axes[1, 0], k[keep], 2*sp["v"][keep]*k[keep]**(5/3), m, markers=False);curve(axes[1, 1], k[keep], 2*sp["b"][keep], m, markers=False)
    for ax, lab, letter in ((axes[0, 0], r"$\langle u^2\rangle/2$", "(a)"), (axes[0, 1], r"$\langle B^2\rangle/2$", "(b)")):
        ax.set_yscale("log");ax.set_xlim(0, t_end);ax.set_xlabel("$t$");ax.set_ylabel(lab, labelpad=2);ticks(ax);panel_letter(ax, letter)
    for ax, lab, letter in ((axes[1, 0], r"$k^{5/3}P(u, k)$", "(c)"), (axes[1, 1], r"$P(B, k)$", "(d)")):
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(1, n/2);ax.set_xlabel(r"$k/k_{\min}$");ax.set_ylabel(lab + f", $t = {t_end:g}$", labelpad=2);ticks(ax);panel_letter(ax, letter)
        ys = np.concatenate([l.get_ydata() for l in ax.lines]) if ax.lines else np.array([1.0]);ys = ys[np.isfinite(ys) & (ys > 0)]
        if ys.size:ax.set_ylim(max(ys.min(), ys.max()*10**-5.5)/1.5, ys.max()*2)
    fig.legend(handles=line_handles(drawn, markers=False), loc="center left", bbox_to_anchor=(.755, .55), handlelength=2.6, fontsize=7, title=f"${n}^3$, to $t = {t_end:g}$", title_fontsize=7.5)
    return fig


# ---------------------------------------------------------------- AREPO: moving and static mesh
def arepo_steps(sw, test, n, label="arepo_static"):
    """(time, timestep) at every synchronisation point of an AREPO run, from its own log; None when the log is not here
    (it is fetched for the runs AREPO ended itself)."""
    p = sw.path(test, label, n, "arepo.log")
    if p is None or not p.is_file():return None
    rows = re.findall(r"Sync-Point \d+, Time: ([-+0-9.eE]+), (?:Redshift: [-+0-9.eE]+, )?Systemstep: ([-+0-9.eE]+)", p.read_text(errors="replace"))
    if not rows:return None
    a = np.array(rows, dtype=float);return a[:, 0], a[:, 1]


def arepo(sw):
    F = Data(sw);ochre = pc.FAMILY["arepo"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.75), gridspec_kw=dict(wspace=.36, left=.075, right=.985, top=.9, bottom=.3))
    # (a) the field loop in the plane: magnetic energy over its initial value, static mesh at four grids, moving mesh
    test = "field_loop_novz";grids = (64, 128, 256, 512);shade = dict(zip(grids, pc.shades("arepo_static", len(grids))))
    for n in grids:
        sr = F.series(test, "arepo_static", n, "magnetic")
        if sr is None:continue
        t, e = sr;a.plot(t, e/e[0], color=shade[n], lw=1.1)
        if F.stopped(test, "arepo_static", n):a.plot(t[-1], e[-1]/e[0], marker="x", color=shade[n], ms=4.5, mew=1.1)
        a.annotate(f"${n}^2$", (t[-1], e[-1]/e[0]), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=shade[n], va="center")
    sr = F.series(test, "arepo_moving", 256, "magnetic")
    if sr is not None:a.plot(sr[0], sr[1]/sr[1][0], color=INK, lw=1.1, ls=(0, (4, 2)))
    a.set_yscale("log");a.set_xlim(0, 2.5);a.set_xticks([0, .5, 1, 1.5, 2]);a.set_xlabel("$t$");a.set_ylabel("magnetic energy / initial", labelpad=2);ticks(a)
    panel_letter(a, "(a) Field loop advection")
    # (b) the timestep of the static-mesh runs AREPO ended itself, over its first value, against the part of the run done
    stopped = [(t, n) for (t, l, n, s), r in sw.arepo.items() if l == "arepo_static" and r["state"] == "stopped"]
    names = dict(orszag_tang="Orszag–Tang", field_loop="Field loop, $v_z \\neq 0$", field_loop_novz="Field loop", current_sheet="Current sheet")
    styles = dict(orszag_tang="-", field_loop=(0, (1, 1)), field_loop_novz=(0, (4, 2)), current_sheet="-.")
    drawn = []
    for t_, n in sorted(stopped):
        st = arepo_steps(sw, t_, n)
        if st is None:continue
        t, dt = st;keep = dt > 0;t, dt = t[keep], dt[keep];dt0 = np.median(dt[:max(5, len(dt)//50)])
        pick = np.unique(np.r_[np.arange(0, len(t), max(1, len(t)//400)), np.arange(max(0, len(t) - 400), len(t))])      # thin the line, keep its end
        b.plot(t[pick]/float(sw.configs[t_]["t_end"]), dt[pick]/dt0, color=INK if n == 256 else ochre, ls=styles.get(t_, "-"), lw=1.0)
        if t_ not in drawn:drawn.append(t_)
    b.set_yscale("log");b.set_xlim(0, 1);b.set_xlabel("time / length of the run");b.set_ylabel("timestep / first timestep", labelpad=2);ticks(b)
    panel_letter(b, "(b) Static mesh: runs that stopped")
    hb = [Line2D([], [], color=MUTED, ls=styles[t_], lw=1.0, label=names.get(t_, t_)) for t_ in drawn]
    hb += [Line2D([], [], color=INK, lw=1.0, label="$256^2$"), Line2D([], [], color=ochre, lw=1.0, label="$512^2$")]
    # (c) AREPO's own measure of the divergence of B at the last output, against the grid
    tests = (("orszag_tang", "Orszag–Tang", "o"), ("rotor", "Rotor", "s"), ("blast_2d_b50", "Blast, $B = 50$", "D"), ("kh_uniform", "Kelvin–Helmholtz", "v"))
    for m in pc.AREPO:
        for test, title, marker in tests:
            pts = []
            for n in sw.grids(test, m):
                p = sw.path(test, m, n, "arepo_divergence.csv")
                d = sweep_data.read_csv(p) if p is not None and p.is_file() and sw.state_of(test, m, n) == "done" else None
                if d is not None:pts.append((n, d["normalized_div_rms"][-1]))
            if pts:c.plot(*zip(*pts), color=ochre if m == "arepo_moving" else MUTED, ls=pc.STYLE[m][3], marker=marker, markeredgewidth=.9,
                          markerfacecolor="white", lw=1.0)
    c.set_xscale("log");c.set_yscale("log");c.set_xticks([32, 64, 128, 256, 512]);c.set_xticklabels(["32", "64", "128", "256", "512"])
    c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());c.set_xlabel("$N$");c.set_ylim(1e-3, 1e-1)
    c.set_ylabel(r"rms $|\nabla\!\cdot\!\mathbf{B}|\,\Delta x\,/\,$rms $|\mathbf{B}|$", labelpad=2);ticks(c);panel_letter(c, "(c) Divergence of $\\mathbf{B}$, end of run")
    hc = [Line2D([], [], color=ochre, ls=pc.STYLE["arepo_moving"][3], lw=1.0, label="moving mesh"), Line2D([], [], color=MUTED, ls=pc.STYLE["arepo_static"][3], lw=1.0, label="static mesh")]
    hc += [Line2D([], [], color=MUTED, ls="", marker=mk, markerfacecolor="white", markeredgewidth=.9, label=title) for _, title, mk in tests]
    ha = [Line2D([], [], color=shade[256], lw=1.1, label="static mesh"), Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2)), label="moving mesh, $256^2$"),
          Line2D([], [], color=shade[256], ls="", marker="x", mew=1.1, ms=4.5, label="AREPO stopped")]
    for ax, hs, ncol in ((a, ha, 1), (b, hb, 2), (c, hc, 2)):
        box = ax.get_position()
        fig.legend(handles=hs, loc="upper center", bbox_to_anchor=((box.x0 + box.x1)/2, .185), ncol=ncol, fontsize=6.5, handlelength=2.2, columnspacing=1.0, labelspacing=.3)
    return fig


FIGURES = dict(cost_vs_error=cost_vs_error, speed=speed, robustness=robustness, convergence=convergence, leakage=leakage,
               divergence=divergence, field_loop=field_loop, turbulence=turbulence, arepo=arepo)


def main(argv):
    preview = None
    if "--preview" in argv:
        i = argv.index("--preview");preview = Path(argv[i + 1]);argv = argv[:i] + argv[i + 2:]
    if "--list" in argv:
        print("\n".join(FIGURES));return
    todo = argv or list(FIGURES)
    for name in todo:
        if name not in FIGURES:sys.exit(f"Unknown figure {name}; known: {', '.join(FIGURES)}")
    style();OUT.mkdir(exist_ok=True)
    if preview:preview.mkdir(parents=True, exist_ok=True)
    sw = sweep_data.Sweep()
    for name in todo:
        fig = FIGURES[name](sw)
        if isinstance(fig, tuple):fig = fig[0]
        fig.savefig(OUT/f"{name}.pdf")
        if preview:fig.savefig(preview/f"{name}.png", dpi=220, facecolor="white")
        plt.close(fig);print(f"wrote paper_figures/{name}.pdf")


if __name__ == "__main__":
    main(sys.argv[1:])
