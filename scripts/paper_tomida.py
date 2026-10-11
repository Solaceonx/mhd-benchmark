"""Tomida et al. (2026, arXiv:2605.07928), sec. III.1-3: Kelvin-Helmholtz instability with a uniform field, with a
field confined to one layer, and with a sudden timestep change. Their Figs. 1, 2, 4, 5, 6-8 with our codes.

Their box is [-1/2, 1/2] x [-1, 1] with one shear layer and outflow in y; ours is the periodic unit square with two
layers (y = 1/4, 3/4). At t = 3 the lower layer's vortex is centred at x = 1/2 and the upper layer's at x = 0. With
the confined field only the lower layer is magnetized, so the upper vortex is our field-free region: field found
inside it that the reference does not have is the paper's leakage. N = 128 and 256 are their h = 1/128 and 1/256.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

import sweep_data
from plot_style import field_map
from paper_common import (ALL, GPU, OTHER, GROUPS, INK, MUTED, REFERENCE_INK, STYLE, arrows, blank, bmag, color, handles, image_axes,
                          label, line_kw, nice_limit, plain, shades, show, spread, style_axes)

PAPER = "tomida"
# Colour ranges: |B| as in the paper's figures; psi as in the paper for the uniform field (+-0.005). For the confined
# field the paper uses +-2e-5, which would saturate every panel of ours, so the range is taken from the data.
KH = dict(kh_uniform=dict(b=(1e-4, 3e-1), psi=5e-3, name="uniform field"),
          kh_localized=dict(b=(1e-7, 2e-1), psi_q=75.0, name="field confined to the lower layer"),
          kh_localized_dtjump=dict(b=(1e-7, 2e-1), psi_q=97.0, name="timestep change"))
B_MAP = field_map("magnetic")       # the notebook's map for field strength (the paper's own figures are black - orange - white)
PSI_MAP = field_map("signed")       # blue negative, black zero, red positive
GLM = tuple(m for m in GPU if m.startswith("glm"))
VARIABLE = tuple(m for m in GLM if m != "glm_constch")       # cleaning speed follows 1/dt (the paper's "v" models)
CORE = (0.0, 0.75, 0.06)    # centre and radius of the disc inside the upper, field-free vortex
BOX = ("Our box is the periodic unit square with two shear layers (y = 1/4 and 3/4) instead of one layer with outflow boundaries, "
       "so two vortices appear, half a box apart in x; CFL 0.25 instead of 0.4.")
BAND = (" The confined field is a sech² band with the full width at half maximum of the paper's Gaussian; it leaves a 2×10⁻⁶ tail at "
        "the upper layer, where the paper's field is zero.")


def colorbar(fig, im, rect, text, sci=False):
    cb = fig.colorbar(im, cax=fig.add_axes(rect));cb.set_label(text)
    if sci:cb.formatter.set_powerlimits((-2, 2));cb.update_ticks()
    cb.outline.set_linewidth(.5);cb.ax.tick_params(labelsize=7.5, length=2, width=.6)
    return cb


def psi_limit(F, test, parts):
    """Symmetric colour limit of psi: the test's fixed value, or a percentile over the (method, grid, frame) panels."""
    spec = KH[test]
    if "psi" in spec:return spec["psi"]
    vals = []
    for m, n, f in parts:
        u = F.sw.field(test, m, n, f)
        if u is not None and m.startswith("glm"):vals.append(np.percentile(np.abs(u[..., 8]), spec["psi_q"]))
    return nice_limit(max(vals)) if vals else 1.0


def kh_maps(F, test, group, grids=(128, 256), frame="end"):
    """Figs. 1 and 4: |B| with field directions, and the cleaning variable."""
    sw = F.sw;spec = KH[test];gpu = group == "gpu"
    methods = list(GPU if gpu else OTHER + sweep_data.REFERENCES)
    kinds = ["b", "psi"] if gpu else ["b"]
    rows = [(n, k) for n in grids for k in kinds];nc = len(methods)
    fig, axes = plt.subplots(len(rows), nc, figsize=(1.72*nc + 1.0, 1.78*len(rows) + .5), squeeze=False,
                             gridspec_kw=dict(wspace=.05, hspace=.07))
    lim = psi_limit(F, test, [(m, n, frame) for m in methods for n in grids]) if gpu else None
    norm_b = LogNorm(*spec["b"]);ims = {}
    for i, (n, kind) in enumerate(rows):
        for j, m in enumerate(methods):
            ax = axes[i, j];image_axes(ax)
            if i == 0:ax.set_title(label(m, True), fontsize=8.5, pad=4)
            if kind == "psi" and not m.startswith("glm"):
                blank(ax, "no cleaning\nvariable");continue
            if m in sweep_data.REFERENCES:
                u = sw.ref_field(test, m, n, frame);big = sw.reference_n(test, m)
                if u is None:blank(ax, F.why(test, m, big));F.miss(test, m, big);continue
            else:
                u = sw.field(test, m, n, frame)
                if u is None:blank(ax, F.why(test, m, n));F.miss(test, m, n);continue
            if kind == "b":
                ims["b"] = show(ax, np.clip(bmag(u), *spec["b"]), cmap=B_MAP, norm=norm_b);arrows(ax, u)
            else:
                ims["psi"] = show(ax, u[..., 8], cmap=PSI_MAP, norm=Normalize(-lim, lim))
        what = r"$|\mathbf{B}|$" if kind == "b" else r"$\psi$"
        axes[i, 0].set_ylabel(f"N = {n}\n{what}" if kind == "b" else what, fontsize=9, color=INK, labelpad=5)
    fig.subplots_adjust(right=.9)
    top, bottom = axes[0, -1].get_position(), axes[-1, -1].get_position()
    if gpu and "psi" in ims and "b" in ims:
        mid = (top.y1 + bottom.y0)/2
        colorbar(fig, ims["b"], [.915, mid + .02, .012, top.y1 - mid - .02], r"$|\mathbf{B}|$")
        colorbar(fig, ims["psi"], [.915, bottom.y0, .012, mid - .02 - bottom.y0], r"cleaning variable $\psi$", sci=True)
    elif "b" in ims:
        colorbar(fig, ims["b"], [.915, bottom.y0, .012, top.y1 - bottom.y0], r"$|\mathbf{B}|$")
    return fig, lim


def best_reference(sw, test, grids, frame):
    for rm in sweep_data.REFERENCES:
        for m in sorted(grids, reverse=True):
            r = sw.ref_field(test, rm, m, frame)
            if r is not None:return rm, m, r
    return None


def kh_profiles(F, test, grids=(128, 256, 512), frame="end", x0=.5, yrange=(0, 1)):
    """Figs. 2 and 5: |B| along y at one x, one panel per method, one line per grid."""
    sw = F.sw;log = test == "kh_localized"
    fig, axes = plt.subplots(len(GPU), 2, figsize=(10.2, 1.45*len(GPU) + .8), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=.06, hspace=.3))
    ref = best_reference(sw, test, grids, frame)
    if ref is None:
        for rm in sweep_data.REFERENCES:
            big = sw.reference_n(test, rm)
            if big:F.miss(test, rm, big)
    first = None
    for n in sorted(grids, reverse=True):
        if first is None:first = sw.field(test, sweep_data.FIRST_FROM, n, 0)
    top = 0.0
    for col, (_, _, methods) in enumerate(GROUPS):
        for row, m in enumerate(methods):
            ax = axes[row, col];style_axes(ax);cs = shades(m, len(grids));drew = False
            if log and first is not None:
                n0 = first.shape[0];y = (np.arange(n0) + .5)/n0
                ax.plot(y, bmag(first)[int(x0*n0), :], color=MUTED, lw=.9, ls=(0, (1, 1.5)))
            if ref is not None:
                n0 = ref[1];y = (np.arange(n0) + .5)/n0;b = bmag(ref[2])[int(x0*n0), :]
                ax.plot(y, b, color=REFERENCE_INK, lw=.9);top = max(top, b[(y >= yrange[0]) & (y <= yrange[1])].max())
            for c, n in zip(cs, grids):
                u = sw.field(test, m, n, frame)
                if u is None:F.miss(test, m, n);continue
                y = (np.arange(n) + .5)/n;b = bmag(u)[int(x0*n), :];ax.plot(y, b, color=c, lw=1.25);drew = True
                top = max(top, b[(y >= yrange[0]) & (y <= yrange[1])].max())
            ax.set_title(label(m, True), fontsize=8.5, pad=2, loc="left")
            if not drew:ax.text(.5, .45, F.why(test, m, grids[0]).replace("\n", " "), transform=ax.transAxes, ha="center", color=MUTED, fontsize=8)
            ax.set_xlim(*yrange)
    if log:axes[0, 0].set_yscale("log");axes[0, 0].set_ylim(1e-7, .5)
    else:axes[0, 0].set_ylim(0, 1.08*top if top else 1)
    where = "x = 1/2" if x0 == .5 else "x = 0"
    for ax in axes[-1]:
        ax.set_xlabel(r"$y$" + ("  (lower shear layer at $y = 1/4$)" if yrange[1] <= .5 else
                               "  (magnetized layer at $y = 1/4$; centre of the field-free vortex at $y = 3/4$)" if log and x0 == 0 else
                               "  (shear layers at $y = 1/4$ and $3/4$)"))
    fig.supylabel(rf"magnetic field strength $|\mathbf{{B}}|$ at ${where}$", fontsize=9.5, x=.06)
    hs = [Line2D([], [], color=str(.75 - .6*i/(max(1, len(grids) - 1))), lw=1.4, label=f"N = {n}") for i, n in enumerate(grids)]
    if ref is not None:hs.append(Line2D([], [], color=REFERENCE_INK, lw=.9, label=f"{label(ref[0])} ({sw.reference_n(test, ref[0])}$^2$, averaged to {ref[1]}$^2$)"))
    if log and first is not None:hs.append(Line2D([], [], color=MUTED, lw=.9, ls=(0, (1, 1.5)), label="initial field"))
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(.5, .95), handlelength=2.2, columnspacing=1.6)
    return fig


# ---------------------------------------------------------------- field inside the field-free vortex
def core_median(u):
    """Median |B| in the disc CORE of a planar state: inside the upper vortex, where the reference has no field."""
    n = u.shape[0];x = (np.arange(n) + .5)/n;X, Y = np.meshgrid(x, x, indexing="ij")
    dx = np.minimum(np.abs(X - CORE[0]), 1 - np.abs(X - CORE[0]))
    return float(np.median(bmag(u)[dx**2 + (Y - CORE[1])**2 < CORE[2]**2]))


def core_table(F, test="kh_localized", grids=(64, 128, 256, 512), frame="end"):
    """{method: {n: median}}, {n: reference median}, {n: initial median}."""
    sw = F.sw;ref, first, out = {}, {}, {}
    for n in grids:
        for rm in sweep_data.REFERENCES:
            r = sw.ref_field(test, rm, n, frame)
            if r is not None and n not in ref:ref[n] = core_median(r)
        u0 = sw.field(test, sweep_data.FIRST_FROM, n, 0)
        if u0 is not None:first[n] = core_median(u0)
        for m in ALL:
            u = sw.field(test, m, n, frame)
            if u is not None:out.setdefault(m, {})[n] = core_median(u)
    return out, ref, first


def core_convergence(F, grids=(64, 128, 256, 512)):
    """Field inside the field-free vortex at t = 3 against resolution: the leak, as one number per run."""
    test = "kh_localized";vals, ref, first = core_table(F, test, grids)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.0), sharey=True, gridspec_kw=dict(wspace=.78))
    for ax, (_, _, methods) in zip(axes, GROUPS):
        style_axes(ax)
        if ref:
            ns = sorted(ref);ax.plot(ns, [ref[n] for n in ns], color=REFERENCE_INK, lw=2.2, alpha=.5, solid_capstyle="round")
        if first:
            ns = sorted(first);ax.plot(ns, [first[n] for n in ns], color=MUTED, lw=.9, ls=(0, (1, 1.5)))
        for m in methods:
            for n in grids:
                if n not in vals.get(m, {}):F.miss(test, m, n)
            pts = sorted(vals.get(m, {}).items())
            if pts:ax.plot(*zip(*pts), marker=STYLE[m][4], ms=4, **line_kw(m))
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xticks(list(grids));ax.set_xticklabels([str(n) for n in grids]);ax.minorticks_off()
        ax.set_xlabel("N")
        hs = handles(methods, markers=True)
        if ref:hs.append(Line2D([], [], color=REFERENCE_INK, lw=2.2, alpha=.5, label="reference (4096², averaged to N²)"))
        if first:hs.append(Line2D([], [], color=MUTED, lw=.9, ls=(0, (1, 1.5)), label="initial field there"))
        ax.legend(handles=hs, loc="center left", bbox_to_anchor=(1.02, .5), handlelength=2.8)
        ax.tick_params(labelleft=True)
    axes[0].set_ylabel(r"median $|\mathbf{B}|$ inside the field-free vortex, $t = 3$")
    rows = []
    early = core_table(F, test, (256,), 12)
    for m in ALL:
        row = dict(method=plain(label(m)))
        for n in grids:
            v = vals.get(m, {}).get(n);row[f"b{n}"] = v;row[f"r{n}"] = None if v is None or n not in ref else v/ref[n]
        v = early[0].get(m, {}).get(256);row["e256"] = None if v is None or 256 not in early[1] else v/early[1][256]
        rows.append(row)
    return fig, rows, ref


def core_finding(rows, n=512):
    """One sentence from the table: how far each family is above the reference at grid n."""
    def span(keys, fmt):
        return spread([r[f"r{n}"] for r in rows if r["key"] in keys and r.get(f"r{n}") is not None], fmt)
    one, whole = (lambda x: f"{x:.1f}"), (lambda x: f"{x:.0f}")
    ext, glm, ct = span(OTHER, one), span(GLM, whole), span(("ct",), lambda x: f"{x:.3g}")
    parts = []
    if ext:parts.append(f"the six AthenaK, RAMSES and Athena++ set-ups have {ext} times the reference's field there")
    if glm:parts.append(f"the GLM kernels {glm} times")
    if ct:parts.append(f"the CT prototype {ct} times")
    return f"At {n}², " + "; ".join(parts) + "." if parts else ""


# ---------------------------------------------------------------- sudden timestep change
def dtjump_maps(F, n=128, frames=(30, 31, 32)):
    """Figs. 6 and 8: |B| and psi before and after the timestep change, every GPU-driver method."""
    sw = F.sw;test = "kh_localized_dtjump";spec = KH[test];times = sw.times(test)
    cols = [("b", f) for f in frames] + [("psi", f) for f in frames]
    fig, axes = plt.subplots(len(GPU), len(cols), figsize=(1.72*len(cols) + 1.6, 1.78*len(GPU) + .5), squeeze=False,
                             gridspec_kw=dict(wspace=.05, hspace=.07))
    lim = psi_limit(F, test, [(m, n, f) for m in GLM for f in frames]);ims = {}
    for i, m in enumerate(GPU):
        for j, (kind, f) in enumerate(cols):
            ax = axes[i, j];image_axes(ax)
            if i == 0:
                what = r"$|\mathbf{B}|$" if kind == "b" else r"$\psi$"
                when = "before the change" if f == frames[0] else "after"
                ax.set_title(f"{what},  t = {times[f]:.1f}\n{when}", fontsize=8.5, pad=4)
            if kind == "psi" and not m.startswith("glm"):blank(ax, "no cleaning\nvariable");continue
            u = sw.field(test, m, n, f)
            if u is None:blank(ax, F.why(test, m, n));F.miss(test, m, n);continue
            if kind == "b":ims["b"] = show(ax, np.clip(bmag(u), *spec["b"]), cmap=B_MAP, norm=LogNorm(*spec["b"]));arrows(ax, u)
            else:ims["psi"] = show(ax, u[..., 8], cmap=PSI_MAP, norm=Normalize(-lim, lim))
        axes[i, 0].set_ylabel(label(m, True), fontsize=9, color=INK, labelpad=5)
    fig.subplots_adjust(right=.9)
    top, bottom = axes[0, -1].get_position(), axes[-1, -1].get_position();mid = (top.y1 + bottom.y0)/2
    if "b" in ims:colorbar(fig, ims["b"], [.915, mid + .02, .012, top.y1 - mid - .02], r"$|\mathbf{B}|$")
    if "psi" in ims:colorbar(fig, ims["psi"], [.915, bottom.y0, .012, mid - .02 - bottom.y0], r"cleaning variable $\psi$", sci=True)
    return fig


def with_dense(F, test, m, n, column):
    """A diagnostics column at the regular outputs merged with the step-by-step record around the timestep change."""
    s = F.series(test, m, n, column);d = F.sw.dense(test, m, n)
    if s is None:return None
    t, v = s
    if d is not None and column in d:
        t = np.concatenate([t, d["time"]]);v = np.concatenate([v, d[column]]);order = np.argsort(t, kind="stable");t, v = t[order], v[order]
    return t, v


def jump_numbers(F, grids=(128, 256, 512)):
    """Per GPU-driver method and grid: the jump of the largest |psi| at the change, and the field inside the field-free
    vortex before (the run without the change at t = 3.0: the same solution up to then) and after (t = 3.2)."""
    sw = F.sw;test = "kh_localized_dtjump";t0 = float(sw.configs[test]["dt_jump"][0]);out = {}
    for m in GPU:
        for n in grids:
            rec = {}
            d = sw.diag(test, m, n);dense = sw.dense(test, m, n)
            if m.startswith("glm") and d is not None and dense is not None and "psi_max" in dense:
                before = d["psi_max"][d["time"] <= t0 + 1e-9]
                if before.size and before[-1] > 0:rec["psi"] = float(dense["psi_max"].max()/before[-1])
            a = sw.field("kh_localized", m, n, "end");b = sw.field(test, m, n, "end")
            if a is not None:rec["before"] = core_median(a)
            if b is not None:rec["after"] = core_median(b)
            out[(m, n)] = rec
    return out


def psi_history(F, n=256, zoom=(2.995, 3.06)):
    """Fig. 7: largest |psi| against time, with the steps around the timestep change resolved; and what the change
    does to the field inside the field-free vortex."""
    sw = F.sw;test = "kh_localized_dtjump";jump = sw.configs[test]["dt_jump"];t_end = float(sw.configs[test]["t_end"])
    fig, (ax, zx, bx) = plt.subplots(1, 3, figsize=(12.6, 3.9), gridspec_kw=dict(wspace=.27, width_ratios=(1.25, 1, 1.1)))
    for a in (ax, zx):style_axes(a);a.axvline(jump[0], color=MUTED, lw=.7, ls=(0, (4, 3)))
    low, high = np.inf, 0.0
    for m in GLM:
        s = with_dense(F, test, m, n, "psi_max")
        if s is None:F.miss(test, m, n);continue
        t, v = s;keep = v > 0;t, v = t[keep], v[keep]
        ax.plot(t, v, **line_kw(m));zx.plot(t, v, **line_kw(m))
        if F.stopped(test, m, n):ax.plot(t[-1], v[-1], marker="x", color=color(m), ms=6, mew=1.5)
        inside = v[(t >= zoom[0]) & (t <= zoom[1])]
        if inside.size:low, high = min(low, inside.min()), max(high, inside.max())
    ax.set_yscale("log");zx.set_yscale("log");ax.set_xlim(0, t_end);zx.set_xlim(*zoom)
    if high > 0:zx.set_ylim(low/1.6, high*1.6)
    ax.set_xlabel("t");zx.set_xlabel("t");ax.set_ylabel(r"largest $|\psi|$")
    ax.set_title(f"cleaning variable, N = {n}", loc="left");zx.set_title("the steps around the change", loc="left")
    # right: the field inside the field-free vortex before and after, one pair of marks per method
    style_axes(bx);numbers = jump_numbers(F, (n,));ref = None
    for rm in sweep_data.REFERENCES:
        r = sw.ref_field("kh_localized", rm, n, "end")
        if r is not None and ref is None:ref = core_median(r)
    if ref:bx.axhline(ref, color=REFERENCE_INK, lw=2.2, alpha=.5)
    for i, m in enumerate(GPU):
        rec = numbers[(m, n)]
        if "before" not in rec or "after" not in rec:
            if not m.startswith("glm") or "after" not in rec:F.miss(test, m, n)
            continue
        bx.plot([i, i], [rec["before"], rec["after"]], color=color(m), lw=1.5)
        bx.plot(i, rec["before"], marker="o", ms=6.5, mfc="white", mec=color(m), mew=1.5)
        bx.plot(i, rec["after"], marker="o", ms=6.5, color=color(m))
        bx.annotate(f"×{rec['after']/rec['before']:.3g}", (i, max(rec["before"], rec["after"])), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
    bx.set_yscale("log");bx.set_xticks(range(len(GPU)));bx.set_xticklabels([label(m, True).replace("GLM ", "") for m in GPU], rotation=30, ha="right", fontsize=7.5)
    bx.set_xlim(-.6, len(GPU) - .4);lo, hi = bx.get_ylim();bx.set_ylim(lo, hi*2.2);bx.grid(axis="x", visible=False)
    bx.set_ylabel(r"median $|\mathbf{B}|$ inside the field-free vortex");bx.set_title("field where there should be none", loc="left")
    hs = [Line2D([], [], marker="o", ms=6.5, mfc="white", mec=INK, mew=1.5, lw=0, label="t = 3.0, before"),
          Line2D([], [], marker="o", ms=6.5, color=INK, lw=0, label="t = 3.2, after")]
    if ref:hs.append(Line2D([], [], color=REFERENCE_INK, lw=2.2, alpha=.5, label="reference at t = 3"))
    bx.legend(handles=hs, loc="upper left", fontsize=7.5, handlelength=1.6, borderaxespad=.2)
    fig.legend(handles=handles(GPU), loc="upper center", ncol=6, bbox_to_anchor=(.5, 1.06), handlelength=2.6, columnspacing=1.4)
    return fig


def figures():
    def maps(test, group, ident, fig_no):
        def run(F):
            spec = KH[test];gpu = group == "gpu"
            fig, lim = kh_maps(F, test, group)
            cap = ("Field strength |B| (logarithmic colour; arrows: field direction) at t = 3 on 128² and 256² cells, the paper's h = 1/128 "
                   "and 1/256, with the paper's colour range. ")
            if gpu and test == "kh_uniform":cap += "Below each: the divergence-cleaning variable ψ of the GLM kernels, also in the paper's range (CT has none)."
            elif gpu:cap += (f"Below each: the cleaning variable ψ of the GLM kernels within ±{lim:.0e}, ten times the paper's range, which would "
                             "saturate every panel here. The upper vortex (centred at the left and right edges, y = 3/4) starts without field: it "
                             "stays dark in the codes of the next figure and fills with field in the GLM kernels.")
            else:cap += "The two right columns are the references (AthenaK 4096², Athena++ 2048²) averaged to the same cells."
            F.save(fig, ident, paper=PAPER, paper_fig=fig_no, kind="reproduced", test=test,
                   title=f"Kelvin–Helmholtz, {spec['name']}: field strength" + (" and cleaning variable" if gpu else "") + " at t = 3, "
                         + ("GLM kernels and CT prototype" if gpu else "other codes"),
                   caption=cap, differs=BOX + (BAND if test == "kh_localized" else ""))
        return run

    def uniform_profiles(F):
        F.save(kh_profiles(F, "kh_uniform", yrange=(0, .5)), "tomida-2", paper=PAPER, paper_fig="Fig. 2", kind="reproduced", test="kh_uniform",
               title="Kelvin–Helmholtz, uniform field: field strength across the lower layer at t = 3",
               caption="|B| along y at x = 1/2, through the centre of the lower vortex: one panel per method, one line per grid (light to dark).",
               differs="The paper cuts its single layer at x = 0; we cut the lower of our two layers at its vortex centre and add 512².")

    def confined_profiles(F):
        F.save(kh_profiles(F, "kh_localized", x0=0.0), "tomida-5", paper=PAPER, paper_fig="Fig. 5", kind="reproduced", test="kh_localized",
               title="Kelvin–Helmholtz, confined field: field strength along a cut through the field-free vortex at t = 3",
               caption=("|B| along y at x = 0 on a logarithmic axis: the cut crosses the magnetized layer (y = 1/4) and the centre of the upper "
                        "vortex (y = 3/4), which starts with only the 2×10⁻⁶ tail of the field. Field at y ≈ 3/4 above the reference line has "
                        "leaked there. One panel per method, one line per grid (light to dark)."),
               differs=("The paper's field-free region is everything outside its single layer; ours is the inside of the second vortex, because "
                        "by t = 3 the flow has carried field from the magnetized layer to the rest of the periodic box." + BAND))

    def core(F):
        fig, rows, ref = core_convergence(F)
        for r, m in zip(rows, ALL):r["key"] = m
        finding = " ".join(x for x in (core_finding(rows, 128), core_finding(rows, 512)) if x)
        early = [r["e256"] for r in rows if r["key"] in GLM and r.get("e256") is not None]
        if early:finding += f" At t = 1.2, before the vortices have rolled up, the GLM kernels are still within a factor {max(early):.1f} of the reference (256²)."
        F.save(fig, "tomida-5-core", paper=PAPER, paper_fig="Fig. 5", kind="extension", test="kh_localized",
               title="Kelvin–Helmholtz, confined field: field inside the field-free vortex against resolution",
               caption=("Median |B| at t = 3 in a disc of radius 0.06 around the centre of the upper vortex, for each grid. Grey: the 4096² "
                        "reference averaged to the same cells; dotted: the field that was there at the start. A method on the grey line "
                        "keeps the region as clean as the reference; one above it has put field there."),
               differs="Not a figure of the paper: its leakage (Fig. 5) reduced to one number per run, so that all grids and methods fit on one axis.",
               finding=finding,
               table=dict(columns=["Method", "64²", "128²", "256²", "512²", "÷ reference, 128²", "÷ reference, 256²", "÷ reference, 512²",
                                   "÷ reference at t = 1.2, 256²"],
                          keys=["method", "b64", "b128", "b256", "b512", "r128", "r256", "r512", "e256"], rows=rows))

    def jump_maps(F):
        F.save(dtjump_maps(F), "tomida-6", paper=PAPER, paper_fig="Figs. 6 and 8", kind="reproduced", test="kh_localized_dtjump",
               title="Kelvin–Helmholtz with a sudden timestep change: field strength and cleaning variable, N = 128",
               caption=("At t = 3 the timestep is cut to 1/40 for 100 steps, then restored. Left: |B| just before the change and at the next two "
                        "outputs; right: the cleaning variable ψ at the same times. Kernels whose cleaning speed follows 1/Δt (LLF, f16/HLL, HLLD, "
                        "PPM) are the paper's \"v\" models; constant c_h is its D1c; CT is its CT."),
               differs=BOX + " Our outputs are 0.1 apart, so the paper's snapshots 10⁻⁴ to 4×10⁻² after the change are not available as images; "
                             "the next figure resolves those steps.")

    def jump_history(F):
        fig = psi_history(F);numbers = jump_numbers(F);rows = []
        for m in GPU:
            row = dict(method=plain(label(m)))
            for n in (128, 256, 512):
                rec = numbers[(m, n)];row[f"p{n}"] = rec.get("psi")
                row[f"c{n}"] = rec["after"]/rec["before"] if "before" in rec and "after" in rec else None
            rows.append(row)
        psi_v = [numbers[(m, n)]["psi"] for m in VARIABLE for n in (128, 256, 512) if "psi" in numbers[(m, n)]]
        psi_c = [numbers[("glm_constch", n)]["psi"] for n in (128, 256, 512) if "psi" in numbers[("glm_constch", n)]]
        ratio = lambda ms, grids=(128, 256, 512): [numbers[(m, n)]["after"]/numbers[(m, n)]["before"] for m in ms for n in grids
                                                   if "before" in numbers[(m, n)] and "after" in numbers[(m, n)]]
        f_c, f_ct = ratio(("glm_constch",)), ratio(("ct",));one = lambda x: f"{x:.1f}";whole = lambda x: f"{x:.3g}" if x < 10 else f"{x:.0f}"
        finding = ""
        if psi_v and psi_c and f_c and f_ct and ratio(VARIABLE, (512,)):
            finding = (f"With a cleaning speed that follows 1/Δt the largest ψ jumps {min(psi_v):.0f}- to {max(psi_v):.0f}-fold, and at the end of the "
                       f"run the field inside the field-free vortex is {spread(ratio(VARIABLE, (512,)), whole)} times what it was before the change "
                       f"on 512² ({spread(ratio(VARIABLE, (256,)), whole)} times on 256², {spread(ratio(VARIABLE, (128,)), whole)} on 128²). With the "
                       f"constant speed ψ changes by a factor {spread(psi_c, one)} and that field by {spread(f_c, one)}; in the CT prototype the "
                       f"field changes by {spread(f_ct, one)}. The paper reports about a 40-fold jump of ψ in its variable-speed models and no "
                       "effect with a constant speed or CT.")
        F.save(fig, "tomida-7", paper=PAPER, paper_fig="Fig. 7", kind="reproduced", test="kh_localized_dtjump",
               title="Sudden timestep change: the cleaning variable against time, and the field it leaves behind, N = 256",
               caption=("Left: largest |ψ| at every output and, from the timestep change (dashed line) to 300 steps after it, at every step. "
                        "Middle: the same around the change. Right: median |B| inside the field-free upper vortex just before the change "
                        "(open marks) and at the end of the run (filled), with their ratio; grey line: the reference at t = 3."),
               differs=("The paper uses a fixed timestep before the change; ours is the CFL step (0.25), cut to 1/40 for 100 steps. \"Before\" is "
                        "taken from the run without the change, which is the same solution up to t = 3 (bit for bit at 128², where both states "
                        "are kept)."),
               finding=finding,
               table=dict(columns=["Method", "ψ jump, 128²", "ψ jump, 256²", "ψ jump, 512²", "field after ÷ before, 128²", "256²", "512²"],
                          keys=["method", "p128", "p256", "p512", "c128", "c256", "c512"], rows=rows))

    return {
        "tomida-1-gpu": maps("kh_uniform", "gpu", "tomida-1-gpu", "Fig. 1"),
        "tomida-1-other": maps("kh_uniform", "other", "tomida-1-other", "Fig. 1"),
        "tomida-2": uniform_profiles,
        "tomida-4-gpu": maps("kh_localized", "gpu", "tomida-4-gpu", "Fig. 4"),
        "tomida-4-other": maps("kh_localized", "other", "tomida-4-other", "Fig. 4"),
        "tomida-5": confined_profiles,
        "tomida-5-core": core,
        "tomida-6": jump_maps,
        "tomida-7": jump_history,
    }
