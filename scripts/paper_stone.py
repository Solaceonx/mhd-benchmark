"""Gardiner & Stone (2005, JCP 205, 509) and (2008, JCP 227, 4123): the test figures of the two CTU + CT papers with
our codes. 2005: field-loop advection (their Figs. 2/3/6, 7, 8), circularly polarized Alfven wave (9), rotated shock
tube (10), convergence (11), current sheet (12), blast wave (13). 2008: 3D field loop (1, 2), inclined Riemann
problem (7), 3D blast wave (8).

Our versions of their problems live on the periodic unit square or cube (src/problems.jl); where the parameters are
not theirs, each figure's `differs` says so.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sweep_data
from plot_style import field_map
from paper_common import (ALL, GROUPS, INK, MUTED, REFERENCE_INK, STYLE, blank, bmag, by_family, color, handles, image_axes, label, line_kw,
                          plain, potential, pressure, sci, shades, show, spread, style_axes, two)

GREY = field_map("magnetic")   # the notebook's map for field strength: dark = low (the papers' own figures are grey, white = low)
GAS = field_map("density")     # and for gas quantities


def map_of(key):
    """The map of an image row: magnetic quantities (keys b2, b3, pb) in the field map, gas quantities in the density map."""
    return GREY if key.startswith("b") or key == "pb" else GAS
B0 = 1e-3                      # field strength inside the loops
# Numbers the papers print: fits of the loop's magnetic energy to C (1 - (t/tau)^alpha), as (tau, alpha).
FIT_2005 = (1.0614e4, 0.2914)                                             # 2D loop, N = 64 (2005, sec. 5.1)
FIT_2008 = {32: (3.22e2, 0.365), 64: (3.68e3, 0.328), 128: (2.65e4, 0.320)}   # 3D loop by N (2008, sec. 6.1)


def fit_curve(t, fit):
    tau, alpha = fit;return 1 - (np.asarray(t, float)/tau)**alpha


def gamma_of(sw, test):
    g = sw.configs[test].get("gamma")
    if g is None:return 2.0 if test.startswith("brio_wu") else 1.4 if test.startswith(("rotor", "blast")) else 5/3
    if isinstance(g, str) and "/" in g:
        a, b = g.split("/");return float(a)/float(b)
    return float(g)


def get(F, test, m, n, frame, kind="state"):
    """A frame, or None with the reason recorded."""
    u = F.sw.field(test, m, n, frame, kind=kind)
    if u is None:F.miss(test, m, n)
    return u


def method_axes(rows, lead=False, w=1.62, h=1.68, **kw):
    """rows x 6 image panels (one column per method of a group); lead adds a first column for a common panel."""
    nc = 6 + int(lead)
    fig, axes = plt.subplots(rows, nc, figsize=(w*nc + .9, h*rows + .45), squeeze=False, gridspec_kw=dict(wspace=.05, hspace=.08), **kw)
    return fig, axes


def greybar(fig, im, axes, text):
    fig.subplots_adjust(right=.9);top, bottom = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cb = fig.colorbar(im, cax=fig.add_axes([.915, bottom.y0, .012, top.y1 - bottom.y0]));cb.set_label(text)
    cb.outline.set_linewidth(.5);cb.ax.tick_params(labelsize=7.5, length=2, width=.6)


# ---------------------------------------------------------------- 2005: field loop
def loop_images(F, group, test="field_loop_novz", n=64, frames=(2, 20)):
    """Figs. 2, 3, 6: magnetic pressure of the advected loop."""
    sw = F.sw;methods = dict((g, ms) for g, _, ms in GROUPS)[group];times = sw.times(test)
    fig, axes = method_axes(len(frames), lead=True);im = None
    first = sw.field(test, sweep_data.FIRST_FROM, n, 0)
    for i, f in enumerate(frames):
        ax = axes[i, 0];image_axes(ax)
        if i == 0 and first is not None:
            im = show(ax, (first[..., 5]**2 + first[..., 6]**2)/B0**2, cmap=GREY, vmin=0, vmax=1.05);ax.set_title("initial", fontsize=8.5, pad=4)
        else:ax.set_visible(False)
        for j, m in enumerate(methods):
            ax = axes[i, j + 1];image_axes(ax)
            if i == 0:ax.set_title(label(m, True), fontsize=8.5, pad=4)
            u = sw.field(test, m, n, f)
            if u is None:blank(ax, F.why(test, m, n));F.miss(test, m, n);continue
            im = show(ax, (u[..., 5].astype(float)**2 + u[..., 6].astype(float)**2)/B0**2, cmap=GREY, vmin=0, vmax=1.05)
        axes[i, 1].set_ylabel(f"t = {times[f]:.1f}", fontsize=9, color=INK)
    if im is not None:greybar(fig, im, axes, r"$(B_x^2 + B_y^2)/B_0^2$")
    for i in range(len(frames)):        # room for the time labels between the common first panel and the methods
        p = axes[i, 0].get_position();axes[i, 0].set_position([p.x0 - .02, p.y0, p.width, p.height])
    return fig


def loop_lines(F, test="field_loop_novz", n=64, frame="end"):
    """Fig. 8: field lines (contours of A_z) after two crossings."""
    sw = F.sw;fig, axes = method_axes(2, lead=True);levels = np.linspace(.08, .92, 8)*B0*.3
    first = sw.field(test, sweep_data.FIRST_FROM, n, 0);x = (np.arange(n) + .5)/n
    for i, (_, _, methods) in enumerate(GROUPS):
        ax = axes[i, 0];image_axes(ax)
        if i == 0 and first is not None:
            a = potential(first);ax.contour(x, x, (a - a.min()).T, levels=levels, colors=INK, linewidths=.6, linestyles="solid");ax.set_aspect("equal");ax.set_title("initial", fontsize=8.5, pad=4)
            ax.set_xlim(0, 1);ax.set_ylim(0, 1)
        else:ax.set_visible(False)
        for j, m in enumerate(methods):
            ax = axes[i, j + 1];image_axes(ax);ax.set_title(label(m, True), fontsize=8.5, pad=4)
            u = sw.field(test, m, n, frame)
            if u is None:blank(ax, F.why(test, m, n));F.miss(test, m, n);continue
            a = potential(u);ax.contour(x, x, (a - a.min()).T, levels=levels, colors=INK, linewidths=.6, linestyles="solid");ax.set_aspect("equal")
            ax.set_xlim(0, 1);ax.set_ylim(0, 1)
    return fig


def power_law(t, e):
    """Fit e/e0 = 1 - (t/tau)^alpha (the papers' form); returns (alpha, tau) or None."""
    loss = 1 - e/e[0];ok = (t > 0) & (loss > 0)
    if ok.sum() < 3:return None
    alpha, c = np.polyfit(np.log(t[ok]), np.log(loss[ok]), 1)
    return float(alpha), float(np.exp(-c/alpha))


def loop_energy(F, test="field_loop_novz", grids=(32, 64, 128)):
    """Fig. 7: volume-averaged magnetic energy of the loop against time."""
    fig, axes = plt.subplots(2, len(grids), figsize=(3.5*len(grids) + 2.4, 5.6), sharex=True, sharey=True, gridspec_kw=dict(wspace=.07, hspace=.12))
    fits = []
    for row, (_, _, methods) in enumerate(GROUPS):
        for col, n in enumerate(grids):
            ax = axes[row, col];style_axes(ax)
            for m in methods:
                s = F.series(test, m, n, "magnetic")
                if s is None:F.miss(test, m, n);continue
                t, e = s;ax.plot(t, e/e[0], **line_kw(m))
                if F.stopped(test, m, n):ax.plot(t[-1], e[-1]/e[0], marker="x", color=color(m), ms=6, mew=1.5)
                fit = power_law(t, e)
                fits.append(dict(key=m, method=plain(label(m)), n=n, final=float(e[-1]/e[0]), alpha=fit and fit[0], tau=fit and fit[1],
                                 ended=F.stopped(test, m, n)))
            if n == 64:
                t = np.linspace(0, float(F.sw.configs[test]["t_end"]), 200);ax.plot(t, fit_curve(t, FIT_2005), color=REFERENCE_INK, lw=1.1, ls=(0, (4, 2)))
            if row == 0:ax.set_title(f"N = {n}")
            if row == 1:ax.set_xlabel("t")
        hs = handles(methods) + [Line2D([], [], color=REFERENCE_INK, lw=1.1, ls=(0, (4, 2)), label="Gardiner & Stone (2005): their fit, N = 64")]
        axes[row, -1].legend(handles=hs, loc="center left", bbox_to_anchor=(1.02, .5), handlelength=2.6)
    fig.supylabel(r"magnetic energy $\langle B^2\rangle$ / initial", fontsize=9.5, x=.045)
    return fig, fits


def loop_bz(F, test="field_loop", grids=(32, 64, 128)):
    """Out-of-plane field of the in-plane loop advected with v_z != 0 (Gardiner & Stone 2008, sec. 3)."""
    fig, axes = plt.subplots(2, len(grids), figsize=(3.5*len(grids) + 2.4, 5.6), sharex=True, sharey="row", gridspec_kw=dict(wspace=.07, hspace=.12))
    final = {}                                         # {method: rms B_z/B0 at the end, finest grid shown}
    for row, (_, _, methods) in enumerate(GROUPS):
        for col, n in enumerate(grids):
            ax = axes[row, col];style_axes(ax)
            for m in methods:
                s = F.series(test, m, n, "bz_rms")
                if s is None:F.miss(test, m, n);continue
                t, b = s;keep = b > 0
                if keep.any():ax.plot(t[keep], b[keep]/B0, **line_kw(m))
                else:ax.plot([], [], **line_kw(m))
                if n == grids[-1] and len(b) and not F.stopped(test, m, n):final[m] = float(b[-1]/B0)
            if row == 0:ax.set_title(f"N = {n}")
            if row == 1:ax.set_xlabel("t")
        axes[row, -1].legend(handles=handles(methods), loc="center left", bbox_to_anchor=(1.02, .5), handlelength=2.6)
        axes[row, 0].set_yscale("log")
    fig.supylabel(r"rms $B_z$ / $B_0$   (exact solution: 0; each row has its own scale)", fontsize=9.5, x=.045)
    return fig, final


# ---------------------------------------------------------------- 2005: waves, shock tube, convergence
def phase_mean(values, phase_index, count):
    """Mean, minimum and maximum of a field over the cells that share a phase index."""
    v = values.ravel();i = phase_index.ravel()
    mean = np.bincount(i, weights=v, minlength=count)/np.maximum(np.bincount(i, minlength=count), 1)
    lo = np.full(count, np.inf);hi = np.full(count, -np.inf);np.minimum.at(lo, i, v);np.maximum.at(hi, i, v)
    return mean, lo, hi


def alfven_profiles(F, test="alfven_oblique", grids=(32, 64, 128)):
    """Fig. 9: transverse in-plane field of the oblique circularly polarized Alfven wave after one period."""
    sw = F.sw;fig, axes = plt.subplots(2, 6, figsize=(12.6, 4.4), sharex=True, sharey=True, gridspec_kw=dict(wspace=.06, hspace=.2))
    s5 = np.sqrt(5.0);xi = np.linspace(0, 1, 400)
    for row, (_, _, methods) in enumerate(GROUPS):
        for col, m in enumerate(methods):
            ax = axes[row, col];style_axes(ax);ax.set_title(label(m, True), fontsize=8.5, pad=3)
            ax.plot(xi, .1*np.cos(2*np.pi*xi), color=REFERENCE_INK, lw=.9)
            for c, n in zip(shades(m, len(grids)), grids):
                u = sw.field(test, m, n, "end")
                if u is None:F.miss(test, m, n);continue
                i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij");idx = (2*i + 4*j + 3) % (2*n)       # phase (x + 2y) in units of 1/(2n)
                b2 = (-2*u[..., 5].astype(float) + u[..., 6].astype(float))/s5
                mean, lo, hi = phase_mean(b2, idx, 2*n);have = np.isfinite(lo)
                ax.plot((np.arange(2*n)[have] + 0.0)/(2*n), mean[have], color=c, lw=1.2)
            if row == 1:ax.set_xlabel(r"phase $(x + 2y)$ mod 1")
    axes[0, 0].set_ylim(-.115, .115)
    fig.supylabel(r"in-plane field across the wave, $B_2$", fontsize=9.5, x=.065)
    hs = [Line2D([], [], color=str(.75 - .6*i/(max(1, len(grids) - 1))), lw=1.4, label=f"N = {n}") for i, n in enumerate(grids)]
    hs.append(Line2D([], [], color=REFERENCE_INK, lw=.9, label="exact (the initial wave)"))
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(.5, 1.03), handlelength=2.2)
    return fig


def normal_profile(u, gamma):
    """Brio-Wu rotated by 45 degrees: per-diagonal mean, min, max of (rho, p, v_n, v_t, B_n, B_t) against (x + y) mod 1."""
    n = u.shape[0];i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij");idx = (i + j + 1) % n
    rho = u[..., 0].astype(float);r2 = np.sqrt(2.0)
    q = dict(rho=rho, p=pressure(u, gamma), vn=(u[..., 1] + u[..., 2])/rho/r2, vt=(u[..., 2] - u[..., 1])/rho/r2,
             bn=(u[..., 5].astype(float) + u[..., 6])/r2, bt=(u[..., 6].astype(float) - u[..., 5])/r2)
    return (np.arange(n) + 0.0)/n, {k: phase_mean(np.asarray(v, float), idx, n) for k, v in q.items()}


def shock_tube_bn(F, test="brio_wu_rotated", n=256):
    """Fig. 10: the field component along the shock normal, which should stay constant."""
    sw = F.sw;gamma = gamma_of(sw, test);worst = {}
    fig, axes = plt.subplots(2, 6, figsize=(12.6, 4.4), sharex=True, sharey=True, gridspec_kw=dict(wspace=.06, hspace=.2))
    for row, (_, _, methods) in enumerate(GROUPS):
        for col, m in enumerate(methods):
            ax = axes[row, col];style_axes(ax);ax.set_title(label(m, True), fontsize=8.5, pad=3)
            ax.axhline(.75, color=REFERENCE_INK, lw=.8)
            u = sw.field(test, m, n, "end")
            if u is None:
                F.miss(test, m, n);ax.text(.5, .5, F.why(test, m, n), transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=7.5);continue
            xi, q = normal_profile(u, gamma);mean, lo, hi = q["bn"]
            ax.fill_between(xi, lo, hi, color=color(m), alpha=.3, lw=0);ax.plot(xi, mean, color=color(m), lw=1.0)
            worst[m] = float(max(abs(lo.min() - .75), abs(hi.max() - .75)))
            if row == 1:ax.set_xlabel(r"$(x + y)$ mod 1")
    fig.supylabel(r"field along the normal, $B_1$   (exact: 0.75)", fontsize=9.5, x=.065)
    return fig, worst


def riemann_profiles(F, test="brio_wu_rotated", n=256):
    """2008 Fig. 7: every variable of the Riemann problem inclined to the grid, against the normal coordinate."""
    sw = F.sw;gamma = gamma_of(sw, test)
    names = (("rho", r"density"), ("p", r"gas pressure"), ("vn", r"velocity along the normal"), ("vt", r"transverse velocity"),
             ("bn", r"field along the normal"), ("bt", r"transverse field"))
    fig, axes = plt.subplots(2, len(names), figsize=(2.55*len(names) + .6, 5.4), sharex=True, sharey="col", gridspec_kw=dict(wspace=.3, hspace=.14))
    ref = None
    for rm in sweep_data.REFERENCES:
        r = sw.ref_field(test, rm, n, "end")
        if r is not None:ref = (rm, normal_profile(r, gamma));break
    if ref is None:
        for rm in sweep_data.REFERENCES:F.miss(test, rm, sw.reference_n(test, rm))
    for row, (_, _, methods) in enumerate(GROUPS):
        profiles = {}
        for m in methods:
            u = sw.field(test, m, n, "end")
            if u is None:F.miss(test, m, n);continue
            profiles[m] = normal_profile(u, gamma)
        for col, (key, name) in enumerate(names):
            ax = axes[row, col];style_axes(ax)
            if ref is not None:ax.plot(ref[1][0], ref[1][1][key][0], color=REFERENCE_INK, lw=2.0, alpha=.5)
            for m, (xi, q) in profiles.items():ax.plot(xi, q[key][0], **line_kw(m, lw=1.0))
            ax.set_xlim(.25, .75)
            if row == 0:ax.set_title(name, fontsize=8.5)
            if row == 1:ax.set_xlabel(r"$(x + y)$ mod 1")
        hs = handles(methods)
        if ref is not None:hs.append(Line2D([], [], color=REFERENCE_INK, lw=2.0, alpha=.5, label=f"{label(ref[0])} ({sw.reference_n(test, ref[0])}$^2$)"))
        axes[row, -1].legend(handles=hs, loc="center left", bbox_to_anchor=(1.04, .5), handlelength=2.6)
    return fig


def error_norm(meta):
    """Norm of the four L1 errors (density, momentum, energy, field) at the last output, or None."""
    if not meta or meta.get("status") != "complete" or not meta.get("frames"):return None
    e = meta["frames"][-1].get("errors") or {}
    keys = [f"{q}_absolute_L1" for q in ("density", "momentum", "energy", "magnetic")]
    if not all(k in e for k in keys):return None
    return float(np.sqrt(sum(float(e[k])**2 for k in keys)))


def convergence(F, tests=("entropy", "alfven", "alfven_oblique"), names=("entropy wave", "Alfvén wave along x", "Alfvén wave along (1, 2)")):
    """Fig. 11: L1 error against resolution for the smooth wave problems."""
    sw = F.sw;fig, axes = plt.subplots(2, len(tests), figsize=(3.5*len(tests) + 2.4, 6.0), sharex=True, sharey="col", gridspec_kw=dict(wspace=.22, hspace=.1))
    orders = []
    for col, (test, name) in enumerate(zip(tests, names)):
        for row, (_, _, methods) in enumerate(GROUPS):
            ax = axes[row, col];style_axes(ax);top = 0.0
            for m in methods:
                pts = [(n, error_norm(sw.meta(test, m, n))) for n in sw.grids(test, m)]
                for n, e in pts:
                    if e is None:F.miss(test, m, n)
                pts = [(n, e) for n, e in pts if e is not None and e > 0]
                if not pts:continue
                ns, es = map(np.array, zip(*pts));ax.loglog(ns, es, marker=STYLE[m][4], ms=3.5, **line_kw(m, lw=1.2));top = max(top, es[0])
                if len(ns) >= 2:orders.append(dict(test=name, key=m, method=plain(label(m)), order=float(-np.polyfit(np.log(ns), np.log(es), 1)[0]),
                                                   last=float(-np.log(es[-1]/es[-2])/np.log(ns[-1]/ns[-2])), finest=float(es[-1]), n=int(ns[-1])))
            if top:
                ns = np.array([32, 512.0])
                for p, ls in ((1, (0, (1, 2))), (2, (0, (4, 2)))):ax.loglog(ns, top*1.6*(32/ns)**p, color=MUTED, lw=.8, ls=ls)
            if row == 0:ax.set_title(name)
            if row == 1:ax.set_xlabel("N")
            ax.set_xticks([32, 64, 128, 256, 512]);ax.set_xticklabels(["32", "64", "128", "256", "512"]);ax.minorticks_off()
    for row, (_, _, methods) in enumerate(GROUPS):
        hs = handles(methods, markers=True) + [Line2D([], [], color=MUTED, lw=.8, ls=(0, (1, 2)), label=r"$\propto N^{-1}$"),
                                               Line2D([], [], color=MUTED, lw=.8, ls=(0, (4, 2)), label=r"$\propto N^{-2}$")]
        axes[row, -1].legend(handles=hs, loc="center left", bbox_to_anchor=(1.02, .5), handlelength=2.6)
    fig.supylabel("norm of the L1 errors after one period", fontsize=9.5, x=.04)
    return fig, orders


# ---------------------------------------------------------------- 2005: current sheet, blast
def current_sheet(F, group, test="current_sheet", n=256, frames=(4, 8, 12, 16, 20)):
    """Fig. 12: field lines of the current sheet problem in time."""
    sw = F.sw;methods = dict((g, ms) for g, _, ms in GROUPS)[group];times = sw.times(test)
    fig, axes = plt.subplots(len(methods), len(frames), figsize=(1.62*len(frames) + .8, 1.68*len(methods) + .4), squeeze=False,
                             gridspec_kw=dict(wspace=.05, hspace=.06))
    x = (np.arange(n) + .5)/n;levels = np.linspace(-.25, .25, 31)        # A_z of the initial sheets spans -1/4 .. 1/4
    for i, m in enumerate(methods):
        for j, f in enumerate(frames):
            ax = axes[i, j];image_axes(ax)
            if i == 0:ax.set_title(f"t = {times[f]:.1f}  (paper: {2*times[f]:.0f})", fontsize=8.5, pad=4)
            u = sw.field(test, m, n, f)
            if u is None:
                blank(ax, F.why(test, m, n))
                if j == 0:F.miss(test, m, n)
                continue
            a = potential(u);ax.contour(x, x, (a - a.mean()).T, levels=levels, colors=INK, linewidths=.4, linestyles="solid");ax.set_aspect("equal")
            ax.set_xlim(0, 1);ax.set_ylim(0, 1)
        axes[i, 0].set_ylabel(label(m, True), fontsize=9, color=INK)
    return fig


def image_rows(F, group, test, n, frame, kind, rows, ranges):
    """rows of images (one per quantity) x the six methods of a group; ranges: {key: (lo, hi)} shared by both groups."""
    sw = F.sw;methods = dict((g, ms) for g, _, ms in GROUPS)[group]
    fig, axes = method_axes(len(rows));x = (np.arange(n) + .5)/n
    for j, m in enumerate(methods):
        u = sw.field(test, m, n, frame, kind=kind)
        for i, (key, name, f) in enumerate(rows):
            ax = axes[i, j];image_axes(ax)
            if i == 0:ax.set_title(label(m, True), fontsize=8.5, pad=4)
            if u is None:
                blank(ax, F.why(test, m, n))
                if i == 0:F.miss(test, m, n)
                continue
            if key == "lines":
                a = potential(u);ax.contour(x, x, a.T, levels=30, colors=INK, linewidths=.4, linestyles="solid");ax.set_aspect("equal");ax.set_xlim(0, 1);ax.set_ylim(0, 1)
            else:
                lo, hi = ranges[key];show(ax, f(u), cmap=map_of(key), vmin=lo, vmax=hi)
    for i, (key, name, f) in enumerate(rows):
        lo, hi = ranges.get(key, (0, 0));lo = 0.0 if abs(lo) < 1e-6*abs(hi) else lo
        text = name if key == "lines" else f"{name}\n{lo:.3g} (dark) – {hi:.3g} (light)"
        axes[i, 0].set_ylabel(text, fontsize=8, color=INK)
    return fig


def shared_ranges(F, test, n, frame, kind, rows, q=(.1, 99.9)):
    """Colour range of each quantity: over every method that has the frame, the extremes of its q-th percentiles
    (a few cells at a floor or a spike would otherwise set the scale for every panel)."""
    out = {}
    for key, name, f in rows:
        if key == "lines":continue
        lo, hi = np.inf, -np.inf
        for m in ALL:
            u = F.sw.field(test, m, n, frame, kind=kind)
            if u is None:continue
            a, b = np.nanpercentile(f(u), q);lo, hi = min(lo, float(a)), max(hi, float(b))
        lo = max(lo, 0.0)                              # density, pressure and energies: a solver's negative cells do not set the scale
        out[key] = (lo, hi) if np.isfinite(lo) and hi > lo else (0.0, 1.0)
    return out


def disturbed_outside(F, test, n, kind, radius=.4, change=.05):
    """{method: share of the cells further than `radius` from the centre whose density differs from the ambient 1 by more than `change`}."""
    x = (np.arange(n) + .5)/n;X, Y = np.meshgrid(x, x, indexing="ij");outside = np.hypot(X - .5, Y - .5) > radius;out = {}
    for m in ALL:
        u = F.sw.field(test, m, n, "end", kind=kind)
        if u is not None:out[m] = float(np.mean(np.abs(u[..., 0][outside] - 1) > change))
    return out


def blast_rows(sw, test, kinetic):
    g = gamma_of(sw, test)
    rows = [("rho", "density", lambda u: u[..., 0]), ("p", "gas pressure", lambda u: pressure(u, g)),
            ("pb", "magnetic energy density", lambda u: bmag(u)**2/2)]
    if kinetic:rows.append(("ke", "kinetic energy density", lambda u: (u[..., 1:4].astype(float)**2).sum(-1)/(2*u[..., 0])))
    else:rows.append(("lines", "field lines", None))
    return rows


# ---------------------------------------------------------------- 2008: 3D field loop
def loop3d_history(F, test="field_loop_3d", grids=(32, 64, 128)):
    """2008 Fig. 1: magnetic energy and the field component along the loop axis against time, by resolution."""
    sw = F.sw;fig, axes = plt.subplots(4, 6, figsize=(12.6, 8.2), sharex=True, gridspec_kw=dict(wspace=.06, hspace=.16))
    r2 = np.sqrt(2.0);times = sw.times(test);energy, axial = {}, {}      # at the end, finest grid shown
    for g, (_, _, methods) in enumerate(GROUPS):
        for col, m in enumerate(methods):
            a, b = axes[g, col], axes[2 + g, col];style_axes(a);style_axes(b);a.set_title(label(m, True), fontsize=8.5, pad=3)
            for i, n in enumerate(grids):                      # the paper's fits for its own scheme, same cell sizes
                if n in FIT_2008:a.plot(times, fit_curve(times, FIT_2008[n]), color=str(.72 - .5*i/max(1, len(grids) - 1)), lw=.9, ls=(0, (1, 1.2)))
            for c, n in zip(shades(m, len(grids)), grids):
                s = F.series(test, m, n, "magnetic")
                if s is None:F.miss(test, m, n)
                else:
                    a.plot(s[0], s[1]/s[1][0], color=c, lw=1.2)
                    if n == grids[-1] and not F.stopped(test, m, n):energy[m] = float(s[1][-1]/s[1][0])
                pts = []
                for f in sw.slice_frames(test):
                    u = sw.field(test, m, n, f, kind="slice")
                    if u is not None:pts.append((times[f], np.abs((u[..., 7].astype(float) - u[..., 5])/r2).mean()/B0))
                if pts:
                    t, v = map(np.array, zip(*pts));keep = v > 0;b.plot(t[keep], v[keep], color=c, lw=1.2, marker="o", ms=2.5)
                    if n == grids[-1] and t[-1] == times[-1]:axial[m] = float(v[-1])
            if col:a.tick_params(labelleft=False);b.tick_params(labelleft=False, which="both")
    for row in (0, 1):
        lo = min(ax.get_ylim()[0] for ax in axes[:2].flat);hi = max(ax.get_ylim()[1] for ax in axes[:2].flat)
        for ax in axes[row]:ax.set_ylim(lo, hi)
    for ax in axes[2:].flat:ax.set_yscale("log")
    los = [ax.get_ylim() for ax in axes[2:].flat if ax.lines]
    if los:
        lo, hi = min(l[0] for l in los), max(l[1] for l in los)
        ticks = [v for v in (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2) if lo <= v <= hi]
        for ax in axes[2:].flat:
            ax.set_ylim(lo, hi);ax.set_yticks(ticks);ax.set_yticklabels([f"{v:g}" for v in ticks]);ax.minorticks_off()
    for ax in axes[2:, 1:].flat:ax.tick_params(labelleft=False, which="both")
    for ax in axes[-1]:ax.set_xlabel("t")
    axes[0, 0].set_ylabel(r"$\langle B^2\rangle$ / initial");axes[1, 0].set_ylabel(r"$\langle B^2\rangle$ / initial")
    axes[2, 0].set_ylabel(r"$\langle|B_3|\rangle/B_0$");axes[3, 0].set_ylabel(r"$\langle|B_3|\rangle/B_0$")
    hs = [Line2D([], [], color=str(.75 - .6*i/(max(1, len(grids) - 1))), lw=1.4, label=f"N = {n}") for i, n in enumerate(grids)]
    hs.append(Line2D([], [], color=".45", lw=.9, ls=(0, (1, 1.2)), label="Gardiner & Stone (2008): their fits for the same N (energy)"))
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(.5, .955), handlelength=2.2)
    return fig, energy, axial


def figures():
    P5, P8 = "stone2005", "stone2008"
    LOOP = ("Same loop (B₀ = 10⁻³, R = 0.3, v = (2, 1)) and the same cell size as the paper's 2N × N grid on a 2 × 1 box, on the unit square: "
            "four crossings in x and two in y by t = 2.")

    def loop_fig(group):
        def run(F):
            F.save(loop_images(F, group), f"stone05-loop-{group}", paper=P5, paper_fig="Figs. 2, 3, 6", kind="reproduced", test="field_loop_novz",
                   title=f"Field loop advection: magnetic pressure at t = 0.2 and t = 2, N = 64 ({'GLM kernels and CT prototype' if group == 'gpu' else 'other codes'})",
                   caption="B_x² + B_y² in units of the loop's B₀² (dark 0, light 1) early (the paper's t = 0.19) and after the loop has returned to the centre.",
                   differs=LOOP)
        return run

    def loop_energy_fig(F):
        fig, fits = loop_energy(F)
        left = by_family({r["key"]: r["final"] for r in fits if r["n"] == 64 and not r["ended"]})
        finding = (f"Energy left at t = 2 on the paper's grid (N = 64): CT prototype {spread(left['ct'], two)}; GLM kernels "
                   f"{spread(left['glm'], two)}; AthenaK, RAMSES and Athena++ {spread(left['other'], two)}. The paper's own scheme: "
                   f"{fit_curve(2.0, FIT_2005):.2f}.")
        F.save(fig, "stone05-loop-energy", paper=P5, paper_fig="Fig. 7", kind="reproduced", test="field_loop_novz", finding=finding,
               title="Field loop advection: magnetic energy against time",
               caption=("Volume-averaged magnetic energy over its initial value. Dashed grey in the N = 64 panels: the curve the paper fits to its "
                        "own result at this cell size, 1 − (t/τ)^α with α = 0.291, τ = 1.06×10⁴ (0.92 left at t = 2). The table gives the same fit "
                        "for each method."),
               differs=LOOP, table=dict(columns=["Method", "N", "energy left at t = 2", "α", "τ"], keys=["method", "n", "final", "alpha", "tau"], rows=fits))

    def loop_lines_fig(F):
        F.save(loop_lines(F), "stone05-loop-lines", paper=P5, paper_fig="Fig. 8", kind="reproduced", test="field_loop_novz",
               title="Field loop advection: field lines at t = 2, N = 64", caption="Contours of A_z (computed from the cell-centred field), same levels in every panel.",
               differs=LOOP)

    def loop_bz_fig(F):
        fig, final = loop_bz(F);fam = by_family(final)
        finding = (f"rms B_z/B₀ at t = 2 and N = 128: CT prototype {spread(fam['ct'])}; GLM kernels {spread(fam['glm'])}; RAMSES "
                   f"{spread(fam['ramses'])}; AthenaK and Athena++ at most {sci(max(fam['athena']))} (rounding)." if fam["athena"] and fam["ct"] else "")
        F.save(fig, "stone08-loop-bz", paper=P8, paper_fig="sec. 3 (no figure)", kind="extension", test="field_loop", finding=finding,
               title="In-plane field loop advected with v_z ≠ 0: growth of the out-of-plane field",
               caption=("The paper's thought experiment: a loop in the (x, y) plane carried with v = (2, 1, √5). B_z must stay zero; schemes that "
                        "do not keep ∇·B = 0 generate it. A method that keeps B_z at exactly zero has no curve on this logarithmic axis."),
               differs="The paper describes this test and reports that its scheme keeps B_z = 0; it shows no figure for it.")

    def alfven_fig(F):
        F.save(alfven_profiles(F), "stone05-alfven", paper=P5, paper_fig="Fig. 9", kind="analogue", test="alfven_oblique",
               title="Circularly polarized Alfvén wave oblique to the grid: profile after one period",
               caption="In-plane field component across the wave against phase, averaged over cells of equal phase, for three grids (light to dark).",
               differs=("Travelling wave only, along (1, 2) on the unit square (the paper's angle) after one period; the paper shows five periods on "
                        "2N × N grids with N = 4 … 64, where the damping is much larger."))

    def shock_fig(F):
        fig, worst = shock_tube_bn(F);fam = by_family(worst)
        finding = (f"Largest departure of B₁ from 0.75: GLM kernels {spread(fam['glm'])}; CT prototype {spread(fam['ct'])}; AthenaK, RAMSES and "
                   f"Athena++ at most {sci(max(fam['other']))}, the precision of the saved states." if fam["other"] and fam["glm"] else "")
        F.save(fig, "stone05-shock-bn", paper=P5, paper_fig="Fig. 10", kind="analogue", test="brio_wu_rotated", finding=finding,
               title="Rotated shock tube: the field component along the shock normal, N = 256",
               caption="Mean over each diagonal (line) and the spread across its cells (band). The exact solution keeps it at 0.75.",
               differs="Brio–Wu states rotated by 45°, periodic (two mirrored Riemann problems); the paper rotates the Ryu–Jones 2a problem by tan⁻¹(1/2).")

    def convergence_fig(F):
        fig, orders = convergence(F);stalled = {}
        for r in orders:
            if r["n"] == 512 and r["last"] < .5:stalled.setdefault(r["test"], []).append(f'{plain(label(r["key"], True))} ({sci(r["finest"])})')
        finding = ("Between N = 256 and 512 the error no longer falls (order below 0.5; error at 512 in brackets) for: " +
                   "; ".join(f"{t}: {', '.join(ms)}" for t, ms in stalled.items()) + ".") if stalled else ""
        F.save(fig, "stone05-convergence", paper=P5, paper_fig="Fig. 11", kind="analogue", test="alfven", finding=finding,
               title="Convergence on smooth waves: L1 error against resolution",
               caption="Norm of the L1 errors of density, momentum, energy and field after one period. Dotted and dashed: first and second order.",
               differs=("Finite-amplitude waves with exact solutions (entropy wave, circularly polarized Alfvén wave, aligned and oblique) instead of "
                        "the paper's four linear modes of amplitude 10⁻⁶."),
               table=dict(columns=["Test", "Method", "order (fit over all grids)", "order between the two finest grids", "error at finest grid", "finest N"],
                          keys=["test", "method", "order", "last", "finest", "n"], rows=orders))

    def sheet_fig(group):
        def run(F):
            F.save(current_sheet(F, group), f"stone05-sheet-{group}", paper=P5, paper_fig="Fig. 12", kind="reproduced", test="current_sheet",
                   title=f"Current sheet: field lines in time, 256² ({'GLM kernels and CT prototype' if group == 'gpu' else 'other codes'})",
                   caption="Contours of A_z at the same levels in every panel. Reconnection here is numerical: more islands, later, means less resistivity.",
                   differs="The paper's [0, 2]² box halved to the unit square, so our t is half the paper's; same 256² cells, β = 0.2, v₀ = 0.1.")
        return run

    def blast_fig(group, test, n, kind, ident, paper, fig_no, title):
        def run(F):
            rows = blast_rows(F.sw, test, kinetic=kind == "slice");ranges = shared_ranges(F, test, n, "end", kind, rows)
            share = disturbed_outside(F, test, n, kind);mine = dict((g, ms) for g, _, ms in GROUPS)[group]
            hit = [f"{plain(label(m, True))} has changed the density by more than 5 % in {100*v:.0f} % of the cells" for m, v in share.items()
                   if m in mine and v > .001]
            clean = [plain(label(m, True)) for m in mine if share.get(m, 1) <= .001]
            finding = ("Outside the blast (further than 0.4 from the centre) " + "; ".join(hit) + "; " + " and ".join(clean) + " in none.") if hit else ""
            F.save(image_rows(F, group, test, n, "end", kind, rows, ranges), f"{ident}-{group}", paper=paper, paper_fig=fig_no, kind="analogue", test=test,
                   finding=finding,
                   title=f"{title} ({'GLM kernels and CT prototype' if group == 'gpu' else 'other codes'})",
                   caption="Linear colour scale, the same range in every panel of a row (given at the left); gas quantities and field quantities each have their own map. The paper's figures are grey.",
                   differs=("A stronger blast than the paper's: pressure 1000 inside r = 0.1 against 0.1 outside (the paper: 100 against 1, r = 0.125), "
                            "field along x with B = 50/√4π, γ = 7/5, to t = 0.01, periodic box." + (" Slice through the centre containing the field." if kind == "slice" else "")))
        return run

    def loop3d_fig(F):
        fig, energy, axial = loop3d_history(F);e, a = by_family(energy), by_family(axial)
        finding = (f"At 128³ and t = 1, energy left: CT prototype {spread(e['ct'], two)}; GLM kernels {spread(e['glm'], two)}; AthenaK, RAMSES "
                   f"and Athena++ {spread(e['other'], two)}; the paper's own scheme {fit_curve(1.0, FIT_2008[128]):.2f}. Mean |B₃|/B₀: CT prototype "
                   f"{spread(a['ct'])}; GLM kernels {spread(a['glm'])}; the other codes {spread(a['other'])}." if e["other"] and a["other"] else "")
        F.save(fig, "stone08-loop3d", paper=P8, paper_fig="Fig. 1", kind="reproduced", test="field_loop_3d", finding=finding,
               title="3D field loop: magnetic energy and the field along the loop axis against time",
               caption=("Top two rows: volume-averaged magnetic energy over its initial value; dotted grey: the paper's fits for its own scheme "
                        "at N = 32, 64, 128 (τ = 322, 3.68×10³, 2.65×10⁴; α = 0.365, 0.328, 0.320). Bottom two: mean |B₃|/B₀, the component along "
                        "the loop axis, which should stay zero; the paper reports that this error converges at about first order. One panel per "
                        "method, one line per grid (light to dark)."),
               differs=("Loop axis (−1, 0, 1)/√2 in the unit cube with v = (1, 1, 2), instead of the paper's 1 × 1 × 2 box with the loop along its "
                        "diagonal. ⟨|B₃|⟩ is taken over the z midplane at six outputs; by the loop's symmetry every plane is a shifted copy."))

    def loop3d_images(group):
        def run(F):
            r2 = np.sqrt(2.0);test, n = "field_loop_3d", 128
            rows = [("b2", r"$B^2/B_0^2$", lambda u: bmag(u)**2/B0**2), ("b3", r"$B_3^2/B_0^2$ (along the loop axis)", lambda u: ((u[..., 7].astype(float) - u[..., 5])/r2)**2/B0**2)]
            ranges = shared_ranges(F, test, n, "end", "slice", rows);ranges["b2"] = (0.0, max(1.05, ranges["b2"][1]))
            F.save(image_rows(F, group, test, n, "end", "slice", rows, ranges), f"stone08-loop3d-images-{group}", paper=P8, paper_fig="Fig. 2", kind="analogue",
                   test=test, title=f"3D field loop at t = 1, 128³: magnetic energy and its part along the loop axis ({'GLM kernels and CT prototype' if group == 'gpu' else 'other codes'})",
                   caption="The z midplane, which cuts the inclined loop in an ellipse.",
                   differs="The paper shows thresholded 3D renderings; these are plane cuts with a linear colour scale.")
        return run

    def riemann_fig(F):
        F.save(riemann_profiles(F), "stone08-riemann", paper=P8, paper_fig="Fig. 7", kind="analogue", test="brio_wu_rotated",
               title="Riemann problem inclined to the grid: every variable against the normal coordinate, N = 256",
               caption="Means over each diagonal. Top: GLM kernels and CT prototype; bottom: other codes; grey: the 4096² reference averaged to 256².",
               differs="Brio–Wu states at 45° in 2D (γ = 2) instead of the paper's Ryu–Jones 2a problem inclined in 3D.")

    out = {"stone05-loop-gpu": loop_fig("gpu"), "stone05-loop-other": loop_fig("other"), "stone05-loop-energy": loop_energy_fig,
           "stone05-loop-lines": loop_lines_fig, "stone05-alfven": alfven_fig, "stone05-shock-bn": shock_fig, "stone05-convergence": convergence_fig,
           "stone05-sheet-gpu": sheet_fig("gpu"), "stone05-sheet-other": sheet_fig("other")}
    for group in ("gpu", "other"):
        out[f"stone05-blast-{group}"] = blast_fig(group, "blast_2d_b50", 256, "state", "stone05-blast", P5, "Fig. 13",
                                                  "MHD blast wave, 256²: density, pressure, magnetic energy and field lines at the end")
    out.update({"stone08-loop3d": loop3d_fig, "stone08-loop3d-images-gpu": loop3d_images("gpu"), "stone08-loop3d-images-other": loop3d_images("other"),
                "stone08-loop-bz": loop_bz_fig, "stone08-riemann": riemann_fig})
    for group in ("gpu", "other"):
        out[f"stone08-blast-{group}"] = blast_fig(group, "blast_3d_b50", 256, "slice", "stone08-blast", P8, "Fig. 8",
                                                  "3D MHD blast wave, 256³: density, pressure, magnetic and kinetic energy at the end")
    return out
