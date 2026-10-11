"""Birn et al. (2001, JGR 106, 3715), the GEM magnetic reconnection challenge: their Fig. 1, the reconnected flux
against time, with our codes.

The challenge compares resistive, Hall and kinetic models; its MHD runs use a resistivity eta = 0.005. Ours are
ideal MHD (no explicit resistivity: stage 2 of the sweep adds it), so reconnection here is each code's numerical
resistivity and the flux falls with resolution. Geometry: the paper's 25.6 x 12.8 d_i box with conducting walls
becomes a periodic double sheet in the unit square (each half box is 25.6 x 12.8).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_common import ALL, GROUPS, MUTED, REFERENCE_INK, by_family, color, handles, label, line_kw, spread, style_axes, two

PAPER = "gem"
TEST = "gem"


def flux(F, m, n):
    """(Omega_i t, reconnected flux in B0 d_i): the mean of the two sheets."""
    d = F.sw.diag(TEST, m, n)
    if d is None or "recon_flux_lower" not in d:return None
    c = F.sw.configs[TEST];lx = float(c["gem_lx"])
    return d["time"]*float(c.get("display_time_scale", lx)), .5*(d["recon_flux_lower"] + d["recon_flux_upper"])*lx


def flux_figure(F, grids=(64, 128, 256, 512)):
    fig, axes = plt.subplots(2, len(grids), figsize=(3.05*len(grids) + 2.6, 5.8), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=.07, hspace=.12))
    refs = []
    for rm, kw in (("athenak_reference", dict(lw=2.2, alpha=.5)), ("athenapp_reference", dict(lw=1.0, ls=(0, (4, 2))))):
        big = F.sw.reference_n(TEST, rm);s = flux(F, rm, big) if big else None
        if s is None:
            if big:F.miss(TEST, rm, big)
            continue
        refs.append((rm, big, s, kw))
    for row, (_, title, methods) in enumerate(GROUPS):
        for col, n in enumerate(grids):
            ax = axes[row, col];style_axes(ax)
            for rm, big, s, kw in refs:ax.plot(*s, color=REFERENCE_INK, solid_capstyle="round", **kw)
            for m in methods:
                s = flux(F, m, n)
                if s is None:F.miss(TEST, m, n);continue
                ax.plot(*s, **line_kw(m))
                if F.stopped(TEST, m, n):ax.plot(s[0][-1], s[1][-1], marker="x", color=color(m), ms=6, mew=1.5);F.miss(TEST, m, n)
            if row == 0:ax.set_title(f"{n} × {n} cells per unit box")
            if row == 1:ax.set_xlabel(r"$\Omega_i t$")
            ax.set_xlim(0, 40)
        hs = handles(methods)
        for rm, big, s, kw in refs:hs.append(Line2D([], [], color=REFERENCE_INK, label=f"{label(rm)} ({big}$^2$)", **kw))
        hs.append(Line2D([], [], color=MUTED, marker="x", ms=6, mew=1.5, lw=0, label="solver stopped here"))
        axes[row, -1].legend(handles=hs, loc="center left", bbox_to_anchor=(1.02, .5), handlelength=2.6)
    axes[0, 0].set_ylim(0, None)
    fig.supylabel(r"reconnected flux  [$B_0 d_i$]", fontsize=9.5, x=.055)
    return fig


def figures():
    def run(F):
        fig = flux_figure(F);n = 512;last = F.sw.frames(TEST) - 1
        end = {}
        for m in ALL:
            s = flux(F, m, n)
            if s is not None and len(s[0]) > last and not F.stopped(TEST, m, n):end[m] = float(s[1][last])
        fam = by_family(end);big = F.sw.reference_n(TEST, "athenak_reference");ref = flux(F, "athenak_reference", big) if big else None
        finding = ""
        if ref is not None and fam["other"]:
            finding = (f"Flux at Ω_i t = 40 on 512²: CT prototype {spread(fam['ct'], two)}; the GLM kernels that reach the end {spread(fam['glm'], two)}; "
                       f"AthenaK, RAMSES and Athena++ {spread(fam['other'], two)}. The {big}² reference stays at {ref[1][0]:.2f} until "
                       f"Ω_i t ≈ {ref[0][int((ref[1] > 1.1*ref[1][0]).argmax())]:.0f}, rises to {ref[1].max():.2f} and ends at {ref[1][-1]:.2f}; "
                       "without reconnection the flux would stay at 0.20.")
        F.save(fig, "gem-1", paper=PAPER, paper_fig="Fig. 1", kind="analogue", test=TEST, finding=finding,
               title="GEM reconnection challenge: reconnected flux against time",
               caption=("Reconnected flux (half the integral of |B_y| along a current sheet, mean of the two sheets) against time in inverse ion "
                        "cyclotron times, for each grid. The flux starts at 0.2 B₀ d_i, the size of the challenge's initial perturbation."),
               differs=("Ideal MHD: no Hall term and no explicit resistivity (the challenge's MHD runs use η = 0.005), so reconnection is numerical "
                        "and should vanish with resolution; the paper's Fig. 1 compares physical models. Periodic double sheet instead of "
                        "conducting walls. The resistive runs are stage 2 of the sweep."))
    return {"gem-1": run}
