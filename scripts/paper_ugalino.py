"""Ugalino et al. (2026, arXiv:2610.02308), "Towards robust simulations of the galactic dynamo: a cross-code
comparison between Eulerian and Lagrangian schemes": their Fig. 5, with our codes on our nearest test.

The paper runs an isolated Milky-Way-mass disc (AGORA initial conditions, self-gravity, cooling, a pressure floor,
40-80 pc cells, 7 Gyr) with RAMSES (constrained transport), ART (Dedner cleaning), AREPO (Powell) and SWIFT (Dedner and
Powell), and has no idealized tests. None of that can be run with uniform periodic boxes. What the sweep does have is
a flow that amplifies a weak field in every code: the Mach 9 decay (decay_supersonic). Fig. 5 of the paper shows, for
the disc, the magnetic-to-kinetic energy ratio, a dimensionless divergence error and the kinetic energy against
time; the same three panels are drawn here for the box.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_common import ALL, MUTED, by_family, color, handles, label, line_kw, sci, spread, style_axes, two

PAPER = "ugalino"
TEST = "decay_supersonic"


def histories(F, m, n):
    """time, magnetic/kinetic energy, kinetic energy, and the divergence measure of a run (None where it has none)."""
    k = F.series(TEST, m, n, "kinetic");b = F.series(TEST, m, n, "magnetic")
    if k is None or b is None:return None
    t = k[0];out = dict(t=t, ratio=b[1]/k[1], kinetic=k[1], div=None)
    if m.startswith("glm"):                          # the cleaning codes: centred differences of the cell-centred field
        d = F.series(TEST, m, n, "normalized_div_center_rms")
        if d is not None:out["div"] = d[1]
    elif m == "ct":                                  # the CT prototype: the face-centred divergence it keeps
        d = F.series(TEST, m, n, "div_face_rms")
        if d is not None:out["div"] = d[1]/(n*np.sqrt(2*b[1]))
    return out


def figure(F, n):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), gridspec_kw=dict(wspace=.27));drawn = [];end = {}
    for m in ALL:
        h = histories(F, m, n)
        if h is None:
            F.miss(TEST, m, n);continue
        drawn.append(m);stopped = F.stopped(TEST, m, n)
        for ax, key in zip(axes, ("ratio", "div", "kinetic")):
            y = h[key]
            if y is None:continue
            keep = np.isfinite(y) & (y > 0)
            ax.plot(h["t"][keep], y[keep], **line_kw(m))
            if stopped and keep.any():ax.plot(h["t"][keep][-1], y[keep][-1], marker="x", color=color(m), ms=6, mew=1.5)
        if stopped:F.miss(TEST, m, n)
        else:end[m] = dict(ratio=float(h["ratio"][-1]), div=None if h["div"] is None else float(h["div"][-1]), growth=float(h["ratio"][-1]/h["ratio"][0]))
    for ax, name in zip(axes, (r"magnetic / kinetic energy", r"divergence error  rms$(\nabla\cdot\mathbf{B})\,\Delta x$ / rms$(B)$", r"kinetic energy $\langle\rho u^2\rangle/2$")):
        style_axes(ax);ax.set_yscale("log");ax.set_xlabel("t");ax.set_ylabel(name);ax.set_xlim(0, float(F.sw.configs[TEST]["t_end"]))
    hs = handles([m for m in ALL if m in drawn]) + [Line2D([], [], color=MUTED, marker="x", ms=6, mew=1.5, lw=0, label="run ended here (solver stop or NaN)")]
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(.5, .93), handlelength=3.2, columnspacing=1.4)
    return fig, end


def finding(end, n):
    if not end:return ""
    ratio = by_family({m: v["ratio"] for m, v in end.items()});growth = [v["growth"] for v in end.values()]
    glm = [v["div"] for m, v in end.items() if m.startswith("glm") and v["div"]];ct = [v["div"] for m, v in end.items() if m == "ct" and v["div"]]
    allr = [v["ratio"] for v in end.values()]
    parts = [f"At {n}³ the magnetic-to-kinetic energy ratio grows {spread(growth, lambda x: f'{x:.0f}')}-fold by the end of the run and ends at "
             f"{spread(allr, two)} over the {len(end)} set-ups that finish (largest over smallest: {max(allr)/min(allr):.2f})"]
    fam = [f"{name} {spread(ratio[key], two)}" for key, name in (("glm", "GLM"), ("ct", "CT prototype"), ("athena", "AthenaK and Athena++"), ("ramses", "RAMSES")) if ratio[key]]
    parts.append("by code: " + ", ".join(fam))
    if glm:parts.append(f"the divergence measure of the GLM kernels ends at {spread(glm, sci)}" + (f", the CT prototype's face divergence at {spread(ct, sci)}" if ct else ""))
    return "; ".join(parts) + (". The paper's Fig. 5 has, for the disc, about 10⁻¹⁶ for RAMSES (CT), 10⁻² for ART (Dedner) and 10⁻¹ for AREPO and SWIFT, "
                               "and the two grid codes several orders of magnitude below equipartition where AREPO and SWIFT reach 10–20 %.")


def figures():
    def make(n):
        def run(F):
            fig, end = figure(F, n)
            F.save(fig, f"ugalino-5-{n}", paper=PAPER, paper_fig="Fig. 5", kind="analogue", test=TEST, finding=finding(end, n),
                   title=f"Field amplification in the Mach 9 decay at {n}³: energy ratio, divergence error and kinetic energy against time",
                   caption=("Left: volume-averaged magnetic over kinetic energy. Centre: a dimensionless divergence error, for the codes "
                            "that carry one: the GLM kernels (centred differences of their cell-centred field) and the CT prototype (the "
                            "face-centred divergence it preserves, at single-precision rounding). AthenaK, RAMSES and Athena++ keep their "
                            "face-centred divergence at rounding by construction; it is not in the saved cell-centred states and is not drawn. "
                            "Right: kinetic energy."),
                   differs=("The paper's Fig. 5 is for the gas disc of an isolated galaxy (radius 15 kpc, height 0.5 kpc) over 7 Gyr, with "
                            "self-gravity, cooling and a pressure floor, where shear winds up a 10⁻¹² G seed field; ours is a periodic box of "
                            "decaying Mach 9 turbulence with a uniform field (β₀ = 22) over 0.2 box sound-crossing times, with no gravity or "
                            "cooling. The paper compares grid, moving-mesh and particle codes; all of ours are fixed grids, so this shows only the "
                            "spread among grid codes and where a GLM code falls in it. Their divergence error is the volume mean of "
                            "|∇·B|Δx/|B| in the disc; ours is rms(∇·B)Δx over rms(B) in the whole box."))
        return run
    return {"ugalino-5-512": make(512), "ugalino-5-256": make(256)}
