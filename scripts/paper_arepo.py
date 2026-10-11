"""AREPO on the suite's planar tests (sherlock_arepo/): the figures of its section of the Papers page.

AREPO ran after the sweep, outside its plan: 17 planar tests at seed 42 on a moving mesh and on a static Cartesian
mesh, with Powell's scheme for the divergence of B and the HLLD solver. The figures are those of the manuscript
(scripts/manuscript_figures.py), drawn here as images with the numbers behind them written out.
"""
import numpy as np
import matplotlib

import manuscript_figures as mf
import sweep_data
from paper_common import sci, spread

PAPER = "arepo"


def drawn(function, sw):
    """A manuscript figure in the manuscript's own type sizes."""
    with matplotlib.rc_context():
        mf.style();out = function(sw)
    return out[0] if isinstance(out, tuple) else out


def counts(sw):
    by = {}
    for (test, label, n, seed), row in sw.arepo.items():
        c = by.setdefault(label, dict(done=0, stopped=[], open=0))
        if row["state"] == "done":c["done"] += 1
        elif row["state"] == "stopped":c["stopped"].append((test, n, row.get("reached")))
        else:c["open"] += 1
    return by


def mesh_finding(sw):
    c = counts(sw);parts = []
    loop = {}
    for n in sw.grids("field_loop_novz", "arepo_static"):
        d = sw.diag("field_loop_novz", "arepo_static", n)
        if d is not None and len(d["magnetic"]) > 1:loop[n] = float(np.nanmax(d["magnetic"])/d["magnetic"][0])
    if loop:
        parts.append("On the static mesh the magnetic energy of the advected field loop grows instead of decaying: by a factor of "
                     + ", ".join(f"{sci(v)} at {n}²" for n, v in sorted(loop.items())) + " at its largest")
    d = sw.diag("field_loop_novz", "arepo_moving", 256)
    if d is not None:parts.append(f"on the moving mesh at 256² it ends at {d['magnetic'][-1]/d['magnetic'][0]:.4f} of its initial value")
    stopped = sorted(c.get("arepo_static", {}).get("stopped", []))
    if stopped:
        parts.append(f"AREPO ended {len(stopped)} static-mesh runs itself when its timestep fell to the smallest its time line holds ("
                     + "; ".join(f"{t.replace('_', ' ')} {n}² at t = {r:.3g}" for t, n, r in stopped) + ")")
    div = {}
    for m in sweep_data.AREPO:
        v = []
        for test in ("orszag_tang", "rotor", "blast_2d_b50", "kh_uniform"):
            for n in sw.grids(test, m):
                p = sw.path(test, m, n, "arepo_divergence.csv")
                if p is not None and p.is_file() and sw.state_of(test, m, n) == "done":v.append(float(sweep_data.read_csv(p)["normalized_div_rms"][-1]))
        div[m] = v
    if div.get("arepo_moving"):
        parts.append(f"AREPO's own divergence measure at the end of the four tests shown is {spread(div['arepo_moving'], sci)} on the moving mesh "
                     f"at every grid and {spread(div['arepo_static'], sci)} on the static mesh")
    return "; ".join(parts) + "." if parts else ""


def convergence_finding(sw):
    out = []
    for test, key, name in (("entropy", "density_absolute_L1", "entropy wave"), ("alfven_oblique", "magnetic_relative_L1", "oblique Alfvén wave")):
        for m, mesh in (("arepo_moving", "moving"), ("arepo_static", "static")):
            e = {n: mf.final_error(sw, test, m, n, key) for n in (32, 512)}
            if e[32] and e[512]:out.append(f"{name}, {mesh} mesh: {sci(e[32])} at 32² to {sci(e[512])} at 512² (order {np.log2(e[32]/e[512])/4:.1f})")
    return ("Errors at the end of the run: " + "; ".join(out) + ". On the moving mesh the entropy wave is carried by the mesh itself, so its error "
            "is at the precision of the saved single-precision states at every grid.") if out else ""


def status_finding(sw):
    c = counts(sw)
    return "; ".join(f"{label.replace('arepo_', '')} mesh: {v['done']} of {v['done'] + len(v['stopped']) + v['open']} runs reached the end"
                     for label, v in sorted(c.items())) + ". A run that reached the end is not thereby right: see the field loop above."


def figures():
    def mesh(F):
        F.save(drawn(mf.arepo, F.sw), "arepo-mesh", paper=PAPER, paper_fig="Added", kind="extension", test="field_loop_novz",
               title="AREPO, static against moving mesh: the field loop, the timestep, and the divergence of B",
               finding=mesh_finding(F.sw),
               caption=("(a) Magnetic energy of the field loop advected in the plane, over its initial value, on AREPO's static Cartesian mesh at four "
                        "grids and on its moving mesh; a cross is where AREPO ended the run. (b) The timestep of the static-mesh runs AREPO ended, "
                        "over its first value, against the part of the run done (from AREPO's own log). (c) AREPO's own measure of the divergence "
                        "of B at the last output, rms(∇·B) × cell size over rms(B) on its own mesh, against the grid."),
               differs=("AREPO was run with Powell's eight-wave scheme, the HLLD solver, one global timestep and Courant factor 0.3, on the suite's "
                        "periodic unit box. A static mesh with Powell's scheme is not how AREPO is normally run; its published MHD results use the "
                        "moving mesh. Two checks of Orszag–Tang at 256² on the static mesh: on one rank it stops at the same step, and with half "
                        "the Courant factor it stops slightly later (t = 0.454 against 0.422)."))

    def conv(F):
        F.save(drawn(mf.convergence, F.sw), "arepo-convergence", paper=PAPER, paper_fig="Added", kind="extension", test="entropy",
               title="Convergence on the wave tests, with AREPO among the other codes", finding=convergence_finding(F.sw),
               caption=("Error at the end of the run against the grid for the entropy wave and the two Alfvén waves. Top: the GLM kernels and the CT "
                        "prototype. Bottom: AthenaK, RAMSES, Athena++ and AREPO (ochre; solid the moving mesh, dashed the static mesh)."))

    def status(F):
        F.save(drawn(mf.robustness, F.sw), "arepo-status", paper=PAPER, paper_fig="Added", kind="extension", test="",
               title="Every test, grid and code, with AREPO: finished, or how far the solver got", finding=status_finding(F.sw),
               caption=("One mark per run: the five grids of a test from left to right. AREPO (right) ran the planar tests at one seed; the rows it has "
                        "no marks in are the 3D tests and the further seeds."))
    return {"arepo-mesh": mesh, "arepo-convergence": conv, "arepo-status": status}
