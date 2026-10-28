"""Accuracy against cost: what a higher-order set-up on a coarser grid buys, and the time to a given error.

    python3 scripts/work_precision.py            -> table on stdout, progress/media/analysis/work_precision.json

For every set-up and test of the sweep (seed 42): the error at the end time against the test's truth (the exact
solution, or AthenaK at 4096^2 averaged to the grid; progress/media/analysis/<test>/data.json, test_diagnostics.py) and
a production cost at every grid,

    cost(N) = steps(N) * N^d / rate,

steps(N): the steps the run took (each code's own timestep); N^d: the cells of the problem; rate: the set-up's
cell-updates per second (its record's million_cell_steps_per_second, which counts the cells it computed: the GPU codes
ran the planar tests on N^3 cells that are the same at every z) on the largest grid of that test it finished with a valid clock (not resumed), where
start-up no longer counts. GPU: one H100; CPU: per core, and the cost is then given on 128 cores, assuming they scale
perfectly, as the speed figure compares one H100 with 128 cores. So the cost is that of the problem at each code's
full speed, as in a 3D production run; a 2D run this small would not reach that speed on a GPU.

Then, for an error eps: the grid N(eps) and cost T(eps) each set-up needs: the first grid whose error is at most eps,
interpolated log-log from the best error of the coarser grids (on the shock and reconnection tests the error hardly
moves from 32^2 to 64^2, where the structure is not yet resolved; a floor at the fine end, as Float16's, means eps is
not reached). eps of a test: the error of AthenaK PLM/HLLE/RK2 (second order, the sweep's
matched set-up) at 256^2, i.e. "what does it cost to be as good as a standard second-order code at 256^2".
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_data as sd  # noqa: E402

ANALYSIS = sd.SITE/"media"/"analysis"
CORES = 128                     # a CPU node, as the speed figure: one H100 against 128 cores
BASE = ("athenak_matched", 256)
# Tests and the quantity whose error is measured (as manuscript_figures.ACCURACY / convergence)
TESTS = (("entropy", "density"), ("alfven", "magnetic"), ("alfven_oblique", "magnetic"), ("field_loop_novz", "magnetic"),
         ("brio_wu", "density"), ("brio_wu_rotated", "density"), ("orszag_tang", "density"), ("rotor", "density"),
         ("blast_2d_b0", "density"), ("blast_2d_b50", "density"), ("kh_uniform", "density"), ("current_sheet", "density"),
         ("gem", "magnetic"), ("field_loop_3d", "magnetic"))
# What changes between the two set-ups of a pair. ORDER: only the reconstruction (Abel et al.'s f16/HLL kernel, PLM ->
# PPM, both fixed); ALL: reconstruction, Riemann solver and time integrator at once (the sweep's matched -> best set-ups);
# SOLVER: only the Riemann solver, at the same order (a control: how much of ALL a better solver alone gives).
PAIRS = (("order", "glm_fix_hll16", "glm_fix_ppm", "Abel GLM f16/HLL: PLM → PPM"),
         ("all", "athenak_matched", "athenak_best", "AthenaK: PLM/HLLE/RK2 → PPM/HLLD/RK3"),
         ("all", "athenapp_matched", "athenapp_best", "Athena++: PLM/HLLE/VL2 → PPM/HLLD/RK3"),
         ("solver", "ramses_matched", "ramses_best", "RAMSES: HLL → HLLD"),
         ("solver", "glm", "glm_fix_hlld", "Abel GLM Float32: LLF → HLLD"))
# The order x solver study (sherlock_order/): each code's four pairs (PLM, HLL-family), (PPM, HLL-family), (PLM, HLLD),
# (PPM, HLLD) at one integrator (Abel et al.'s kernels: also one precision and tile, Float32 4-cell).
GRID = (("Abel GLM, Float32", ("glm_fix_hll32", "glm_fix_ppm32", "glm_fix_hlld", "glm_fix_ppm_hlld")),
        ("AthenaK, RK3", ("athenak_plm_hlle_rk3", "athenak_ppm_hlle", "athenak_plm_hlld", "athenak_best")),
        ("Athena++, RK3", ("athenapp_plm_hlle_rk3", "athenapp_ppm_hlle", "athenapp_plm_hlld", "athenapp_best")))


def ranks(resource):
    m = re.search(r"(\d+) CPU rank", resource or "");return int(m.group(1)) if m else None


def curves(sw, test, kind):
    """{method: [(N, error, steps, cost seconds, unit), ...]} of the complete, finite runs of a test, seed 42."""
    path = ANALYSIS/test/"data.json"
    if not path.is_file():return {}
    d = json.loads(path.read_text())
    runs = {(r["method"], r["N"]): r for r in d["runs"] if r.get("seed", 42) == 42}
    pts = {}
    for g in d["diagnostics"]:
        r = runs.get((g["method"], g["N"]))
        if g.get("seed", 42) != 42 or not g.get("errors") or not r or r["status"] != "complete" or not r.get("finite_complete"):continue
        err = g["errors"][-1]["values"].get(f"{kind}_relative_L1");meta = sw.meta(test, g["method"], g["N"]) or {}
        steps, mcs = meta.get("steps"), meta.get("million_cell_steps_per_second")
        if not err or not steps or not mcs:continue
        cpu = ranks(r.get("resource"))
        rate = 1e6*float(mcs)/(cpu or 1)                                 # cell-updates per second, per core on CPUs
        pts.setdefault(g["method"], []).append(dict(N=g["N"], error=err, steps=int(steps), rate=rate, cpu=cpu is not None,
                                                    timed=meta.get("timing_valid", True) is not False))
    out = {}
    for m, p in pts.items():
        p.sort(key=lambda x: x["N"])
        timed = [x for x in p if x["timed"]]                              # a resumed run's clock covers only its last part
        if not timed:continue
        rate = timed[-1]["rate"]                                          # production rate: the largest grid finished
        d_ = 3 if test.endswith("_3d") else 2
        for x in p:x["cost"] = x["steps"]*x["N"]**d_/rate/(CORES if x["cpu"] else 1)
        out[m] = dict(points=p, rate=rate, cpu=p[0]["cpu"])
    return out


def at_error(curve, eps):
    """(N, cost, at most): the grid and cost at which a set-up's error falls to eps, or None if no grid of the sweep
    reaches it. At most: its coarsest grid already does (it would need fewer cells still)."""
    p = curve["points"]
    if p[0]["error"] <= eps:return p[0]["N"], p[0]["cost"], True
    best = p[0]
    for x in p[1:]:
        if x["error"] <= eps:                     # from the best of the coarser grids to this one, log-log
            f = np.log(best["error"]/eps)/np.log(best["error"]/x["error"])
            return float(np.exp(np.log(best["N"]) + f*np.log(x["N"]/best["N"]))), float(np.exp(np.log(best["cost"]) + f*np.log(x["cost"]/best["cost"]))), False
        if x["error"] < best["error"]:best = x
    return None


def order(curve):
    """Measured order between the two finest grids (None if the error did not fall there)."""
    p = curve["points"]
    if len(p) < 2 or p[-1]["error"] >= p[-2]["error"]:return None
    return -float(np.log(p[-1]["error"]/p[-2]["error"])/np.log(p[-1]["N"]/p[-2]["N"]))


def split(plm_hll, ppm_hll, plm_hlld, ppm_hlld):
    """The gains of a code's four pairs (a cost or an error each; above 1: the change lowers it): order at each solver,
    solver at each order, both, and the interaction (both over the product of the two alone; 1 when they multiply)."""
    return dict(order_hll=plm_hll/ppm_hll, order_hlld=plm_hlld/ppm_hlld, solver_plm=plm_hll/plm_hlld, solver_ppm=ppm_hll/ppm_hlld,
                both=plm_hll/ppm_hlld, interaction=plm_hll*ppm_hlld/(ppm_hll*plm_hlld))


def factorial(row, cells):
    """The split of the cost at a test's eps. None when a pair misses eps; at_most: some pair reached eps on its
    coarsest grid already, so its cost is an upper bound."""
    m = [row["methods"].get(c) or {} for c in cells];c = [x.get("cost") for x in m]
    if not all(c):return None
    return dict(split(*c), at_most=any(x.get("at_most") for x in m))


def errors(row, method):
    return {N: e for N, e, _ in (row["methods"].get(method) or {}).get("points", [])}


def error_factorial(row, cells):
    """The same split of the error at every grid all four pairs ran: no cost model and no eps, so it exists on every
    test, but says nothing of the time (PPM and HLLD cost more per step)."""
    e = [errors(row, c) for c in cells]
    return {N: split(*(x[N] for x in e)) for N in sorted(set.intersection(*(set(x) for x in e)))}


def across(row, i):
    """Pair i of GRID in every code at every grid they all ran: {N: {code: error}}."""
    e = {name: errors(row, cells[i]) for name, cells in GRID}
    return {N: {name: x[N] for name, x in e.items()} for N in sorted(set.intersection(*(set(x) for x in e.values())))}


def table(sw=None):
    sw = sw or sd.Sweep();rows = {}
    for test, kind in TESTS:
        c = curves(sw, test, kind)
        if not c:continue
        base = c.get(BASE[0]) or c.get("athenapp_matched")
        ref = next((x for x in (base or {}).get("points", []) if x["N"] == BASE[1]), None)
        if ref is None:continue
        eps = ref["error"];row = dict(kind=kind, eps=eps, methods={})
        for m, cur in c.items():
            hit = at_error(cur, eps)
            row["methods"][m] = dict(N=hit[0] if hit else None, cost=hit[1] if hit else None, at_most=bool(hit and hit[2]), order=order(cur),
                                     rate=cur["rate"], cpu=cur["cpu"], points=[(x["N"], x["error"], x["cost"]) for x in cur["points"]])
        row["error_split"] = {name: error_factorial(row, cells) for name, cells in GRID}
        row["across"] = {pair: across(row, i) for i, pair in enumerate(("plm_hll", "ppm_hll", "plm_hlld", "ppm_hlld"))}
        rows[test] = row
    return rows


def main():
    rows = table()
    (ANALYSIS/"work_precision.json").write_text(json.dumps(rows, indent=1) + "\n")
    for test, row in rows.items():
        print(f"== {test}: eps = {row['eps']:.3g} ({row['kind']} L1, AthenaK PLM/HLLE at {BASE[1]})")
        for _, lo, hi, label in PAIRS:
            a, b = row["methods"].get(lo), row["methods"].get(hi)
            if not (a and b):continue
            f = lambda x: "–" if x["N"] is None else f"N {'≤' if x['at_most'] else ''}{x['N']:.0f}, {x['cost']:.3g} s"
            gain = f"{a['cost']/b['cost']:.2f}x faster, {a['N']/b['N']:.2f}x fewer cells per side" if a["cost"] and b["cost"] else ""
            print(f"   {label:42s} {f(a):22s} -> {f(b):22s} {gain}   (orders {a['order'] or float('nan'):.2f} -> {b['order'] or float('nan'):.2f})")
        for name, cells in GRID:
            g = factorial(row, cells)
            if g:print(f"   {name:18s} faster by: PPM {g['order_hll']:.2f}x with HLL, {g['order_hlld']:.2f}x with HLLD; HLLD {g['solver_plm']:.2f}x "
                       f"with PLM, {g['solver_ppm']:.2f}x with PPM; both {g['both']:.2f}x; interaction {g['interaction']:.2f}" + (" (upper bounds)" if g["at_most"] else ""))
            for N, g in error_factorial(row, cells).items():
                print(f"   {name:18s} N {N:4d} error lower by: PPM {g['order_hll']:.2f}x with HLL, {g['order_hlld']:.2f}x with HLLD; HLLD "
                      f"{g['solver_plm']:.2f}x with PLM, {g['solver_ppm']:.2f}x with PPM; both {g['both']:.2f}x; interaction {g['interaction']:.2f}")
        for i, pair in enumerate(("PLM/HLL", "PPM/HLL", "PLM/HLLD", "PPM/HLLD")):
            for N, e in across(row, i).items():
                print(f"   {pair:9s} N {N:4d} error: " + ", ".join(f"{k} {v:.3g}" for k, v in e.items()))


if __name__ == "__main__":
    main()
