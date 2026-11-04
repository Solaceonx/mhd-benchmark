"""Convergence without a truth: each set-up against its own finest grid, the differences between grids, and Richardson
extrapolation; with the error against the test's truth beside them, to see where each of these can be trusted.

    .venv-movie/bin/python scripts/self_convergence.py [TEST ...]
        -> table on stdout, progress/media/analysis/self_convergence.json,
           progress/media/analysis/<test>/09-self-<group>.png and .svg

For every set-up and planar test of work_precision.TESTS (seed 42) with final states on the Mac (32^2 to 512^2; the 3D
states are too large to fetch), in the test's quantity (density or magnetic field), relative L1 as test_diagnostics:

    own(N)    the error at N against the set-up's own finest grid, averaged to N: its own "gold standard"
    diff(N)   the difference between grids N and 2N, the finer averaged to N
    p(N)      the order the differences show from grids N, 2N, 4N: log2(diff(N)/diff(2N))
    rich(N)   the error at N against the Richardson extrapolation of the three finest grids N1 < N2 < N3 whose
              differences are above the Float32 floor of the snapshots (FLOOR):
              u_inf = u3 + (u3 - u2)/(2^p - 1) on grid N2 (u3 averaged to it), p = p(N1)
    truth(N)  the error against the test's truth (the exact solution, or AthenaK at 4096^2 averaged to N), as
              work_precision.py and the convergence figures

and, on grid N2, the error of u3 (averaged) and of u_inf against the truth: gain = the first over the second, how much
nearer the truth the extrapolation is than the finest grid. Richardson assumes the error falls as N^-p with one p; it is
not applied when p is outside 0.5-6 (the differences did not fall, or fell by roundoff), and across shocks, where
the error is first order and concentrated at the discontinuities, it is expected to gain little.
The truth on a grid is the reference a record was scored against (its ref_*.bin: the exact cell averages), or AthenaK
at 4096^2 averaged to the grid.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_diagnostics as td  # noqa: E402
from work_precision import TESTS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (test_diagnostics set the backend)

EXACT = ("entropy", "alfven", "alfven_oblique", "field_loop_novz")       # back to the initial state at t_end (exact_reference)
TRUTH_FROM = ("glm", "athenapk_matched", "cholla_plmc", "enzo_ct")     # their initial states are the suite's
P_RANGE = (.5, 6.)
FLOOR = 1e-6        # the snapshots are Float32: differences below this are roundoff, not convergence
td.TITLES["09-self"] = "Convergence against each code's own grids"


def down(a, n):
    """Cell averages of a planar state on n x n cells (the grids are nested: 2^k cells of the finer per side)."""
    r = a.shape[0]//n;return a.reshape(n, r, n, r, a.shape[-1]).mean(axis=(1, 3))


def frame(run, time):
    f = next((f for f in run.meta.get("frames", []) if abs(f["time"] - time) <= 1e-6*max(1, abs(time)) and f.get("file")), None)
    return run.path/f["file"] if f else None


def error(a, ref, kind):
    return td.compute_errors(a, ref).get(f"{kind}_relative_L1")


class Test:
    def __init__(self, data, test, kind):
        self.data, self.test, self.kind = data, test, kind;self.end = float(data.sw.configs[test]["t_end"])

    def state(self, run, time):
        p = frame(run, time)
        a = self.data.load_array(str(p), run.n) if p else None
        return a.astype(float) if a is not None and a.ndim == 3 else None

    def truth(self, run):
        """The truth on the run's grid. Exact tests: the suite's initial state there, from the first of these runs that
        has it on the Mac (the reference files, ref_*.bin, are not fetched; the sweep fetched initial states at 128^2 only,
        the five new codes at every grid, all equal to Float32 roundoff; the initial state scores the grid codes as their
        records do, to 4 digits; a particle code's own is interpolated, so not its own). Else AthenaK at 4096^2."""
        if self.test in EXACT:
            for r in self.data.select(self.test, run.n, methods=TRUTH_FROM):
                if (a := self.state(r, 0.0)) is not None:return a
            return None
        ref, _ = self.data.reference(self.test, run.n, 42, self.end)
        return ref.astype(float) if ref is not None else None

    def method(self, m):
        runs = {r.n: r for r in self.data.select(self.test, methods=[m]) if r.completed}
        u = {n: a for n, r in sorted(runs.items()) if (a := self.state(r, self.end)) is not None}
        if len(u) < 2:return None
        grids = sorted(u);top = grids[-1]
        out = dict(grids=grids, finest=top, own={}, diff={}, p={}, rich={}, truth={})
        for n in grids:
            e = self.data.endpoint(runs[n], self.kind)
            if e is not None:out["truth"][n] = e
            if n < top:out["own"][n] = error(u[n], down(u[top], n), self.kind)
            if 2*n in u:out["diff"][n] = error(u[n], down(u[2*n], n), self.kind)
        for n in out["diff"]:
            if 2*n in out["diff"] and min(out["diff"][n] or 0, out["diff"][2*n] or 0) >= FLOOR:out["p"][n] = float(np.log2(out["diff"][n]/out["diff"][2*n]))
        # Richardson from the three finest grids whose differences are above the floor, if they fell at a plausible order
        n1 = max(out["p"], default=None);p = out["p"].get(n1)
        if p is not None and P_RANGE[0] <= p <= P_RANGE[1]:
            n2 = 2*n1;u3 = down(u[4*n1], n2);inf = u3 + (u3 - u[n2])/(2**p - 1)
            out["rich"] = {n: error(u[n], down(inf, n), self.kind) for n in grids if n <= n2}
            out["richardson"] = dict(p=p, grids=[n1, n2, 4*n1], estimated_error_finest=out["diff"][n2]/(2**p - 1))
            t = self.truth(runs[n2])
            if t is not None:
                fine, extrapolated = error(u3, t, self.kind), error(inf, t, self.kind)
                out["richardson"].update(finest_vs_truth=fine, extrapolated_vs_truth=extrapolated, gain=fine/extrapolated if extrapolated else None)
        return out

    def table(self):
        ms = sorted({m for (t, m, n, s) in self.data.runs if t == self.test and m != "athenak_reference"},
                    key=lambda m: td.METHODS.index(m) if m in td.METHODS else len(td.METHODS))
        return {m: r for m in ms if (r := self.method(m))}


def figures(test, rows):
    fig_ = td.Figures.__new__(td.Figures);fig_.test = test;fig_.root = td.OUT/test;fig_.entries = []
    fig_.root.mkdir(parents=True, exist_ok=True)
    panels = (("truth", "against the test's truth"), ("own", "against its own finest grid"),
              ("diff", "between grid N and 2N"), ("rich", "against its Richardson extrapolation"))
    kind = dict(TESTS)[test]
    for gi, (name, methods) in enumerate(td.groups([m for m in rows])):
        fig, axs = plt.subplots(2, 2, figsize=(13, 9.5), sharey=True)
        for ax, (key, title) in zip(axs.flat, panels):
            for m in methods:
                pts = sorted((int(n), 100*float(e)) for n, e in rows[m][key].items() if e)
                if pts:td.draw(ax, *zip(*pts), m, points=True)
            if not ax.lines:td.blank(ax, "No set-up has the grids this needs.");continue
            ax.set_xscale("log", base=2);ax.set_yscale("log");ax.set_title(title, fontsize=12)
            ax.set_xlabel("Cells per side, N");ax.set_ylabel(f"{kind.title()} relative L1 [%]")
        fig_.save(fig, "09-self", str(gi), f"{name} · endpoint, seed 42",
                  "Top left: the error against the test's truth (exact, or AthenaK 4096² averaged to N). Top right: against the "
                  "set-up's own finest grid averaged to N. Bottom left: the difference between grid N and 2N. Bottom right: against "
                  "the Richardson extrapolation of its three finest grids, at the order their differences show (scripts/self_convergence.py).")
    return fig_.entries


def main(tests):
    data = td.Data();out = {}
    for test, kind in TESTS:
        if tests and test not in tests or test.endswith("_3d"):continue
        rows = Test(data, test, kind).table()
        if not rows:continue
        out[test] = dict(kind=kind, methods={m: {k: ({str(n): v for n, v in x.items()} if isinstance(x, dict) and k != "richardson" else x)
                                               for k, x in r.items()} for m, r in rows.items()})
        figures(test, rows)
        print(f"== {test} ({kind} relative L1): set-up, finest grid, own error / truth error at half of it, Richardson's order "
              "and the grids it used, the estimated and true error of its finest, and its gain over that grid")
        for m, r in rows.items():
            half = r["finest"]//2;own, tru = r["own"].get(half), r["truth"].get(half);ri = r.get("richardson") or {}
            f = lambda v, s="{:.3g}": "–" if v is None else s.format(v)
            print(f"   {td.style(m)['label'][:44]:44s} N {r['finest']:4d}  own/truth {f(own/tru if own and tru else None, '{:.2f}'):>5s}  "
                  f"p {f(ri.get('p'), '{:.2f}'):>5s} {'/'.join(map(str, ri.get('grids', []))):12s} estimated {f(ri.get('estimated_error_finest'), '{:.2e}'):>8s} "
                  f"true {f(ri.get('finest_vs_truth'), '{:.2e}'):>8s}  gain {f(ri.get('gain'), '{:.2f}'):>5s}")
    (td.OUT/"self_convergence.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
