"""Dedicated Mach 30 and 50 result pages from the existing driven-hydro records.

    .venv-movie/bin/python scripts/build_highmach_results.py [--page-only]
"""
import csv
import html
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
import matplotlib
import numpy as np

import turb_figures as tf
import build_turbulence as bt
import test_diagnostics as diagnostics
from plot_style import apply_plot_style

SITE = tf.ROOT / "progress"
PLOTS = ((tf.mach_time, "turb-mach-time", "Mach number versus time", "Both grids and all available seeds; the shaded startup interval is excluded from steady-state averages."),
         (tf.spectra, "turb-spectra", "Velocity power spectra", "Time averages over 2T–10T, compensated by k². The two grids are shown separately; these are statistical comparisons, without a pointwise truth realization."),
         (tf.dissipation, "turb-dissipation", "Effective Reynolds number", "Inferred from measured power and resolved velocity gradients. Small marks show individual seeds; two resolutions do not establish convergence."),
         (tf.slices, "turb-slices", "Density at 10 driving times", "256³ midplane, seed 1. One common colour scale across codes, with densities clipped to the floor for display. The force realizations differ across codes, so individual structures should not coincide."))
LABELS = {"gpu-hydro": "Abel et al. Euler / HLL", "gpu-glm": "Abel et al. zero-field MHD / LLF",
          "ramsesfix-matched": "RAMSES HLL", "ramsesfix-best": "RAMSES HLLC"}
e = html.escape


def range_text(values, fmt):
    lo, hi = fmt(min(values)), fmt(max(values))
    return lo if lo == hi else f"{lo}–{hi}"


def nonpositive_count(run):
    values = run.series.get("negative_density_cells")
    return int(values[run.window].max()) if values is not None else None


def make_plots(mach, runs):
    out = SITE / "media/turb" / f"mach-{mach}"; out.mkdir(parents=True, exist_ok=True)
    original = tf.save
    def save(fig, name):
        for ax in fig.axes:
            ax.grid(False, which="both")
        for suffix in ("png", "svg"):
            fig.savefig(out/f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
        tf.plt.close(fig)
    tf.save = save
    try:
        for function, _, _, _ in PLOTS: function(runs)
    finally:
        tf.save = original


def summary_table(runs):
    rows = []
    for n in sorted({r.n for r in runs}):
        for kind in tf.LIKE:
            group = [r for r in runs if r.kind == kind and r.n == n]
            if not group: continue
            cells = [LABELS[kind], f"{n}³", str(len(group)),
                     range_text([r.mean("mach_vol") for r in group], lambda x:f"{x:.2f}"),
                     range_text([r.power() for r in group], lambda x:f"{x:.3g}"),
                     range_text([r.re_eff() for r in group], lambda x:f"{x:.0f}"),
                     f"{100*max(bt.floor_share(r) for r in group):.3f}%",
                     str(max(nonpositive_count(r) for r in group if nonpositive_count(r) is not None))
                     if any(nonpositive_count(r) is not None for r in group) else "not recorded"]
            rows.append("<tr>"+"".join(f"<td>{e(c)}</td>" for c in cells)+"</tr>")
    headers = ["Set-up", "Grid", "Seeds", "Achieved Mach", "Power", "Effective Re", "Max. floor share", "Max. nonpositive cells"]
    return '<div class="table-scroll"><table class="metrics-table sweep-text paper-table"><thead><tr>'+"".join(f"<th>{e(x)}</th>" for x in headers)+"</tr></thead><tbody>"+"".join(rows)+"</tbody></table></div>"


def build(mach, runs):
    floor = max(bt.floor_share(r) for r in runs)
    bad = max(nonpositive_count(r) for r in runs if nonpositive_count(r) is not None)
    drift = max(abs(bt.halves(r)) for r in runs)
    other = 50 if mach == 30 else 30
    movie = bt.movie_card(mach)
    movie_link = '<a href="#turb-movie">Movie</a> · ' if movie else ''
    panels = []
    for _, name, title, caption in PLOTS:
        base = f"media/turb/mach-{mach}/{name}"
        version = int((SITE/f"{base}.png").stat().st_mtime)
        panels.append(f'<article class="card paper-fig" id="{name}"><h3>{e(title)}</h3><a href="{base}.png?v={version}"><img style="width:100%;height:auto" src="{base}.png?v={version}" alt="Mach {mach}: {e(title)}" loading="lazy"></a><p>{e(caption)}</p><p><a href="{base}.svg?v={version}">Download vector SVG</a></p></article>')
    root = SITE / "media/analysis" / f"driven_M{mach}"
    root.mkdir(parents=True, exist_ok=True)
    records = [dict(run=r.name,kind=r.kind,N=r.n,seed=r.seed,metadata=r.meta,series=r.series) for r in runs]
    (root/"data.json").write_text(json.dumps(diagnostics.clean(records),allow_nan=False,separators=(",",":"))+"\n")
    with (root/"runs.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=["run","kind","N","seed","achieved_mach","power","effective_Re","max_floor_fraction","max_nonpositive_cells"])
        writer.writeheader()
        for r in runs:writer.writerow(dict(run=r.name,kind=r.kind,N=r.n,seed=r.seed,achieved_mach=r.mean("mach_vol"),power=r.power(),effective_Re=r.re_eff(),
            max_floor_fraction=bt.floor_share(r),max_nonpositive_cells=nonpositive_count(r)))
    main = f'''<div class="page-heading"><div><div class="eyebrow"><a href="turbulence.html">Driven turbulence</a><span class="badge">{len(runs)} completed runs</span></div>
<h1>Driven turbulence · Mach {mach}</h1><p>Zero-field hydrodynamic turbulence, driven for 10 stirring times on 128³ and 256³.
Abel et al. Euler and zero-field MHD kernels, compared with RAMSES HLL and HLLC using the corrected solenoidal force.</p></div><a class="method-link" href="test-driven-mach-{other}.html">Mach {other} →</a></div>
<div class="chips"><span class="chip">γ − 1 = 10⁻⁶</span><span class="chip">B = 0</span><span class="chip">128³ and 256³</span><span class="chip">Average: 2T–10T</span><span class="chip">Float32 GPU states</span></div>
<p class="small">{movie_link}<a href="#turb-mach-time">Mach histories</a> · <a href="#turb-spectra">Spectra</a> · <a href="#turb-dissipation">Dissipation</a> · <a href="#turb-slices">Density images</a> · <a href="#analysis">Shared diagnostics</a> · <a href="problems.html#driven">Setup source</a></p>{movie}
<article class="card"><h3>What the records show</h3><p>All {len(runs)} runs reached 10T. The first and second halves of the averaging window differ in mean Mach by at most {100*drift:.2f}%.
Each table entry averages one run over 2T–10T; ranges show the two GPU seeds. RAMSES has one seed per set-up and grid.</p>{summary_table(runs)}
<p>Floor share and nonpositive-cell counts are the largest saved observations during the averaging window. Power is the slope of total energy versus time in code units; it is not wall-clock power.</p></article>
<article class="card"><h3>Density statistics require care</h3><p>Up to {100*floor:.2f}% of cells lie at the density floor in a saved observation, and RAMSES has up to {bad:,} nonpositive-density cells.
The density distribution is affected by numerical floors, so its width is not a reliable physical measurement here. Completion alone does not establish physical validity.</p></article>
<article class="card"><h3>How this differs from Mach 9 decay</h3><p>The <a href="test-decay-supersonic.html#failure-audit">Mach 9 problem</a> is decaying MHD with a mean magnetic field and γ = 1.01.
These Mach {mach} runs are driven hydro with B = 0 and γ − 1 = 10⁻⁶. Abel et al.'s runs here use Float32 shared storage; the failing half-precision production kernels from the Mach 9 sweep are not these configurations.
The two experiments test different numerical regimes.</p></article>
<article class="card"><h3>Why AthenaK is absent</h3><p>This pilot first matched the existing Ornstein–Uhlenbeck driving in Abel et al. and RAMSES.
AthenaK supports driven turbulence, but its driving has not yet been matched and validated for this experiment, so there are no AthenaK runs in this comparison.
Adding it requires matching the forcing spectrum, correlation time, solenoidal projection and normalization, along with the gas model and measurements.</p></article>{''.join(panels)}'''
    main += diagnostics.section("sherlock_turb",figure_prefix=f"media/analysis/driven_M{mach}/",data_test=f"driven_M{mach}")
    registry=json.loads((SITE/"tests.json").read_text())
    page=bt.render(main,registry).replace('<title>Turbulence ·',f'<title>Mach {mach} turbulence ·')
    page=page.replace('<meta name="color-scheme"','<meta name="robots" content="noindex, nofollow"><meta name="color-scheme"',1)
    (SITE/f"test-driven-mach-{mach}.html").write_text(page)
    print(f"Mach {mach}: {len(runs)} records, 4 focused figures, shared diagnostics, page and downloads.")


def main():
    apply_plot_style()
    matplotlib.rcParams.update({"font.size":9,"axes.titlesize":10,"axes.labelsize":10,"legend.fontsize":8,"figure.dpi":100,"savefig.dpi":220,"legend.frameon":False})
    tf.KINDS={k:tuple(value.replace("Tom's", "Abel et al.") if isinstance(value,str) else value for value in values) for k,values in tf.KINDS.items()}
    runs=tf.load()
    for mach in (30,50):
        selected=[r for r in runs if r.mach==mach]
        if not selected:raise SystemExit(f"No fetched Mach {mach} records")
        if "--page-only" not in sys.argv:make_plots(mach,selected)
        build(mach,selected)


if __name__=="__main__":main()
