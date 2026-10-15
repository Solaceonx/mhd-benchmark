"""The "Turbulence" page of the local notebook: first look at the driven-turbulence pilot (sherlock_turb/README.md).

  python3 scripts/build_turbulence.py [--page-only]     -> progress/turbulence.html, progress/media/turb/*.png

Draws the figures (scripts/turb_figures.py) from turb_results/ and writes the page; every number in the text is
computed here from the same files. Fetch command for turb_results/: see the page.
"""
import html
import json
import os
import struct
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import turb_figures as tf  # noqa: E402
import test_diagnostics as diagnostics  # noqa: E402

SITE = tf.ROOT/"progress"
PAGE = dict(id="turbulence", page="turbulence.html", kind="turbulence", number="09", title="Driven turbulence", nav="Turbulence",
            status="Waiting for data", document="../sherlock_turb/README.md",
            description="Driven hydrodynamic turbulence in Tom Abel's kernels and RAMSES: the pilot.", chips=[])
e = html.escape
FETCH = ("mkdir -p ~/Projects/research/abel_mhd/turb_results && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu "
         "'cd /scratch/users/lyxia/mhd-results-turb && find . -maxdepth 3 \\( -name run.toml -o -name series.csv -o -name spectra.csv -o -name pdf.csv "
         "-o -name slice_040.bin -o -name mach.csv -o -name forcing.toml -o -name summary.csv \\) -not -path \"./trial/*\" | tar -czf - -T -' "
         "| tar -xzf - -C ~/Projects/research/abel_mhd/turb_results")


def png_width(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">I", head[16:20])[0] if head[:8] == b"\x89PNG\r\n\x1a\n" else 0


def pct(x, digits=1):
    return f"{100*x:+.{digits}f} %".replace("-", "−")


def rng(values, fmt):
    lo, hi = fmt(min(values)), fmt(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


def halves(r):
    m = r.series["mach_vol"][r.window];h = len(m)//2
    return m[h:].mean()/m[:h].mean() - 1


def floor_share(r, column="density_floor_cells"):
    """Largest share of the cells at (or, with the other column, below) the density floor at any saved time of the window."""
    return float(r.series[column][r.window].max())/r.n**3 if column in r.series else 0.0


def findings(runs):
    """The sentences under the figures, from the data."""
    cs = tf.cases(runs);pick = tf.pick;out = {}
    out["mach"] = (f"All {len(runs)} runs pass the check: the mean Mach number of the second half of the averaging window differs from the first half's by "
                   f"{rng([abs(halves(r)) for r in runs], lambda v: f'{100*v:.1f} %')}, and the Mach number wanders by "
                   f"{rng([float(r.series['mach_vol'][r.window].std()/r.mean('mach_vol')) for r in runs], lambda v: f'{100*v:.1f} %')} about its mean. "
                   "The paler curves (RAMSES as released) run lower from the start.")
    gap = {k: [] for k in ("rel_m", "rel_p", "fix_m", "fix_p", "glm_m", "glm_p", "seed_p", "cross_m", "cross_p")}
    for m, n in cs:
        ref = pick(runs, "gpu-hydro", m, n);rm = np.mean([r.mean("mach_vol") for r in ref]);rp = np.mean([r.power() for r in ref])
        if len(ref) > 1:gap["seed_p"].append(abs(ref[0].power()/ref[1].power() - 1))
        for kind, key in (("ramses-matched", "rel"), ("ramses-best", "rel"), ("ramsesfix-matched", "fix"), ("ramsesfix-best", "fix"), ("gpu-glm", "glm"),
                          ("gpu-hydro_ramsesproj", "cross")):
            for r in pick(runs, kind, m, n):gap[key + "_m"].append(r.mean("mach_vol")/rm - 1);gap[key + "_p"].append(r.power()/rp - 1)
    out["gap"] = (f"RAMSES as released sits {rng(gap['rel_m'], pct)} in Mach number and {rng(gap['rel_p'], lambda v: pct(v, 0))} in power against Tom's Euler kernel. "
                  f"With the projection corrected it is within {rng(gap['fix_m'], pct)} and {rng(gap['fix_p'], lambda v: pct(v, 0))}; "
                  f"two seeds of the Euler kernel differ by {rng(gap['seed_p'], lambda v: f'{100*v:.0f} %')} in power. "
                  f"Tom's MHD kernel with zero field: {rng(gap['glm_m'], pct)} and {rng(gap['glm_p'], lambda v: pct(v, 0))}.")
    if gap["cross_m"]:
        out["gap"] += (f" The reverse test (diamonds): Tom's Euler kernel driven with RAMSES's released projection sits {rng(gap['cross_m'], pct)} and "
                       f"{rng(gap['cross_p'], lambda v: pct(v, 0))}, where released RAMSES sits. So the gap is the projection, not the solver.")
    n = max(r.n for r in runs);parts = []
    for m in sorted({r.mach for r in runs}):
        k, h = tf.mean_spectrum(runs, "gpu-hydro", m, n, "v");_, rh = tf.mean_spectrum(runs, "ramsesfix-matched", m, n, "v")
        _, rc = tf.mean_spectrum(runs, "ramsesfix-best", m, n, "v");_, g = tf.mean_spectrum(runs, "gpu-glm", m, n, "v")
        i = int(np.argmin(abs(k - n/4)));inner = k >= 4
        parts.append(f"Mach {m}: the two HLL codes differ by at most {100*np.max(abs(h[inner]/rh[inner] - 1)):.0f} % at any k ≥ 4; at k = {int(k[i])} "
                     f"RAMSES HLLC holds {rc[i]/rh[i]:.1f} times and the MHD kernel (LLF) {g[i]/h[i]:.2f} times the power of its HLL counterpart")
    out["spectra"] = f"On {n}³, " + "; ".join(parts) + "."
    parts = []
    for m in sorted({r.mach for r in runs}):
        def share(kind, kk=2):
            k, c = tf.mean_spectrum(runs, kind, m, n, "v_compressive");_, v = tf.mean_spectrum(runs, kind, m, n, "v");return c[int(kk) - 1]/v[int(kk) - 1]
        if not pick(runs, "ramses-matched", m, n):continue          # released RAMSES ran at the lower Mach numbers only
        parts.append(f"Mach {m}: at k = 2, {share('ramses-matched'):.2f} and {share('ramses-best'):.2f} as released against "
                     f"{share('ramsesfix-matched'):.2f} and {share('ramsesfix-best'):.2f} corrected and {share('gpu-hydro'):.2f} for Tom's Euler kernel")
    out["share"] = ("The released projection shows only where the driving acts: " + "; ".join(parts) +
                    ". Past the driven band the released and corrected curves of each solver join, and what separates the curves is the solver.")
    parts = [];floored = {m: max(floor_share(r) for r in runs if r.mach == m) for m in sorted({r.mach for r in runs})}
    clean = [m for m in floored if floored[m] < 1e-4]
    for m in clean:
        b = {kind: np.mean([np.sqrt(np.expm1(r.mean("sigma_s")**2))/r.mean("mach_vol") for r in pick(runs, kind, m, n)]) for kind in tf.KINDS if pick(runs, kind, m, n)}
        parts.append(f"Mach {m}: b = " + ", ".join(f"{b[k]:.2f} ({tf.KINDS[k][4]})" for k in b))
    def b_of(kind, m, g):
        return np.mean([np.sqrt(np.expm1(r.mean("sigma_s")**2))/r.mean("mach_vol") for r in pick(runs, kind, m, g)])
    grids = sorted({r.n for r in runs})
    trend = "; ".join(f"Mach {m}: {b_of('gpu-hydro', m, grids[0]):.2f} → {b_of('gpu-hydro', m, grids[-1]):.2f} (Euler kernel), "
                      f"{b_of('ramsesfix-best', m, grids[0]):.2f} → {b_of('ramsesfix-best', m, grids[-1]):.2f} (RAMSES HLLC)" for m in clean)
    out["pdf"] = (f"The width σ of the distribution of s gives b = √(exp σ² − 1)/Mach, about 1/3 for solenoidal driving in the literature. On {n}³, " + "; ".join(parts) +
                  f". From {grids[0]}³ to {grids[-1]}³: {trend}. Where it still moves with resolution it is not converged.")
    if len(clean) < len(floored):
        hit = [m for m in floored if m not in clean];below = max(floor_share(r, "negative_density_cells") for r in runs if r.mach in hit)
        out["pdf"] += (f" At Mach {' and '.join(str(m) for m in hit)} the distribution runs into the density floor of 10⁻⁶ in every code (the spike at the left): up to "
                       f"{100*max(floored[m] for m in hit):.1f} % of the cells sit there, and in RAMSES up to {100*below:.2f} % are at or below zero. "
                       "The width is then set by the floor, so no b is given for these Mach numbers.")
    parts = []
    for m in sorted({r.mach for r in runs}):
        re = {kind: [np.mean([r.re_eff() for r in pick(runs, kind, m, g)]) for g in sorted({r.n for r in runs})] for kind in tf.LIKE}
        parts.append(f"Mach {m}: Euler kernel {re['gpu-hydro'][0]:.0f} → {re['gpu-hydro'][-1]:.0f}, RAMSES HLL {re['ramsesfix-matched'][0]:.0f} → {re['ramsesfix-matched'][-1]:.0f}, "
                     f"RAMSES HLLC {re['ramsesfix-best'][0]:.0f} → {re['ramsesfix-best'][-1]:.0f} ({re['ramsesfix-best'][-1]/re['ramsesfix-matched'][-1]:.1f} times HLL), "
                     f"MHD kernel {re['gpu-glm'][0]:.0f} → {re['gpu-glm'][-1]:.0f} ({pct(re['gpu-glm'][-1]/re['gpu-hydro'][-1] - 1, 0)} against the Euler kernel)")
    out["re"] = "From 128³ to 256³. " + "; ".join(parts) + "."
    return out


FIGURES = (   # file, title, caption, key of the finding
    ("turb-mach-time", "Steady state", "Rms Mach number (volume-weighted, with the sound speed of the gas at that time) against time in driving times T. "
     "Averages on this page are taken from 2 T on; the check is that the second half of that window agrees with the first.", "mach"),
    ("turb-projection-gap", "What RAMSES's projection does", "Time-averaged Mach number and power taken up from the driving (the slope of the total energy), "
     "each relative to Tom's Euler kernel at the same Mach number and grid (mean of its two seeds). Every run of a Mach number uses the same driving strength.", "gap"),
    ("turb-spectra", "Velocity spectra", "Time-averaged velocity power spectra, multiplied by k² (flat for a field of shocks). Runs that share one driving only; "
     "Tom's kernels are the mean of two seeds. The two set-ups that run the same scheme in two codes are dashed with shifted dashes: where they coincide the "
     "line alternates green and purple.", "spectra"),
    ("turb-compressive-share", "Compressive share, scale by scale", "The part of the velocity power at each wavenumber that is compressive (along k). "
     "The driving acts at 1 < k < 3 and is meant to be solenoidal.", "share"),
    ("turb-density-pdf", "Density distribution", "Time-averaged distribution of s = ln(ρ/⟨ρ⟩) by volume on the finer grid. The vertical line is the density floor.", "pdf"),
    ("turb-dissipation", "Numerical dissipation", "Effective Reynolds number v L/ν, with ν the viscosity that would turn the measured power into heat through the "
     "velocity gradients the grid holds: ν = power / (⟨ρ|∇×v|²⟩ + 4/3 ⟨ρ(∇·v)²⟩). It grows with resolution; on one grid a larger value means sharper gradients "
     "for the same dissipation. Short bars: the single seeds.", "re"),
    ("turb-slices", "What it looks like", "Density in the midplane at the end of each run (time 10 T) on the finer grid; light is dense, and each row has its own "
     "colour scale; cells below the density floor are drawn at it. The runs of a row have the same statistics of the driving but not the same random pattern (two codes, two random number generators; Tom's two "
     "kernels share theirs, and so do RAMSES's two solvers), so structures are not expected to match from code to code.", None),
)


def figure_html(i, name, title, caption, finding):
    file = f"media/turb/{name}.png";width = png_width(SITE/file)//2
    src = f"{file}?v={int((SITE/file).stat().st_mtime)}"        # a redrawn figure gets a new address, so a browser does not show its old copy
    out = [f'<article class="card paper-fig" id="{name}"><div class="fig-head"><span class="fig-no">{i}</span><h3>{e(title)}</h3></div>',
           f'<div class="fig-pair"><figure class="ours" style="flex-basis:{int(.8*width)}px;max-width:{width}px"><a class="image-link" href="{src}">'
           f'<img src="{src}" alt="{e(title)}" loading="lazy"></a></figure></div><p>{e(caption)}</p>']
    if finding:out.append(f'<p class="finding"><strong>In these runs:</strong> {e(finding)}</p>')
    return "".join(out) + "</article>"


def table_html(runs):
    cols = ("run", "Mach", "second half vs first", "mass-weighted Mach", "σ of ln ρ", "power", "effective Re", "sound speed at end", "lowest density", "cells at the floor", "cell updates / s")
    rows = []
    for r in sorted(runs, key=lambda r: (r.mach, r.n, r.name)):
        speed = r.meta.get("cell_updates_per_second") or r.meta.get("perf", {}).get("cell_updates_per_second_per_rank")
        unit = "" if r.meta.get("code") == "gpu" else " per core"
        cells = (r.name, f"{r.mean('mach_vol'):.2f}", pct(halves(r)), f"{r.mean('mach_mass'):.2f}", f"{r.mean('sigma_s'):.3f}", f"{r.power():.3g}", f"{r.re_eff():.0f}",
                 f"{r.series['cs'][-1]:.4f}", f"{r.series['rho_min'][r.window].min():.1e}", (f"{100*floor_share(r):.2f} %" if floor_share(r) else "0"), (f"{speed:.2e}{unit}" if speed else "–"))
        rows.append("<tr>" + "".join(f"<td>{e(c)}</td>" for c in cells) + "</tr>")
    return ('<div class="table-scroll"><table class="metrics-table sweep-text paper-table"><thead><tr>' + "".join(f"<th>{e(c)}</th>" for c in cols) +
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def build(runs):
    if not runs:
        return (f'<div class="page-heading"><div><div class="eyebrow"><span>{PAGE["number"]} · Driven turbulence</span><span class="badge">Waiting for data</span></div>'
                '<h1>Driven turbulence: the pilot</h1><p>The result files are not on this machine yet.</p></div></div>'
                f'<div class="card paper-state"><h3>Fetch them</h3><pre class="command">{e(FETCH)}</pre><p>Then: <code>python3 scripts/build_turbulence.py</code></p></div>'), "Waiting for data"
    f = findings(runs);machs = sorted({r.mach for r in runs});grids = sorted({r.n for r in runs});kinds = [k for k in tf.KINDS if any(r.kind == k for r in runs)]
    chips = [f"{len(runs)} runs", "Mach " + (", ".join(str(m) for m in machs[:-1]) + " and " + str(machs[-1]) if len(machs) > 1 else str(machs[0])), " and ".join(f"{n}³" for n in grids), f"{len(kinds)} set-ups in 2 codes", "10 driving times each",
             "solenoidal driving"]
    out = [f'''
<div class="page-heading"><div><div class="eyebrow"><span>{PAGE["number"]} · Driven turbulence</span><span class="badge">{len(runs)} runs</span></div>
<h1>Driven turbulence: the pilot</h1>
<p>Hydrodynamic turbulence in a periodic box, stirred until it is statistically steady, in Tom Abel's GPU kernels and in RAMSES with one definition of the
stirring. A first look: {len(machs)} Mach numbers, {len(grids)} grids. It also found a fault in RAMSES's driving, which is why RAMSES appears twice at the lower two
Mach numbers.</p></div>
<a class="method-link" href="../sherlock_turb/README.md">Set-up and tests <span>↗</span></a></div>
<div class="chips">{"".join(f'<span class="chip">{e(c)}</span>' for c in chips)}</div>
<section class="test-family"><h2>Focused high-Mach results</h2><div class="test-grid">
<a class="test-card" href="test-driven-mach-30.html"><strong>Mach 30</strong><span>12 completed runs on 128³ and 256³: spectra, density, dissipation and validity.</span><em>Open results →</em></a>
<a class="test-card" href="test-driven-mach-50.html"><strong>Mach 50</strong><span>12 completed runs on 128³ and 256³: spectra, density, dissipation and validity.</span><em>Open results →</em></a>
</div></section>
<div class="card"><h3>How to read this page</h3>
<p>Units: box length 1, mean density 1, sound speed 1. The gas is nearly isothermal (γ − 1 = 10⁻⁵, and 10⁻⁶ at Mach 30 and 50; no cooling), so the total energy rises at exactly the rate the
driving puts it in: that slope is the power, and in a steady state the dissipation rate. The driving is a random acceleration on the largest scales (1 &lt; k &lt; 3) with
memory time T = L/(2 M c<sub>s</sub>), scaled to the same strength at every moment in both codes. The strength was found once for each target Mach number and is used
unchanged by every run, so a run is named by its target and the Mach number it reaches is a result.</p>
<p>Colour is the code: green Tom's kernels, purple RAMSES. Line style is the set-up. <strong>The paler purple is RAMSES as released</strong>: its projection of the driving
takes the direction of a wave vector from the absolute values of its components, so a fifth of the power of its "solenoidal" force is compressive (measured from the force
fields it saves). The full purple has those three lines corrected and is the RAMSES to compare with Tom's kernels; at Mach 30 and 50 only the corrected RAMSES ran.
The green diamonds of figure 2 are the reverse test: Tom's Euler kernel driven with RAMSES's released projection. His Float16 Euler kernel is not here: it stops in its first
steps, because it stores the energy flux in half precision and this gas has an internal energy of 10⁵.</p>
<p>Density images use the notebook's map for gas quantities (teal, dark where the density is low), as on the other pages.</p></div>''']
    movie = SITE/"media"/"tests"/"driven_turbulence-density.mp4"
    if movie.is_file():           # scripts/make_turb_movie.py, from the movie runs (sherlock_turb/movie.sh)
        out.append(f'<article class="card test-movie"><div class="fig-head"><h3>Movie</h3></div><video controls loop muted playsinline preload="metadata" '
                   f'src="media/tests/{movie.name}?v={int(movie.stat().st_mtime)}"></video><p>Density in the midplane of Tom\'s Euler kernel on 256³ at each Mach number, '
                   "20 pictures per driving time, from the start of the driving to 10 driving times. Each panel has its own colour scale, fixed over the run. "
                   "These are the runs of seed 1 on this page, repeated with the density saved more often.</p></article>")
    out += [figure_html(i + 1, name, title, caption, f.get(key)) for i, (name, title, caption, key) in enumerate(FIGURES) if (SITE/"media"/"turb"/f"{name}.png").is_file()]
    out.append(f'<article class="card paper-fig" id="turb-runs"><div class="fig-head"><span class="fig-no">{len(FIGURES) + 1}</span><h3>Every run</h3></div>'
               f'<p>Averages over time ≥ {tf.FROM:g} T. Power in units of ρ c<sub>s</sub>³/L. Cell updates per second: the whole GPU for Tom\'s kernels (single precision), '
               f'one core for RAMSES (double precision).</p>{table_html(runs)}</article>')
    out.append(f'<div class="card paper-state"><h3>Where the data comes from</h3><p>turb_results/: the small files of every run, copied from Sherlock with</p>'
               f'<pre class="command">{e(FETCH)}</pre><p>Then <code>python3 scripts/build_turbulence.py</code>.</p></div>')
    return "".join(out), f"{len(runs)} runs"


def nav_html(tests, current):
    return "".join(f'<a href="{e(t["page"])}"' + (' aria-current="page"' if t["id"] == current else "") + f'><span>{t["number"]}</span>{e(t["nav"])}</a>' for t in tests)


def render(main_html, tests):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="Driven hydrodynamic turbulence in Tom Abel's GPU kernels and RAMSES: steady state, spectra, dissipation.">
<title>Turbulence · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="turbulence"><a class="skip-link" href="#main">Skip to the figures</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> DRIVEN TURBULENCE</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav_html(tests, PAGE["id"])}</nav>
<main id="main">{main_html}</main>
<footer><span>Drawn by scripts/turb_figures.py from turb_results/ (copied from Sherlock; sherlock_turb/README.md).</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main(argv):
    have = tf.DATA.is_dir() and any(tf.DATA.glob("*/series.csv"))
    runs = tf.load() if have else []
    if runs and "--page-only" not in argv:tf.main()
    main_html, status = build(runs)
    main_html += diagnostics.section("sherlock_turb")
    tests = json.loads((SITE/"tests.json").read_text());page = dict(PAGE, status=status)
    if not any(t["id"] == PAGE["id"] for t in tests):tests.append(page)
    tests = [page if t["id"] == PAGE["id"] else t for t in tests]
    (SITE/"tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=2) + "\n")
    (SITE/PAGE["page"]).write_text(render(main_html, tests))
    link = f'<a href="{PAGE["page"]}"><span>{PAGE["number"]}</span>{PAGE["nav"]}</a>'
    for t in tests:                       # the navigation of the pages already built
        path = SITE/t["page"]
        if t["id"] == PAGE["id"] or not path.is_file():continue
        text = path.read_text()
        if PAGE["page"] not in text and "</nav>" in text:path.write_text(text.replace("</nav>", link + "</nav>", 1))
    print(f"Turbulence page: {status}; {SITE/PAGE['page']}")


if __name__ == "__main__":
    main(sys.argv[1:])
