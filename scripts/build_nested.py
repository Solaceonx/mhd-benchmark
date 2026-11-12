"""The "Nested turbulence" page: Tom Abel's nested-grid code (NestedMHD) on Sherlock's H100s, a nest against a uniform grid
at its finest resolution in driven Mach-10 MHD.
    python3 scripts/build_nested.py [--page-only]
Draws the figures and movies first (scripts/nested_figures.py) unless --page-only; every number on the page comes from
progress/media/nested/nested.json, which that script writes from the records in nested_results/."""
import html
import json
import struct
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
MEDIA = SITE/"media"/"nested"
sys.path.insert(0, str(ROOT/"scripts"))
from publish_site import PAGES  # noqa: E402

PAGE = "nested.html"
e = html.escape


def png_width(path):
    head = path.read_bytes()[:24]
    return struct.unpack(">I", head[16:20])[0] if head[:8] == b"\x89PNG\r\n\x1a\n" else 0


def figure(i, name, title, caption, finding=None):
    f = MEDIA/f"{name}.png"
    if not f.is_file():
        return ""
    width = png_width(f) // 2; src = f"media/nested/{name}.png?v={int(f.stat().st_mtime)}"
    out = [f'<article class="card paper-fig" id="{name}"><div class="fig-head"><span class="fig-no">{i}</span><h3>{e(title)}</h3></div>',
           f'<div class="fig-pair"><figure class="ours" style="flex-basis:{int(.8 * width)}px;max-width:{width}px"><a class="image-link" href="{src}">'
           f'<img src="{src}" alt="{e(title)}" loading="lazy"></a></figure></div><p>{caption}</p>']
    if finding:
        out.append(f'<p class="finding"><strong>In these runs:</strong> {finding}</p>')
    return "".join(out) + "</article>"


def movie(case, m, title, text):
    mp4 = MEDIA/f"{case}-movie.mp4"
    if not mp4.is_file():
        return ""
    v = int(mp4.stat().st_mtime); src = f"media/nested/{case}-movie"
    return f'''<article class="card test-movie" id="movie-{case}"><h3>{e(title)}</h3>
<video controls loop muted playsinline preload="metadata" poster="{src}.png?v={v}" style="width:100%;height:auto;max-height:80vh;object-fit:contain"
aria-label="{e(title)}"><source src="{src}.mp4?v={v}" type="video/mp4"><a href="{src}.mp4">Download the movie</a></video>
<p>{text} {m["frames"]} frames, one every {m["dt"]:g} from t = 0 to {m["t_last"]:g}{"" if m["finished"] else " (the run stopped there)"}, played at 25 per second.
Every level is drawn in its own place at the finest resolution (coarser levels as blocks); dashed squares mark levels 2 and 3. The colour scales are fixed.
This is a rerun of the same set-up with outputs every 0.002; stopping that often changes the time steps slightly, so it does not follow the run of the figures
below in detail.</p>
<p><a href="{src}.mp4?v={v}" download>Download MP4 · {m["mb"]:g} MB</a></p></article>'''


def f2(x, d=2):
    return f"{x:.{d}f}"


def build(J):
    R = J["runs"]; tu = J["t_uniform"]; cu = J["cell_updates"]; dt = J["dt_finest"]; sr, sb = J["spectra_ratio_central"], J["spectra_ratio_box"]
    wall = {c: R[c]["wall"] for c in R}
    agree = J["gpu"]["agreement"]; speed = J["gpu"]["speed"]
    unif32 = [s["rate"] for s in speed if s["setup"].startswith("uniform") and s["T"] == "Float32"]
    nest_ct = next(s["rate"] for s in speed if s["setup"].startswith("nest") and s["T"] == "Float32" and s["mode"] == "ct")
    nest_glm = next(s["rate"] for s in speed if s["setup"].startswith("nest") and s["T"] == "Float32" and s["mode"] == "glm")
    ratio_wall = J["wall_02_to_tu"]["uni256"] / J["wall_02_to_tu"]["nest64"]
    ratio_steps = J["steps_to_tu"]["uni256"][0] / J["steps_to_tu"]["nest64"][-1]
    bt = J["brms_box_at_tu"]; bc = J["brms_central_range"]; mb = J["mach_box"]
    number = next(i for i, (src, _, _) in enumerate(PAGES, 1) if src == PAGE)
    chips = ["3 levels, refinement ratio 2", "driven Mach 10, plasma β 200", "constrained transport, HLL, Courant 0.25",
             f"nest 3 × 64³ to t = 1 in {wall['nest64'] / 60:.0f} min", f"nest 3 × 128³ to t = 0.5 in {wall['nest128'] / 60:.0f} min",
             f"div B ≤ {max(J['divb'].values()):.0e}", "one H100 per run"]
    out = [f'''
<div class="page-heading"><div><div class="eyebrow"><span>{number:02d} · Nested turbulence</span><span class="badge">3 runs + movies</span></div>
<h1>Nested turbulence: Tom Abel's NestedMHD on a GPU</h1>
<p>A nest puts a finer grid inside a coarser one: here three levels, each twice as fine as the one around it, centred in a periodic box, so the central
quarter of the box is computed at four times the resolution of the rest. Tom's question: does the finest level of a nest reproduce what a uniform grid at
that resolution gives, for much less computing? We ran his code (<a href="https://github.com/yipihey/NestedTurbulence">NestedTurbulence</a>, MIT) on
Sherlock's H100s, which it had not run on before, fixed what stopped it in 3-D, and set a nest against a uniform grid in driven Mach-10 MHD turbulence.</p></div>
<a class="method-link" href="../sherlock_nested/README.md">Set-up, fixes and jobs <span>↗</span></a></div>
<div class="chips">{"".join(f'<span class="chip">{e(c)}</span>' for c in chips)}</div>
<div class="card"><h3>The short answer so far</h3><ul>
<li><strong>It runs, and it is cheap.</strong> The 3 × 64³ nest (finest 256³-equivalent) reached t = 1 in {wall["nest64"]:.0f} s. Over t = 0.2–{tu:g} it took
{J["wall_02_to_tu"]["nest64"]:.0f} s where the uniform 256³ grid took {J["wall_02_to_tu"]["uni256"] / 60:.0f} min ({ratio_wall:.0f}×). Constrained transport keeps div B at
round-off ({max(J["divb"].values()):.0e}) on every level.</li>
<li><strong>Large-scale statistics agree.</strong> Over the whole box the rms Mach number is {f2(mb["nest64"])} (nest) and {f2(mb["uni256"])} (uniform), and at t = {tu:g} the
rms field is {f2(bt["nest64"])} and {f2(bt["uni256"])}. The density and speed PDFs of the central cube match.</li>
<li><strong>Small scales do not, yet.</strong> At wave numbers 32–96 the nest's finest level carries {sr["v"] * 100:.0f} % of the uniform grid's velocity power,
{sr["B"] * 100:.0f} % of its field power and {sr["rho"] * 100:.0f} % of its density power at the same resolution (figure 4).</li>
<li><strong>The uniform grids are the slow part.</strong> The uniform 256³ run stalled at t = {tu:g}; the uniform 512³ run reached no output in about 2 hours. Nearly empty cells
(ρ near the floor of 10⁻⁶) make the Alfvén speed, and so the time step, collapse. The 3 × 128³ nest meets them too (figure 5).</li></ul></div>''']
    M = J.get("movies", {})
    if "nest128" in M:
        out.append(movie("nest128", M["nest128"], "Movie: nest 3 × 128³ (finest 512³-equivalent)",
                         "Density and field strength in the mid-plane z = 1/2, from the uniform start to developed turbulence."))
    if "nest64" in M:
        out.append(movie("nest64", M["nest64"], "Movie: nest 3 × 64³ (finest 256³-equivalent)",
                         "Density and field strength in the mid-plane z = 1/2, from the uniform start to developed turbulence."))
    if not M:
        out.append('<div class="card"><h3>Movies</h3><p>The movie runs (a mid-plane slice of every level every 0.002, from t = 0 to 1) are on Sherlock now; '
                   'they are added here when they finish.</p></div>')
    figs = [
        ("nested-overview", "What the 3 × 128³ nest looks like",
         "The mid-plane at t = 0.5. Left: the whole box, level 1 (128³ across the box) with level 3 (the central quarter, 512³-equivalent) pasted in; level 2 "
         "is computed but was not saved in these runs (the movies show it). Right: level 3 alone. Top ln ρ, bottom log₁₀ |B|.", None),
        ("nested-slices", f"Nest and uniform grid at the same time (t = {tu:g})",
         f"Mid-plane slices of the 3 × 64³ nest (top) and the uniform 256³ grid (bottom): the whole box at 64³ (the uniform grid averaged over 4³ blocks) and the central "
         f"cube at 256³ cells across the box, the nest's finest level. Both runs have the same driving with the same random seed.",
         "the pictures differ: a turbulent flow is chaotic, so small differences between the two grids grow until the flows no longer match point by point, within "
         "a few crossing times (L / 10 c<sub>s</sub> = 0.1). They are compared by their statistics."),
        ("nested-stats", "Statistics",
         "Top: PDFs over the central cube (the nest's finest level, and the same cells of the uniform grid). Bottom: rms speed and field over the whole box at the "
         "nest's root resolution, and the rms field of the central cube.",
         f"the Mach number holds at about 10 in every run (whole-box averages {f2(mb['nest64'])}, {f2(mb['uni256'])} and {f2(mb['nest128'])}). The whole-box field grows "
         f"in the same way in the nest and the uniform grid ({f2(bt['nest64'])} and {f2(bt['uni256'])} at t = {tu:g}). The central cube is too small to compare its field: "
         f"its rms swings between {f2(bc['nest64'][0])} and {f2(bc['nest64'][1])} in the nest and {f2(bc['uni256'][0])} and {f2(bc['uni256'][1])} in the uniform grid. "
         "The uniform grid has more weak-field cells (log₁₀ |B| below −2)."),
        ("nested-spectra", "Power spectra",
         "Spectra of the velocity (times k²), the magnetic field and the density, averaged over the snapshots (every 0.05). Top: the whole periodic box at the nest's root "
         "resolution. Bottom: the central cube at the finest resolution; it is not periodic, so it is multiplied by a Hann window first, which sets the lowest "
         "wave numbers (shaded). The nest 3 × 128³ is averaged to t = 0.5, the other two to the uniform run's last output; the field is still growing, so its "
         "spectra are not compared across those times.",
         f"over the whole box, outside its finer levels the nest is a 64³ simulation: at k = 8–24 it has {sb['v'] * 100:.0f} % of the uniform grid's velocity power. "
         f"In the central cube, at the same resolution, the nest's finest level has {sr['v'] * 100:.0f} % of the velocity power, {sr['B'] * 100:.0f} % of the field "
         f"power and {sr['rho'] * 100:.0f} % of the density power at k = 32–96. Gas crosses the central cube (a quarter of the box) in about 0.025 at Mach 10, "
         "probably too short for the cascade to rebuild the small scales the coarser levels never had. That is the next thing to test (a larger finest level, "
         "or more levels)."),
        ("nested-cost", "Cost and time step",
         "Left: wall-clock time on one H100. Middle: the mean time step of the finest grid between outputs. Right: the lowest density in the central cube, and over "
         "the whole uniform 256³ grid.",
         f"to t = {tu:g} the uniform 256³ grid took {J['steps_to_tu']['uni256'][0]:,} steps, {ratio_steps:.1f} times as many as the nest's finest level "
         f"({J['steps_to_tu']['nest64'][-1]:,}). Its step fell to {dt['uni256']['min']:.0e} while the 3 × 64³ nest's never went below {dt['nest64']['min']:.0e}. "
         f"The cause is nearly empty cells: where ρ approaches the floor (10⁻⁶) the Alfvén speed B/√ρ becomes huge. The 3 × 128³ nest, with a finer finest level, "
         f"meets them as well (its step fell to {dt['nest128']['min']:.0e} near t = 0.29), so they come with resolution, not with the uniform grid itself."),
    ]
    out += [figure(i + 1, *f) for i, f in enumerate(figs)]
    rows = [("nest 3 × 64³", "256³ in the central 1/4", "Float64", "1 (finished)", f"{wall['nest64']:.0f} s", " / ".join(f"{s:,}" for s in R["nest64"]["steps"]),
             f"{cu['nest64']:.2e}", f"{cu['nest64'] / wall['nest64']:.2e}"),
            ("uniform 256³", "256³ everywhere", "Float64", f"{R['uni256']['t']:g} (stalled; time limit)", f"{wall['uni256']:.0f} s", f"{R['uni256']['steps'][0]:,}",
             f"{R['uni256']['steps'][0] * 256 ** 3:.2e}", f"{R['uni256']['steps'][0] * 256 ** 3 / wall['uni256']:.2e}"),
            ("nest 3 × 128³", "512³ in the central 1/4", "Float32", "0.5 (finished)", f"{wall['nest128']:.0f} s", " / ".join(f"{s:,}" for s in R["nest128"]["steps"]),
             f"{cu['nest128']:.2e}", f"{cu['nest128'] / wall['nest128']:.2e}"),
            ("uniform 512³", "512³ everywhere", "Float32", "below 0.2 (no output; pre-empted after about 2 h)", "–", "–", "–", "–")]
    cols = ("run", "finest resolution", "precision", "reached t", "wall", "steps per level (coarsest first)", "cell updates", "cell updates / s")
    out.append(f'<article class="card paper-fig" id="nested-runs"><div class="fig-head"><span class="fig-no">{len(figs) + 1}</span><h3>Every run</h3></div>'
               '<p>One H100 each. The nests advance each level with its own time step, so the finest level takes the most steps. The rates include the driving and '
               'the outputs every 0.01; small grids do not fill the GPU.</p><div class="table-scroll"><table class="metrics-table sweep-text paper-table"><thead><tr>'
               + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr></thead><tbody>"
               + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div></article>")
    out.append(f'''<div class="card"><h3>Set-up</h3>
<p>Box of side 1, mean density 1, sound speed 1, isothermal. A uniform field along x with plasma β = 200 (B₀ = 0.1). A solenoidal random acceleration on
1 ≤ |k| ≤ 3, scaled continuously to hold the rms Mach number at 10, random seed 1 in every run. Tom's MUSCL-Hancock scheme with HLL fluxes and constrained
transport, Courant 0.25, density floor 10⁻⁶. The nests: each level has as many cells as the root (64³ or 128³) and covers the central half of the level
around it; each level takes its own time step (Berger–Oliger), with refluxing and an EMF correction at the level boundaries, so mass and div B = 0 are kept.</p></div>
<div class="card"><h3>What it took to run in 3-D on a GPU</h3><ul>
<li><strong>The GPU agrees with the CPU</strong> to {max(a["diff"] for a in agree if a["T"] == "Float64"):.0e} in double precision and {max(a["diff"] for a in agree if a["T"] == "Float32"):.0e}
in single, with the same number of steps on every level (a nested Orszag–Tang, CT and GLM). Speed: a uniform grid {min(unif32) / 1e9:.1f}–{max(unif32) / 1e9:.1f} × 10⁹ cell updates
per second in single precision; the 3 × 128³ nest {nest_ct / 1e9:.2f} × 10⁹ with CT and {nest_glm / 1e9:.2f} × 10⁹ with GLM.</li>
<li><strong>Courant 0.4 is unstable in 3-D</strong> with these kernels, even on a uniform grid (they add the three directions' signal speeds); 0.25 is used. Tom's own runs were 2-D.</li>
<li><strong>Ghost cells could go negative.</strong> Filling a fine level's ghost cells from its parent added three independently limited slopes, so next to a nearly empty
cell a corner ghost could reach −½ of its parent's density; every 3-D nest failed at t ≈ 0.034. Fixed by one factor that scales all the slopes of a cell so the fine
density stays above the lowest of its six coarse neighbours.</li>
<li><strong>Faster driving.</strong> The driving's totals are now summed on the GPU instead of copying the root level to the CPU every step.</li></ul>
<p>The fixes are on our branch <code>gpu-3d-fixes</code> of his code; they are not in Tom's repository yet.</p></div>
<div class="card"><h3>Open</h3><ul>
<li>A fair uniform reference at 256³ and 512³: either a longer run, or the same safeguard for nearly empty cells in both (a higher floor, or a cap on the Alfvén speed).</li>
<li>Why the finest level is short of small-scale power: a larger finest level or more levels would tell.</li>
<li>Longer runs of the nests, for time-averaged statistics once the field has saturated.</li></ul></div>
<div class="card paper-state"><h3>Where the data comes from</h3><p>nested_results/ (copied from Sherlock): compare/ (histograms every 0.01, snapshots every 0.05),
movie/ (mid-plane slices every 0.002), logs/ and gpu_check/. Figures: <code>python3 scripts/nested_figures.py</code>; page: <code>python3 scripts/build_nested.py</code>.</p></div>''')
    return "".join(out)


def render(main_html):
    nav = "".join(f'<a href="{src}"' + (' aria-current="page"' if src == PAGE else "") + f'><span>{i:02d}</span>{e(label)}</a>'
                  for i, (src, _, label) in enumerate(PAGES, 1))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="Tom Abel's nested-grid MHD code on a GPU: a nest against a uniform grid in driven Mach-10 turbulence.">
<title>Nested turbulence · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="nested"><a class="skip-link" href="#main">Skip to the figures</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> NESTED TURBULENCE</div></header>
<nav class="test-nav" aria-label="Pages">{nav}</nav>
<main id="main">{main_html}</main>
<footer><span>Drawn by scripts/nested_figures.py from nested_results/ (copied from Sherlock; sherlock_nested/README.md).</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main(argv):
    if "--page-only" not in argv:
        import nested_figures
        nested_figures.main()
    J = json.loads((MEDIA/"nested.json").read_text())
    (SITE/PAGE).write_text(render(build(J)))
    print("Nested turbulence page:", SITE/PAGE)


if __name__ == "__main__":
    main(sys.argv[1:])
