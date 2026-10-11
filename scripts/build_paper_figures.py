"""Build the "Papers" sheet of the progress site (progress/paper-figures.html): the figures of the comparison papers
redrawn from the comprehensive sweep, with what each one reproduces and where our set-up is not the paper's.

    python3 scripts/sweep_fetch.py ...        copy the sweep's result files here (see that script; one command)
    python3 scripts/build_paper_figures.py    draw every figure (scripts/paper_figures.py) and write the page
    python3 scripts/build_paper_figures.py --page-only   rewrite the page from the figures already drawn

The page lists every figure of every paper: drawn here, drawn in a different form, or not possible yet and why.
Also adds the page to progress/tests.json and to the navigation of the pages already built.
"""
import html
import json
import re
import struct
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common  # noqa: E402
import paper_figures  # noqa: E402
import sweep_data  # noqa: E402

SITE = paper_common.SITE
PAGE = dict(id="paper-figures", page="paper-figures.html", kind="papers", number="08", title="Paper figures", nav="Papers",
            status="Waiting for data", document="../scripts/paper_figures.py",
            description="The figures of the comparison papers, redrawn with every code.", chips=[])
e = html.escape
KIND = dict(reproduced=("Same test", "the paper's problem and parameters on our periodic unit box"),
            analogue=("Same measurement, different set-up", "the paper's quantity and layout; our problem differs as stated"),
            extension=("Added here", "not a figure of the paper: a measurement our set-up needs or allows"))

# Every figure of every paper: (figure, what the paper shows, ids of our figures or None, why not / note).
PAPERS = [
    dict(id="tomida", cite="Tomida et al. (2026)", ref="arXiv:2605.07928", url="https://arxiv.org/abs/2605.07928",
         title="Constrained transport against mixed divergence cleaning",
         about=("Kelvin–Helmholtz instability with a weak field: uniform, confined to one layer, and with a sudden timestep change. "
                "The paper's CT, D1v/Dhv (cleaning speed following the timestep) and D1c (constant speed) correspond to our CT prototype and "
                "AthenaK/RAMSES/Athena++, the four GLM kernels with c_h ∝ 1/Δt, and the constant-c_h kernel."),
         rows=[("Fig. 1", "KH, uniform field: |B| and ψ at t = 3, h = 1/128 and 1/256", ["tomida-1-gpu", "tomida-1-other"], ""),
               ("Fig. 2", "KH, uniform field: |B| along y through the vortex", ["tomida-2"], ""),
               ("Fig. 3", "Constant cleaning speed c_h = 1, 2, 4, 8", None, "The sweep has one constant speed (the GLM constant c_h column of Fig. 1); a scan needs new runs."),
               ("Fig. 4", "KH, confined field: |B| and ψ at t = 3", ["tomida-4-gpu", "tomida-4-other"], ""),
               ("Fig. 5", "KH, confined field: |B| along y (leakage)", ["tomida-5", "tomida-5-core"], ""),
               ("Figs. 6, 8", "Sudden timestep change: |B| and ψ before and after", ["tomida-6"], ""),
               ("Fig. 7", "Sudden timestep change: largest ψ against time", ["tomida-7"], ""),
               ("Figs. 9–15", "Collapsing clouds; convective dynamo", None, "Not in the test suite (self-gravity, AMR, stratified convection)."),
               ("Figs. 16–18", "Appendix: Powell's method; thermal energy error of extended GLM", None, "Not in the test suite.")]),
    dict(id="kritsuk", cite="Kritsuk et al. (2011)", ref="ApJ 737, 13", url="https://arxiv.org/abs/1103.5525",
         title="Decay of supersonic magnetized turbulence",
         about=("Nine codes decay one Mach 9 turbulent state for 0.2 crossing times. Ours start from a smooth random velocity field at the same "
                "Mach number and mean field rather than the paper's driven, saturated state (that start is stage 3 of the sweep), so every figure "
                "here is the same measurement on a different flow."),
         rows=[("Fig. 1", "Kinetic and magnetic energy, sonic and Alfvénic Mach numbers against time, 512³", ["kritsuk-1-512", "kritsuk-1-256"], ""),
               ("Fig. 2", "Enstrophy + dilatation and mean squared current against time", ["kritsuk-2-512", "kritsuk-2-256"], ""),
               ("Fig. 3", "Velocity and magnetic power spectra at three times", ["kritsuk-3-512", "kritsuk-3-256"], ""),
               ("Fig. 4", "Spectra over a reference: effective bandwidth, 256³", ["kritsuk-4"], ""),
               ("Fig. 5", "Dilatational over solenoidal velocity power", ["kritsuk-5-512", "kritsuk-5-256"], ""),
               ("Table 1", "Selected numbers per code", ["kritsuk-table-1"], "")]),
    dict(id="gem", cite="Birn et al. (2001)", ref="JGR 106, 3715", url="https://doi.org/10.1029/1999JA900449",
         title="GEM magnetic reconnection challenge",
         about=("A Harris sheet with a prescribed perturbation, run with resistive MHD, Hall MHD, hybrid and particle codes. Ours is ideal MHD in "
                "every code, so the reconnected flux measures numerical resistivity."),
         rows=[("Fig. 1", "Reconnected flux against time for each model", ["gem-1"], "")]),
    dict(id="stone2005", cite="Gardiner & Stone (2005)", ref="JCP 205, 509", url="https://arxiv.org/abs/astro-ph/0501557",
         title="An unsplit Godunov method for ideal MHD via constrained transport",
         about="The 2D test suite of the CTU + CT scheme: field loop advection, Alfvén waves, a rotated shock tube, a current sheet and a blast wave.",
         rows=[("Fig. 1", "Stationary field loop with three CT variants", None, "No stationary loop in the sweep; the three variants are internal to the paper's scheme."),
               ("Figs. 2, 3, 6", "Advected field loop: magnetic pressure at t = 0.19 and t = 2", ["stone05-loop-gpu", "stone05-loop-other"], ""),
               ("Figs. 4, 5", "Circularly polarized Alfvén wave: three CT variants, standing and travelling", None, "Variants internal to the paper's scheme; our travelling wave is under Fig. 9."),
               ("Fig. 7", "Loop magnetic energy against time", ["stone05-loop-energy"], ""),
               ("Fig. 8", "Loop field lines at t = 0 and t = 2", ["stone05-loop-lines"], ""),
               ("Fig. 9", "Alfvén wave profile after five periods by resolution", ["stone05-alfven"], ""),
               ("Fig. 10", "Rotated shock tube: field along the normal", ["stone05-shock-bn"], ""),
               ("Fig. 11", "Linear wave convergence", ["stone05-convergence"], ""),
               ("Fig. 12", "Current sheet: field lines in time", ["stone05-sheet-gpu", "stone05-sheet-other"], ""),
               ("Fig. 13", "Blast wave: density, pressure, magnetic energy, field lines", ["stone05-blast-gpu", "stone05-blast-other"], "")]),
    dict(id="stone2008", cite="Gardiner & Stone (2008)", ref="JCP 227, 4123", url="https://arxiv.org/abs/0712.2634",
         title="An unsplit Godunov method for ideal MHD via constrained transport in three dimensions",
         about="The 3D test suite: an inclined field loop, linear and Alfvén waves, an inclined Riemann problem and a blast wave.",
         rows=[("Fig. 1", "3D field loop: magnetic energy and ⟨|B₃|⟩ against time by resolution", ["stone08-loop3d"], ""),
               ("Fig. 2", "3D field loop: thresholded images at t = 1", ["stone08-loop3d-images-gpu", "stone08-loop3d-images-other"], ""),
               ("sec. 3", "In-plane loop advected with v_z ≠ 0 (described, no figure)", ["stone08-loop-bz"], ""),
               ("Figs. 3–6", "3D linear wave convergence; 3D circularly polarized Alfvén wave; helicity", None, "No 3D wave tests in the sweep (our waves are 2D: Gardiner & Stone 2005, Figs. 9 and 11)."),
               ("Fig. 7", "Riemann problem inclined to the grid", ["stone08-riemann"], ""),
               ("Fig. 8", "3D blast wave: slices", ["stone08-blast-gpu", "stone08-blast-other"], "")]),
    dict(id="ugalino", cite="Ugalino et al. (2026)", ref="arXiv:2610.02308", url="https://arxiv.org/abs/2610.02308",
         title="Towards robust simulations of the galactic dynamo: a cross-code comparison between Eulerian and Lagrangian schemes",
         about=("An isolated Milky-Way-mass disc (AGORA initial conditions, self-gravity, cooling, a pressure floor, 40–80 pc cells, 7 Gyr) run "
                "with RAMSES (constrained transport), ART (Dedner cleaning), AREPO (Powell) and SWIFT (Dedner and Powell). The two grid "
                "codes agree with each other, cleaning or not; the moving-mesh and particle codes amplify the field by orders of magnitude "
                "more, and AREPO with a slowed mesh returns to the grid result. The paper has no idealized tests; our nearest is the "
                "amplification of a weak field in decaying Mach 9 turbulence."),
         rows=[("Fig. 5", "Magnetic-to-kinetic energy ratio, dimensionless divergence error and kinetic energy against time", ["ugalino-5-512", "ugalino-5-256"], ""),
               ("Figs. 1–4, 6–11 and later", "The disc: density and field maps, growth by radius and resolution, terms of the induction equation", None,
                "Not in the test suite (self-gravity, collisionless particles, cooling, adaptive mesh)."),
               ("sec. IV.4", "AREPO with a reduced mesh velocity", None,
                "Not this experiment: AREPO ran on the suite's planar tests with its moving mesh and with a static one (its section below), "
                "not on the disc and not with a slowed mesh.")]),
    dict(id="arepo", own=True, cite="AREPO, run after the sweep", ref="arepo-code.org", url="https://arepo-code.org",
         title="The public AREPO on the suite's planar tests: moving and static mesh, Powell's scheme",
         about=("AREPO (Springel 2010; Pakmor & Springel 2013; public release of Weinberger et al. 2020) is the code of the moving-mesh runs of "
                "Ugalino et al. (2026). It was run on the 17 planar tests of the suite, at every grid, on its moving mesh and on a static "
                "Cartesian mesh, with Powell's eight-wave scheme for the divergence of B and the HLLD solver: 164 runs on CPUs. It is not in the "
                "figures of the other sections, which hold the twelve set-ups of the sweep."),
         rows=[("Added", "Static against moving mesh: field loop, timestep, divergence of B", ["arepo-mesh"], ""),
               ("Added", "Convergence on the wave tests", ["arepo-convergence"], ""),
               ("Added", "Which runs finished", ["arepo-status"], "")]),
    dict(id="pakmor", cite="Pakmor et al. (2026)", ref="arXiv:2607.06660", url="https://arxiv.org/abs/2607.06660",
         title="A comparison of numerical schemes for driven subsonic MHD turbulence",
         about=("Driven subsonic turbulence (Mach 0.2) amplifying a seed field to saturation, with identical driving in every code. "
                "None of it can be drawn yet: the driving and the runs to t = 100 are stage 3 of the sweep, which is not built."),
         rows=[("all figures", "Growth rate, saturation level and spectra of the small-scale dynamo", None, "Needs the driven runs of stage 3.")]),
]


# The papers' own figures. Only Tomida et al. (2026) is published under a licence that allows copies (CC BY 4.0), so
# only its figures are stored here (progress/media/papers/original/tomida2026_*.png, from arxiv.org/html/2605.07928v2).
# For the others the page links to the figure in the paper; a copy the reader saves as
# progress/media/papers/original/<figure id>.png (or .jpg) is shown next to ours.
ORIGINAL_DIR = "media/papers/original"
CC_BY = "Tomida et al. (2026), arXiv:2605.07928, CC BY 4.0"
ORIGINALS = {
    "tomida-1-gpu": [("tomida2026_khuni.png", f"Fig. 1 of {CC_BY}")], "tomida-1-other": [("tomida2026_khuni.png", f"Fig. 1 of {CC_BY}")],
    "tomida-2": [("tomida2026_profu.png", f"Fig. 2 of {CC_BY}")],
    "tomida-4-gpu": [("tomida2026_khloc.png", f"Fig. 4 of {CC_BY}")], "tomida-4-other": [("tomida2026_khloc.png", f"Fig. 4 of {CC_BY}")],
    "tomida-5": [("tomida2026_profloc.png", f"Fig. 5 of {CC_BY}")],
    "tomida-6": [("tomida2026_khdt1_upright.png", f"Fig. 6 of {CC_BY}; turned upright"), ("tomida2026_khdt2_upright.png", f"Fig. 8 of {CC_BY}; turned upright")],
    "tomida-7": [("tomida2026_psimax.png", f"Fig. 7 of {CC_BY}")],
}
TOMIDA, KRITSUK = "https://arxiv.org/html/2605.07928v2#S3.F{}", "https://arxiv.org/pdf/1103.5525#page={}"
STONE05, STONE08 = "https://arxiv.org/pdf/astro-ph/0501557#page={}", "https://arxiv.org/pdf/0712.2634#page={}"
LINKS = {   # our figure -> (where the paper's figure is, link)
    "tomida-1-gpu": ("Fig. 1", TOMIDA.format(1)), "tomida-1-other": ("Fig. 1", TOMIDA.format(1)), "tomida-2": ("Fig. 2", TOMIDA.format(2)),
    "tomida-4-gpu": ("Fig. 4", TOMIDA.format(4)), "tomida-4-other": ("Fig. 4", TOMIDA.format(4)), "tomida-5": ("Fig. 5", TOMIDA.format(5)),
    "tomida-6": ("Figs. 6 and 8", TOMIDA.format(6)), "tomida-7": ("Fig. 7", TOMIDA.format(7)),
    "kritsuk-1-512": ("Fig. 1, p. 7", KRITSUK.format(7)), "kritsuk-1-256": ("Fig. 1, p. 7", KRITSUK.format(7)),
    "kritsuk-2-512": ("Fig. 2, p. 7", KRITSUK.format(7)), "kritsuk-2-256": ("Fig. 2, p. 7", KRITSUK.format(7)),
    "kritsuk-3-512": ("Fig. 3, p. 9", KRITSUK.format(9)), "kritsuk-3-256": ("Fig. 3, p. 9", KRITSUK.format(9)),
    "kritsuk-4": ("Fig. 4, p. 10", KRITSUK.format(10)), "kritsuk-5-512": ("Fig. 5, p. 10", KRITSUK.format(10)), "kritsuk-5-256": ("Fig. 5, p. 10", KRITSUK.format(10)),
    "kritsuk-table-1": ("Table 1, p. 6", KRITSUK.format(6)),
    "gem-1": ("Fig. 1", "https://doi.org/10.1029/1999JA900449"),
    "ugalino-5-512": ("Fig. 5", "https://arxiv.org/html/2610.02308v1"), "ugalino-5-256": ("Fig. 5", "https://arxiv.org/html/2610.02308v1"),
    "stone05-loop-gpu": ("Figs. 2, 3, 6, pp. 21, 22, 31", STONE05.format(21)), "stone05-loop-other": ("Figs. 2, 3, 6, pp. 21, 22, 31", STONE05.format(21)),
    "stone05-loop-energy": ("Fig. 7, p. 31", STONE05.format(31)), "stone05-loop-lines": ("Fig. 8, p. 31", STONE05.format(31)),
    "stone05-alfven": ("Fig. 9, p. 32", STONE05.format(32)), "stone05-shock-bn": ("Fig. 10, p. 34", STONE05.format(34)),
    "stone05-convergence": ("Fig. 11, p. 36", STONE05.format(36)),
    "stone05-sheet-gpu": ("Fig. 12, p. 38", STONE05.format(38)), "stone05-sheet-other": ("Fig. 12, p. 38", STONE05.format(38)),
    "stone05-blast-gpu": ("Fig. 13, p. 39", STONE05.format(39)), "stone05-blast-other": ("Fig. 13, p. 39", STONE05.format(39)),
    "stone08-loop3d": ("Fig. 1, p. 20", STONE08.format(20)),
    "stone08-loop3d-images-gpu": ("Fig. 2, p. 21", STONE08.format(21)), "stone08-loop3d-images-other": ("Fig. 2, p. 21", STONE08.format(21)),
    "stone08-loop-bz": ("the text on p. 19", STONE08.format(19)), "stone08-riemann": ("Fig. 7, p. 29", STONE08.format(29)),
    "stone08-blast-gpu": ("Fig. 8, p. 31", STONE08.format(31)), "stone08-blast-other": ("Fig. 8, p. 31", STONE08.format(31)),
}


def own_copy(ident):
    """A copy of the paper's figure that the reader saved for this figure (or for its -gpu/-other pair), or None."""
    for stem in (ident, re.sub(r"-(gpu|other|512|256)$", "", ident)):
        for ext in ("png", "jpg", "jpeg", "webp", "gif"):
            if (SITE/ORIGINAL_DIR/f"{stem}.{ext}").is_file():return f"{ORIGINAL_DIR}/{stem}.{ext}"
    return None


def image_width(path):
    """Pixel width of a PNG, JPEG, GIF or WebP file (0 if unknown)."""
    try:
        from PIL import Image
        with Image.open(path) as im:return im.size[0]
    except Exception:
        return png_width(path)


def png_width(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">I", head[16:20])[0] if head[:8] == b"\x89PNG\r\n\x1a\n" else 0


def cell(v):
    if v is None:return "—"
    if isinstance(v, float):return f"{v:.3g}"
    return e(str(v))


def table_html(t):
    head = "".join(f"<th>{e(c)}</th>" for c in t["columns"])
    body = "".join("<tr>" + "".join(f"<td>{cell(r.get(k))}</td>" for k in t["keys"]) + "</tr>" for r in t["rows"])
    return f'<div class="table-scroll"><table class="metrics-table sweep-text paper-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure_html(f):
    label, tip = KIND[f["kind"]];ident = f["id"]
    out = [f'<article class="card paper-fig" id="{e(ident)}"><div class="fig-head"><span class="fig-no">{e(f["paper_fig"])}</span>'
           f'<h3>{e(f["title"])}</h3><span class="kind kind-{f["kind"]}" title="{e(tip)}">{e(label)}</span></div>']
    theirs = [(f"{ORIGINAL_DIR}/{name}", credit) for name, credit in ORIGINALS.get(ident, []) if (SITE/ORIGINAL_DIR/name).is_file()]
    if not theirs and f["kind"] != "extension" and own_copy(ident):theirs = [(own_copy(ident), "the paper's figure (your copy)")]
    ours = ""
    if f.get("file"):
        width = png_width(SITE/f["file"])//2
        src = f'{f["file"]}?v={int((SITE/f["file"]).stat().st_mtime)}'       # a redrawn figure gets a new address, so a browser does not show its old copy
        ours = (f'<figure class="ours" style="flex-basis:{int(.8*width)}px;max-width:{width}px"><a class="image-link" href="{e(src)}">'
                f'<img src="{e(src)}" alt="{e(f["title"])}" loading="lazy"></a>' + ('<figcaption>Here</figcaption>' if theirs else '') + '</figure>')
    if theirs:
        blocks = []
        for path, credit in theirs:
            w = image_width(SITE/path);shown = min(w//2 if w else 460, 520 if len(theirs) == 1 else 760)
            blocks.append(f'<figure class="theirs" style="flex-basis:{shown}px;max-width:{max(shown, 320)}px"><a class="image-link" href="{e(path)}">'
                          f'<img src="{e(path)}" alt="{e(credit)}" loading="lazy"></a><figcaption>In the paper: {e(credit)}</figcaption></figure>')
        out.append(f'<div class="fig-pair">{"".join(blocks)}{ours}</div>')
    elif ours:out.append(f'<div class="fig-pair">{ours}</div>')
    out.append(f'<p>{e(f["caption"])}</p>')
    if f.get("finding"):out.append(f'<p class="finding"><strong>In these runs:</strong> {e(f["finding"])}</p>')
    if f.get("table"):
        t = f["table"];inner = table_html(t)
        out.append(inner if len(t["rows"]) <= 14 else f'<details><summary>Table: {len(t["rows"])} rows</summary>{inner}</details>')
    if f.get("differs"):out.append(f'<p class="small"><strong>Where this is not the paper\'s set-up:</strong> {e(f["differs"])}</p>')
    if ident in LINKS and f["kind"] != "extension" or ident == "stone08-loop-bz":
        where, url = LINKS[ident]
        hint = "" if theirs or ident == "stone08-loop-bz" else (f' To see it here next to ours, save your own copy of it as <code>progress/{ORIGINAL_DIR}/'
                                                                 f'{e(re.sub(r"-(gpu|other|512|256)$", "", ident))}.png</code> and rebuild with <code>--page-only</code>.')
        out.append(f'<p class="small paper-link">In the paper: <a href="{e(url)}" target="_blank" rel="noopener">{e(where)} ↗</a>.{hint}</p>')
    if f.get("missing"):
        items = "".join(f"<li>{e(m)}</li>" for m in f["missing"])
        out.append(f'<details class="missing"><summary>{len(f["missing"])} part{"s" if len(f["missing"]) != 1 else ""} not drawn or cut short</summary><ul>{items}</ul></details>')
    out.append("</article>")
    return "".join(out)


def coverage_html(paper, by_id):
    rows = []
    for fig, what, ids, note in paper["rows"]:
        if ids is None:state = '<span class="kind kind-none">Not possible yet</span>';ours = e(note)
        else:
            drawn = [by_id[i] for i in ids if i in by_id]
            if drawn:
                kinds = sorted({d["kind"] for d in drawn}, key=list(KIND).index)
                state = " ".join(f'<span class="kind kind-{k}">{e(KIND[k][0])}</span>' for k in kinds)
                ours = " · ".join(f'<a href="#{e(d["id"])}">{e(d["title"])}</a>' for d in drawn)
            else:state = '<span class="kind kind-wait">Waiting for data</span>';ours = "Drawn once the results are here."
        rows.append(f"<tr><td>{e(fig)}</td><td>{e(what)}</td><td>{state}</td><td>{ours}</td></tr>")
    return ('<div class="table-scroll"><table class="metrics-table sweep-text paper-cover"><thead><tr><th>In the paper</th><th>Shows</th><th>Here</th>'
            f'<th>Our figure</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def fetched():
    """(files here, files wanted) of the sweep results on this machine."""
    sw = sweep_data.Sweep();w = sw.wanted()
    have = sum(1 for p, s, a in w if (sw.results/p).is_file())
    return have, sum(1 for p, s, a in w if not a), sw


def build(entries, status_time):
    by_id = {f["id"]: f for f in entries}
    have, wanted, sw = fetched()
    drawn = [f for f in entries if f.get("file") or f.get("table")]
    counts = {k: sum(1 for f in drawn if f["kind"] == k) for k in KIND}
    not_possible = sum(1 for p in PAPERS for r in p["rows"] if r[2] is None)
    papers = [p for p in PAPERS if not p.get("own")]              # the AREPO section is ours, not a paper's
    badge = "Waiting for data" if not drawn else f"{len(drawn)} figures"
    if drawn:
        lede = (f"{len(drawn)} figures and tables from the sweep's own output, one for each figure of the comparison papers that our tests can make, "
                "each with the five GLM kernels and the CT prototype of T. Abel, AthenaK, RAMSES and Athena++ in place of the paper's codes. "
                "AREPO, run after the sweep, has a section of its own.")
        state = (f'<div class="card paper-state"><h3>What is drawn from</h3><p>{have:,} of {wanted:,} result files of the sweep are on this machine '
                 f'(run status as of {e(status_time or "?")}). A panel whose run is unfinished or was stopped by its solver says so; each figure lists those parts.</p></div>')
    else:
        lede = ("One figure for each figure of the comparison papers that our tests can make, with every code. The figures are drawn from the "
                "sweep's result files, which are not on this machine yet.")
        state = ('<div class="card paper-state"><h3>Waiting for the results</h3><p>The figures need about 1.3 GB of the sweep\'s output '
                 f'({wanted:,} files: every run\'s diagnostics and spectra, and the frames the images show). {have:,} are here. One command on the Mac '
                 'copies them (one password and Duo prompt; running it again fetches only what is missing):</p>'
                 '<pre class="command">cd /Users/lyxia/Projects/research/abel_mhd &amp;&amp; python3 scripts/sweep_fetch.py list | ssh lyxia@login.sherlock.stanford.edu '
                 '\'cd /scratch/users/lyxia/mhd-results-sweep &amp;&amp; while IFS= read -r f; do [ -f "$f" ] &amp;&amp; echo "$f"; done | tar -cf - -T -\' '
                 '| python3 scripts/sweep_fetch.py receive</pre><p>Then: <code>python3 scripts/build_paper_figures.py</code></p></div>')
    words = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    chips = [f"{sum(len(p['rows']) for p in papers)} figures and tables in {words.get(len(papers), len(papers))} papers", f"{counts['reproduced']} same test" if drawn else "",
             f"{counts['analogue']} same measurement, different set-up" if drawn else "", f"{counts['extension']} added here" if drawn else "",
             f"{not_possible} not possible yet", "12 methods + 2 references"]
    out = [f'''
<div class="page-heading"><div><div class="eyebrow"><span>{PAGE["number"]} · Paper figures</span><span class="badge">{e(badge)}</span></div>
<h1>The papers' figures, with every code</h1>
<p>{e(lede)}</p></div>
<a class="method-link" href="../scripts/paper_figures.py">How they are drawn <span>↗</span></a></div>
<div class="chips">{"".join(f'<span class="chip">{e(c)}</span>' for c in chips if c)}</div>
{state}
<div class="card"><h3>How to read this page</h3>
<p><span class="kind kind-reproduced">Same test</span> the paper's problem and parameters, on our periodic unit box.
<span class="kind kind-analogue">Same measurement, different set-up</span> the paper's quantity and layout, on a problem that differs as each figure states.
<span class="kind kind-extension">Added here</span> not in the paper: a measurement our set-up needs.
<span class="kind kind-none">Not possible yet</span> needs runs or tests the sweep does not have.</p>
<p>Colour is the code (green GLM, orange CT prototype, blue AthenaK, purple RAMSES, raspberry Athena++, ochre AREPO); line style is the variant within a code;
grey is a reference run. Image grids have one column per method. A cross on a curve marks where a solver stopped.</p>
<p>Images use the notebook's colour maps, the same on every page (scripts/plot_style.py): field strength in blue, gas quantities in teal, both dark where the value
is low; signed quantities such as ψ from blue (negative) through black (zero) to orange. The papers' own figures use other colours (Tomida et al.: black to orange
to white; Gardiner &amp; Stone: grey).</p>
<p><strong>The papers' own figures.</strong> Tomida et al. (2026) is published under CC BY 4.0, so its figures are shown beside ours. The other
papers carry no such licence: each figure links to its place in the paper, the numbers a paper prints (fitted curves, table values) are drawn on
our axes, and a copy you save yourself under <code>progress/{ORIGINAL_DIR}/</code> with the name given on the figure is shown beside ours.</p></div>''']
    for p in PAPERS:
        figs = [by_id[i] for r in p["rows"] if r[2] for i in r[2] if i in by_id and (by_id[i].get("file") or by_id[i].get("table"))]
        out.append(f'<section class="paper-section" id="{p["id"]}"><div class="paper-title"><h2>{e(p["cite"])}</h2>'
                   f'<a href="{e(p["url"])}">{e(p["ref"])} ↗</a></div><p class="paper-about"><strong>{e(p["title"])}.</strong> {e(p["about"])}</p>'
                   f'<div class="card">{coverage_html(p, by_id if drawn else {})}</div>{"".join(figure_html(f) for f in figs)}</section>')
    return "".join(out), dict(drawn=len(drawn), have=have, wanted=wanted)


def nav_html(tests, current):
    return "".join(f'<a href="{e(t["page"])}"' + (' aria-current="page"' if t["id"] == current else "") +
                   f'><span>{t["number"]}</span>{e(t["nav"])}</a>' for t in tests)


def render(main_html, tests):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="Figures of Tomida et al. 2026, Kritsuk et al. 2011, the GEM challenge and Gardiner &amp; Stone 2005/2008 redrawn with GLM, CT, AthenaK, RAMSES and Athena++.">
<title>Papers · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="paper-figures"><a class="skip-link" href="#main">Skip to the figures</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> PAPER FIGURES</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav_html(tests, PAGE["id"])}</nav>
<main id="main">{main_html}</main>
<footer><span>Drawn by scripts/paper_figures.py from sweep_results/ (copied from Sherlock by scripts/sweep_fetch.py).</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main(argv):
    record = SITE/"paper_figures.json"
    have, wanted, _ = fetched()
    if "--page-only" not in argv:
        if have < 100:
            print(f"Only {have} of {wanted} result files are here: nothing to draw yet (fetch them with scripts/sweep_fetch.py). Writing the page without figures.")
            record.write_text(json.dumps(dict(status_time="", figures=[]), indent=1))
        else:paper_figures.main([])
    data = json.loads(record.read_text()) if record.is_file() else dict(status_time="", figures=[])
    main_html, info = build(data["figures"], data.get("status_time", ""))
    tests = json.loads((SITE/"tests.json").read_text())
    page = dict(PAGE, status=f'{info["drawn"]} figures' if info["drawn"] else "Waiting for data")
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
    print(f'Paper figures: {info["drawn"]} drawn; {info["have"]:,} of {info["wanted"]:,} result files here; page {SITE/PAGE["page"]}')


if __name__ == "__main__":
    main(sys.argv[1:])
