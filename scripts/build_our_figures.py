"""The "Our figures" page: the draft figures of our own paper (paper_figures/), with the tests behind each claim.

  python3 scripts/build_our_figures.py [--page-only]

Redraws every figure with scripts/manuscript_figures.py (PDFs into paper_figures/, picture copies into
progress/media/ours/, with the PDFs copied beside them for download), then writes progress/our-figures.html. The
numbers in the claims table are computed here from the same records the figures use.
"""
import html
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manuscript_figures as mf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
MEDIA = SITE/"media"/"ours"
PAGE = dict(id="our-figures", page="our-figures.html", kind="ours", number="11", title="Our figures", nav="Our figures", status="", document="",
            description="The draft figures of our paper, and the tests behind each claim.", chips=[])
e = html.escape
def TEST_PAGE(t):
    """A test's name, linked to its page where it has one (the timing test has none)."""
    page = f'test-{t.replace("_", "-")}.html'
    return f'<a href="{page}">{e(t)}</a>' if (SITE/page).is_file() else f"<code>{e(t)}</code>"

# name, title, tests, what it shows, how to read it
FIGURES = (
    ("speed", "Speed", ["throughput_ot3d", "orszag_tang_3d"],
     "(a) Cell updates per second on one H100, on 3D Orszag–Tang run for a short time, 64³ to 512³, the fastest of the repeats (some of the "
     "rerun's repeats were slowed 3 to 12 times by something else on the node; the originals' repeats agree to 0.1 %); filled symbols are grids "
     "divisible by 6, where the Float16 kernels use their larger tile. (b) Time to t = 1 of 3D Orszag–Tang at 512³: one H100 for Abel et al.'s kernels and AthenaK, "
     "128 CPU cores for RAMSES and Athena++.",
     "Abel et al.'s kernels ran at Courant number 0.25 in their own rule (the sum over the three directions), the other codes at 0.3 against the "
     "fastest single direction, so the kernels took about 3.3 times more steps; the times in (b) carry that handicap."),
    ("cost_vs_error", "Accuracy for the GPU time spent", ["entropy", "alfven_oblique", "field_loop"],
     "Since the full rerun (2026-10-21) the fixed kernels of the sweep-fixes branch are drawn in full colour and the originals they fix in a pale tint, marked “original”. Error at the end against the exact answer, against the GPU time the run took, for the codes that ran on the H100. Each point is one grid, "
     "32² to 512². A run that took longer than the same test on twice the grid is left out: its time is not the solver's (three 32² runs of the "
     "fixed CT prototype, where compiling leaked into the timing).",
     "Lower and further left is better. A curve that turns up stops converging: the original half-precision kernels. Fixed, they converge."),
    ("accuracy", "Accuracy on every planar test", [t for t, _, _ in mf.ACCURACY],
     "Relative L1 error at the end of each test (density, or the field where the density does not change, and for GEM), divided by the median of the "
     "same-order CT set-ups: AthenaK PLM/HLLE, RAMSES HLL and Athena++ PLM/HLLE. Against the exact answer where there is one, otherwise against a "
     "time-matched AthenaK run at 4096² averaged to the grid. Faint symbols are the external codes' best set-ups and AREPO, for context. "
     "A missing symbol is a run that did not finish (see Robustness). Since the full rerun (2026-10-21) the fixed kernels of the sweep-fixes branch are drawn in full colour and the originals they fix in a pale tint, marked “original”. ",
     "1 means as accurate as the same-order CT codes; the shaded band is ±25 %. Kelvin–Helmholtz at t = 3 is turbulent, so its error also measures "
     "where the vortices happen to be."),
    ("convergence", "Order of convergence", ["entropy", "alfven", "alfven_oblique"],
     "L1 error against grid size on the three smooth waves with exact answers. Top row: Abel et al.'s kernels and the CT prototype; bottom row: "
     "the external codes. Dotted and dashed lines: first and second order. Since the full rerun (2026-10-21) the fixed kernels of the sweep-fixes branch are drawn in full colour and the originals they fix in a pale tint, marked “original”. ",
     "Second-order schemes follow the dashed slope. The original half-precision kernels level off or rise past 64 to 128 cells and the original "
     "CT prototype follows the first-order slope; every fixed kernel follows the second-order slope to 512²."),
    ("robustness", "Robustness", ["every test"],
     "Every test, grid and set-up of the sweep: a full square is a run that reached the end time; a partly filled one shows how far a run got before "
     "its own solver stopped it. The fixed kernels take the columns of the originals they fix; the originals' counts are under the key. A run whose "
     "magnetic or kinetic energy ends above 10 times AthenaK PLM/HLLE's on the same grid counts as failed from the time it got there: the fixed PPM "
     "kernel at 512² on the field loop and the confined-field Kelvin–Helmholtz test (field energy 92 and 156 times AthenaK's at the end; the "
     "original stopped on both). No other run of the original or fixed kernels trips this.",
     "Columns are set-ups, rows are tests, and the five squares of a cell are the grids 32 to 512. The counts at the bottom are the runs that finished."),
    ("leakage", "Leakage into a field-free region", ["kh_localized", "kh_localized_dtjump"],
     "(a) Field strength at t = 3 on 512², with the field-free vortex circled; the last two maps are fixed kernels. (b) The field inside that vortex "
     "over the reference's, against grid size. (c) The largest cleaning potential ψ through a sudden 40-fold cut of the timestep, at 256². Since the full rerun (2026-10-21) the fixed kernels of the sweep-fixes branch are drawn in full colour and the originals they fix in a pale tint, marked “original”. ",
     "In (b) the CT codes sit at 1, and so does the fixed CT prototype (0.9 to 1.7; the original was 1600 times at 64²). The GLM kernels sit 7.5 "
     "to 13 times above at 512² and level off rather than converge; the Float16 kernel with cell-centred CT in place of GLM sits at 5.7 to 6.3 at "
     "every grid. In (c) ψ jumps with the cleaning speed, which follows the timestep, except in the constant-c_h variant."),
    ("divergence", "Two tests of the divergence constraint", ["brio_wu_rotated", "field_loop"],
     "(a) Rotated shock tube at 256²: the largest error in the field normal to the shock, which must stay exactly 0.75. (b) Field loop with motion "
     "out of the plane at 128²: the rms of the field B_z that grows only from divergence errors.",
     "Further left is better. Constrained-transport codes keep both at round-off; cleaning keeps them small but finite. The Float16 kernel with "
     "cell-centred CT keeps the normal field to 5×10⁻⁵ but lets B_z grow to 4×10⁻² of B₀, six times the fixed f16/HLL kernel; the CT prototype, "
     "original and fixed, keeps the normal field near 10⁻⁶ but its B_z is 1 to 4×10⁻³, not round-off."),
    ("field_loop", "Advecting a field loop", ["field_loop_novz", "field_loop_3d"],
     "Magnetic energy of the advected loop over its starting value: (a) in the plane at 64², (b) the inclined loop in 3D at 128³, with Gardiner & "
     "Stone's fitted decay at those grids.",
     "The exact answer keeps the energy constant; the decay is numerical diffusion. The fixed CT prototype loses energy like the other "
     "second-order codes; the original lost two thirds by t = 2 in the plane."),
    ("turbulence", "Supersonic decaying turbulence", ["decay_supersonic"],
     "Mach 9 decay with a mean field at 512³ to t = 0.2: (a) kinetic and (b) magnetic energy against time; (c) velocity and (d) magnetic spectra at "
     "the end.",
     "The codes agree on the energies; at high wavenumber the spectra show each scheme's dissipation. No fixed kernel reached the end at 512³ "
     "(seed 1): the fixed CT prototype stops at Mach 9, the Float16 kernels stop as their originals did."),
    ("arepo", "AREPO", ["field_loop", "orszag_tang", "rotor", "blast_2d_b50", "kh_uniform", "current_sheet"],
     "(a) Field-loop magnetic energy on AREPO's static and moving meshes. (b) The static-mesh runs that stopped: their timestep against time. "
     "(c) AREPO's own divergence measure at the end against grid size.",
     "On the static mesh the field loop's energy grows and the timestep collapses; the moving mesh finishes every test."),
)


def robust_note():
    """Runs that reached the end, of those planned, per set-up, from the sweep's status (as the robustness figure counts them)."""
    import sweep_data
    sw = sweep_data.Sweep();count = {}
    for (test, label, n, seed), row in sw.rows.items():
        if label not in ("glm", "glm_hll16", "glm_fix_hll16", "ct", "ct_fix", "athenak_best", "ramses_best"):continue
        done, total = count.get(label, (0, 0));count[label] = (done + (row.get("state") == "done"), total + 1)
    parts = [f"{mf.name(m)} {count[m][0]} of {count[m][1]}" for m in ("glm", "glm_hll16", "glm_fix_hll16", "ct", "ct_fix", "athenak_best", "ramses_best") if m in count]
    return "Runs that reached the end: " + ", ".join(parts) + "."


def claims():
    """Rows of the claims table, with numbers computed from the records the accuracy figure uses."""
    def ratios(method, n):
        out = []
        for test, _, kind in mf.ACCURACY:
            err = mf.endpoint_errors(test, n, kind);base = [err[m] for m in mf.MATCHED if m in err]
            if len(base) >= 2 and method in err:out.append((err[method]/statistics.median(base), test))
        return out
    def span(r, tests=None):
        v = [x for x, t in r if tests is None or t in tests]
        fmt = lambda x: f"{x:.2g}" if x < 10 else f"{x:.0f}"
        return f"{fmt(min(v))} to {fmt(max(v))}" if v else "–"
    smooth = ("entropy", "alfven", "alfven_oblique", "field_loop_novz")
    nonlinear = [t for t, _, _ in mf.ACCURACY if t not in smooth]
    llf, hlld, f16 = ratios("glm", 512), ratios("glm_hlld", 512), ratios("glm_hll16", 512)
    f16fix, ctfix = ratios("glm_fix_hll16", 512), ratios("ct_fix", 512)
    tl = lambda ts: ", ".join(TEST_PAGE(t) for t in ts)
    return [
        ("Fast", "Speed", tl(["throughput_ot3d", "orszag_tang_3d"]), "One H100 against one H100 (AthenaK) and 128 CPU cores (RAMSES, Athena++)."),
        ("As accurate as same-order CT codes on shocks and nonlinear flows", "Accuracy", tl(nonlinear),
         f"At 512², error over the same-order CT codes: LLF {span(llf, nonlinear)}, HLLD {span(hlld, nonlinear)}; fixed: Float16 HLL "
         f"{span(f16fix, nonlinear)}, CT prototype {span(ctfix, nonlinear)}."),
        ("Second order on smooth problems", "Order of convergence, Accuracy for the GPU time spent", tl(["entropy", "alfven", "alfven_oblique", "field_loop"]),
         f"With the fixes also in half precision: the fixed production kernel is {span(f16fix, smooth)} times the CT codes' error on these "
         f"tests at 512² (the original {span(f16, smooth)}), the fixed CT prototype {span(ctfix, smooth)}."),
        ("Robust", "Robustness", "all 25 tests at every grid", robust_note()),
        ("Weak point: field leaks where there should be none", "Leakage", tl(["kh_localized", "kh_localized_dtjump"]),
         "Not removed by tuning the damping. Cell-centred CT in the Float16 kernel holds it at about 6 times the reference at every grid; the fixed "
         "CT prototype removes it."),
        ("Divergence kept small", "Two tests of the divergence constraint, Advecting a field loop", tl(["brio_wu_rotated", "field_loop", "field_loop_novz", "field_loop_3d"]),
         "Constrained-transport codes keep the normal field and B_z at round-off; cleaning keeps them small but finite."),
        ("Supersonic turbulence", "Supersonic decaying turbulence", tl(["decay_supersonic"]), "Driven turbulence has its own page."),
    ]


def card(i, name, title, tests, shows, read):
    png, pdf = f"media/ours/{name}.png", f"media/ours/{name}.pdf"
    stamp = int((SITE/png).stat().st_mtime)
    tlinks = ", ".join(t if t == "every test" else TEST_PAGE(t) for t in tests)
    return (f'<article class="card paper-fig" id="{name}"><div class="fig-head"><span class="fig-no">{i}</span><h3>{e(title)}</h3>'
            f'<a class="method-link" href="{pdf}">PDF <span>↓</span></a></div>'
            f'<div class="fig-pair"><figure class="ours"><a class="image-link" href="{png}?v={stamp}"><img src="{png}?v={stamp}" alt="{e(title)}" loading="lazy"></a></figure></div>'
            f'<p><strong>Tests:</strong> {tlinks}</p><p>{e(shows)}</p><p class="finding"><strong>Read it:</strong> {e(read)}</p></article>')


def page_html():
    rows = "".join(f"<tr><td><strong>{e(c)}</strong></td><td>{e(f)}</td><td>{t}</td><td>{e(n)}</td></tr>" for c, f, t, n in claims())
    out = [f'''
<div class="page-heading"><div><div class="eyebrow"><span>{PAGE["number"]} · Our figures</span><span class="badge">{len(FIGURES)} drafts</span></div>
<h1>The figures of our paper</h1><p>Drafts, drawn from the sweep's records by <code>scripts/manuscript_figures.py</code>. Each says which tests it rests on;
the table says which figure and which tests stand behind each claim.</p></div></div>
<article class="card"><h3>Claims and the tests behind them</h3><div class="table-scroll"><table class="metrics-table sweep-text paper-table our-claims">
<thead><tr><th>claim</th><th>figure</th><th>tests</th><th>note</th></tr></thead><tbody>{rows}</tbody></table></div></article>''']
    out += [card(i + 1, *f) for i, f in enumerate(FIGURES)]
    return "".join(out)


def render(main_html, registry):
    nav = "".join(f'<a href="{e(t["page"])}"' + (' aria-current="page"' if t["id"] == PAGE["id"] else "") + f'><span>{t["number"]}</span>{e(t["nav"])}</a>' for t in registry)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="The draft figures of our MHD comparison paper and the tests behind each claim.">
<title>Our figures · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="our-figures"><a class="skip-link" href="#main">Skip to the figures</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> OUR FIGURES</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav}</nav>
<main id="main">{main_html}</main>
<footer><span>Drawn by scripts/manuscript_figures.py; page by scripts/build_our_figures.py.</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main(argv):
    MEDIA.mkdir(parents=True, exist_ok=True)
    if "--page-only" not in argv:
        subprocess.run([sys.executable, str(ROOT/"scripts"/"manuscript_figures.py"), "--preview", str(MEDIA)], check=True)
    for name, *_ in FIGURES:shutil.copyfile(ROOT/"paper_figures"/f"{name}.pdf", MEDIA/f"{name}.pdf")
    registry = json.loads((SITE/"tests.json").read_text());page = dict(PAGE, status=f"{len(FIGURES)} drafts")
    registry = [page if t["id"] == PAGE["id"] else t for t in registry] if any(t["id"] == PAGE["id"] for t in registry) else registry + [page]
    (SITE/"tests.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    (SITE/PAGE["page"]).write_text(render(page_html(), registry))
    link = f'<a href="{PAGE["page"]}"><span>{PAGE["number"]}</span>{PAGE["nav"]}</a>'
    for t in registry:                       # the navigation of the pages already built
        path = SITE/t["page"]
        if t["id"] == PAGE["id"] or not path.is_file():continue
        text = path.read_text()
        if f'href="{PAGE["page"]}"' not in text and "</nav>" in text:path.write_text(text.replace("</nav>", link + "</nav>", 1))
    print(f"Our figures: {len(FIGURES)} figures; {SITE/PAGE['page']}")


if __name__ == "__main__":
    main(sys.argv[1:])
