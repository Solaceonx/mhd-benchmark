"""Weekly progress: a page per week to present what was done, and an index of the weeks.

    python3 scripts/build_weekly.py [WEEK ...]     -> progress/week-WEEK.html, progress/weekly.html (+ weekly.json)

A week is named by its last day (YYYY-MM-DD) and written by its function in WEEKS, from the records as they stand when it
is built (work_precision.json, self_convergence.json, the fetched run records): its numbers are a snapshot of that day,
so build a past week again only to correct it. progress/weekly.json keeps each built week's title and summary for the
index. Without arguments: the latest week and the index. Run after work_precision.py and self_convergence.py.
"""
import collections
import html
import json
import math
import statistics
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_benchmark as bb  # noqa: E402
import paper_common as pc  # noqa: E402
import sweep_data as sd  # noqa: E402
import work_precision as wp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
ANALYSIS = SITE/"media"/"analysis"
INDEX = SITE/"weekly.json"
PAGE = dict(id="weekly", page="weekly.html", kind="weekly", number="W", title="Weekly progress", nav="Weekly", status="",
            document="", description="What was done each week, page by page, to present.", chips=[])
NEWCODES = ("athenapk", "cholla", "pluto", "enzo", "phantom")
# self_convergence.py's tests, by the kind of flow (planar only)
KINDS = (("Smooth waves", ("entropy", "alfven", "alfven_oblique")),
         ("Shocks", ("brio_wu", "brio_wu_rotated", "orszag_tang", "rotor", "blast_2d_b0", "blast_2d_b50")),
         ("Kelvin–Helmholtz, current sheet", ("kh_uniform", "current_sheet")))
e = html.escape
STYLE = ('<style>.week-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:12px;margin-top:18px}'
         '.week-stat{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 16px}.week-stat strong{display:block;font:600 24px/1.2 var(--font);'
         'color:#202a2a;letter-spacing:-.02em}.week-stat span{font-size:11px;color:#64716f}.week-card h3 .no{font:11px ui-monospace,monospace;color:#64716f;margin-right:8px}'
         '.week-card .lede{font-size:13px!important;color:#202a2a!important}.week-table{border-collapse:collapse;font-size:12px;margin-top:12px}'
         '.week-table th,.week-table td{padding:5px 10px;border-bottom:1px solid #edf0e9;text-align:right;white-space:nowrap}.week-table th:first-child,'
         '.week-table td:first-child{text-align:left}.week-table th{font-size:10px;font-weight:500;color:#64716f}.week-scroll{overflow-x:auto}'
         '.week-figure{margin:14px 0 0}.week-figure img{display:block;width:100%;max-width:1100px;border:1px solid #edf0e9;border-radius:4px}'
         '.week-figure figcaption{font-size:11px;color:#64716f;margin-top:6px;line-height:1.6}.week-more{font-size:12px;margin-top:12px!important}'
         '.week-list{list-style:none;padding:0;margin:0}.week-list li{padding:16px 0;border-top:1px solid #edf0e9}.week-list li:first-child{border-top:0}'
         '.week-list h3{margin:0 0 4px}.week-list .when{font:11px ui-monospace,monospace;color:#64716f}</style>')


def span(first, last):
    a, b = date.fromisoformat(first), date.fromisoformat(last)
    return f"{a.day} {a:%B}" + (f" {a.year}" if a.year != b.year else "") + f" – {b.day} {b:%B %Y}"


def gain(x):
    """A ratio of times (above 1: the change saves time) as words."""
    return "–" if x is None else f"{x:.2g}× faster" if x >= 1 else f"{1/x:.2g}× slower"


def card(no, headline, lede, *body):
    return f'<article class="card week-card"><h3><span class="no">{no:02d}</span>{headline}</h3><p class="lede">{lede}</p>{"".join(body)}</article>'


def bullets(items):
    return '<ul class="checks plain">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def table(head, rows):
    return ('<div class="week-scroll"><table class="week-table"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")


def figure(src, caption, link=None):
    img = f'<img src="{src}" alt="{e(caption)}" loading="lazy">'
    return f'<figure class="week-figure">{f"<a href={chr(34)}{link}{chr(34)}>{img}</a>" if link else img}<figcaption>{caption}</figcaption></figure>'


def more(href, text):
    return f'<p class="week-more"><a href="{href}">{text} →</a></p>'


def stats(items):
    return '<div class="week-stats">' + "".join(f'<div class="week-stat"><strong>{v}</strong><span>{k}</span></div>' for v, k in items) + "</div>"


def self_rows(sc):
    """Per kind of flow (KINDS): medians over set-ups and tests of self_convergence.py's numbers: the error against the
    set-up's own finest grid over the true one (at half the finest grid), the order the grid differences show and the
    order against the truth on the same grids, Richardson's estimate of the finest grid's error over its true error, and
    the extrapolation's gain over the finest grid."""
    out = []
    for kind, tests in KINDS:
        own, pd, pt, est, g = [], [], [], [], []
        for t in tests:
            for r in (sc.get(t) or {}).get("methods", {}).values():
                h = str(r["finest"]//2)
                if r["own"].get(h) and r["truth"].get(h):own.append(r["own"][h]/r["truth"][h])
                ri = r.get("richardson") or {}
                if not ri:continue
                n2, n3 = (str(n) for n in ri["grids"][1:]);pd.append(ri["p"])
                if r["truth"].get(n2) and r["truth"].get(n3):pt.append(math.log2(r["truth"][n2]/r["truth"][n3]))
                if ri.get("finest_vs_truth"):est.append(ri["estimated_error_finest"]/ri["finest_vs_truth"])
                if ri.get("gain"):g.append(ri["gain"])
        m = lambda v, f="{:.2f}": f.format(statistics.median(v)) if v else "–"
        out.append((kind, m(own), m(pd, "{:.1f}"), m(pt, "{:.1f}"), m(est, "{:.2g}"), m(g, "{:.2g}")))
    return out


def week_2026_11_04():
    sw = sd.Sweep();rows = json.loads(bb.DATA.read_text());tests, methods, fastest = bb.setups(rows)
    sc = json.loads((ANALYSIS/"self_convergence.json").read_text())
    fam = lambda key: pc.STYLE[key[1]][0]
    new = collections.Counter((fam(k), r["state"]) for k, r in sw.newcodes.items())
    total = lambda c, fams: {s: sum(n for (f, s2), n in c.items() if f in fams and s2 == s) for s in ("done", "stopped", "open")}
    nc = total(new, NEWCODES);planned = sum(nc.values());finished = nc["done"] + nc["stopped"]
    order = collections.Counter(r["state"] for r in sw.order.values());giz = collections.Counter(r["state"] for r in sw.gizmo.values())
    where = {f: w for f, _, w, _ in bb.CODES};names = {f: n for f, n, _, _ in bb.CODES}

    # 1. the five codes: runs by state
    code_rows = [(f"{e(names[f])} <span class='small'>· {where[f]}</span>", *(new.get((f, s), 0) for s in ("done", "stopped", "open"))) for f in NEWCODES]
    code_rows.append(("<strong>All five</strong>", *(f"<strong>{nc[s]}</strong>" for s in ("done", "stopped", "open"))))

    # 2. speed against time to accuracy; the Orszag-Tang example
    a, b, rate, rel = bb.extremes(rows, tests, methods, fastest)
    ot = rows["orszag_tang"]["methods"];ex = [m for m in ("glm", "glm_fix_hlld", "cholla_ppmc") if (ot.get(m) or {}).get("cost")]
    ex_rows = [(e(bb.full_name(m)), f"{ot[m]['N']:.0f}²", bb.mcell(ot[m]["rate"]), bb.seconds(ot[m]["cost"])) for m in ex]
    jobs = [f"<strong>{e(job)}</strong>: " + ", ".join(f"{e(bb.full_name(m))} ({bb.times(s)})" for m, s in top) for job, _, top, _ in bb.best(rows, methods, fastest)]

    # 3. the order x solver study: medians over the tests of the time to the accuracy, each change alone and both
    split_rows = []
    for name, cells in wp.GRID:
        g = [x for r in rows.values() if (x := wp.factorial(r, cells))]
        if not g:continue
        med = lambda k: statistics.median(x[k] for x in g)
        sp = [bb.speed(rows, tests, c) for c in cells[:2]]
        cost = f"{sp[0][0]/sp[1][0]:.2g}×" if all(sp) else "–"
        split_rows.append((e(name), gain(med("order_hll")), gain(med("solver_plm")), gain(med("both")), cost, len(g)))

    # 4. self-convergence; 5. the data audit's numbers
    n_self = sum(len(v["methods"]) for v in sc.values());selfc = self_rows(sc)
    under = lambda row: f"{100*(1 - float(row[1])):.0f} %" if row[1] != "–" else "–"
    giz_alf = (sc.get("alfven") or {}).get("methods", {}).get("gizmo_mfm", {}).get("truth", {})
    arepo = (sc.get("field_loop_novz") or {}).get("methods", {}).get("arepo_static", {}).get("truth", {})
    open_by = {f: new.get((f, "open"), 0) for f in NEWCODES if new.get((f, "open"))}

    summary = [f"Five more codes (AthenaPK, Cholla, gPLUTO, Enzo, Phantom) on every test: {finished:,} of {planned:,} runs finished.",
               f"A benchmark page: time to a given accuracy, order of accuracy and speed for {len(methods)} set-ups on {len(tests)} tests.",
               "Convergence without a truth (own finest grid, grid differences, Richardson), which found a bug in our GIZMO conversion."]
    body = f'''{STYLE}
<div class="page-heading"><div><div class="eyebrow"><span>W · Weekly progress</span><span class="badge">{span("2026-10-28", "2026-11-04")}</span><a href="weekly.html">All weeks</a></div>
<h1>Five more codes, a benchmark, and convergence without a truth</h1>
<p>The week in five parts: the codes added to the suite, the benchmark of which code is good at what, the study of what higher order and a
better Riemann solver buy, measuring convergence without the true solution, and what we corrected in our own data.</p>
{stats([(f"{finished:,} / {planned:,}", "runs of the five new codes finished"), (f"{order['done']} / {sum(order.values())}", "runs of the order × solver study complete"),
        (f"{len(methods)} × {len(tests)}", "set-ups × tests on the benchmark"), (f"{n_self}", "set-up × test convergence analyses without a truth")])}</div></div>
{card(1, "Five more codes run the whole suite",
      "AthenaPK, Cholla and gPLUTO on H100s, Enzo and Phantom on CPUs: each a matched set-up (second order, an HLL-type solver) and its best "
      "one, from the suite's own starting states, on every test and grid.",
      table(("code", "complete", "stopped", "still to run"), code_rows),
      bullets(["A <em>stopped</em> run is the code's own stop or a non-finite state, recorded with the code's reason: e.g. Cholla stops itself "
               "when div B passes its fixed limit, and AthenaPK PPM/HLLD turns non-finite on the decay test on every grid and seed.",
               "Phantom's own divergence projection removed 26 % of the magnetic energy by the first output, so it runs without it; its stability "
               "check stops the field loops, and we keep that check on.",
               "3D at 512³ does not fit one H100 in Cholla or gPLUTO: those runs use two."]),
      more("sweep-plan.html", "Every run's state on the Sweep page"))}
{card(2, "Which code is good at what: the Benchmark page",
      "For every test, the time each set-up needs to reach the error a standard second-order code has at 256², and the order its error falls "
      "with resolution; new this week, its speed in cell updates per second beside that time.",
      "<p>Least time to the accuracy, per job (× the fastest on each test):</p>", bullets(jobs),
      f"<p>Speed per cell is not speed to an answer. {e(bb.full_name(a))} makes the most cell updates per second, {rate[a]/rate[b]:.2g}× as many as "
      f"{e(bb.full_name(b))}, yet {e(bb.full_name(b))} reaches the accuracy in the least time (median {bb.times(rel[b][0])} the fastest, against "
      f"{bb.times(rel[a][0])}). A more diffusive solver needs a finer grid, and in 2D the work grows as the grid's cube (cells × steps). Orszag–Tang:</p>",
      table(("set-up", "grid needed", "million cell updates / s", "time on one H100"), ex_rows),
      more("benchmark.html", "The Benchmark page"))}
{card(3, "What higher order and a better solver buy",
      "The order × solver study: PLM and PPM, each with an HLL-type and the HLLD Riemann solver, in AthenaK, Athena++ and Abel et al.'s kernels "
      f"({order['done']} complete runs). The time to the same accuracy, median over the tests:",
      table(("code", "PLM → PPM", "HLL → HLLD", "both", "PPM's cost per cell update", "tests"), split_rows),
      bullets(["In AthenaK and Athena++ PPM pays: the coarser grid it needs more than covers its cost per cell (on the smooth waves Athena++ "
               "PPM converges at about fourth order).",
               "In Abel et al.'s kernels PPM is several times slower per cell update than PLM, so it does not pay on most tests; HLLD does."]),
      figure("media/ours/order_tradeoff.png", "Each change's time to the same error (a) and the cells per side it needs (b), test by test.", "our-figures.html"),
      more("our-figures.html", "Our figures page"))}
{card(4, "Convergence without a truth",
      "Each set-up against its own finest grid, the differences between successive grids, and Richardson extrapolation, set beside its error "
      f"against the true solution, for the {len(sc)} planar tests (scripts/self_convergence.py). Medians over the set-ups:",
      table(("tests", "own finest ÷ true error", "order from differences", "order against truth", "Richardson estimate ÷ true error", "Richardson gain"), selfc),
      bullets([f"Against its own finest grid a set-up looks better than it is: its error seems {under(selfc[0])} smaller on smooth waves, "
               f"{under(selfc[1])} with shocks, {under(selfc[2])} on Kelvin–Helmholtz and the current sheet.",
               f"Grid differences give the true order on smooth waves. On Kelvin–Helmholtz and the current sheet the differences shrink at order "
               f"{selfc[2][2]} while the error against the truth barely falls (order {selfc[2][3]}): the grids agree with each other, not with the truth.",
               "Richardson extrapolation estimates the error well on smooth waves, but does not bring the answer nearer the truth: the limiters "
               "change the error's pattern from grid to grid.",
               "An own-grid reference hides a code that converges to the wrong answer: GIZMO's Alfvén-wave error stops falling at about 0.1 %, "
               "and AREPO's static mesh on the field loop agrees with itself while its field blows up."]),
      figure("media/analysis/alfven/09-self-0.png", "Alfvén wave: against the truth (top left) GIZMO's error (gold) is flat from 64²; against its own "
             "finest grid (top right) it seems to converge.", "test-alfven.html"),
      more("test-alfven.html", "The Alfvén-wave test page"))}
{card(5, "Getting the data right",
      "An audit of our own records and conversions before the final rebuild.",
      bullets(["The new codes' records held srun's wall time, start-up included (gPLUTO Brio–Wu 32²: 456 s, of which 18 s computing): every "
               "record now has the code's own clock.",
               "Speeds are taken from each set-up's largest 3D run: a planar run leaves a GPU mostly idle, and Abel et al.'s kernels run the "
               "planar tests on N³ cells, so planar speeds favoured them.",
               "Our GIZMO conversion made the field on the grid 0.05 % too strong (it divided by the cell volume, but GIZMO's particle volumes "
               f"add up to 1.0005 of the box). Fixed, and all {giz['done']} complete runs reconverted from GIZMO's snapshots: its Alfvén-wave error "
               f"at 512² went from 1.16×10⁻³ to {pc.sci(giz_alf.get('512', float('nan')), 2)}; the floor that is left is GIZMO's own.",
               "A stop on the second of two GPUs was read as a lost job (gPLUTO, Cholla): now the code's own stop and reason.",
               f"AREPO's static mesh on the field loop finishes, but its field error is {arepo.get('64', 0):.0f}× the field at 64² and "
               f"{arepo.get('128', 0):.0f}× at 128²: flagged, not yet marked invalid."]))}
{card(6, "Still running, and next",
      "On Sherlock as of this page: " + (", ".join(f"{e(names[f])} {n} runs" for f, n in open_by.items()) or "nothing") + ".",
      bullets(["Then: the last fetch, every analysis and page rebuilt, and the site published.",
               "Convergence without a truth in 3D: the 3D states are too large to fetch, so a job on Sherlock.",
               "Whether to mark the AREPO static field loop invalid."]))}'''
    return dict(first="2026-10-28", last="2026-11-04", title="Five more codes, a benchmark, and convergence without a truth", summary=summary, html=body)


WEEKS = {"2026-11-04": week_2026_11_04}


def render(main_html, title, registry, current):
    nav = "".join(f'<a href="{e(t["page"])}"' + (' aria-current="page"' if t["id"] == current else "") + f'><span>{t["number"]}</span>{e(t["nav"])}</a>' for t in registry)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="{e(PAGE["description"])}">
<title>{e(title)} · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="weekly"><a class="skip-link" href="#main">Skip to the content</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> WEEKLY PROGRESS</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav}</nav>
<main id="main">{main_html}</main>
<footer><span>Numbers from the records as of the build; page by scripts/build_weekly.py.</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def index(weeks):
    items = "".join(f'<li><div class="when">{span(w["first"], w["last"])}</div><h3><a href="week-{k}.html">{e(w["title"])}</a></h3>{bullets(w["summary"])}</li>'
                    for k, w in sorted(weeks.items(), reverse=True))
    return f'''{STYLE}
<div class="page-heading"><div><div class="eyebrow"><span>W · Weekly progress</span><span class="badge">{len(weeks)} week{"s" if len(weeks) != 1 else ""}</span></div>
<h1>Weekly progress</h1><p>What was done each week, one page per week, to present. Each page's numbers are those of the records on the day it was built.</p></div></div>
<article class="card"><ul class="week-list">{items}</ul></article>'''


def main(argv):
    registry = json.loads((SITE/"tests.json").read_text())
    if not any(t["id"] == PAGE["id"] for t in registry):        # after the Benchmark page
        at = next((i + 1 for i, t in enumerate(registry) if t["id"] == "benchmark"), 0);registry.insert(at, PAGE)
        (SITE/"tests.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    weeks = json.loads(INDEX.read_text()) if INDEX.is_file() else {}
    for k in argv or [max(WEEKS)]:
        if k not in WEEKS:sys.exit(f"no week {k}: one of {', '.join(WEEKS)}")
        w = WEEKS[k]();(SITE/f"week-{k}.html").write_text(render(w.pop("html"), w["title"], registry, PAGE["id"]));weeks[k] = w
        print(f"Week: {SITE/f'week-{k}.html'}")
    INDEX.write_text(json.dumps(weeks, ensure_ascii=False, indent=1) + "\n")
    (SITE/PAGE["page"]).write_text(render(index(weeks), PAGE["title"], registry, PAGE["id"]))
    print(f"Index: {SITE/PAGE['page']}")


if __name__ == "__main__":
    main(sys.argv[1:])
