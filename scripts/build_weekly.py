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
from build_test_pages import TITLES  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (paper_common set the backend)
from matplotlib.lines import Line2D  # noqa: E402

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
         '.week-list h3{margin:0 0 4px}.week-list .when{font:11px ui-monospace,monospace;color:#64716f}'
         '.wpt{margin-top:14px}.wpt-tabs{display:grid;gap:6px}.wpt-group{display:flex;flex-wrap:wrap;gap:6px;align-items:center}'
         '.wpt-group span{font-size:10px;color:#64716f;min-width:11rem}.wpt-group button{font:12px var(--font);padding:4px 10px;border:1px solid #d9dedd;'
         'border-radius:999px;background:#fff;color:#202a2a;cursor:pointer}.wpt-group button[aria-pressed="true"]{background:#202a2a;color:#fff;'
         'border-color:#202a2a}@media(max-width:640px){.wpt-group span{min-width:100%}}</style>')


def span(first, last):
    a, b = date.fromisoformat(first), date.fromisoformat(last)
    return f"{a.day} {a:%B}" + (f" {a.year}" if a.year != b.year else "") + f" – {b.day} {b:%B %Y}"


def gain(x):
    """A ratio of times (above 1: the change saves time) as words."""
    return "–" if x is None else f"{x:.2g}× faster" if x >= 1 else f"{1/x:.2g}× slower"


def card(no, title, *body):
    return f'<article class="card week-card"><h3><span class="no">{no:02d}</span>{e(title)}</h3>{"".join(body)}</article>'


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


def save(fig, week, name):
    """A figure of a week's page, PNG and SVG, in progress/media/weekly/WEEK/; returns the PNG's address on the site."""
    out = SITE/"media"/"weekly"/week;out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):fig.savefig(out/f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig);return f"media/weekly/{week}/{name}.png"


def speed_figure(week, rows, tests, methods, fastest, labels):
    """Each set-up's cell updates per second against its median time to the accuracy (× the fastest set-up on each test).
    Hue: the code; filled: a GPU code (one H100), open: a CPU code (128 cores)."""
    pc.apply_plot_style();plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(8.4, 5.0));names = {f: n for f, n, _, _ in bb.CODES};seen = {}
    for m in methods:
        s = bb.speed(rows, tests, m);med = bb.time_ratio(rows, tests, fastest, m)[0]
        if not s or not med:continue
        c = pc.color(m);seen.setdefault(pc.STYLE[m][0], c)
        ax.scatter(s[0]/1e6, med, s=36, facecolor="none" if s[2] else c, edgecolor=c, lw=1.3, zorder=3)
        if m in labels:     # to the left of the point when the offset is negative; a leader line when far from it
            (text, (dx, dy)) = labels[m]
            ax.annotate(text, (s[0]/1e6, med), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=pc.INK, ha="right" if dx < 0 else "left",
                        va="center", arrowprops=dict(arrowstyle="-", color=pc.MUTED, lw=.6, shrinkA=1, shrinkB=4) if abs(dy) > 12 else None)
    ax.set_xscale("log");ax.set_yscale("log");pc.style_axes(ax)
    ax.set_xlabel("Speed: million cell updates per second (GPU codes on one H100, CPU codes on 128 cores)")
    ax.set_ylabel("Time to the accuracy: median × the fastest (1 = fastest)")
    ax.text(.99, .98, "faster per cell →\n↓ less time to the accuracy", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=pc.MUTED)
    hs = [Line2D([], [], ls="", marker="o", color=c, label=names[f]) for f, c in seen.items()]
    hs += [Line2D([], [], ls="", marker="o", color=pc.MUTED, label="filled: GPU"), Line2D([], [], ls="", marker="o", mfc="none", color=pc.MUTED, label="open: CPU")]
    ax.legend(handles=hs, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return save(fig, week, "speed_vs_accuracy")


def test_speed_figure(week, rows, t):
    """One test: each set-up's cell updates per second against its time to the test's target error, in seconds on one H100
    or 128 cores. Set-ups that finished the test but reached the target on no grid sit in a band above; the three fastest
    are numbered. Returns (image, [(set-up, seconds)] of the fastest three)."""
    pc.apply_plot_style();plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(8.4, 4.6));names = {f: n for f, n, _, _ in bb.CODES};seen = {}
    pts = [(m, s[0]/1e6, x.get("cost"), s[2], x.get("at_most")) for m, x in rows[t]["methods"].items()
           if m in pc.STYLE and (s := bb.speed(rows, [t], m))]
    hit = sorted((p for p in pts if p[2]), key=lambda p: p[2]);miss = [p for p in pts if not p[2]]
    lo, hi = min(p[2] for p in hit), max(p[2] for p in hit);band = hi*6
    for m, x, y, cpu, at_most in hit + [(m, x, band, cpu, False) for m, x, _, cpu, _ in miss]:
        c = pc.color(m);seen.setdefault(pc.STYLE[m][0], c)
        ax.scatter(x, y, s=34, marker="v" if at_most else "o", facecolor="none" if cpu else c, edgecolor=c, lw=1.3, zorder=3)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_ylim(lo/3, band*2.5);pc.style_axes(ax)
    top = math.floor(math.log10(band/2.4));ticks = [10.0**k for k in range(math.ceil(math.log10(lo/3)), top + 1)]
    ax.set_yticks(ticks + ([band] if miss else []), [f"$10^{{{round(math.log10(v))}}}$" for v in ticks] + ([f"not reached ({len(miss)})"] if miss else []))
    if miss:ax.axhline(band/2.4, color=pc.RULE, lw=.8, zorder=1)
    # the three fastest numbered, each number on the side of its point farthest from the other points and numbers
    xy = ax.transData.transform([(p[1], p[2] if p[2] else band) for p in pts]);px = fig.dpi/72;taken = list(xy)
    for i, (m, x, y, *_) in enumerate(hit[:3], 1):
        at = ax.transData.transform((x, y))
        sides = [(dx, dy) for dx in (-9, 0, 9) for dy in (-9, 0, 9) if dx or dy]
        dx, dy = max(sides, key=lambda d: min(math.dist(at + px*np.array(d), q) for q in taken))
        ax.annotate(str(i), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, fontweight="bold", color=pc.INK, ha="center", va="center")
        taken.append(at + px*np.array((dx, dy)))
    eps = rows[t]["eps"]
    ax.set_title(f"{TITLES.get(t, t)}: time to a {rows[t]['kind']} error of {100*eps:.2g} %", loc="left", fontsize=10)
    ax.set_xlabel("Speed: million cell updates per second (GPU codes on one H100, CPU codes on 128 cores)")
    ax.set_ylabel("Time to the target error [s]")
    hs = [Line2D([], [], ls="", marker="o", color=seen[f], label=n) for f, n, _, _ in bb.CODES if f in seen]
    hs += [Line2D([], [], ls="", marker="o", color=pc.MUTED, label="filled: GPU"), Line2D([], [], ls="", marker="o", mfc="none", color=pc.MUTED, label="open: CPU")]
    if any(p[4] for p in hit):hs.append(Line2D([], [], ls="", marker="v", color=pc.MUTED, label="at most: below it on its coarsest grid"))
    ax.legend(handles=hs, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return save(fig, week, f"speed_vs_accuracy_{t}"), [(m, y) for m, _, y, *_ in hit[:3]]


def test_switcher(figs):
    """The per-test figures behind a row of buttons, a kind of test per row; without script every figure shows."""
    by = {t: (src, top) for t, src, top in figs};groups = []
    for job, ts in bb.JOBS:
        ts = [t for t in ts if t in by]
        if ts:groups.append(f'<div class="wpt-group"><span>{e(job)}</span>' + "".join(
            f'<button type="button" data-t="{t}" aria-pressed="false">{e(TITLES.get(t, t))}</button>' for t in ts) + "</div>")
    panels = "".join(f'<figure class="week-figure wpt-panel" data-t="{t}"><img src="{src}" alt="{e(TITLES.get(t, t))}: speed against time to the '
                     f'target error" loading="lazy"><figcaption>Fastest to the target: ' + "; ".join(
                         f"{i}. {e(bb.full_name(m))}, {bb.seconds(y)}" for i, (m, y) in enumerate(top, 1)) +
                     f' · <a href="{bb.test_page(t)}#analysis-03-cost">the test\'s page</a></figcaption></figure>' for t, src, top in figs)
    script = ('<script>(()=>{const w=document.currentScript.parentElement,b=[...w.querySelectorAll("button")],p=[...w.querySelectorAll(".wpt-panel")];'
              'const show=t=>{p.forEach(x=>x.hidden=x.dataset.t!==t);b.forEach(x=>x.setAttribute("aria-pressed",x.dataset.t===t))};'
              'b.forEach(x=>x.addEventListener("click",()=>show(x.dataset.t)));show(b[0].dataset.t)})()</script>')
    return f'<div class="wpt"><div class="wpt-tabs">{"".join(groups)}</div>{panels}{script}</div>'


def convergence_figure(week, sc, show):
    """Alfvén wave and Orszag-Tang: a few set-ups' error against the true solution (solid) and against their own finest
    grid (dashed)."""
    pc.apply_plot_style();plt.rcParams.update({"font.size": 9})
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, (test, title) in zip(axs, (("alfven", "Alfvén wave (smooth)"), ("orszag_tang", "Orszag–Tang (shocks)"))):
        for m in show:
            r = sc[test]["methods"].get(m)
            if not r:continue
            for key, ls in (("truth", "-"), ("own", "--")):
                pts = sorted((int(n), 100*v) for n, v in r[key].items() if v)
                if pts:ax.plot(*zip(*pts), ls=ls, marker=pc.STYLE[m][4] if key == "truth" else "", ms=4, lw=1.4, color=pc.color(m))
        ax.set_xscale("log", base=2);ax.set_yscale("log");pc.style_axes(ax);ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("cells per side, N");ax.set_ylabel(f"{sc[test]['kind']} error, relative L1 [%]")
    hs = [Line2D([], [], color=pc.color(m), marker=pc.STYLE[m][4], ms=4, label=label) for m, label in show.items()]
    hs += [Line2D([], [], color=pc.MUTED, label="against the true solution"), Line2D([], [], color=pc.MUTED, ls="--", label="against its own finest grid (512²)")]
    fig.legend(handles=hs, ncol=3, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.12))
    return save(fig, week, "convergence_own_vs_truth")


def week_2026_11_04():
    week = "2026-11-04"
    sw = sd.Sweep();rows = json.loads(bb.DATA.read_text());tests, methods, fastest = bb.setups(rows)
    sc = json.loads((ANALYSIS/"self_convergence.json").read_text())
    fam = lambda key: pc.STYLE[key[1]][0]
    new = collections.Counter((fam(k), r["state"]) for k, r in sw.newcodes.items())
    by_setup = collections.Counter((k[1], r["state"]) for k, r in sw.newcodes.items())
    total = lambda c, fams: {s: sum(n for (f, s2), n in c.items() if f in fams and s2 == s) for s in ("done", "stopped", "open")}
    nc = total(new, NEWCODES);planned = sum(nc.values());finished = nc["done"] + nc["stopped"]
    order = collections.Counter(r["state"] for r in sw.order.values());giz = collections.Counter(r["state"] for r in sw.gizmo.values())
    where = {f: w for f, _, w, _ in bb.CODES};names = {f: n for f, n, _, _ in bb.CODES}

    def accuracy(m):
        """time to the accuracy (median × the fastest, tests reached of those finished), order on the smooth waves, speed"""
        med, n, ran = bb.time_ratio(rows, tests, fastest, m)
        smooth = [rows[t]["methods"][m]["order"] for t in bb.JOBS[0][1] if t in rows and (rows[t]["methods"].get(m) or {}).get("order") is not None]
        s = bb.speed(rows, tests, m)
        return (f"{bb.times(med)} <span class='small'>({n}/{ran})</span>" if med else f"– ({n}/{ran})",
                f"{statistics.mean(smooth):.2f}" if smooth else "–", bb.mcell(s[0]) if s else "–")

    # 1. the five codes, set-up by set-up, with AthenaK PPM/HLLD for comparison
    new_setups = [m for m in methods if pc.STYLE[m][0] in NEWCODES]
    setup_rows = [(e(bb.name(m)), where[pc.STYLE[m][0]], by_setup.get((m, "done"), 0), by_setup.get((m, "stopped"), 0), by_setup.get((m, "open"), 0), *accuracy(m))
                  for m in new_setups]
    setup_rows.append(("<em>for comparison: AthenaK PPM/HLLD/RK3</em>", "GPU", "", "", "", *accuracy("athenak_best")))

    # 2. speed against time to accuracy; the best per job; the Orszag-Tang example
    a, b, rate, rel = bb.extremes(rows, tests, methods, fastest)
    speed_png = speed_figure(week, rows, tests, methods, fastest, {
        "glm": ("Abel GLM LLF (original)", (-20, 34)), "glm_hlld": ("Abel GLM HLLD, f16/HLL", (-4, -30)), "ct": ("Abel CT prototype", (-8, 0)),
        "cholla_ppmc": ("Cholla PPMC", (8, 0)), "athenak_best": ("AthenaK PPM/HLLD", (8, 0)), "athenak_matched": ("AthenaK PLM/HLLE", (-8, 6)),
        "athenapk_best": ("AthenaPK PPM/HLLD", (-8, 0)), "gizmo_mfm": ("GIZMO", (8, 0)), "phantom_sph": ("Phantom", (8, 0))})
    per_test = [(t, *test_speed_figure(week, rows, t)) for t in tests]
    planar_only = sorted({names[pc.STYLE[m][0]] for m in methods if (s := bb.speed(rows, tests, m)) and "³" not in s[1]})
    job_rows = [(e(job), *(f"{e(bb.full_name(m))} <span class='small'>({bb.times(s)})</span>" for m, s in top), f"{n} of {len(methods)}")
                for job, _, top, n in bb.best(rows, methods, fastest)]
    ot = rows["orszag_tang"]["methods"];ex = [m for m in ("glm", "glm_fix_hlld", "cholla_ppmc") if (ot.get(m) or {}).get("cost")]
    ex_rows = [(e(bb.full_name(m)), f"{ot[m]['N']:.0f}²", bb.mcell(ot[m]["rate"]), bb.seconds(ot[m]["cost"])) for m in ex]

    # 3. the order x solver study: medians over the tests of the time to the accuracy, each change alone and both
    split_rows = []
    for name, cells in wp.GRID:
        g = [x for r in rows.values() if (x := wp.factorial(r, cells))]
        if not g:continue
        med = lambda k: statistics.median(x[k] for x in g)
        sp = [bb.speed(rows, tests, c) for c in cells[:2]]
        split_rows.append((e(name), gain(med("order_hll")), gain(med("solver_plm")), gain(med("both")), f"{sp[0][0]/sp[1][0]:.2g}×" if all(sp) else "–", len(g)))

    # 4. convergence without the true solution; 5. the corrections' numbers
    n_self = sum(len(v["methods"]) for v in sc.values());selfc = self_rows(sc)
    under = lambda row: f"{100*(1 - float(row[1])):.0f} %" if row[1] != "–" else "–"
    conv_png = convergence_figure(week, sc, {"glm_fix_hlld": "Abel et al. GLM HLLD, fixed", "athenak_best": "AthenaK PPM/HLLD/RK3",
                                             "ramses_best": "RAMSES HLLD", "gizmo_mfm": "GIZMO MFM"})
    giz_alf = (sc.get("alfven") or {}).get("methods", {}).get("gizmo_mfm", {})
    arepo = (sc.get("field_loop_novz") or {}).get("methods", {}).get("arepo_static", {}).get("truth", {})
    fixes = [("The new codes' run times were srun's wall clock, start-up included",
              "gPLUTO Brio–Wu 32²: 456 s recorded, 18 s of it computing", "Each record now has the code's own clock"),
             ("Speeds were taken from the planar runs", "A planar run leaves a GPU mostly idle; Abel et al.'s kernels run the planar tests on N³ cells, "
              "so they were favoured", "Each set-up's speed from its largest finished 3D run"),
             ("Our GIZMO conversion divided the field by the cell volume", "GIZMO's particle volumes add up to 1.0005 of the box, so the field on "
              "the grid was 0.05 % too strong at every grid", f"Divided by the particle volume each cell receives; all {giz['done']} complete runs "
              f"reconverted: Alfvén-wave error at 512² from 1.16×10⁻³ to {pc.sci(giz_alf.get('truth', {}).get('512', float('nan')), 2)}"),
             ("A code's stop on the second of two GPUs was read as a lost job (gPLUTO, Cholla)", "No record, and the task's later runs not started",
              "The stop is read from every rank's log and recorded with the code's reason"),
             ("AREPO's static mesh finishes the field loop", f"Its field error is {arepo.get('64', 0):.0f}× the field at 64², {arepo.get('128', 0):.0f}× at 128²",
              "Flagged; not yet marked invalid")]
    open_rows = [(e(names[f]), where[f], new.get((f, "open"), 0)) for f in NEWCODES if new.get((f, "open"))]

    title = "Progress, 28 October – 4 November 2026"
    summary = [f"AthenaPK, Cholla, gPLUTO, Enzo and Phantom on every test: {finished:,} of {planned:,} runs finished.",
               f"Time to a given accuracy, order and speed for {len(methods)} set-ups on {len(tests)} tests.",
               "Convergence measured without the exact solution; a bug in our GIZMO conversion found and fixed."]
    body = f'''{STYLE}
<div class="page-heading"><div><div class="eyebrow"><span>W · Weekly progress</span><span class="badge">{span("2026-10-28", week)}</span><a href="weekly.html">All weeks</a></div>
<h1>{e(title)}</h1></div></div>
{stats([(f"{finished:,} / {planned:,}", "runs of the five new codes finished"), (f"{order['done']} / {sum(order.values())}", "runs of the PLM/PPM × HLL/HLLD study complete"),
        (f"{len(methods)} × {len(tests)}", "set-ups × tests on the Benchmark page"), (f"{n_self}", "set-up × test convergence checks without the exact solution")])}
{card(1, "New codes: AthenaPK, Cholla, gPLUTO, Enzo and Phantom",
      table(("set-up", "", "complete", "stopped", "still to run", "time to accuracy", "order, smooth waves", "M cell updates / s"), setup_rows),
      f"<p>Five more codes now run the whole suite from the same starting states as the others: AthenaPK, Cholla and gPLUTO on H100s, Enzo and "
      f"Phantom on CPUs, each with a <em>matched</em> set-up (second order, an HLL-type solver) and its <em>best</em> one. {finished:,} of their "
      f"{planned:,} runs have finished.</p>"
      "<p><strong>How to read it.</strong> <em>Stopped</em>: the code stopped itself or its state turned non-finite; each is recorded with the "
      "code's own reason. <em>Time to accuracy</em>: how long the set-up takes to reach a fixed error, as a multiple of the fastest set-up on each "
      "test, median over the tests it reached (in brackets: tests reached of those it finished; section 2 explains the measure). <em>Order</em>: "
      "how fast the error falls with resolution on the smooth-wave tests (2 for a second-order scheme).</p>"
      "<p><strong>What stands out.</strong> Cholla PPMC and AthenaPK PPM/HLLD are among the fastest of all to the accuracy. Most stops are the "
      "codes' own checks: Cholla stops when div B passes a fixed limit; AthenaPK PPM/HLLD turns non-finite on the decay test on every grid and seed; "
      "Phantom's stability check stops the field loops (its own divergence projection removed 26 % of the magnetic energy by the first output, so "
      "it runs without it). Cholla and gPLUTO need two H100s for 3D at 512³.</p>",
      more("sweep-plan.html", "Every run's state on the Sweep page"))}
{card(2, "Time to a given accuracy, and speed per cell",
      figure(speed_png, f"All {len(tests)} tests together. Each point is a set-up: its speed against its median time to the accuracy over the tests "
             "(× the fastest set-up on each test, so 1 = the fastest)."),
      f"<p><strong>How it is measured.</strong> For each test we fix a target error: the error AthenaK PLM/HLLE/RK2, a standard second-order "
      f"set-up, has at 256². For each set-up we find the grid on which its error falls to that target (interpolated between the grids it ran), and "
      f"the time it needs there: its steps times the cells, divided by its speed (cell updates per second on its largest 3D run, on one H100 or 128 "
      f"cores). The time is then divided by that of the fastest set-up on the test; the plot shows the median over the {len(tests)} tests.</p>"
      f"<p><strong>What it shows.</strong> Speed per cell does not decide the time to an answer. {e(bb.full_name(a))} makes the most cell updates "
      f"per second, {rate[a]/rate[b]:.2g}× as many as {e(bb.full_name(b))}, but needs {bb.times(rel[a][0])} the fastest time against "
      f"{bb.times(rel[b][0])} for {e(bb.full_name(b))}. A more diffusive scheme needs a finer grid for the same error, and in 2D the work grows as "
      f"the cube of the grid (cells × steps). On Orszag–Tang:</p>",
      table(("set-up", "grid needed", "M cell updates / s", "time on one H100"), ex_rows),
      "<p>The least time to the accuracy, per kind of test (× the fastest on each test, geometric mean over the kind's tests; only set-ups that "
      "reached the accuracy on all of them):</p>",
      table(("tests", "first", "second", "third", "reached all"), job_rows),
      f"<p><strong>Test by test.</strong> The same plot for each test, with the time in seconds to that test's target error. Speeds are the same "
      f"on every test except for codes with no 3D run ({", ".join(planar_only)}), so mostly the points move up and down: the order of the set-ups changes from test "
      "to test. Set-ups that finished a test but reached its target on none of their grids are in the band at the top.</p>",
      test_switcher(per_test),
      more("benchmark.html", "Every set-up and test on the Benchmark page"))}
{card(3, "Higher order and a better Riemann solver: PLM vs PPM, HLL vs HLLD",
      table(("code", "PLM → PPM", "HLL → HLLD", "both", "PPM's cost per cell update", "tests"), split_rows),
      f"<p><strong>What was run.</strong> In AthenaK, Athena++ and Abel et al.'s kernels, all four pairs of reconstruction (PLM, piecewise linear; "
      f"PPM, piecewise parabolic) and Riemann solver (an HLL-type one, or HLLD), at one time integrator: {order['done']} complete runs.</p>"
      "<p><strong>How to read it.</strong> Each entry is how much faster the set-up reaches the target error of section 2 after the change, "
      "median over the tests: PPM in place of PLM (with the HLL-type solver), HLLD in place of HLL (with PLM), and both. The last but one column is "
      "how much more a cell update costs with PPM than with PLM.</p>"
      "<p><strong>What it shows.</strong> In AthenaK and Athena++ PPM pays: the coarser grid it needs more than covers its cost per cell (on the "
      "smooth waves Athena++ PPM converges at about fourth order). In Abel et al.'s kernels a PPM cell update costs several times a PLM one, so PPM "
      "does not pay on most tests; HLLD does, in every code. The figure shows each test.</p>",
      figure("media/ours/order_tradeoff.png", "(a) Each change's time to the same error, test by test (right of 1: the change is faster); (b) the "
             "cells per side it saves.", "our-figures.html"),
      more("our-figures.html", "Our figures page"))}
{card(4, "Convergence measured without the exact solution",
      figure(conv_png, "Solid: the error against the true solution. Dashed: against the set-up's own 512² run, averaged to each grid (so up to 256²)."),
      f"<p><strong>Why.</strong> In a real problem the true solution is unknown, and convergence is judged from the runs themselves. We tried three "
      f"ways on the {len(sc)} planar tests and compared each with the true error, which is known there (the exact solution, or AthenaK at 4096²): "
      "(1) the set-up's own finest grid as the reference; (2) the difference between grid N and 2N, whose fall with N gives the order; (3) Richardson "
      "extrapolation, which combines the three finest grids, assuming the error falls as N<sup>−p</sup>, into an estimate of the converged answer.</p>",
      table(("tests", "own finest ÷ true error", "order from differences", "order against truth", "Richardson estimate ÷ true error",
             "Richardson gain"), selfc),
      f"<p><strong>What it shows.</strong> Against its own finest grid a set-up looks better than it is: its error seems {under(selfc[0])} smaller "
      f"on smooth waves, {under(selfc[1])} with shocks, {under(selfc[2])} on Kelvin–Helmholtz and the current sheet. Grid differences give the true "
      f"order on smooth waves; on Kelvin–Helmholtz and the current sheet they shrink at order {selfc[2][2]} while the true error barely falls (order "
      f"{selfc[2][3]}): the grids agree with each other, not with the truth. Richardson's estimate of the error is good on smooth waves (ratio "
      f"{selfc[0][4]}), but its extrapolated answer is no nearer the truth than the finest grid (gain {selfc[0][5]}; below 1 with shocks): the "
      "limiters change the error's pattern from grid to grid. In the figure, GIZMO's true error on the Alfvén wave stops falling at 64² (about "
      "0.1 %), while against its own finest grid it seems to converge: an own-grid reference cannot see a code converging to a wrong answer.</p>",
      more("test-alfven.html", "All set-ups on the Alfvén-wave test page"))}
{card(5, "Corrections to our data",
      table(("what was wrong", "effect", "fix"), fixes),
      "<p>Found in an audit of our own records and conversions before the final rebuild. The GIZMO error was found by the convergence checks of "
      "section 4: every grid of GIZMO's had the same error floor, which its own-grid comparison could not show.</p>")}
{card(6, "Runs still going, and next steps",
      table(("code", "", "runs still to run"), open_rows),
      bullets(["When they are in: the last fetch, every analysis and page rebuilt, and the site published again.",
               "The convergence checks of section 4 in 3D (the 3D states are too large to fetch: a job on Sherlock).",
               "Decide whether to mark AREPO's static field loop invalid."]))}'''
    return dict(first="2026-10-28", last=week, title=title, summary=summary, html=body)

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
