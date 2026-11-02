"""The benchmark page: which code is good at what, test by test, measured two ways.

    python3 scripts/build_benchmark.py          -> progress/benchmark.html (after scripts/work_precision.py)

1. Time to a given accuracy: the time each set-up needs for its error at the end to fall to that of a standard second-order
   code (AthenaK PLM/HLLE/RK2) at 256^2, on one H100 or 128 CPU cores (work_precision.py: cost model and interpolation).
2. Order against resolution: the error's slope between the two finest grids a set-up finished.
Both come from progress/media/analysis/work_precision.json; nothing here is measured anew.
"""
import html
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as pc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
DATA = SITE/"media"/"analysis"/"work_precision.json"
PAGE = dict(id="benchmark", page="benchmark.html", kind="benchmark", number="00", title="Benchmark", nav="Benchmark", status="",
            document="", description="Which code is good at what: the time to reach a given accuracy and the order of accuracy, test by test.", chips=[])
e = html.escape
# The jobs a reader picks a code for, and the tests that measure each (work_precision.TESTS)
JOBS = (("Smooth waves", ("entropy", "alfven", "alfven_oblique")),
        ("Field transport", ("field_loop_novz", "field_loop_3d")),
        ("Shocks", ("brio_wu", "brio_wu_rotated", "blast_2d_b0", "blast_2d_b50")),
        ("Vortices and instabilities", ("orszag_tang", "rotor", "kh_uniform")),
        ("Reconnection", ("current_sheet", "gem")))
# Codes in the order of the rows: (family in paper_common.STYLE, name, where it runs, precision)
CODES = (("glm", "Abel et al., GLM kernels", "GPU", "Float32 state; the f16 kernels compute in Float16"), ("ct", "Abel et al., CT prototype", "GPU", "Float32"),
         ("athenak", "AthenaK", "GPU", "fp64"), ("athenapk", "AthenaPK", "GPU", "fp64"), ("cholla", "Cholla", "GPU", "fp64"),
         ("pluto", "gPLUTO", "GPU", "fp64"), ("athenapp", "Athena++", "CPU", "fp64"), ("ramses", "RAMSES", "CPU", "fp64"),
         ("enzo", "Enzo", "CPU", "fp64"), ("arepo", "AREPO", "CPU", "fp64"), ("gizmo", "GIZMO", "CPU", "fp64"),
         ("phantom", "Phantom", "CPU", "fp64"))


def test_page(t):
    return f'test-{t.replace("_", "-")}.html'


def name(m):
    return pc.plain(pc.STYLE[m][1])


def ratio_colour(r):
    """Sequential blue by how many times the fastest: 1x darkest, 100x and slower lightest (text stays readable)."""
    f = min(1.0, math.log10(r)/2.0)
    dark, light = (28, 92, 171), (232, 240, 251)
    c = tuple(round(d + (l - d)*f) for d, l in zip(dark, light))
    return f"rgb{c}", "#fff" if f < 0.45 else "#202a2a"


def order_colour(o):
    """Sequential green by the measured order: 0 lightest, 3 and above darkest."""
    f = max(0.0, min(1.0, o/3.0))
    dark, light = (14, 110, 88), (229, 244, 239)
    c = tuple(round(l + (d - l)*f) for d, l in zip(dark, light))
    return f"rgb{c}", "#fff" if f > 0.55 else "#202a2a"


def seconds(x):
    if x < 0.01:return f"{x*1e3:.2g} ms"
    if x < 1:return f"{x*1e3:.0f} ms"
    return f"{x:.3g} s" if x < 600 else f"{x/60:.0f} min"


def times(r):
    return f"{r:.2g}×" if r < 10 else f"{r:.0f}×"


def full_name(m):
    """With the authors for Abel et al.'s kernels, whose labels (GLM HLLD, ...) another code's could share."""
    return ("Abel et al. " if pc.STYLE[m][0] in ("glm", "ct") else "") + name(m)


def build():
    rows = json.loads(DATA.read_text())
    tests = [t for _, ts in JOBS for t in ts if t in rows]
    methods = [m for fam, *_ in CODES for m in pc.STYLE if pc.STYLE[m][0] == fam and any(m in rows[t]["methods"] for t in tests)]
    fastest = {t: min((x["cost"] for x in rows[t]["methods"].values() if x.get("cost")), default=None) for t in tests}

    def cell_time(t, m):
        x = rows[t]["methods"].get(m)
        if x is None:return '<td class="na" title="no finished run of this test">n/a</td>'
        if not x.get("cost"):
            return f'<td class="miss" title="{e(name(m))}: no grid reached the error (finest error {x["points"][-1][1]:.3g})">not reached</td>'
        r = x["cost"]/fastest[t];bg, fg = ratio_colour(r)
        tip = (f'{full_name(m)} on {t}: error {rows[t]["eps"]:.3g} at N ≈ {x["N"]:.0f}' + (" or coarser" if x["at_most"] else "") +
               f'; {seconds(x["cost"])} on {"128 cores" if x["cpu"] else "one H100"}; speed measured on {x["rate_from"]}')
        return (f'<td style="background:{bg};color:{fg}" title="{e(tip)}"><a href="{test_page(t)}#analysis-03-cost" style="color:inherit">'
                f'{"≤ " if x["at_most"] else ""}{seconds(x["cost"])}<small>{times(r)}</small></a></td>')

    def cell_order(t, m):
        x = rows[t]["methods"].get(m)
        if x is None:return '<td class="na">n/a</td>'
        o = x.get("order");pts = x["points"]
        if o is None:return f'<td class="miss" title="the error did not fall between the two finest grids ({pts[-2][0] if len(pts) > 1 else "–"} → {pts[-1][0]})">—</td>'
        bg, fg = order_colour(o)
        tip = f'{name(m)} on {t}: error ' + " → ".join(f"{err:.3g} at {n}" for n, err, _ in pts) + f"; order {o:.2f} between {pts[-2][0]} and {pts[-1][0]}"
        return f'<td style="background:{bg};color:{fg}" title="{e(tip)}"><a href="{test_page(t)}#analysis-02-convergence" style="color:inherit">{o:.2f}</a></td>'

    head = "".join(f'<th colspan="{len([t for t in ts if t in rows])}" class="job">{e(job)}</th>' for job, ts in JOBS)
    sub = "".join(f'<th title="error to reach: {rows[t]["eps"]:.3g} ({rows[t]["kind"]} L1, relative)"><a href="{test_page(t)}">{e(t.replace("_", " "))}</a></th>' for t in tests)

    def table(cell, extra_head, extra):
        body = []
        for fam, code, where, precision in CODES:
            mine = [m for m in methods if pc.STYLE[m][0] == fam]
            if not mine:continue
            body.append(f'<tr class="family-row"><th colspan="{len(tests) + 2}">{e(code)} <span class="small">· {where} · {e(precision)}</span></th></tr>')
            for m in mine:
                body.append(f'<tr><th scope="row" class="setup">{e(name(m))}</th>{"".join(cell(t, m) for t in tests)}{extra(m)}</tr>')
        return (f'<div class="matrix-scroll"><table class="bench"><thead><tr><th></th>{head}<th></th></tr><tr><th>set-up</th>{sub}'
                f'<th>{extra_head}</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>')

    def overall_time(m):
        rs = [rows[t]["methods"][m]["cost"]/fastest[t] for t in tests if (rows[t]["methods"].get(m) or {}).get("cost")]
        ran = sum(m in rows[t]["methods"] for t in tests)
        return f'<td class="num" title="median over the tests it reached; reached {len(rs)} of the {ran} it finished">{statistics.median(rs):.2g}× · {len(rs)}/{ran}</td>' if rs else f'<td class="num">0/{ran}</td>'

    def overall_order(m):
        smooth = [rows[t]["methods"][m]["order"] for t in JOBS[0][1] if t in rows and (rows[t]["methods"].get(m) or {}).get("order") is not None]
        return f'<td class="num" title="mean over the smooth-wave tests">{statistics.mean(smooth):.2f}</td>' if smooth else '<td class="num">—</td>'

    best = []
    for job, ts in JOBS:
        ts = [t for t in ts if t in rows];score = {}
        for m in methods:
            rs = [rows[t]["methods"][m]["cost"]/fastest[t] for t in ts if (rows[t]["methods"].get(m) or {}).get("cost")]
            if len(rs) == len(ts):score[m] = math.exp(sum(map(math.log, rs))/len(rs))
        top = sorted(score.items(), key=lambda kv: kv[1])[:3]
        best.append(f'<li><strong>{e(job)}</strong> ({", ".join(t.replace("_", " ") for t in ts)}): ' +
                    ", ".join(f"{e(full_name(m))} ({times(s)})" for m, s in top) + f'; {len(score)} of {len(methods)} set-ups reached the accuracy on every one.</li>')

    style = ('<style>.bench{border-collapse:collapse;font-size:.82rem}.bench th,.bench td{padding:4px 6px;border:1px solid #fff;text-align:center;white-space:nowrap}'
             '.bench td small{display:block;font-size:.7rem;opacity:.85}.bench th.setup{text-align:left;font-weight:500}.bench th.job{border-bottom:2px solid #202a2a}'
             '.bench td.na{color:#9aa5a0;background:#f3f5f4}.bench td.miss{color:#64716f;background:#f3f5f4;font-style:italic}.bench a{text-decoration:none}'
             '.bench tr.family-row th{text-align:left;background:#e9edeb;padding-top:8px}.bench td.num{background:#fff;color:#202a2a}'
             '.key{display:inline-block;width:14px;height:12px;vertical-align:middle;margin:0 3px 0 10px;border-radius:2px}</style>')
    key_time = "".join(f'<i class="key" style="background:{ratio_colour(r)[0]}"></i>{r:g}×' for r in (1, 3, 10, 30, 100))
    key_order = "".join(f'<i class="key" style="background:{order_colour(o)[0]}"></i>{o:g}' for o in (0.5, 1, 1.5, 2, 3))
    fetched = (ROOT/"newcodes_results"/"FETCHED").read_text().strip() if (ROOT/"newcodes_results"/"FETCHED").is_file() else ""
    return f'''{style}
<div class="page-heading"><div><div class="eyebrow"><span>{PAGE["number"]} · Benchmark</span><span class="badge">{len(methods)} set-ups · {len(tests)} tests</span></div>
<h1>Which code is good at what</h1><p>Twelve codes on the same tests, from the same starting states, measured the same way. Two questions per test:
how long a set-up takes to reach a given accuracy, and how fast its error falls as the grid is refined.</p></div></div>
<article class="card"><h3>Best at each job: least time to the accuracy</h3><ul class="checks plain">{"".join(best)}</ul>
<p class="small">Geometric mean, over the job's tests, of the time relative to the fastest set-up on each test; only set-ups that reached the accuracy on all of them.</p></article>
<article class="card" id="time"><h3>1 · Time to a given accuracy</h3>
<p>For each test, the error to reach is that of a standard second-order code, AthenaK PLM/HLLE/RK2, at 256² (its value is in each column's tooltip).
Each cell is the time a set-up needs for that error at the end of the test: on one H100 for the GPU codes, on 128 cores for the CPU codes
(their speed per core, assuming perfect scaling). The small number is how many times the fastest set-up on that test. A set-up's speed is
measured on its largest finished 3D run, where a GPU is fully used; the grid it needs is read from its errors on the grids it ran.
Click a cell for the test's accuracy-against-cost figure.</p>
<p class="small">Shade: {key_time} the fastest on the test · <em>not reached</em>: no grid of the sweep got that accurate · n/a: no finished run.</p>
{table(cell_time, "median × fastest · tests reached", overall_time)}</article>
<article class="card" id="order"><h3>2 · Order of accuracy against resolution</h3>
<p>The measured order between the two finest grids a set-up finished: the error falls as N<sup>−p</sup>. A second-order scheme shows p ≈ 2 on the
smooth waves; on shocks and other discontinuities every scheme falls to about 1 or below. The tooltip lists the error on every grid; click a cell for
the test's error-against-resolution figure.</p>
<p class="small">Shade: {key_order} · —: the error did not fall between the two finest grids.</p>
{table(cell_order, "mean on smooth waves", overall_order)}</article>
<article class="card"><h3>How this is measured</h3><ul class="checks plain">
<li>Every code starts from the suite's own initial state and writes the same outputs; the error is against the exact solution or a 4096² reference run.</li>
<li>Each code runs a <em>matched</em> set-up (second order, an HLL-type solver, as close to Abel et al.'s kernels as it allows) and a <em>best</em> one (its strongest standard choice).</li>
<li>Precision: Abel et al.'s kernels keep their state in Float32 (the f16 kernels compute in Float16); every other code runs in fp64. GPU: one NVIDIA H100; set-ups on two GPUs are counted per GPU.</li>
<li>A run that stopped or turned non-finite counts as not reaching the accuracy at that grid; the test pages say why each stopped.</li>
<li>Seed 42, the end time of each test. Data: <a href="media/analysis/work_precision.json">work_precision.json</a>{f"; the five newest codes as fetched {e(fetched)}" if fetched else ""}.</li></ul></article>'''


def render(main_html, registry):
    nav = "".join(f'<a href="{e(t["page"])}"' + (' aria-current="page"' if t["id"] == PAGE["id"] else "") + f'><span>{t["number"]}</span>{e(t["nav"])}</a>' for t in registry)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="{e(PAGE["description"])}">
<title>Benchmark · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="benchmark"><a class="skip-link" href="#main">Skip to the benchmark</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> BENCHMARK</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav}</nav>
<main id="main">{main_html}</main>
<footer><span>From work_precision.json; page by scripts/build_benchmark.py.</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main():
    registry = json.loads((SITE/"tests.json").read_text())
    registry = [PAGE if t["id"] == PAGE["id"] else t for t in registry] if any(t["id"] == PAGE["id"] for t in registry) else [PAGE] + registry
    (SITE/"tests.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    (SITE/PAGE["page"]).write_text(render(build(), registry))
    print(f"Benchmark: {SITE/PAGE['page']}")


if __name__ == "__main__":
    main()
