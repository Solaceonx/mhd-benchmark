"""The benchmark page: which code is good at what, test by test, measured two ways.

    python3 scripts/build_benchmark.py          -> progress/benchmark.html (after scripts/work_precision.py)

1. Time to a given accuracy: the time each set-up needs for its error at the end to fall to that of a standard second-order
   code (AthenaK PLM/HLLE/RK2) at 256^2, on one H100 or 128 CPU cores (work_precision.py: cost model and interpolation).
2. Order against resolution: the error's slope between the two finest grids a set-up finished.
3. Speed: each set-up's cell updates per second (the rate of the cost model in 1), beside its time to the accuracy.
All come from progress/media/analysis/work_precision.json; nothing here is measured anew.
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
         ("pluto", "gPLUTO", "GPU", "fp64"), ("miniramses", "RAMSES II (E. Moseley)", "GPU", "fp64"), ("athenapp", "Athena++", "CPU", "fp64"), ("ramses", "RAMSES", "CPU", "fp64"),
         ("enzo", "Enzo", "CPU", "fp64"), ("arepo", "AREPO", "CPU", "fp64"), ("gizmo", "GIZMO", "CPU", "fp64"),
         ("mhdale", "MHD-ALE + our fixes", "CPU", "fp64"),
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


def precision(m):
    """The arithmetic of a set-up (CODES gives it per code): Abel et al.'s f16 kernels compute in Float16."""
    if pc.STYLE[m][0] not in ("glm", "ct"):return "fp64"
    return "Float16" if m.endswith("16") else "Float32"


def speed(rows, tests, m):
    """(cell updates per second, where measured): one H100, or 128 cores at the speed per core of a CPU code (perfect
    scaling, as the cost model). The rate is the set-up's largest 3D run (work_precision.full_rate), the same on every
    test; set-ups with no 3D run (AREPO, GIZMO) have each test's largest grid: their median over the tests."""
    x = [rows[t]["methods"][m] for t in tests if m in rows[t]["methods"]]
    if not x:return None
    cores = 128 if x[0]["cpu"] else 1
    if "³" in x[0]["rate_from"]:return x[0]["rate"]*cores, x[0]["rate_from"], x[0]["cpu"]
    return statistics.median(r["rate"] for r in x)*cores, f"its planar runs (median of {len(x)} tests at their largest grid)", x[0]["cpu"]


def mcell(rate):
    r = rate/1e6;return f"{r:,.0f}" if r >= 10 else f"{r:.2g}"


def setups(rows):
    """The page's tests (in the order of JOBS), its set-ups (in the order of CODES) and the least time to the accuracy on each test."""
    tests = [t for _, ts in JOBS for t in ts if t in rows]
    methods = [m for fam, *_ in CODES for m in pc.STYLE if pc.STYLE[m][0] == fam and any(m in rows[t]["methods"] for t in tests)]
    fastest = {t: min((x["cost"] for x in rows[t]["methods"].values() if x.get("cost")), default=None) for t in tests}
    return tests, methods, fastest


def best(rows, methods, fastest):
    """Per job: (job, its tests, the three set-ups with the least time to the accuracy over all of them, as the geometric
    mean of the time over the fastest's, and how many set-ups reached the accuracy on every one of them)."""
    out = []
    for job, ts in JOBS:
        ts = [t for t in ts if t in rows];score = {}
        for m in methods:
            rs = [rows[t]["methods"][m]["cost"]/fastest[t] for t in ts if (rows[t]["methods"].get(m) or {}).get("cost")]
            if len(rs) == len(ts):score[m] = math.exp(sum(map(math.log, rs))/len(rs))
        out.append((job, ts, sorted(score.items(), key=lambda kv: kv[1])[:3], len(score)))
    return out


def time_ratio(rows, tests, fastest, m):
    """(median, over the tests it reached, of its time to the accuracy over the fastest set-up's; tests reached; tests finished)"""
    rs = [rows[t]["methods"][m]["cost"]/fastest[t] for t in tests if (rows[t]["methods"].get(m) or {}).get("cost")]
    return (statistics.median(rs) if rs else None), len(rs), sum(m in rows[t]["methods"] for t in tests)


def extremes(rows, tests, methods, fastest):
    """(the set-up with the most cell updates per second, the one with the least median time to the accuracy of those that
    reached at least three quarters of the tests they finished, {set-up: speed}, {set-up: time_ratio}); None if one is both."""
    rate = {m: s[0] for m in methods if (s := speed(rows, tests, m))};rel = {m: time_ratio(rows, tests, fastest, m) for m in rate}
    a = max(rate, key=rate.get);ok = [m for m, (med, n, ran) in rel.items() if med and n >= .75*ran]
    b = min(ok, key=lambda m: rel[m][0]) if ok else None
    return (a, b, rate, rel) if b and b != a else None


def build():
    rows = json.loads(DATA.read_text())
    tests, methods, fastest = setups(rows)

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

    def relative(m):
        return time_ratio(rows, tests, fastest, m)

    def overall_time(m):
        med, n, ran = relative(m)
        return f'<td class="num" title="median over the tests it reached; reached {n} of the {ran} it finished">{times(med)} · {n}/{ran}</td>' if med else f'<td class="num">0/{ran}</td>'

    def bars(cpu):
        """One bar per set-up: its cell updates per second, codes in the order of their fastest set-up; beside it the
        median time to the accuracy of section 1. Hue: the code; hatching: the precision. GPU and CPU on one scale: section
        1 sets one H100 against 128 cores."""
        every = [s for m in methods if (s := speed(rows, tests, m))]
        got = {m: s for m in methods if (s := speed(rows, tests, m)) and s[2] == cpu}
        if not got:return ""
        top = max(s[0] for s in every)
        fams = sorted({pc.STYLE[m][0] for m in got}, key=lambda f: -max(got[m][0] for m in got if pc.STYLE[m][0] == f))
        out = [f'<div class="speed-row speed-head"><span>set-up</span><span>million cell updates per second, {"128 cores" if cpu else "one H100"}</span>'
               f'<span></span><span title="section 1: median over the tests it reached of the time to the accuracy over the fastest set-up\'s; tests reached of those finished">time to accuracy</span></div>']
        for f in fams:
            for m in sorted((m for m in got if pc.STYLE[m][0] == f), key=lambda m: -got[m][0]):
                rate, where, _ = got[m];med, n, ran = relative(m);p = precision(m)
                tip = (f"{full_name(m)}: {mcell(rate)} million cell updates per second on {'128 cores (per core × 128)' if cpu else 'one H100'}, {p}; "
                       f"measured on {where}. Time to the accuracy: " + (f"{times(med)} the fastest (median), reached {n} of {ran} tests" if med else f"reached none of {ran} tests"))
                acc = f'{times(med)}<small>{n}/{ran}</small>' if med else f'—<small>0/{ran}</small>'
                out.append(f'<div class="speed-row" title="{e(tip)}"><span class="speed-name">{e(full_name(m))}</span><span class="speed-track">'
                           f'<i class="speed-bar pat-{p.lower()}" style="width:{100*rate/top:.1f}%;--c:{pc.color(m)}"></i></span>'
                           f'<span class="speed-num">{mcell(rate)}</span><span class="speed-acc">{acc}</span></div>')
        return f'<div class="speed">{"".join(out)}</div>'

    def contrast():
        x = extremes(rows, tests, methods, fastest)
        if not x:return ""
        a, b, rate, rel = x
        return (f"{e(full_name(a))} makes the most cell updates per second, {rate[a]/rate[b]:.2g}× as many as {e(full_name(b))}; yet "
                f"{e(full_name(b))} reaches the accuracy in the least time (a median {times(rel[b][0])} the fastest on each test, against {times(rel[a][0])}).")

    def overall_order(m):
        smooth = [rows[t]["methods"][m]["order"] for t in JOBS[0][1] if t in rows and (rows[t]["methods"].get(m) or {}).get("order") is not None]
        return f'<td class="num" title="mean over the smooth-wave tests">{statistics.mean(smooth):.2f}</td>' if smooth else '<td class="num">—</td>'

    tops = [f'<li><strong>{e(job)}</strong> ({", ".join(t.replace("_", " ") for t in ts)}): ' +
            ", ".join(f"{e(full_name(m))} ({times(s)})" for m, s in top) + f'; {n} of {len(methods)} set-ups reached the accuracy on every one.</li>'
            for job, ts, top, n in best(rows, methods, fastest)]

    style = ('<style>.bench{border-collapse:collapse;font-size:.82rem}.bench th,.bench td{padding:4px 6px;border:1px solid #fff;text-align:center;white-space:nowrap}'
             '.bench td small{display:block;font-size:.7rem;opacity:.85}.bench th.setup{text-align:left;font-weight:500}.bench th.job{border-bottom:2px solid #202a2a}'
             '.bench td.na{color:#9aa5a0;background:#f3f5f4}.bench td.miss{color:#64716f;background:#f3f5f4;font-style:italic}.bench a{text-decoration:none}'
             '.bench tr.family-row th{text-align:left;background:#e9edeb;padding-top:8px}.bench td.num{background:#fff;color:#202a2a}'
             '.key{display:inline-block;width:14px;height:12px;vertical-align:middle;margin:0 3px 0 10px;border-radius:2px}'
             '.speed{display:grid;gap:2px;margin-top:12px}.speed-row{display:grid;grid-template-columns:minmax(8rem,17rem) minmax(4rem,1fr) 4rem 5.5rem;'
             'align-items:center;column-gap:10px;font-size:11px;color:#202a2a;padding:2px 4px;border-radius:3px}.speed-row:hover{background:#f3f5f4}'
             '.speed-head{font-size:10px;color:#64716f}.speed-head:hover{background:none}.speed-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}'
             '.speed-bar{display:block;height:12px;min-width:2px;border-radius:0 3px 3px 0}.pat-fp64{background:var(--c)}'
             '.pat-float32{background:repeating-linear-gradient(135deg,transparent 0 3px,rgba(255,255,255,.65) 3px 5px),var(--c)}'
             '.pat-float16{background:repeating-linear-gradient(135deg,transparent 0 3px,rgba(255,255,255,.65) 3px 5px),repeating-linear-gradient(45deg,transparent 0 3px,rgba(255,255,255,.65) 3px 5px),var(--c)}'
             '.speed-num,.speed-acc{font:11px ui-monospace,monospace;text-align:right;white-space:nowrap}.speed-acc small{color:#64716f;margin-left:4px;font-size:10px}'
             '.speed-group{margin-top:16px!important;font-weight:600;color:#202a2a!important}'
             '.swatch{display:inline-block;width:16px;height:11px;border-radius:2px;vertical-align:-1px;margin:0 4px 0 10px}'
             '@media(max-width:640px){.speed-row{grid-template-columns:minmax(3rem,1fr) 3.5rem 5rem}.speed-name{grid-column:1/-1}.speed-head span:first-child{display:none}}</style>')
    key_time = "".join(f'<i class="key" style="background:{ratio_colour(r)[0]}"></i>{r:g}×' for r in (1, 3, 10, 30, 100))
    key_order = "".join(f'<i class="key" style="background:{order_colour(o)[0]}"></i>{o:g}' for o in (0.5, 1, 1.5, 2, 3))
    fetched = (ROOT/"newcodes_results"/"FETCHED").read_text().strip() if (ROOT/"newcodes_results"/"FETCHED").is_file() else ""
    return f'''{style}
<div class="page-heading"><div><div class="eyebrow"><span>{PAGE["number"]} · Benchmark</span><span class="badge">{len(methods)} set-ups · {len(tests)} tests</span></div>
<h1>Which code is good at what</h1><p>Twelve codes on the same tests, from the same starting states, measured the same way. Two questions per test:
how long a set-up takes to reach a given accuracy, and how fast its error falls as the grid is refined.</p></div></div>
<article class="card"><h3>Best at each job: least time to the accuracy</h3><ul class="checks plain">{"".join(tops)}</ul>
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
<article class="card" id="speed"><h3>3 · Speed: cell updates per second</h3>
<p>The speed behind the times of section 1: how many cells a set-up advances by one step each second, on its largest finished 3D run (the
median over the 3D tests at that grid, by the code's own clock), on one H100; a set-up on two GPUs is counted per GPU. CPU codes: their speed
per core times 128, assuming perfect scaling, as in section 1; AREPO and GIZMO ran no 3D test, so theirs is from their planar runs. Beside each
bar: the set-up's time to the accuracy, from section 1 (median × the fastest, and the tests it reached of those it finished). GPU and CPU
bars share one scale.</p>
<p>A fast cell update is not a fast answer: a step does different work in each set-up (reconstruction, Riemann solver, stages per step), the
steps are of different length, and each cell buys a different accuracy. {contrast()}</p>
<p class="small">Hue: the code · <i class="swatch pat-fp64" style="--c:#64716f"></i>fp64 <i class="swatch pat-float32" style="--c:#64716f"></i>Float32
<i class="swatch pat-float16" style="--c:#64716f"></i>Float16 arithmetic (Float32 state) · hover a row for the run it was measured on.</p>
<p class="speed-group">GPU codes, one H100</p>{bars(False)}
<p class="speed-group">CPU codes, 128 cores</p>{bars(True)}</article>
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
