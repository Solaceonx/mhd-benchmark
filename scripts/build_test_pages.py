"""One page per test problem, with its movie and what every code did on it, and the index of them.

  python3 scripts/build_test_pages.py        -> progress/tests.html, progress/test-<name>.html

Reads the sweep's configs (set-up, grids, end time), progress/gpu_tests.json (family, source, what the test shows),
progress/test_movies.json (the movies of progress/media/tests/, drawn by render_suite_movies.py --family-maps) and the
sweep's plan and status through sweep_data.Sweep (who finished, who stopped). The gates and the timing test have no
page. Driven turbulence has its own page (build_turbulence.py) and a card in the index.
"""
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as pc  # noqa: E402
import sweep_data as sd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
CONFIGS = ROOT/"sherlock_sweep"/"configs"
PAGE = dict(id="tests", page="tests.html", kind="tests", number="10", title="Test problems", nav="Tests", status="", document="",
            description="Every test problem of the sweep: its set-up, a movie, and which codes finished it.", chips=[])
e = html.escape
TITLES = {
    "uniform": "Uniform state", "entropy": "Entropy wave", "alfven": "Alfvén wave", "alfven_oblique": "Oblique Alfvén wave",
    "brio_wu": "Brio–Wu shock tube", "brio_wu_rotated": "Brio–Wu shock tube at 45°", "orszag_tang": "Orszag–Tang vortex",
    "orszag_tang_3d": "Orszag–Tang vortex in 3D", "rotor": "Rotor", "current_sheet": "Current sheet", "gem": "GEM reconnection",
    "field_loop": "Field loop", "field_loop_novz": "Field loop, in plane", "field_loop_3d": "Field loop in 3D",
    "blast_2d_b0": "Blast wave, no field", "blast_2d_b50": "Blast wave, Bx = 50/√4π", "blast_2d_b100": "Blast wave, Bx = 100/√4π",
    "blast_3d_b0": "3D blast wave, no field", "blast_3d_b50": "3D blast wave, Bx = 50/√4π", "blast_3d_b100": "3D blast wave, Bx = 100/√4π",
    "kh_uniform": "Kelvin–Helmholtz, uniform field", "kh_localized": "Kelvin–Helmholtz, localized field",
    "kh_localized_dtjump": "Kelvin–Helmholtz with a timestep jump", "decay": "Decaying turbulence from Mach 2",
    "decay_supersonic": "Decaying turbulence from Mach 9",
}
FIELD_NAMES = {"density": "Density", "magnetic": "Field strength |B|", "vorticity": "Vorticity"}
STATE = {"done": "✓", "skipped": "skipped", "open": "not finished", "redo": "to rerun", "unknown": "?"}


def page_of(test):
    return f"test-{test.replace('_', '-')}.html"


def config_of(test):
    """(the comment lines that describe the set-up, the settings) of a sweep config."""
    text = (CONFIGS/f"{test}.toml").read_text()
    notes = [l[1:].strip() for l in text.splitlines() if l.startswith("#") and "Sweep config written" not in l]
    import tomllib
    return " ".join(notes), tomllib.loads(text)


def results_table(sweep, test):
    """Rows: the codes' set-ups; columns: the grids; a cell says whether every seed finished, or where a run stopped."""
    labels = list(sweep.labels(test)) + [a for a in sd.AREPO if test in sd.AREPO_TESTS and sweep.grids(test, a)]
    grids = sorted({n for l in labels for n in sweep.grids(test, l)});seeds = sweep.seeds(test);rows = [];stopped = 0;total = 0
    for label in labels:
        cells = []
        for n in grids:
            states = [(s, sweep.state_of(test, label, n, s)) for s in seeds]
            states = [(s, st) for s, st in states if st is not None]
            if not states:cells.append('<td class="none">·</td>');continue
            total += len(states);bad = [(s, st) for s, st in states if st != "done"]
            if not bad:cells.append(f'<td class="done">✓{"" if len(states) == 1 else f" ×{len(states)}"}</td>');continue
            parts = []
            for s, st in bad:
                if st == "stopped":
                    stopped += 1;rec = sweep.recorded(test, label, n, s) or {};t = rec.get("reached")
                    parts.append("stopped" + (f" at t = {t:.3g}" if isinstance(t, (int, float)) else ""))
                else:parts.append(STATE.get(st, st))
            text = parts[0] if len(set(parts)) == 1 and len(parts) == len(states) else f"{len(states) - len(bad)} of {len(states)} ✓; " + ", ".join(sorted(set(parts)))
            cells.append(f'<td class="stop">{e(text)}</td>')
        name = e(pc.label(label)).replace("$c_h$", "c<sub>h</sub>")        # the labels are written for Matplotlib
        rows.append(f'<tr><th scope="row">{name}</th>{"".join(cells)}</tr>')
    dim = "³" if sweep.configs[test].get("problem", test).endswith("3d") or test.startswith(("decay", "blast_3d", "orszag_tang_3d", "field_loop_3d")) else "²"
    head = "".join(f"<th>{n}{dim}</th>" for n in grids)
    table = (f'<div class="table-scroll"><table class="metrics-table sweep-text test-results"><thead><tr><th>set-up</th>{head}</tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>')
    return table, total, stopped, len(labels)


def movie_card(test, movie):
    if not movie:
        return '<article class="card"><h3>Movie</h3><p>No movie of this test yet: its runs kept too few outputs to make one.</p></article>'
    fields = [f for f in FIELD_NAMES if f in movie["fields"]]
    buttons = "".join(f'<button type="button" data-src="{e(movie["fields"][f])}" aria-pressed="{"true" if i == 0 else "false"}">{e(FIELD_NAMES[f])}</button>' for i, f in enumerate(fields))
    dims = "³, shown in its midplane" if movie.get("midplane") else "²"
    sparse = movie["frames"] < 100
    note = (f'{movie["frames"]} outputs on {movie["grid"]}{dims}, to t = {movie["tEnd"]:.3g}. Left to right: {"; ".join(movie["panels"])}.'
            + (" From the sweep's own runs, which kept every output on this grid." if movie.get("source") == "sweep_results" else
               " A short preview from the ordinary runs of the first campaign; the other movies come from runs that saved 241 outputs." if sparse else
               " From the movie runs of the first campaign, which saved 241 outputs."))
    return (f'<article class="card test-movie"><div class="fig-head"><h3>Movie</h3><div class="field-switch" role="group" aria-label="Quantity shown">{buttons}</div></div>'
            f'<video id="movie" controls loop muted playsinline preload="metadata" src="{e(movie["fields"][fields[0]])}"></video><p>{e(note)}</p></article>'
            "<script>document.querySelectorAll('.field-switch button').forEach(b=>b.addEventListener('click',()=>{const v=document.getElementById('movie'),t=v.currentTime,"
            "playing=!v.paused;v.src=b.dataset.src;v.addEventListener('loadedmetadata',()=>{v.currentTime=Math.min(t,v.duration||t);if(playing)v.play()},{once:true});"
            "document.querySelectorAll('.field-switch button').forEach(x=>x.setAttribute('aria-pressed',x===b))}))</script>")


def test_page(sweep, test, info, movie, order):
    notes, config = config_of(test);table, total, stopped, setups = results_table(sweep, test)
    i = order.index(test);prev, nxt = order[i - 1], order[(i + 1) % len(order)]
    grids = config["grids"];three = test.startswith(("decay", "blast_3d", "orszag_tang_3d", "field_loop_3d"));dim = "³" if three else "²"
    chips = [info["family"], f'{grids[0]}{dim} to {grids[-1]}{dim}', f'to t = {config["t_end"]:g}', f"{setups} set-ups", f"{total} runs",
             "all finished" if not stopped else f"{stopped} stopped by {'its' if stopped == 1 else 'their'} solver"]
    if len(config.get("seeds", [42])) > 1:chips.insert(3, f'{len(config["seeds"])} seeds')
    return f'''
<div class="page-heading"><div><div class="eyebrow"><a href="tests.html">Tests</a><span>{e(info["family"])}</span><span class="badge">{e(test)}</span></div>
<h1>{e(TITLES[test])}</h1><p>{e(info["focus"])}</p></div>
<a class="method-link" href="tests.html">All tests <span>↗</span></a></div>
<div class="chips">{"".join(f'<span class="chip">{e(c)}</span>' for c in chips)}</div>
{movie_card(test, movie)}
<article class="card"><h3>Set-up</h3><p>{e(notes)}</p><p class="small">After: {e(info["source"])}. The same initial state goes to every code.</p></article>
<article class="card"><h3>Who finished it</h3><p>Each set-up on each grid of the sweep. ✓ the run reached the end time; otherwise where its own solver stopped it
(a result, not a failed job); · not run.</p>{table}</article>
<div class="test-turn"><a href="{page_of(prev)}">← {e(TITLES[prev])}</a><a href="{page_of(nxt)}">{e(TITLES[nxt])} →</a></div>'''


def index_page(tests, movies, families):
    out = [f'''
<div class="page-heading"><div><div class="eyebrow"><span>{PAGE["number"]} · Test problems</span><span class="badge">{len(tests)} tests</span></div>
<h1>Every test problem</h1><p>One page a test: what it is for, a movie of it, and which code finished it on which grid. The colours of the movies
change from one family of tests to the next.</p></div></div>''']
    for family in families:
        mine = [t for t in tests if tests[t]["family"] == family]
        cards = "".join(f'<a class="test-card" href="{page_of(t)}"><strong>{e(TITLES[t])}</strong><code>{e(t)}</code><span>{e(tests[t]["focus"].split(". ")[0].rstrip(".") + ".")}</span>'
                        f'<em>{"movie" if t in movies else "no movie yet"}</em></a>' for t in mine)
        if family == "Turbulence":
            cards += ('<a class="test-card" href="turbulence.html"><strong>Driven turbulence</strong><code>sherlock_turb</code>'
                      '<span>Stirred to a steady state at Mach 3, 10, 30 and 50.</span><em>its own page</em></a>')
        out.append(f'<section class="test-family"><h2>{e(family)}</h2><div class="test-grid">{cards}</div></section>')
    return "".join(out)


def nav_html(tests, current):
    return "".join(f'<a href="{e(t["page"])}"' + (' aria-current="page"' if t["id"] == current else "") + f'><span>{t["number"]}</span>{e(t["nav"])}</a>' for t in tests)


def render(main_html, registry, title):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="A test problem of the MHD comparison: set-up, movie, and which codes finished it.">
<title>{e(title)} · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="tests"><a class="skip-link" href="#main">Skip to the content</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> TEST PROBLEMS</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav_html(registry, PAGE["id"])}</nav>
<main id="main">{main_html}</main>
<footer><span>Written by scripts/build_test_pages.py from the sweep's configs, status and movie runs.</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main():
    info = json.loads((SITE/"gpu_tests.json").read_text());tests = {t: v for t, v in info["tests"].items() if t in TITLES}
    missing = [t for t in TITLES if t not in tests]
    if missing:sys.exit(f"no description in gpu_tests.json for {missing}")
    families = [f for f in info["families"] if any(v["family"] == f for v in tests.values())]
    order = [t for f in families for t in tests if tests[t]["family"] == f]
    manifest = SITE/"test_movies.json";movies = json.loads(manifest.read_text()) if manifest.is_file() else {}
    movies = {t: m for t, m in movies.items() if all((SITE/p).is_file() for p in m["fields"].values())}
    sweep = sd.Sweep()
    registry = json.loads((SITE/"tests.json").read_text());page = dict(PAGE, status=f"{len(tests)} tests")
    registry = [page if t["id"] == PAGE["id"] else t for t in registry] if any(t["id"] == PAGE["id"] for t in registry) else registry + [page]
    (SITE/"tests.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    for old in SITE.glob("test-*.html"):old.unlink()
    (SITE/PAGE["page"]).write_text(render(index_page(tests, movies, families), registry, "Tests"))
    for test in order:
        (SITE/page_of(test)).write_text(render(test_page(sweep, test, tests[test], movies.get(test), order), registry, TITLES[test]))
    link = f'<a href="{PAGE["page"]}"><span>{PAGE["number"]}</span>{PAGE["nav"]}</a>'
    for t in registry:                       # the navigation of the pages already built
        path = SITE/t["page"]
        if t["id"] == PAGE["id"] or not path.is_file():continue
        text = path.read_text()
        if f'href="{PAGE["page"]}"' not in text and "</nav>" in text:path.write_text(text.replace("</nav>", link + "</nav>", 1))
    print(f"Test pages: {len(order)} tests, {len(movies)} with a movie ({', '.join(t for t in order if t not in movies) or 'none'} without); {SITE/PAGE['page']}")


if __name__ == "__main__":
    main()
