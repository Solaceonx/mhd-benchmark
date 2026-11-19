"""The home page of the site: what the project is, how much has been run, and every page in its group.
    python3 scripts/build_home.py
Counts every simulation from the records on this Mac (sweep_data's tables, ramses2_results/, the MHD-ALE records, turb_results/,
nested_results/), makes a small JPEG of each page's lead picture (progress/media/home/) and writes progress/home.html, which
scripts/publish_site.py publishes as index.html."""
import glob
import html
import re
import subprocess
import sys
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
THUMBS = SITE/"media"/"home"
sys.path.insert(0, str(ROOT/"scripts"))
import plot_style as ps  # noqa: E402
from publish_site import PAGES, GROUPS, nav_html  # noqa: E402

e = html.escape
# label prefix of a run -> the code (13 codes; Abel et al.'s GLM kernels and CT prototype count as one)
CODES = (("glm", "Abel et al. GPU kernels", "GPU"), ("ct", "Abel et al. GPU kernels", "GPU"), ("athenak", "AthenaK", "GPU"),
         ("athenapk", "AthenaPK", "GPU"), ("cholla", "Cholla", "GPU"), ("pluto", "gPLUTO", "GPU"), ("ramses2", "RAMSES II", "GPU"),
         ("ramses", "RAMSES", "CPU"), ("athenapp", "Athena++", "CPU"), ("arepo", "AREPO", "CPU"), ("gizmo", "GIZMO", "CPU"),
         ("enzo", "Enzo", "CPU"), ("phantom", "Phantom", "CPU"), ("mhdale", "MHD-ALE", "CPU"), ("castro", "Castro", "GPU"), ("nyx", "Nyx", "GPU"))
# page (local name) -> (one line, lead picture or None); the picture is shrunk to a JPEG thumbnail
CARDS = {
    "benchmark.html": ("Which code is good at what: the time each set-up needs to reach a given accuracy, and its order of accuracy, test by test.",
                       "media/ours/work_precision.png"),
    "paper-figures.html": ("The figures of the comparison papers (Tomida et al. 2026, Kritsuk et al. 2011, the GEM challenge, Gardiner & Stone) "
                           "redrawn from our runs, with every code.", "media/papers/tomida-1-gpu.png"),
    "our-figures.html": ("Drafts of our paper's figures, and the tests behind each claim.", "media/ours/accuracy.png"),
    "turbulence.html": ("Driven supersonic turbulence at Mach 3–50 in Abel et al.'s kernels and RAMSES: steady state, spectra, dissipation.",
                        "media/tests/driven_turbulence-density.png"),
    "nested.html": ("Tom Abel's nested-grid code on a GPU: does the finest level of a nest match a uniform grid at its resolution, for much less computing?",
                    "media/nested/deep192-movie.jpg"),
    "tests.html": ("Every test problem: its set-up, movies of every code, and which codes finished it.", "media/tests/orszag_tang-codes-density.mp4"),
    "problems.html": ("How each of the 26 tests is set up: initial state, boundaries, end time and what is measured.", None),
    "solver.html": ("A walk through the central pieces of Abel et al.'s GLM-MHD solver, line by line, with a glossary.", None),
    "sweep-fixes.html": ("Each problem the sweep found in Abel et al.'s kernels, the change on the branch sweep-fixes, and the rerun that shows it fixed.", None),
    "sweep-plan.html": ("Which runs exist and which stopped, and why; the compute used on Sherlock.", None),
    "weekly.html": ("What was done each week, one page a week.", None),
}
NEWS = (   # (date, page, text): what changed, newest first
    ("2026-11-14", "nested.html", "Nested turbulence: nests against uniform grids with the same Alfvén-speed cap, now all run to the end; an 8-seed ergodicity "
                                  "test; a big four-level nest (1024³-equivalent); three movies."),
    ("2026-11-14", "home.html", "The site reorganised: this home page, the pages in groups; the paper figures moved from index.html to papers.html."),
    ("2026-11-14", "benchmark.html", "RAMSES II rerun with the official code (HLL, HLLD and UCT-HLLD, double and single precision): {r2_done} of {r2} runs "
                                     "complete, now on every page in place of the earlier mini-RAMSES runs."),
    ("2026-11-12", "nested.html", "Nested turbulence page added; the Orszag–Tang page and the four first check pages retired."),
    ("queued", "sweep-plan.html", "AthenaK and AthenaPK in single precision, so the GPU codes compare at one precision."),
)


def count():
    """Every simulation on record, by campaign and by code."""
    import sweep_data as sd
    sw = sd.Sweep()
    camp, code, tests = Counter(), Counter(), set()
    later = {name: getattr(sw, name) for name in ("arepo", "gizmo", "fixed", "order", "newcodes", "ramses2")}
    for key, row in sw.rows.items():
        if row.get("state") not in ("done", "stopped") :
            continue
        camp[next((n for n, d in later.items() if key in d), "sweep")] += 1; tests.add(key[0])
        code[next(name for p, name, _ in CODES if key[1].startswith(p))] += 1
    r2 = sorted(glob.glob(str(ROOT/"ramses2_results"/"*-run-N*-s*"/"data"/"*_r1"/"run.toml")))
    r2_done = sum(tomllib.load(open(f, "rb")).get("status") == "complete" for f in r2)
    ale = glob.glob(str(ROOT/"sweep_results"/"*-mhdale-ale-run-N*-s*"))
    camp["mhdale"] = len(ale); code["MHD-ALE"] += len(ale)
    turb = len(glob.glob(str(ROOT/"turb_results"/"**"/"run.toml"), recursive=True))
    nested = len(glob.glob(str(ROOT/"nested_results"/"**"/"run.toml"), recursive=True)) + len(glob.glob(str(ROOT/"nested_results"/"**"/"movie.toml"), recursive=True))
    grids = sorted({k[2] for k, r in sw.rows.items() if r.get("state") in ("done", "stopped") and k[2] <= 512})
    return dict(campaign=camp, code=code, tests=len(tests), bench=sum(camp.values()), turb=turb, nested=nested, r2=len(r2), r2_done=r2_done,
                grids=(grids[0], grids[-1]))


def thumb(src):
    """A 640-pixel JPEG of a page's lead picture (a movie: its last frame)."""
    if not src:
        return None
    f = SITE/src; out = THUMBS/(Path(src).stem + ".jpg")
    if not f.is_file():
        return None
    if not out.is_file() or out.stat().st_mtime < f.stat().st_mtime:
        pre = ["-sseof", "-0.3"] if f.suffix == ".mp4" else []
        subprocess.run([ps.ffmpeg_path(), "-y", "-loglevel", "error", *pre, "-i", str(f), "-frames:v", "1", "-update", "1",
                        "-vf", "scale=640:-2", "-q:v", "4", str(out)], check=True)
    return f"media/home/{out.name}?v={int(out.stat().st_mtime)}"


def build(N):
    number = {src: i for i, (src, _, _) in enumerate(PAGES, 1)}; label = {src: lab for src, _, lab in PAGES}
    total = N["bench"] + N["turb"] + N["nested"]
    gpu = sorted({name for _, name, hw in CODES if hw == "GPU"}); cpu = sorted({name for _, name, hw in CODES if hw == "CPU"})
    tiles = [(f"{total:,}", "simulations on record", f"{N['bench']:,} in the benchmark, {N['turb']} of driven turbulence, {N['nested']} nested"),
             (f"{len(gpu) + len(cpu)}", "codes", f"{len(gpu)} on GPUs, {len(cpu)} on CPUs"),
             (f"{N['tests']}", "test problems", "waves, shocks, field loops, blasts, instabilities, decaying and driven turbulence"),
             (f"{N['grids'][0]}–{N['grids'][1]}", "cells a side", "references at 4096² and 512³"),
             ("H100", "Sherlock (Stanford)", "and 128-core CPU nodes")]
    out = [f'''<div class="page-heading"><div><div class="eyebrow"><span>01 · Home</span><span class="badge">working notes</span></div>
<h1>Benchmarking magnetohydrodynamics codes</h1>
<p>The same magnetohydrodynamics (MHD) test problems run with {len(gpu) + len(cpu)} codes on GPUs and CPUs, on grids from {N["grids"][0]} to {N["grids"][1]}
cells a side, each measured against an exact solution or a much finer reference run, to find which codes and settings give the most accuracy for the
computing they use, problem by problem. It began with Tom Abel's new GPU kernels (Abel et al.) and grew to most public MHD codes. Every number on these
pages is computed from the run records.</p></div></div>
<div class="sweep-tiles">{"".join(f'<div class="tile"><span class="control-label">{e(b)}</span><strong>{e(a)}</strong><small>{e(c)}</small></div>' for a, b, c in tiles)}</div>''']
    groups, cur = [], None
    for src, _, lab in PAGES[1:]:
        if src in GROUPS:
            cur = (GROUPS[src], []); groups.append(cur)
        cur[1].append(src)
    for name, pages in groups:
        cards = []
        for src in pages:
            text, pic = CARDS.get(src, ("", None)); img = thumb(pic)
            cards.append(f'<a class="home-card" href="{src}">' + (f'<img src="{img}" alt="" loading="lazy">' if img else "")
                         + f'<div class="home-body"><h3><span>{number[src]:02d}</span>{e(label[src])}</h3><p>{e(text)}</p></div></a>')
        out.append(f'<section class="home-group"><h2>{e(name)}</h2><div class="home-cards">{"".join(cards)}</div></section>')
    news = "".join(f'<li><time>{e(d)}</time><span>{e(t.format(**N))} <a href="{p}">{e(label.get(p, "Home"))} ↗</a></span></li>' for d, p, t in NEWS)
    col = lambda names: "".join(f'<div><dt>{e(c)}</dt><dd>{N["code"].get(c, 0):,} runs</dd></div>' for c in names)
    rows = f'<div><h4>On GPUs</h4><dl>{col(gpu)}</dl></div><div><h4>On CPUs</h4><dl>{col(cpu)}</dl></div>'
    out.append(f'''<section class="home-group"><h2>Recent</h2><div class="sweep-grid">
<div class="card"><h3>What changed</h3><ul class="home-news">{news}</ul></div>
<div class="card"><h3>The codes</h3><div class="home-codes">{rows}</div><p class="small">Runs of the benchmark on record per code (the driven and nested
turbulence runs not included). Abel et al.'s kernels count their five GLM variants, the CT prototype, the fixed rerun and the order study.</p></div></div></section>''')
    return "".join(out)


def render(main_html):
    nav = nav_html("home.html")
    for src, dst, _ in PAGES:        # local links: the publisher maps them back to the published names
        nav = nav.replace(f'href="{dst}"', f'href="{src}"') if dst != src else nav
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="Benchmarking MHD codes on GPUs and CPUs: what has been run, and every page.">
<title>MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="home"><a class="skip-link" href="#main">Skip to the pages</a>
<header class="masthead"><a class="brand" href="home.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> HOME</div></header>
{nav}
<main id="main">{main_html}</main>
<footer><span>Counted by scripts/build_home.py from the run records on this Mac.</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main():
    THUMBS.mkdir(parents=True, exist_ok=True)
    N = count()
    print({k: (dict(v) if isinstance(v, Counter) else v) for k, v in N.items()})
    (SITE/"home.html").write_text(render(build(N)))
    print("Home page:", SITE/"home.html")


if __name__ == "__main__":
    main()
