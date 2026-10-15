"""Copy the progress site into a clone of the personal website (solaceonx.github.io).

    python3 scripts/publish_site.py SITE_CLONE_DIR [--link]

Writes SITE_CLONE_DIR/mhd/: every page except the GPU suite, whose 38,000 stills are 235 MB (its movies are on the
test pages, test-<name>.html, which are published and reached from tests.html). The Papers
page stays index.html (the address that was given out first) and the Orszag–Tang page becomes orszag-tang.html. The
pages are unlisted: nothing on the website links to them and they ask search engines not to index them. --link adds
links to them in the MHD entry of research.html (lyxia had them removed on 2026-10-04: not for general readers yet).

Files the pages link to in the local repository (../scripts/out/..., method notes) are copied to mhd/files/ and the
links are pointed there. Local how-to text is taken out. notes.html (private notes, encrypted with a password in the
browser) is copied as it is and no page links to it. Commit and push are left to the caller.
"""
import glob
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
PAGES = [   # local name, published name, label in the navigation (in its order)
    ("paper-figures.html", "index.html", "Paper figures"),
    ("sweep-plan.html", "sweep-plan.html", "Sweep status"),
    ("turbulence.html", "turbulence.html", "Turbulence"),
    ("tests.html", "tests.html", "Tests"),
    ("solver.html", "solver.html", "Inside the solver"),
    ("index.html", "orszag-tang.html", "Orszag–Tang"),
    ("hydro-decay.html", "hydro-decay.html", "Hydro decay"),
    ("isothermal-decay.html", "isothermal-decay.html", "Prepared decay"),
    ("acoustic-wave.html", "acoustic-wave.html", "Acoustic check"),
    ("entropy-wave.html", "entropy-wave.html", "Entropy wave"),
]
NAMES = {src: dst for src, dst, _ in PAGES}
NOT_PUBLISHED = ("gpu", "movies")           # directories of progress/media that belong to the GPU-suite page
LOCAL = re.compile(r"\.\./(?!research\.html)[A-Za-z0-9_.\-/${}]+")
MARK = "<!-- mhd-sweep-links -->"
LARGEST = 30e6                              # bytes; a larger file is a mistake in the list, not a page asset


def localise(text, wanted):
    """Point links into the local repository at files/, and note what they need (${...} of a script as *)."""
    def one(m):
        rel = m.group(0)[3:].rstrip(".")
        wanted.add(re.sub(r"\$\{[^}]*\}", "*", rel))
        return "files/" + m.group(0)[3:]
    return LOCAL.sub(one, text)


def rewrite(html, name, wanted):
    html = re.sub(r'href="([a-z-]+\.html)', lambda m: f'href="{NAMES.get(m.group(1), m.group(1))}', html)
    html = localise(html, wanted)
    nav = '<nav class="test-nav" aria-label="Pages"><a href="../research.html"><span>←</span>Research</a>' + "".join(
        f'<a href="{dst}"{" aria-current=\"page\"" if src == name else ""}><span>{i:02d}</span>{label}</a>'
        for i, (src, dst, label) in enumerate(PAGES, 1)) + "</nav>"
    html, n = re.subn(r'<nav class="test-nav".*?</nav>', lambda m: nav, html, count=1, flags=re.S)
    if n != 1:sys.exit(f"{name}: no navigation found")
    html = re.sub(r'<a class="brand" href="[^"]*">', '<a class="brand" href="../research.html">', html, count=1)
    html = html.replace("LOCAL RESEARCH NOTEBOOK", "RESEARCH NOTEBOOK")
    number = next((i for i, (src, _, _) in enumerate(PAGES, 1) if src == name), None)      # the page's number follows the navigation
    if number:html = re.sub(r'(class="eyebrow"><span>)\d\d( · )', rf"\g<1>{number:02d}\2", html, count=1)
    if 'name="robots"' not in html:
        html = html.replace('<meta name="color-scheme"', '<meta name="robots" content="noindex, nofollow"><meta name="color-scheme"', 1)
    if 'name="robots"' not in html:sys.exit(f"{name}: could not add the no-index line")
    # instructions that only make sense on the Mac (saving a local copy of a paper's figure, fetching, rebuilding)
    html = re.sub(r" To see it here next to ours, save your own copy of it as <code>.*?</code> and rebuild with <code>--page-only</code>\.", "", html)
    html = re.sub(r"(<p><strong>The papers' own figures\.</strong>.*?)( The other\s+papers carry no such licence:.*?beside ours\.)</p>",
                  r"\1 The other papers carry no reuse licence; each figure links to its place in the paper, and the numbers a paper prints "
                  r"(fitted curves, table values) are drawn on our axes.</p>", html, flags=re.S)
    html = re.sub(r'<div class="card paper-state"><h3>Where the data comes from</h3>.*?</div>(?=</main>)', "", html, flags=re.S)
    return html


def copy_wanted(wanted, out):
    """Copy what the pages link to in the repository to out/files/. Returns the patterns that matched nothing."""
    missing = []
    for pattern in sorted(wanted):
        found = [Path(p) for p in glob.glob(str(ROOT/pattern))]
        files = [f for p in found for f in ([p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file()))]
        files = [f for f in files if f.name != ".DS_Store"]
        if not files:missing.append(pattern)
        for f in files:
            if f.stat().st_size > LARGEST:sys.exit(f"{f}: {f.stat().st_size/1e6:.0f} MB is too large to publish")
            dest = out/"files"/f.relative_to(ROOT);dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(f, dest)
    return missing


def copy_media(out):
    for f in sorted((SITE/"media").rglob("*")):
        rel = f.relative_to(SITE/"media")
        if not f.is_file() or f.name == ".DS_Store" or rel.parts[0] in NOT_PUBLISHED:continue
        # of other papers' figures only the CC BY ones (Tomida et al. 2026) may be published
        if rel.parts[:2] == ("papers", "original") and not f.name.startswith("tomida2026_"):
            print(f"not published (no reuse licence): {rel}");continue
        dest = out/"media"/rel;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(f, dest)


def check(out):
    """Every address a published page names must exist; nothing may point at the local repository or a page left out."""
    bad = []
    for f in [*out.glob("*.html"), out/"app.js", out/"data.js"]:
        if f.name == 'notes.html':continue  # Copy the private notebook without reading its contents.
        text = f.read_text()
        text = re.sub(r'''https?://[^\s"'<>]+''', '', text)  # External pages are not local assets.
        bad += [f"{f.name}: {x}" for x in set(LOCAL.findall(text))]
        pages = {*NAMES.values(), "research.html", *(q.name for q in out.glob("test-*.html"))}
        if f.name != "data.js":bad += [f"{f.name}: {x}" for x in set(re.findall(r'[a-z0-9-]+\.html(?![A-Za-z])', text)) if x not in pages]
        for rel in set(re.findall(r'(?:files|media)/[A-Za-z0-9_.\-/]+\.[A-Za-z0-9]+', text)):
            if not (out/rel).is_file():bad.append(f"{f.name}: {rel} does not exist")
    if bad:sys.exit("links that would break on the website:\n  " + "\n  ".join(sorted(bad)))


def main(argv):
    link = "--link" in argv;argv = [a for a in argv if a != "--link"]
    if len(argv) != 1:sys.exit(__doc__)
    clone = Path(argv[0]).resolve();out = clone/"mhd"
    if not (clone/"research.html").is_file():sys.exit(f"{clone} does not look like the website clone (no research.html)")
    if out.exists():shutil.rmtree(out)
    out.mkdir();wanted = set()
    for src, dst, _ in PAGES:(out/dst).write_text(rewrite((SITE/src).read_text(), src, wanted))
    for page in sorted(SITE.glob("test-*.html")):(out/page.name).write_text(rewrite(page.read_text(), page.name, wanted))      # one a test, reached from tests.html
    # the script and data of the interactive pages, without the GPU suite
    app = (SITE/"app.js").read_text();gpu_link = "+links([['GPU test suite','gpu-suite.html']])"
    if app.count(gpu_link) != 1:sys.exit("app.js: the link to the GPU suite has changed; adjust publish_site.py")
    (out/"app.js").write_text(localise(app.replace(gpu_link, ""), wanted))
    raw = (SITE/"data.js").read_text();head = raw[:raw.index("{")]
    data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    del data["gpu"];data["tests"] = [t for t in data["tests"] if t["kind"] != "gpu"]
    order = {dst: f"{i:02d}" for i, (_, dst, _) in enumerate(PAGES, 1)}
    for t in data["tests"]:t["page"] = NAMES[t["page"]];t["number"] = order[t["page"]]
    (out/"data.js").write_text(localise(head + json.dumps(data, ensure_ascii=False) + ";\n", wanted))
    for name in ("style.css", "sync-videos.js", "solver-walkthrough.css", "solver-walkthrough.js"):shutil.copy(SITE/name, out/name)
    shutil.copy(SITE/"notes.html", out/"notes.html")       # private notes: encrypted in the browser, not in the navigation
    copy_media(out)
    missing = copy_wanted(wanted, out)
    if missing:print("linked from the pages but not found here (those links will be dead):\n  " + "\n  ".join(missing))
    check(out)
    # research.html: links in the MHD project entry (once)
    research = clone/"research.html";text = research.read_text()
    links = (f'{MARK}<p>Comparison sweep with every code: <a href="mhd/index.html">paper figures redrawn</a> · '
             '<a href="mhd/sweep-plan.html">run status</a>.</p>')
    if not link:pass
    elif MARK in text:text = re.sub(re.escape(MARK) + r"<p>.*?</p>", links, text, flags=re.S)
    else:
        anchor = "benchmarking them on a variety of simulations.</p>"
        if anchor not in text:sys.exit("research.html: the MHD entry's text has changed; add the links by hand")
        text = text.replace(anchor, anchor + "\n                " + links, 1)
    research.write_text(text)
    files = [p for p in out.rglob("*") if p.is_file()];size = sum(p.stat().st_size for p in files)
    print(f"Wrote {out} ({len(list(out.glob('*.html')))} pages, {len(files)} files, {size/1e6:.1f} MB) "
          + ("and the links in research.html" if link else "(unlisted: no link added)"))


if __name__ == "__main__":
    main(sys.argv[1:])
