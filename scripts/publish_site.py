"""Copy the Sweep and Papers pages of the progress site into a clone of the personal website (solaceonx.github.io).

    python3 scripts/publish_site.py SITE_CLONE_DIR [--link]

Writes SITE_CLONE_DIR/mhd/ (paper-figures.html as index.html, sweep-plan.html, style.css and the figure images). The
pages are unlisted: nothing on the website links to them and they ask search engines not to index them. --link adds
links to them in the MHD entry of research.html (lyxia had them removed on 2026-10-04: not for general readers yet). Only these two pages are published: the other pages of the progress
site read local movies and data. Links that point into the local repository are removed, and the navigation is
reduced to the two published pages plus a link back to the research page. Commit and push are left to the caller.
"""
import re
import shutil
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]/"progress"
PAGES = {"paper-figures.html": "index.html", "sweep-plan.html": "sweep-plan.html"}
MARK = "<!-- mhd-sweep-links -->"


def rewrite(html, name):
    # navigation: the two published pages and the way back
    nav = ('<nav class="test-nav" aria-label="Pages"><a href="../research.html"><span>←</span>Research</a>'
           f'<a href="index.html"{" aria-current=\"page\"" if name == "paper-figures.html" else ""}><span>01</span>Paper figures</a>'
           f'<a href="sweep-plan.html"{" aria-current=\"page\"" if name == "sweep-plan.html" else ""}><span>02</span>Sweep status</a></nav>')
    html = re.sub(r'<nav class="test-nav".*?</nav>', nav, html, count=1, flags=re.S)
    html = html.replace('href="paper-figures.html', 'href="index.html')
    html = html.replace('<a class="brand" href="index.html">', '<a class="brand" href="../research.html">')
    # links into the local repository (scripts, run instructions) do not exist on the website
    html = re.sub(r'<a class="method-link" href="\.\./[^"]*">.*?</a>', '', html, flags=re.S)
    html = re.sub(r'<a href="\.\./[^"]*">(.*?)</a>', r'\1', html, flags=re.S)
    html = html.replace("LOCAL RESEARCH NOTEBOOK", "RESEARCH NOTEBOOK")
    html = html.replace('<meta name="color-scheme"', '<meta name="robots" content="noindex, nofollow"><meta name="color-scheme"', 1)
    # instructions that only make sense on the Mac (saving a local copy of a paper's figure, rebuilding the page)
    html = re.sub(r" To see it here next to ours, save your own copy of it as <code>.*?</code> and rebuild with <code>--page-only</code>\.", "", html)
    html = re.sub(r"(<p><strong>The papers' own figures\.</strong>.*?)( The other\s+papers carry no such licence:.*?beside ours\.)</p>",
                  r"\1 The other papers carry no reuse licence; each figure links to its place in the paper, and the numbers a paper prints "
                  r"(fitted curves, table values) are drawn on our axes.</p>", html, flags=re.S)
    return html


def main(argv):
    link = "--link" in argv;argv = [a for a in argv if a != "--link"]
    if len(argv) != 1:sys.exit(__doc__)
    clone = Path(argv[0]).resolve();out = clone/"mhd"
    if not (clone/"research.html").is_file():sys.exit(f"{clone} does not look like the website clone (no research.html)")
    if out.exists():shutil.rmtree(out)
    (out/"media"/"papers"/"original").mkdir(parents=True)
    for src, dst in PAGES.items():
        html = rewrite((SITE/src).read_text(), src)
        leftover = re.findall(r'(?:href|src)="(\.\./[^"]*|[a-z-]+\.html)"', html)
        bad = [x for x in leftover if x not in ("index.html", "sweep-plan.html", "../research.html")]
        if bad:sys.exit(f"{src}: links that would break on the website: {sorted(set(bad))}")
        (out/dst).write_text(html)
    shutil.copy(SITE/"style.css", out/"style.css")
    used = set(re.findall(r'(?:href|src)="(media/papers/[^"]+\.png)"', (out/"index.html").read_text()))
    for rel in sorted(used):shutil.copy(SITE/rel, out/rel)
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
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Wrote {out} ({len(used)} images, {size/1e6:.1f} MB) " + ("and the links in research.html" if link else "(unlisted: no link added)"))


if __name__ == "__main__":
    main(sys.argv[1:])
