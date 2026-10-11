"""Figures of the comparison papers, redrawn from the comprehensive sweep with every code.

    python3 scripts/paper_figures.py [FIGURE_ID ...] [--out DIR] [--list]

Reads sweep_results/ (scripts/sweep_data.py; copied from Sherlock by scripts/sweep_fetch.py) and writes one PNG per
figure to progress/media/papers/ plus progress/paper_figures.json, which scripts/build_paper_figures.py turns into
the "Paper figures" page of the progress site. Each figure follows one figure of a paper: the same quantity, axes
and layout, with our tests, grids and codes; its `differs` text says where our set-up is not the paper's.

A panel whose data is not here says why (the solver stopped, the run is not finished, the file is not fetched), and
the figure's entry lists those parts.

One module per paper: paper_tomida.py (Tomida et al. 2026), paper_kritsuk.py (Kritsuk et al. 2011), paper_gem.py
(Birn et al. 2001), paper_stone.py (Gardiner & Stone 2005, 2008), paper_ugalino.py (Ugalino et al. 2026); paper_arepo.py is the
section on AREPO, which ran after the sweep; shared pieces in paper_common.py.
"""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common  # noqa: E402

MODULES = ("paper_tomida", "paper_kritsuk", "paper_gem", "paper_stone", "paper_ugalino", "paper_arepo")


def registry():
    out = {}
    for name in MODULES:
        try:module = importlib.import_module(name)
        except ModuleNotFoundError as err:
            if err.name != name:raise
            continue
        out.update(module.figures())
    return out


def main(argv):
    out = paper_common.OUT
    if "--out" in argv:
        i = argv.index("--out");out = Path(argv[i + 1]);argv = argv[:i] + argv[i + 2:]
    figures = registry()
    if "--list" in argv:
        print("\n".join(figures));return None
    todo = argv or list(figures)
    for ident in todo:
        if ident not in figures:sys.exit(f"Unknown figure {ident}; known: {', '.join(figures)}")
    F = paper_common.Figures(out)
    for ident in todo:figures[ident](F)
    if not argv:
        status = F.sw.status or {}
        target = paper_common.SITE/"paper_figures.json" if out == paper_common.OUT else out/"paper_figures.json"
        # Figures the last build drew that are no longer in the registry: remove their images (only files that build listed).
        if target.is_file():
            kept = {e["id"] for e in F.entries}
            for old in json.loads(target.read_text()).get("figures", []):
                stale = out/f'{old["id"]}.png'
                if old["id"] not in kept and old.get("file") and stale.is_file():stale.unlink();print(f"removed {stale.name} (no longer drawn)")
        target.write_text(json.dumps(dict(status_time=status.get("time", ""), figures=F.entries), indent=1))
    return F


if __name__ == "__main__":
    main(sys.argv[1:])
