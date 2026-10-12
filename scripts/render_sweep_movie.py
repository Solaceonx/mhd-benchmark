"""A movie of one test from the sweep's own frames (sweep_results/), one panel per set-up.

  .venv-movie/bin/python scripts/render_sweep_movie.py gem --n 128 --labels glm athenak_best ramses_best arepo_moving

For tests the movie runs of the first campaign did not cover. Uses the renderer of render_suite_movies.py with the
colour maps of the test's family, and writes progress/media/tests/<test>-<field>.mp4 and the test's entry in
progress/test_movies.json. A set-up whose frames are not on this machine is left out and named.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_suite_movies as rsm  # noqa: E402
import paper_common as pc  # noqa: E402
import sweep_data as sd  # noqa: E402

suite, SITE = rsm.suite, rsm.SITE
FIELDS = ("density", "magnetic", "vorticity")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test");parser.add_argument("--n", type=int, default=128);parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=SITE/"media"/"tests");parser.add_argument("--manifest", type=Path, default=SITE/"test_movies.json")
    args = parser.parse_args()
    rsm.FAMILY_OF = {name: t["family"] for name, t in json.loads((SITE/"gpu_tests.json").read_text())["tests"].items()}
    sweep = sd.Sweep();frames = range(sweep.frames(args.test));times = [float(t) for t in sweep.times(args.test)]
    states = {}
    for label in args.labels:
        got = [sweep.field(args.test, label, args.n, f) for f in frames]
        if sum(u is not None for u in got) < len(got)//2:print(f"{label}: {sum(u is not None for u in got)} of {len(got)} frames here, left out");continue
        states[label] = [None if u is None else u.transpose(1, 0, 2).astype(float) for u in got]     # as the renderer wants them: [y, x, field]
    if not states:sys.exit("no set-up has its frames on this machine")
    labels = {l: pc.label(l) for l in states};args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {};entry = manifest[args.test] = {"fields": {}}
    fps = max(6, round(len(times)/5))
    for field in FIELDS:
        planes = {l: [None if u is None else suite.field_of(u, field) for u in states[l]] for l in states}
        path = args.out/f"{args.test}-{field}.mp4"
        rsm.render(path, args.test, field, times, planes, labels, args.n, fps, False, False)
        entry["fields"][field] = os.path.relpath(path, SITE)
    entry.update(frames=len(times), tEnd=times[-1], grid=args.n, truth="none: the sweep's runs side by side", panels=[labels[l] for l in states],
                 midplane=False, adaptive=False, source="sweep_results")
    args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"movies {args.test}: {len(times)} outputs at {args.n}, panels {', '.join(states)}")


if __name__ == "__main__":
    main()
