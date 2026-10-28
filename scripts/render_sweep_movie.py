"""A movie of one test from the sweep's own frames (sweep_results/), one panel per set-up.

  .venv-movie/bin/python scripts/render_sweep_movie.py gem --n 128 --labels glm athenak_best ramses_best arepo_moving
  .venv-movie/bin/python scripts/render_sweep_movie.py alfven rotor ... --tag codes --rows 2 \
      --labels glm ct athenak_best ramses_best arepo_moving gizmo_mfm

For tests the movie runs of the first campaign did not cover. Uses the renderer of render_suite_movies.py with the
colour maps of the test's family, and writes progress/media/tests/<test>-<field>.mp4 and the test's entry in
progress/test_movies.json. A set-up whose frames are not on this machine is left out and named. With --tag codes:
progress/media/tests/<test>-codes-<field>.mp4 and progress/test_movies_codes.json, a second movie on the test page
(the codes side by side, in a grid of --rows rows).
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
NAMES = dict(rsm.LABELS, gizmo_mfm="GIZMO meshless finite mass")      # panel titles; the others from paper_common


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="+");parser.add_argument("--n", type=int, default=128);parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--tag", default="", help="a second movie of the test: <test>-<tag>-<field>.mp4, listed in test_movies_<tag>.json")
    parser.add_argument("--rows", type=int, default=1);parser.add_argument("--out", type=Path, default=SITE/"media"/"tests")
    parser.add_argument("--manifest", type=Path, default=None, help="default: progress/test_movies.json, or test_movies_<tag>.json")
    args = parser.parse_args()
    manifest_path = args.manifest or SITE/(f"test_movies_{args.tag}.json" if args.tag else "test_movies.json")
    rsm.FAMILY_OF = {name: t["family"] for name, t in json.loads((SITE/"gpu_tests.json").read_text())["tests"].items()}
    sweep = sd.Sweep();args.out.mkdir(parents=True, exist_ok=True)
    for test in args.tests:
        frames = range(sweep.frames(test));times = [float(t) for t in sweep.times(test)]
        states = {}
        for label in args.labels:
            got = [sweep.field(test, label, args.n, f) for f in frames]
            if sum(u is not None for u in got) < len(got)//2:print(f"{test} {label}: {sum(u is not None for u in got)} of {len(got)} frames here, left out");continue
            states[label] = [None if u is None else u.transpose(1, 0, 2).astype(float) for u in got]     # as the renderer wants them: [y, x, field]
        if not states:print(f"{test}: no set-up has its frames on this machine");continue
        labels = {l: NAMES.get(l) or pc.label(l) for l in states}
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {};entry = {"fields": {}}
        fps = max(6, round(len(times)/5))
        for field in FIELDS:
            planes = {l: [None if u is None else suite.field_of(u, field) for u in states[l]] for l in states}
            path = args.out/(f"{test}-{args.tag}-{field}.mp4" if args.tag else f"{test}-{field}.mp4")
            rsm.render(path, test, field, times, planes, labels, args.n, fps, False, False, rows=args.rows)
            entry["fields"][field] = os.path.relpath(path, SITE)
        entry.update(frames=len(times), tEnd=times[-1], grid=args.n, truth="none: the sweep's runs side by side", panels=[labels[l] for l in states],
                     midplane=False, adaptive=False, source="sweep_results")
        manifest[test] = entry          # read again for each test: the file is written after every movie
        manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        print(f"movies {test}: {len(times)} outputs at {args.n}, panels {', '.join(states)}", flush=True)


if __name__ == "__main__":
    main()
