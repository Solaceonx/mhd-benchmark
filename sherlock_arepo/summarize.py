"""One line per AREPO run under the results directory: python3 summarize.py RESULTS_DIR

Status, grid, how far it got, steps, seconds, and at the last output the kinetic and magnetic energy (the sweep's
diagnostics.csv) and AREPO's own divergence measure (arepo_divergence.csv: rms(div B) x cell size / rms(B)).
"""
import sys
import tomllib
from pathlib import Path


def last_row(path):
    lines = path.read_text().splitlines() if path.is_file() else []
    return dict(zip(lines[0].split(","), lines[-1].split(","))) if len(lines) > 1 else {}


def main(root):
    print(f"{'run':46s} {'status':9s} {'t reached':>10s} {'steps':>7s} {'seconds':>8s} {'kinetic':>11s} {'magnetic':>11s} {'div (AREPO)':>11s}")
    for record in sorted(Path(root).glob("*-arepo-*-run-N*-s*/data/arepo_*_r1/run.toml"), key=lambda p: (p.parts[-4].rsplit("-N", 1)[0], int(p.parts[-4].rsplit("-N", 1)[1].split("-")[0]))):
        m = tomllib.loads(record.read_text());d = last_row(record.parent/"diagnostics.csv");v = last_row(record.parent/"arepo_divergence.csv")
        num = lambda x: f"{float(x):11.4e}" if x not in (None, "") else f"{'-':>11s}"
        print(f"{record.parts[-4]:46s} {m.get('status', '?'):9s} {m.get('time_reached', 0):10.4g} {m.get('steps', 0):7d} {m.get('solve_seconds', 0):8.1f} "
              f"{num(d.get('kinetic'))} {num(d.get('magnetic'))} {num(v.get('normalized_div_rms'))}"
              + ("" if m.get("status") == "complete" else "   " + ((m.get("error") or "").splitlines() or [""])[-1][:150]))


if __name__ == "__main__":
    if len(sys.argv) != 2:sys.exit(__doc__)
    main(sys.argv[1])
