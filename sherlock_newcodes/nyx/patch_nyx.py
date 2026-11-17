"""Prepare a Nyx tree for the comparison suite (ours). Idempotent.    python3 patch_nyx.py NYX_DIR ABEL_PROBLEMS_HPP

1. Copies the problem directory abel_suite/ (Prob.cpp, the spec reader, GNUmakefile) and the suite's abel_problems.hpp
   into Exec/AbelSuite/.
2. Makes time steps land on output times, as every other code of the suite does (Castro: castro.plot_per_is_exact):
   Nyx::computeNewDt and computeInitialDt shorten a step that would pass the next multiple of amr.plot_per, after their
   own limit by stop_time. AMReX then writes the plotfile at that step (Amr::writePlotNow allows for round-off). The
   CFL rule and the solver are untouched.
Every edit is anchored on text of the pinned commit and fails loudly if that text is not there exactly the expected
number of times.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "abel: land on output times"
OLD = """    if (stop_time >= 0.0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }
"""
OLD_INITIAL = """    if (stop_time >= 0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }
"""
ADD = """    {   // abel: land on output times (the comparison suite), as castro.plot_per_is_exact does
        const Real plot_per = parent->plotPer();
        if (plot_per > 0.0) {
            const Real next = (std::floor((cur_time + eps) / plot_per) + 1.0) * plot_per;
            if (cur_time + dt_0 > next - eps) { dt_0 = next - cur_time; }
        }
    }
"""


def main(root, header):
    root = Path(root);ex = root/"Exec"/"AbelSuite"
    if ex.exists():shutil.rmtree(ex)
    shutil.copytree(HERE/"abel_suite", ex);shutil.copy(header, ex/"abel_problems.hpp")
    f = root/"Source"/"Driver"/"Nyx.cpp";t = f.read_text()
    if MARK not in t:
        if t.count(OLD) != 1 or t.count(OLD_INITIAL) != 1:
            sys.exit(f"patch_nyx.py: Nyx.cpp: the stop_time limits were not found once each ({t.count(OLD)}, {t.count(OLD_INITIAL)})")
        t = t.replace(OLD, OLD + ADD).replace(OLD_INITIAL, OLD_INITIAL + ADD)
        f.write_text(t)
    print(f"patched {f} and copied the problem to {ex}")


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
