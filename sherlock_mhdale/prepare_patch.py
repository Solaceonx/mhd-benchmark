"""Copy the MHD-ALE source and add the comparison suite to it.

    python3 prepare_patch.py SOURCE_DIR BUILD_DIR ABEL_PROBLEMS_HPP

SOURCE_DIR is the checkout of T. Abel's fork of MHD-ALE at the pinned commit (never changed); BUILD_DIR receives its C++ files with the
edits below, our testcase_Abel.cpp and abel_dump.cpp/.hpp, and the suite's abel_problems.hpp. Every edit is anchored
on text of the pinned commit and fails loudly if that text is not there once.

Edits (all marked "abel:" in the patched files):
  Problemdata.hpp/.cpp  problem number 100 = the suite's tests (GetProblemAbel in testcase_Abel.cpp)
  makefile              the two new objects
  MHD.cpp               1. the state is written on the suite's grid at t = 0 and at every time of the plot-time file
                        2. the step before an output time is shortened to end on it (the code as published writes
                           at the first step past the time), and the step size from before is restored afterwards
                        3. ABEL_ONLY_INITIAL: stop after writing the initial state; ABEL_DUMP_LAST: also write the
                           last state of a run cut short by -ms
Nothing in the solver is changed.
"""
import shutil
import sys
from pathlib import Path

COMMIT = "2e900cd42d3b9041cd87e7e36e1334f239798438"       # yipihey/MHD-ALE (T. Abel): x-periodic ALE, mean-field remap


def edit(text, old, new, name):
    if text.count(old) != 1:
        sys.exit(f"prepare_patch.py: {name}: expected this text once, found it {text.count(old)} times:\n{old}")
    return text.replace(old, new)


def main(source, build, header):
    source, build, here = Path(source), Path(build), Path(__file__).resolve().parent
    build.mkdir(parents=True, exist_ok=True)
    for f in list(source.glob("*.cpp")) + list(source.glob("*.hpp")) + [source/"makefile"]:
        shutil.copy2(f, build/f.name)
    for f in ("testcase_Abel.cpp", "abel_dump.cpp", "abel_dump.hpp"):
        shutil.copy2(here/f, build/f)
    shutil.copy2(header, build/"abel_problems.hpp")

    p = build/"Problemdata.hpp";t = p.read_text()
    t = edit(t, "   TAYLOR_GREEN = 0,\n", "   TAYLOR_GREEN = 0,\n   ABEL = 100, // abel: the comparison suite (testcase_Abel.cpp)\n", p.name)
    t = edit(t, "ProblemData *GetProblemData(Testcase p, int dim, real_t Bmag);",
             "ProblemData *GetProblemAbel(int dim); // abel\nProblemData *GetProblemData(Testcase p, int dim, real_t Bmag);", p.name)
    p.write_text(t)

    p = build/"Problemdata.cpp";t = p.read_text()
    t = edit(t, "   default:\n      mfem_error(\"Unknown Problem type\");",
             "   case ABEL: // abel\n      return GetProblemAbel(dim);\n      break;\n   default:\n      mfem_error(\"Unknown Problem type\");", p.name)
    p.write_text(t)

    p = build/"makefile";t = p.read_text()
    t = edit(t, "vecgf_error.o mesh_smoother.o", "vecgf_error.o mesh_smoother.o testcase_Abel.o abel_dump.o", p.name)
    p.write_text(t)

    p = build/"MHD.cpp";t = p.read_text()
    t = edit(t, '#include "MHD_solver.hpp"\n', '#include "MHD_solver.hpp"\n#include "abel_dump.hpp" // abel\n', p.name)
    t = edit(t, "   for (int ti = 1; !last_step; ti++)\n",
             "   // abel: the initial state on the suite's grid; abel_next is the next output time, abel_steps the accepted steps\n"
             "   int abel_next = 0, abel_steps = 0;\n"
             "   AbelDump(0, t, 0, *pmesh, rho_gf, v_gf, e_gf, B_gf, mat_gf, pd->mu, outputdir);\n"
             "   if (getenv(\"ABEL_ONLY_INITIAL\")) { last_step = true; }\n"
             "   for (int ti = 1; !last_step; ti++)\n", p.name)
    t = edit(t, "      if (steps == max_tsteps) { last_step = true; }\n",
             "      if (steps == max_tsteps) { last_step = true; }\n"
             "      // abel: end this step on the next output time if it would pass it\n"
             "      const double abel_dt_full = dt; bool abel_clipped = false;\n"
             "      if (!last_step && abel_next < plot_time.Size() && t + dt >= plot_time[abel_next] - 1e-12)\n"
             "      { dt = plot_time[abel_next] - t; abel_clipped = true; }\n", p.name)
    t = edit(t, "         dt *= gamma1;\n",
             "         dt *= gamma1;\n"
             "         if (abel_clipped) { dt = abel_dt_full * gamma1; } // abel: reduce the step size of before the shortening\n", p.name)
    t = edit(t, "         if (dt_est > 1.25 * dt) { dt *= gamma2; }\n",
             "         if (dt_est > 1.25 * dt) { dt *= gamma2; }\n"
             "         if (abel_clipped) { dt = abel_dt_full; } // abel: back to the step size of before the shortening\n"
             "         abel_steps++;\n", p.name)
    t = edit(t, "      if (last_step || CheckPlot(plot_time, t))\n",
             "      // abel: the state on the suite's grid at the output times\n"
             "      while (abel_next < plot_time.Size() && t >= plot_time[abel_next] - 1e-9)\n"
             "      {\n"
             "         AbelDump(abel_next + 1, t, abel_steps, *pmesh, rho_gf, v_gf, e_gf, B_gf, mat_gf, pd->mu, outputdir);\n"
             "         abel_next++;\n"
             "      }\n"
             "      if (last_step && abel_next < plot_time.Size() && getenv(\"ABEL_DUMP_LAST\"))\n"
             "      { AbelDump(999, t, abel_steps, *pmesh, rho_gf, v_gf, e_gf, B_gf, mat_gf, pd->mu, outputdir); }\n"
             "      if (last_step || CheckPlot(plot_time, t))\n", p.name)
    p.write_text(t)
    (build/"ABEL_PATCH.txt").write_text(f"MHD-ALE {COMMIT} with the comparison suite added by prepare_patch.py\n")
    print(f"patched source in {build}")


if __name__ == "__main__":
    if len(sys.argv) != 4:sys.exit(__doc__)
    main(*sys.argv[1:])
