"""Prepare an AthenaPK checkout (with its Parthenon submodule) for the comparison suite. Idempotent.

    python3 patch_athenapk.py ATHENAPK_DIR ABEL_PROBLEMS_HPP

1. Copies abel_suite.cpp and the suite's abel_problems.hpp into src/pgen/ and registers the problem generator
   (problem_id = abel_suite) in src/pgen/CMakeLists.txt, src/pgen/pgen.hpp and src/main.cpp.
2. Makes time steps land on output times, as every other code of the suite does (AthenaK's patch_athenak.py, our
   GPU driver, RAMSES, Athena++): Parthenon only shortens the last step to end on tlim and writes an output at the
   first step past each output time. EvolutionDriver::SetGlobalTimeStep now also shortens a step that would pass the
   next time an output is due, and the time is set to that output time exactly once the step is taken. The CFL rule,
   the solver and the outputs themselves are untouched.
Every edit is anchored on text of the pinned commits and fails loudly if that text is not there exactly once.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "abel_suite"


def edit(path, old, new, done):
    text = path.read_text()
    if done in text:return
    if text.count(old) != 1:sys.exit(f"patch_athenapk.py: {path.name}: expected this text once, found it {text.count(old)} times:\n{old}")
    path.write_text(text.replace(old, new))


def main(root, header):
    root = Path(root);pgen = root/"src/pgen";parthenon = root/"external/parthenon/src"
    shutil.copy(HERE/"abel_suite.cpp", pgen/"abel_suite.cpp")
    shutil.copy(header, pgen/"abel_problems.hpp")
    edit(pgen/"CMakeLists.txt", "    advection.cpp\n", "    abel_suite.cpp\n    advection.cpp\n", "abel_suite.cpp")
    edit(pgen/"pgen.hpp", "namespace linear_wave {\n",
         "namespace abel_suite {\nusing namespace parthenon::driver::prelude;\n"
         "void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);\n} // namespace abel_suite\n\n"
         "namespace linear_wave {\n", "namespace abel_suite")
    edit(root/"src/main.cpp", '  if (problem == "linear_wave") {\n',
         '  if (problem == "abel_suite") {\n    pman.app_input->ProblemGenerator = abel_suite::ProblemGenerator;\n'
         '  } else if (problem == "linear_wave") {\n', 'problem == "abel_suite"')

    # Parthenon: the next output time, and steps that land on it
    edit(parthenon/"outputs/outputs.hpp",
         "  MakeOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm = nullptr,\n"
         "              SignalHandler::OutputSignal signal = SignalHandler::OutputSignal::none);\n",
         "  MakeOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm = nullptr,\n"
         "              SignalHandler::OutputSignal signal = SignalHandler::OutputSignal::none);\n"
         "  // abel_suite: the first time after `now` at which an output at a time cadence is due\n"
         "  Real NextOutputTime(Real now) const;\n", "NextOutputTime")
    text = (parthenon/"outputs/outputs.cpp").read_text()
    if "NextOutputTime" not in text:
        tail = "\n} // namespace parthenon\n"
        if not text.rstrip().endswith("} // namespace parthenon"):sys.exit("patch_athenapk.py: outputs.cpp does not end in its namespace")
        text = text.rstrip()[:-len("} // namespace parthenon")] + (
            "// abel_suite: the first time after `now` at which an output at a time cadence (dt > 0) is due\n"
            "Real Outputs::NextOutputTime(Real now) const {\n"
            "  Real next = std::numeric_limits<Real>::max();\n"
            "  for (auto &ptype : output_types_) {\n"
            "    const Real dt = ptype->output_params.dt;\n"
            "    if (dt <= 0.0) continue;\n"
            "    Real t = ptype->output_params.next_time;\n"
            "    while (t <= now + 1e-12 * std::max(Real(1), std::abs(now))) t += dt;\n"
            "    next = std::min(next, t);\n"
            "  }\n"
            "  return next;\n"
            "}\n" + tail)
        (parthenon/"outputs/outputs.cpp").write_text(text)
    edit(parthenon/"outputs/outputs.cpp", "#include <algorithm>\n", "#include <algorithm>\n#include <cmath>\n#include <limits>\n", "#include <limits>\n#include <cmath>")
    driver = parthenon/"driver/driver.cpp"
    edit(driver, '#include "driver/driver.hpp"\n',
         '#include "driver/driver.hpp"\n\n#include <cmath>\n// abel_suite: the output time the current step was shortened to land on (< 0: none)\n'
         'static parthenon::Real abel_land_time = -1.0;\n', "abel_land_time = -1.0;")
    edit(driver, "      tm.ncycle++;\n      tm.time += tm.dt;\n",
         "      tm.ncycle++;\n      tm.time += tm.dt;\n"
         "      // abel_suite: a step shortened to land on an output time ends on it exactly\n"
         "      if (abel_land_time >= 0.0 &&\n"
         "          std::abs(tm.time - abel_land_time) <= 1e-12 * std::max(Real(1), std::abs(abel_land_time))) {\n"
         "        tm.time = abel_land_time;\n"
         "      }\n", "a step shortened to land on an output time")
    edit(driver, "  if (tm.time < tm.tlim && (tm.tlim - tm.time) < tm.dt) {\n    tm.dt = tm.tlim - tm.time;\n  }\n}\n",
         "  if (tm.time < tm.tlim && (tm.tlim - tm.time) < tm.dt) {\n    tm.dt = tm.tlim - tm.time;\n  }\n"
         "  // abel_suite: nor past the next output time\n"
         "  abel_land_time = -1.0;\n"
         "  if (pouts != nullptr) {\n"
         "    const Real t_out = pouts->NextOutputTime(tm.time);\n"
         "    if (t_out > tm.time && (t_out - tm.time) < tm.dt) {\n"
         "      tm.dt = t_out - tm.time;\n"
         "      abel_land_time = t_out;\n"
         "    }\n"
         "  }\n}\n", "nor past the next output time")
    print(f"patched {root}")


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
