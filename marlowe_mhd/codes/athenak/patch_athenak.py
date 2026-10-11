"""Prepare an AthenaK checkout for the comparison suite (idempotent).

1. Copies abel_suite.cpp and abel_problems.hpp into src/pgen/.
2. Makes timesteps land exactly on output times, as our GPU driver, RAMSES and the
   Athena++ segment runner do, so every code is compared at identical physical times.
   AthenaK otherwise writes the first state *after* each output time. Only the argument of
   the two Mesh::NewTimeStep(tlim) calls in src/driver/driver.cpp changes; the physics,
   CFL rule and the clipping logic in Mesh::NewTimeStep are untouched.

usage: python3 patch_athenak.py ATHENAK_DIR
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "// abel_suite: land on output times"
HELPER = f'''{MARK}
static Real AbelNextStop(Mesh *pmesh, Outputs *pout, Real tlim) {{
  Real stop = tlim;
  for (auto &out : pout->pout_list) {{
    Real dt = out->out_params.dt;
    if (dt <= 0.0) continue;
    Real next = (out->out_params.last_time < 0.0 ? pmesh->time : out->out_params.last_time) + dt;
    while (next <= pmesh->time) next += dt;
    stop = std::min(stop, next);
  }}
  return stop;
}}

'''


def main(root):
    root = Path(root)
    for name in ("abel_suite.cpp",):
        shutil.copy(HERE / name, root / "src/pgen" / name)
    shutil.copy(HERE.parent / "common/abel_problems.hpp", root / "src/pgen/abel_problems.hpp")
    driver = root / "src/driver/driver.cpp"
    text = driver.read_text()
    if MARK in text:
        print("AthenaK already patched");return
    calls = text.count("pmesh->NewTimeStep(tlim);")
    anchor = "void Driver::Initialize("
    if calls != 2 or text.count(anchor) != 1:
        sys.exit(f"Unexpected AthenaK driver layout ({calls} NewTimeStep calls); refusing to patch")
    text = text.replace("pmesh->NewTimeStep(tlim);", "pmesh->NewTimeStep(AbelNextStop(pmesh, pout, tlim));")
    text = text.replace(anchor, HELPER + anchor, 1)
    if "#include <algorithm>" not in text:
        text = "#include <algorithm>\n" + text
    driver.write_text(text)
    print("Patched", driver)


if __name__ == "__main__":
    if len(sys.argv) != 2:sys.exit(__doc__)
    main(sys.argv[1])
