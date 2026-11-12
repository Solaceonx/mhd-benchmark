"""Prepare an AthenaK checkout for the comparison suite (idempotent).

1. Copies abel_suite.cpp and abel_problems.hpp into src/pgen/.
2. Makes timesteps land exactly on output times, as our GPU driver, RAMSES and the
   Athena++ segment runner do, so every code is compared at identical physical times.
   AthenaK otherwise writes the first state *after* each output time. Only the argument of
   the two Mesh::NewTimeStep(tlim) calls in src/driver/driver.cpp changes; the physics,
   CFL rule and the clipping logic in Mesh::NewTimeStep are untouched.

usage: python3 patch_athenak.py ATHENAK_DIR
"""
import re
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
# Single precision (Athena_SINGLE_PRECISION=ON) does not compile at the pinned commit c5a0d7f: double constants and
# double tables meet Real = float in seven files, all in parts the suite does not use (viscosity's time step, the turbulence
# driver, z4c drift control, tabulated and piecewise EOS, two test problems). These edits only make the types agree;
# with Real = double they change nothing. Checked on the Mac (Kokkos serial, clang): the fp32 build then compiles.
SINGLE = [
    ("src/diffusion/viscosity.cpp", "dtnew = std::min(dtnew, 1.0/rate);", "dtnew = std::min(dtnew, static_cast<Real>(1.0/rate));"),
    ("src/srcterms/turb_driver.cpp", "t0 = std::max(t0, 1.0e-20);", "t0 = std::max(t0, static_cast<Real>(1.0e-20));"),
    ("src/srcterms/turb_driver.cpp", "t1 = std::max(t1, 1.0e-20);", "t1 = std::max(t1, static_cast<Real>(1.0e-20));"),
    ("src/z4c/driftcontrol/driftcontrol.cpp", "std::max(dc_budget[a] - dc_budget_used[a], 0.0)",
     "std::max(dc_budget[a] - dc_budget_used[a], static_cast<Real>(0.0))"),
    ("src/z4c/driftcontrol/driftcontrol.cpp", "std::max(dc_budget_tend - pmesh->time, 1.0)",
     "std::max(dc_budget_tend - pmesh->time, static_cast<Real>(1.0))"),
    ("src/eos/primitive-solver/piecewise_polytrope.cpp", "  double densities[MAX_PIECES+1];\n  double gammas[MAX_PIECES+1];",
     "  Real densities[MAX_PIECES+1];\n  Real gammas[MAX_PIECES+1];"),
    ("src/pgen/unit_tests/gauss_legendre.cpp", "double ylmR1,ylmI1,ylmR2,ylmI2;", "Real ylmR1,ylmI1,ylmR2,ylmI2;"),
    ("src/pgen/tests/divb_amr.cpp", "+ std::abs(divb_amr.guide_b3) + 6.0*std::abs(divb_amr.field_amp));",
     "+ std::abs(divb_amr.guide_b3) + static_cast<Real>(6.0)*std::abs(divb_amr.field_amp));"),
]
TABLES = ("src/eos/primitive-solver/eos_compose.cpp", "src/eos/primitive-solver/eos_hybrid.cpp")   # Real * table_x = table["x"] (double)


def single_precision(root):
    for name, old, new in SINGLE:
        f = root/name;text = f.read_text()
        if new in text:continue
        if text.count(old) != 1:sys.exit(f"{name}: the single-precision edit does not apply ({old!r}); refusing to patch")
        f.write_text(text.replace(old, new))
    for name in TABLES:
        f = root/name;text = f.read_text()
        new = re.sub(r'Real \* (table_\w+) = table\["', r'auto * \1 = table["', text)
        if new != text:f.write_text(new)


def main(root):
    root = Path(root)
    single_precision(root)
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
