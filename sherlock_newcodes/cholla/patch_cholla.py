"""Prepare a Cholla checkout for the comparison suite (idempotent).

    python3 patch_cholla.py CHOLLA_DIR ABEL_PROBLEMS_HPP

1. Copies the suite's abel_problems.hpp into src/grid/ and adds the initial condition init=Abel
   (Grid3D::Abel_Suite, MHD builds): the starting state of the spec file in $ABEL_SPEC (tools/write_spec.jl), cell
   averages of rho, momentum and thermal + kinetic energy, and the face fields of the vector potential, as AthenaK's
   abel_suite.cpp. magnetic_x[id] is the field on the upper x face of cell id (Cholla's convention).
2. The builds of our two set-ups (builds/make.type.abel_plmc, abel_ppmc) and of this machine (make.host.sherlock).
Cholla already ends its steps on output times (main.cpp: next_scheduled_time); nothing else changes.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFINITION = r'''
#ifdef MHD
// abel: the comparison suite's starting state (spec file in $ABEL_SPEC, tools/write_spec.jl), from the suite's
// shared port abel_problems.hpp: cell averages, and face fields from the vector potential. Every cell, ghosts
// included, from the periodic global index of its centre; the boundary conditions overwrite the ghosts anyway.
void Grid3D::Abel_Suite()
{
  const char *path = std::getenv("ABEL_SPEC");
  if (path == nullptr) {
    chprintf("abel: ABEL_SPEC is not set\n");
    chexit(-1);
  }
  const abel::Spec spec = abel::ReadSpec(path);
  const int n           = spec.n;
  auto wrap             = [n](int q) { return ((q % n) + n) % n; };
  for (int k = 0; k < H.nz; k++) {
    for (int j = 0; j < H.ny; j++) {
      for (int i = 0; i < H.nx; i++) {
        const int id = i + j * H.nx + k * H.nx * H.ny;
        Real x, y, z;
        Get_Position(i, j, k, &x, &y, &z);
        const int gi = wrap(static_cast<int>(std::floor(x * n)));
        const int gj = wrap(static_cast<int>(std::floor(y * n)));
        const int gk = spec.planar ? 0 : wrap(static_cast<int>(std::floor(z * n)));
        const int ki = spec.planar ? 0 : wrap(gk + 1);
        const abel::Cell c = abel::CellAverage(spec, gi, gj, gk);
        const Real bx0 = abel::FaceB(spec, 0, gi, gj, gk), bx1 = abel::FaceB(spec, 0, wrap(gi + 1), gj, gk);
        const Real by0 = abel::FaceB(spec, 1, gi, gj, gk), by1 = abel::FaceB(spec, 1, gi, wrap(gj + 1), gk);
        const Real bz0 = abel::FaceB(spec, 2, gi, gj, gk), bz1 = abel::FaceB(spec, 2, gi, gj, ki);
        const Real bx = 0.5 * (bx0 + bx1), by = 0.5 * (by0 + by1), bz = 0.5 * (bz0 + bz1);
        C.density[id]    = c.rho;
        C.momentum_x[id] = c.m[0];
        C.momentum_y[id] = c.m[1];
        C.momentum_z[id] = c.m[2];
        C.Energy[id]     = c.ehydro + 0.5 * (bx * bx + by * by + bz * bz);
        C.magnetic_x[id] = bx1;
        C.magnetic_y[id] = by1;
        C.magnetic_z[id] = bz1;
      }
    }
  }
}
#endif  // MHD (abel)
'''


def edit(path, old, new, done):
    text = path.read_text()
    if done in text:return
    if text.count(old) != 1:sys.exit(f"patch_cholla.py: {path.name}: expected this text once, found it {text.count(old)} times:\n{old}")
    path.write_text(text.replace(old, new))


def main(root, header):
    root = Path(root);grid = root/"src/grid"
    shutil.copy(header, grid/"abel_problems.hpp")
    for f in ("make.type.abel_plmc", "make.type.abel_ppmc", "make.host.sherlock"):shutil.copy(HERE/f, root/"builds"/f)
    edit(grid/"grid3D.h", "  void Orszag_Tang_Vortex();\n",
         "  void Orszag_Tang_Vortex();\n  void Abel_Suite();  // abel: the comparison suite (init=Abel)\n", "Abel_Suite()")
    ic = grid/"initial_conditions.cpp"
    edit(ic, '#include "../utils/mhd_utilities.h"\n', '#include "../utils/mhd_utilities.h"\n#include "abel_problems.hpp"  // abel\n', '"abel_problems.hpp"')
    edit(ic, '  } else if (strcmp(P.init, "Riemann") == 0) {\n',
         '  } else if (strcmp(P.init, "Abel") == 0) {  // abel: the comparison suite\n#ifdef MHD\n    Abel_Suite();\n#else\n'
         '    chprintf("abel: the suite needs an MHD build\\n");\n    chexit(-1);\n#endif\n'
         '  } else if (strcmp(P.init, "Riemann") == 0) {\n', 'strcmp(P.init, "Abel")')
    text = ic.read_text()
    if "void Grid3D::Abel_Suite()" not in text:ic.write_text(text.rstrip() + "\n" + DEFINITION)
    print(f"patched {root}")


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
