//========================================================================================
// AthenaPK problem generator for the comparison suite (job/problem_id = abel_suite).
// Builds the starting state from <problem/abel>/spec_file (tools/write_spec.jl) with the
// suite's shared port abel_problems.hpp, as AthenaK's abel_suite.cpp does: cell averages of
// rho, momentum and thermal + kinetic energy, and the cell-centred field as the mean of the
// two divergence-free face fields of the vector potential (problems.jl center_faces).
// Mesh: the periodic unit square or cube, nx1 = nx2 = N (nx3 = N, or 1 for planar tests).
//========================================================================================
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../main.hpp"
#include "abel_problems.hpp"
#include "mesh/mesh.hpp"

namespace abel_suite {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const abel::Spec spec = abel::ReadSpec(pin->GetString("problem/abel", "spec_file"));
  const int n = spec.n;
  if (pin->GetInteger("parthenon/mesh", "nx1") != n || pin->GetReal("parthenon/mesh", "x1min") != 0.0 ||
      pin->GetReal("parthenon/mesh", "x1max") != 1.0) {
    std::cerr << "### FATAL ERROR: abel_suite mesh must be [0,1] with nx1 = spec n = " << n << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (std::abs(pin->GetReal("hydro", "gamma") - spec.gamma) > 1e-12) {
    std::cerr << "### FATAL ERROR: hydro/gamma differs from the spec's " << spec.gamma << std::endl;
    std::exit(EXIT_FAILURE);
  }
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;
  const bool planar = spec.planar;

  pmb->par_for(
      "abel_suite", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // global (0-based) cell index from the cell centre
        const int gi = static_cast<int>(std::floor(coords.Xc<1>(i) * n));
        const int gj = static_cast<int>(std::floor(coords.Xc<2>(j) * n));
        const int gk = planar ? 0 : static_cast<int>(std::floor(coords.Xc<3>(k) * n));
        const abel::Cell c = abel::CellAverage(spec, gi, gj, gk);
        const Real bx = 0.5 * (abel::FaceB(spec, 0, gi, gj, gk) + abel::FaceB(spec, 0, gi + 1, gj, gk));
        const Real by = 0.5 * (abel::FaceB(spec, 1, gi, gj, gk) + abel::FaceB(spec, 1, gi, gj + 1, gk));
        const Real bz = 0.5 * (abel::FaceB(spec, 2, gi, gj, gk) + abel::FaceB(spec, 2, gi, gj, gk + 1));
        u(IDN, k, j, i) = c.rho;
        u(IM1, k, j, i) = c.m[0];
        u(IM2, k, j, i) = c.m[1];
        u(IM3, k, j, i) = c.m[2];
        u(IB1, k, j, i) = bx;
        u(IB2, k, j, i) = by;
        u(IB3, k, j, i) = bz;
        u(IPS, k, j, i) = 0.0;
        u(IEN, k, j, i) = c.ehydro + 0.5 * (bx * bx + by * by + bz * bz);
      });
}
} // namespace abel_suite
