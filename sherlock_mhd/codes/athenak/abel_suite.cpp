//========================================================================================
// AthenaK problem generator for the GLM/CT comparison suite (sherlock_mhd).
// Builds the initial state from <problem>/spec_file written by tools/write_spec.jl, using
// the shared port in abel_problems.hpp. Mesh must be the periodic unit cube with nx1 = N
// (and nx2 = N, nx3 = N or 1 for planar problems).
// Build: copied into athenak/src/pgen/ with abel_problems.hpp; cmake -D PROBLEM=abel_suite.
//========================================================================================
#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "abel_problems.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR: abel_suite requires an <mhd> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const abel::Spec spec = abel::ReadSpec(pin->GetString("problem", "spec_file"));
  auto &ms = pmy_mesh_->mesh_size;
  if (pmy_mesh_->mesh_indcs.nx1 != spec.n || ms.x1min != 0.0 || ms.x1max != 1.0) {
    std::cout << "### FATAL ERROR: abel_suite mesh must be [0,1] with nx1 = spec n = "
              << spec.n << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real gamma = pmbp->pmhd->peos->eos_data.gamma;
  if (std::abs(gamma - spec.gamma) > 1e-12) {
    std::cout << "### FATAL ERROR: <mhd> gamma " << gamma << " differs from spec gamma "
              << spec.gamma << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
  const int n = spec.n;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &size = pmbp->pmb->mb_size;

  // Global (0-based) index of local cell q in a block starting at xmin.
  par_for("abel_faces", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    int gi = static_cast<int>(std::lround(size.d_view(m).x1min*n)) + (i - is);
    int gj = static_cast<int>(std::lround(size.d_view(m).x2min*n)) + (j - js);
    int gk = static_cast<int>(std::lround(size.d_view(m).x3min*n)) + (k - ks);
    b0.x1f(m, k, j, i) = abel::FaceB(spec, 0, gi, gj, gk);
    b0.x2f(m, k, j, i) = abel::FaceB(spec, 1, gi, gj, gk);
    b0.x3f(m, k, j, i) = abel::FaceB(spec, 2, gi, gj, gk);
    if (i == ie) b0.x1f(m, k, j, i+1) = abel::FaceB(spec, 0, gi+1, gj, gk);
    if (j == je) b0.x2f(m, k, j+1, i) = abel::FaceB(spec, 1, gi, gj+1, gk);
    if (k == ke) b0.x3f(m, k+1, j, i) = abel::FaceB(spec, 2, gi, gj, gk+1);
  });

  par_for("abel_cells", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    int gi = static_cast<int>(std::lround(size.d_view(m).x1min*n)) + (i - is);
    int gj = static_cast<int>(std::lround(size.d_view(m).x2min*n)) + (j - js);
    int gk = static_cast<int>(std::lround(size.d_view(m).x3min*n)) + (k - ks);
    abel::Cell c = abel::CellAverage(spec, gi, gj, gk);
    Real bx = 0.5*(b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    Real by = 0.5*(b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
    Real bz = 0.5*(b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
    u0(m, IDN, k, j, i) = c.rho;
    u0(m, IM1, k, j, i) = c.m[0];
    u0(m, IM2, k, j, i) = c.m[1];
    u0(m, IM3, k, j, i) = c.m[2];
    u0(m, IEN, k, j, i) = c.ehydro + 0.5*(bx*bx + by*by + bz*bz);
  });
}
