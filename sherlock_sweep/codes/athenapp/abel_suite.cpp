//========================================================================================
// Athena++ problem generator for the GLM/CT comparison suite (sherlock_mhd).
// Same initial data as the AthenaK generator, from the shared abel_problems.hpp port.
// Build: copy this file and ../common/abel_problems.hpp into athena/src/pgen/, then
//   python configure.py --prob abel_suite -b --flux hlld --nghost 3 [-omp] [-mpi]
// Input: <problem>/spec_file from tools/write_spec.jl; mesh = [0,1]^3 (nx3 = 1 for planar).
//========================================================================================
#include <cmath>
#include <iostream>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "abel_problems.hpp"

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const abel::Spec spec = abel::ReadSpec(pin->GetString("problem", "spec_file"));
  const int n = spec.n;
  if (pmy_mesh->mesh_size.nx1 != n || pmy_mesh->mesh_size.x1min != 0.0 ||
      pmy_mesh->mesh_size.x1max != 1.0) {
    std::cout << "### FATAL ERROR: abel_suite mesh must be [0,1] with nx1 = spec n = " << n << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (std::abs(peos->GetGamma() - spec.gamma) > 1e-12) {
    std::cout << "### FATAL ERROR: <hydro> gamma differs from spec gamma " << spec.gamma << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Global 0-based index from the left face coordinate of local index q.
  auto gidx = [n](Real xleft) {return static_cast<int>(std::lround(xleft*n));};
  for (int k = ks; k <= ke + 1; ++k) {
    for (int j = js; j <= je + 1; ++j) {
      for (int i = is; i <= ie + 1; ++i) {
        int gi = gidx(pcoord->x1f(i)), gj = gidx(pcoord->x2f(j)), gk = gidx(pcoord->x3f(k));
        if (j <= je && k <= ke) pfield->b.x1f(k, j, i) = abel::FaceB(spec, 0, gi, gj, gk);
        if (i <= ie && k <= ke) pfield->b.x2f(k, j, i) = abel::FaceB(spec, 1, gi, gj, gk);
        if (i <= ie && j <= je) pfield->b.x3f(k, j, i) = abel::FaceB(spec, 2, gi, gj, gk);
      }
    }
  }
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        abel::Cell c = abel::CellAverage(spec, gidx(pcoord->x1f(i)), gidx(pcoord->x2f(j)), gidx(pcoord->x3f(k)));
        Real bx = 0.5*(pfield->b.x1f(k, j, i) + pfield->b.x1f(k, j, i+1));
        Real by = 0.5*(pfield->b.x2f(k, j, i) + pfield->b.x2f(k, j+1, i));
        Real bz = 0.5*(pfield->b.x3f(k, j, i) + pfield->b.x3f(k+1, j, i));
        phydro->u(IDN, k, j, i) = c.rho;
        phydro->u(IM1, k, j, i) = c.m[0];
        phydro->u(IM2, k, j, i) = c.m[1];
        phydro->u(IM3, k, j, i) = c.m[2];
        phydro->u(IEN, k, j, i) = c.ehydro + 0.5*(bx*bx + by*by + bz*bz);
      }
    }
  }
}
