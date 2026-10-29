////////////////////////////////////////////////////////////////////////
/*! \file
  \brief The comparison suite's starting states for gPLUTO (abel).

  The state of the spec file in $ABEL_SPEC (tools/write_spec.jl), from the suite's shared port
  abel_problems.hpp, as AthenaK's abel_suite.cpp builds it: cell averages of rho, momentum and
  thermal + kinetic energy (returned here as the primitives that convert back to them), and the
  face fields of the vector potential. gPLUTO calls Init() at cell centres and, with staggered
  (constrained transport) fields and ASSIGN_VECTOR_POTENTIAL NO, at the lower face of each
  direction for that field component: a coordinate on a face (x n an integer) gives that face's
  field, one at a centre the mean of the cell's two faces (the cell-centred field of GLM builds).
  The mesh is the periodic unit square or cube with N zones per side.
*/
////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <cstdlib>

#include "gpluto.hpp"
#include "abel_problems.hpp"

static const abel::Spec &AbelSpec()
{
  static abel::Spec spec;
  static bool read = false;
  if (!read) {
    const char *path = std::getenv("ABEL_SPEC");
    if (path == nullptr) QuitPLUTO("abel: ABEL_SPEC is not set\n", 1);
    spec = abel::ReadSpec(path);
    read = true;
  }
  return spec;
}

// index of the cell whose centre, or of the face, a coordinate lies on; on_face reports which
static int AbelIndex(double x, int n, bool &on_face)
{
  const double q = x*n;
  const double r = std::floor(q + 0.5);
  on_face = std::fabs(q - r) < 1.e-6;
  const int i = on_face ? static_cast<int>(r) : static_cast<int>(std::floor(q));
  return ((i % n) + n) % n;
}

////////////////////////////////////////////////////////////////////////
void Init (double *v, double x1, double x2, double x3, RunConfig &run_config)
////////////////////////////////////////////////////////////////////////
{
  const abel::Spec &s = AbelSpec();
  const int n = s.n;
  run_config.gamma = s.gamma;
  bool f1, f2, f3;
  const int i = AbelIndex(x1, n, f1), j = AbelIndex(x2, n, f2);
  int k = 0;
  f3 = false;
  #if DIMENSIONS == 3
  if (!s.planar) k = AbelIndex(x3, n, f3);
  #endif
  auto up = [n](int q) {return (q + 1) % n;};

  // hydrodynamics: the cell's averages (only cell-centre calls use them)
  const abel::Cell c = abel::CellAverage(s, i, j, k);
  const double kinetic = 0.5*(c.m[0]*c.m[0] + c.m[1]*c.m[1] + c.m[2]*c.m[2])/c.rho;
  v[RHO] = c.rho;
  v[VX1] = c.m[0]/c.rho;
  v[VX2] = c.m[1]/c.rho;
  v[VX3] = c.m[2]/c.rho;
  v[PRS] = (s.gamma - 1.0)*(c.ehydro - kinetic);

  // magnetic field: a face's own value, or the mean of the cell's two faces
  const int kk = s.planar ? 0 : up(k);
  v[BX1] = f1 ? abel::FaceB(s, 0, i, j, k) : 0.5*(abel::FaceB(s, 0, i, j, k) + abel::FaceB(s, 0, up(i), j, k));
  v[BX2] = f2 ? abel::FaceB(s, 1, i, j, k) : 0.5*(abel::FaceB(s, 1, i, j, k) + abel::FaceB(s, 1, i, up(j), k));
  v[BX3] = f3 ? abel::FaceB(s, 2, i, j, k) : 0.5*(abel::FaceB(s, 2, i, j, k) + abel::FaceB(s, 2, i, j, kk));
  #ifdef GLM_MHD
  v[PSI_GLM] = 0.0;
  #endif
  v[AX1] = v[AX2] = v[AX3] = 0.0;
  #if ENTROPY_SWITCH != NO
  v[ENTR] = Entropy(v, run_config);
  #endif
}

void InitDomain (Data *data, Grid *grid) {}
void Analysis (Data *data, Grid *grid) {}
void UserDefBoundary (const Data *data, RBox *box, int side, Grid *grid) {}

#if BODY_FORCE != NO
void BodyForceVector(const State& state, const RunConfig& run_config, double *g)
{
  g[IDIR] = g[JDIR] = g[KDIR] = 0.0;
}
double BodyForcePotential(const State& state, const RunConfig& run_config) {return 0.0;}
#endif
