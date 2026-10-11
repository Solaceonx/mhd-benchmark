// The comparison suite's tests as an MHD-ALE problem (ours; added to the build by prepare_patch.py, -p 100).
//
// Initial state from codes/common/abel_problems.hpp of the sweep bundle, the same functions AthenaK and Athena++
// use: PrimAt (rho, v, p), PotentialAt (periodic part of the vector potential), Uniform (mean field). The spec file
// (tools/write_spec.jl) is named by the environment variable ABEL_SPEC. Units: energy density B^2/2, so mu = 1.
//
// ABEL_BOX chooses the box, all on [0,1]^2:
//   periodic  periodic in x and y (the suite's set-up). Needs the fork yipihey/MHD-ALE for the remap (-mst 1 -pmf).
//   xwalls    periodic in y, walls at x = 0 and 1 (the set-up of MHD-ALE's own shock tubes).
//   walls     walls on all four sides (the set-up of MHD-ALE's own rotor and blast).
// A wall is the same problem as the periodic box only while nothing has reached it; the driver decides per test.
//
// In two dimensions MHD-ALE has no v_z and no B_z (scalar potential A_z, two velocity components): a test that
// needs them is refused here rather than run without them.
#include <cstdlib>
#include <string>
#include "Problemdata.hpp"
#include "abel_problems.hpp"

namespace mfem
{

static abel::Spec abel_spec;
static int abel_box = 0;                 // 0 periodic, 1 xwalls, 2 walls
static bool abel_spec_read = false;

const abel::Spec &AbelSpec()
{
   if (!abel_spec_read)
   {
      const char *path = std::getenv("ABEL_SPEC");
      if (!path) { MFEM_ABORT("abel: set ABEL_SPEC to the spec file written by tools/write_spec.jl"); }
      abel_spec = abel::ReadSpec(path);
      abel_spec_read = true;
   }
   return abel_spec;
}

// Mean pressure and mean density of the suite's cell that holds a point, over the cell's own samples
// (abel::CellAverage uses the same points). An element is two cells wide.
static void abel_cell(const Vector &x, real_t &p, real_t &rho)
{
   const abel::Spec &s = AbelSpec();
   const int m = s.sub > 0 ? s.sub : 1;
   const real_t h = 1.0 / s.n, i = floor(x(0) * s.n), j = floor(x(1) * s.n);
   p = 0.0; rho = 0.0;
   for (int b = 0; b < m; b++) for (int c = 0; c < m; c++)
   {
      const abel::Prim q = abel::PrimAt(s, (i + (c + 0.5) / m) * h, (j + (b + 0.5) / m) * h, 0.0);
      p += q.p; rho += q.rho;
   }
   p /= m * m; rho /= m * m;
}

// Density: the value of the suite's cell that holds the point (ABEL_POINT_VALUES=1: the point value). MHD-ALE
// integrates this function over each element with a symmetric Gauss rule of an even number of points, which gives
// each of the element's four cells a quarter of the weight: the element then holds the mass of its four cells.
// With point values the trials started with up to 0.3 per cent more or less mass than the suite's state at 64^2.
static real_t abel_rho0(const Vector &x)
{
   static const bool point = std::getenv("ABEL_POINT_VALUES") != nullptr;
   if (point) { return abel::PrimAt(AbelSpec(), x(0), x(1), 0.0).rho; }
   real_t p, rho; abel_cell(x, p, rho);
   return rho;
}

static real_t abel_gamma(const Vector &x) { return AbelSpec().gamma; }

static void abel_v0(const Vector &x, Vector &v)
{
   const abel::Prim q = abel::PrimAt(AbelSpec(), x(0), x(1), 0.0);
   v(0) = q.vx; v(1) = q.vy;
}

// Specific internal energy, p = (gamma - 1) rho e. MHD-ALE sets the energy field from this function's values at
// the field's nodes. A point value there misplaces the energy of a jump: in the trials the blast started with 3.5
// per cent less energy than the suite's state at 64^2. So the value returned is that of the suite's cell which holds
// the point: mean pressure over mean density. With the energy of degree 1 an element has two nodes per direction,
// one in each of its two cells per direction, and so holds the thermal energy of its four cells.
static real_t abel_e0(const Vector &x)
{
   const abel::Spec &s = AbelSpec();
   static const bool point = std::getenv("ABEL_POINT_VALUES") != nullptr;
   if (point)
   {
      const abel::Prim q = abel::PrimAt(s, x(0), x(1), 0.0);
      return q.p / (s.gamma - 1.0) / q.rho;
   }
   real_t p, rho; abel_cell(x, p, rho);
   return p / (s.gamma - 1.0) / rho;
}

// A_z. With walls on all sides nothing is periodic and the mean field is carried by the potential as well:
// B = (dA/dy, -dA/dx), so A = Bx y - By x gives the uniform (Bx, By).
static void abel_A0(const Vector &x, Vector &A)
{
   const abel::Spec &s = AbelSpec();
   A(0) = abel::PotentialAt(s, x(0), x(1), 0.0).z;
   if (abel_box == 2) { const abel::Vec b = abel::Uniform(s); A(0) += b.x * x(1) - b.y * x(0); }
}

static void abel_Bmean(const Vector &x, Vector &B)
{
   const abel::Vec b = abel::Uniform(AbelSpec());
   B(0) = b.x; B(1) = b.y;
}

static void abel_B0(const Vector &x, Vector &B) { abel_Bmean(x, B); }     // not used by MHD.cpp for the state itself

static void abel_H0(const Vector &x, Vector &H) { abel_Bmean(x, H); }     // mu = 1; the state at an untouched wall

ProblemData *GetProblemAbel(int dim)
{
   if (dim != 2) { MFEM_ABORT("abel: the suite's planar tests run with -dim 2"); }
   const abel::Spec &s = AbelSpec();
   const char *box = std::getenv("ABEL_BOX");
   const std::string b = box ? box : "periodic";
   if (b == "periodic") { abel_box = 0; }
   else if (b == "xwalls") { abel_box = 1; }
   else if (b == "walls") { abel_box = 2; }
   else { MFEM_ABORT("abel: ABEL_BOX must be periodic, xwalls or walls"); }

   // Out-of-plane parts cannot be represented: look for them on a grid of points.
   {
      real_t vz = 0.0, axy = 0.0;
      const int m = 64;
      for (int j = 0; j < m; j++) for (int i = 0; i < m; i++)
      {
         const real_t x = (i + 0.37) / m, y = (j + 0.61) / m;
         const abel::Prim q = abel::PrimAt(s, x, y, 0.0);
         const abel::Vec a = abel::PotentialAt(s, x, y, 0.0);
         vz = fmax(vz, fabs(q.vz)); axy = fmax(axy, fmax(fabs(a.x), fabs(a.y)));
      }
      if (vz > 0.0 || axy > 0.0 || abel::Uniform(s).z != 0.0)
      {
         if (Mpi::Root()) { printf("abel: NOT REPRESENTABLE in 2D: max |vz| = %g, max |Ax|,|Ay| = %g, mean Bz = %g\n", vz, axy, abel::Uniform(s).z); }
         MFEM_ABORT("abel: this test needs v_z or B_z, which the 2D code does not have");
      }
   }

   ProblemData *pd = new ProblemData(abel_e0, abel_rho0, abel_gamma, abel_v0, abel_A0, abel_B0);
   const char *visc = std::getenv("ABEL_VISC");          // artificial viscosity: on unless ABEL_VISC=0
   pd->SetViscosity(!(visc && std::string(visc) == "0"));
   pd->SetVorticity(true);                               // as in MHD-ALE's own rotor and Brio-Wu
   pd->SetMu(1.0);
   pd->SetDimension(2);
   // Boundary attributes of the mesh (make_mesh.py, MHD-ALE's convention): 1 y=0, 2 x=1, 3 y=1, 4 x=0.
   if (abel_box == 0)
   {
      Array<int> none(4);                                // no walls: as testcase_PeriodicBox.cpp of the fork declares it
      none = 0;
      pd->AddEssentialBoundaryCondition(none, none);
      pd->SetHBoundaryCondition(none, abel_H0);
      pd->SetPeriodic(1.0, 1.0, -1.0, new VectorFunctionCoefficient(2, abel_Bmean));
   }
   else if (abel_box == 1)
   {
      Array<int> ess_bdr_x({0,1,0,1});                   // v_x = 0 on the walls x = 0, 1
      Array<int> ess_bdr_y({0,0,0,0});
      pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y);
      Array<int> ess_bdr_H({0,1,0,1});
      pd->SetHBoundaryCondition(ess_bdr_H, abel_H0);
      pd->SetPeriodic(-1.0, 1.0, -1.0, new VectorFunctionCoefficient(2, abel_Bmean));
   }
   else
   {
      Array<int> ess_bdr_x({0,1,0,1});                   // normal velocity zero on each wall
      Array<int> ess_bdr_y({1,0,1,0});
      pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y);
      Array<int> ess_bdr_H({1,1,1,1});
      pd->SetHBoundaryCondition(ess_bdr_H, abel_H0);
   }
   if (Mpi::Root())
   {
      const abel::Vec bm = abel::Uniform(s);
      printf("abel: problem %d, box %s, gamma %.17g, mean B (%g, %g), viscosity %d\n", s.problem, b.c_str(), s.gamma, bm.x, bm.y, (int) pd->visc);
   }
   return pd;
}

}
