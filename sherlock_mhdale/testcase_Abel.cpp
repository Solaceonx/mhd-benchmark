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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
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

// ABEL_IC=l2 (2026-11-17): the element's density and internal energy density p/(gamma - 1) as polynomials of degree
// ABEL_IC_ORDER (the code's density and energy degree, -ot; default 1) in each direction, fitted by least squares so
// that their averages over the suite's sub-cells (s.sub per cell side; the points abel_cell uses) match the suite's
// sub-cell values, then limited toward the element mean (Zhang & Shu) to stay within the element's range of values. The
// element keeps exactly the mass and thermal energy of its cells (the fit has a constant term, so its residuals sum to
// zero), and a smooth field is represented to order p + 1. With cell values (the default) a smooth field was
// represented to first order: MHD-ALE projects the cell values, constant over half a degree-1 element, onto the element
// and gets 3/4 of the true slope; on the entropy wave that error alone was the whole measured error (4.3e-4 at 128^2,
// the same at t = 0 as at t = 1). The element is ABEL_ELEM_CELLS (default 2) suite cells wide (make_mesh.py: N/K).
struct AbelElemFit { bool done = false; std::vector<real_t> rho, E; };

static bool abel_ic_l2()
{
   static const bool on = std::getenv("ABEL_IC") && std::string(std::getenv("ABEL_IC")) == "l2";
   return on;
}

static int abel_env_int(const char *name, int fallback)
{
   const char *v = std::getenv(name);
   return v ? std::atoi(v) : fallback;
}

static real_t abel_legendre(int a, real_t t)
{
   real_t p0 = 1.0, p1 = t;
   if (a == 0) { return 1.0; }
   for (int k = 1; k < a; k++) { const real_t p2 = ((2 * k + 1) * t * p1 - k * p0) / (k + 1); p0 = p1; p1 = p2; }
   return p1;
}

static real_t abel_poly(const std::vector<real_t> &c, int P, real_t xi, real_t eta)
{
   real_t v = 0.0;
   for (int b = 0; b <= P; b++) for (int a = 0; a <= P; a++) { v += c[b * (P + 1) + a] * abel_legendre(a, xi) * abel_legendre(b, eta); }
   return v;
}

// Scale the non-constant part so that the polynomial stays in [lo, hi] at (M + 1)^2 points from corner to corner.
static void abel_limit(std::vector<real_t> &c, int P, int M, real_t lo, real_t hi)
{
   const real_t mean = c[0];
   real_t pmin = mean, pmax = mean;
   for (int j = 0; j <= M; j++) for (int i = 0; i <= M; i++)
   {
      const real_t v = abel_poly(c, P, -1.0 + 2.0 * i / M, -1.0 + 2.0 * j / M);
      pmin = fmin(pmin, v); pmax = fmax(pmax, v);
   }
   real_t theta = 1.0;
   if (pmax > hi && pmax > mean) { theta = fmin(theta, (hi - mean) / (pmax - mean)); }
   if (pmin < lo && pmin < mean) { theta = fmin(theta, (mean - lo) / (mean - pmin)); }
   theta = fmax(theta, 0.0);
   for (size_t k = 1; k < c.size(); k++) { c[k] *= theta; }
}

static const AbelElemFit &abel_fit(const Vector &x, real_t &xi, real_t &eta, int &P)
{
   const abel::Spec &s = AbelSpec();
   static const int K = abel_env_int("ABEL_ELEM_CELLS", 2);
   static const int order = abel_env_int("ABEL_IC_ORDER", 1);
   P = order;
   if (K < 1 || s.n % K) { MFEM_ABORT("abel: ABEL_ELEM_CELLS must divide the suite's N"); }
   const int ne = s.n / K, m = s.sub > 0 ? s.sub : 1, M = K * m;
   static std::vector<AbelElemFit> fits(ne * ne);
   const int ei = std::min(ne - 1, std::max(0, (int) floor(x(0) * ne)));
   const int ej = std::min(ne - 1, std::max(0, (int) floor(x(1) * ne)));
   const real_t w = 1.0 / ne, x0 = ei * w, y0 = ej * w;
   xi = 2.0 * (x(0) - x0) / w - 1.0; eta = 2.0 * (x(1) - y0) / w - 1.0;
   AbelElemFit &f = fits[ej * ne + ei];
   static std::vector<real_t> B;                          // (P + 1) x M: least squares on sub-interval averages
   if (B.empty())
   {
      std::vector<real_t> A(M * (P + 1));                 // A[q][a]: average of P_a over sub-interval q of [-1, 1]
      auto anti = [](int a, real_t t) { return a == 0 ? t : (abel_legendre(a + 1, t) - abel_legendre(a - 1, t)) / (2 * a + 1); };
      for (int q = 0; q < M; q++) for (int a = 0; a <= P; a++)
      {
         const real_t lo = -1.0 + 2.0 * q / M, hi = -1.0 + 2.0 * (q + 1) / M;
         A[q * (P + 1) + a] = (anti(a, hi) - anti(a, lo)) / (hi - lo);
      }
      const int n1 = P + 1;                               // G = A^T A, then B = G^-1 A^T by Gauss-Jordan
      std::vector<real_t> G(n1 * n1, 0.0), Ginv(n1 * n1, 0.0);
      for (int a = 0; a < n1; a++) for (int b = 0; b < n1; b++) for (int q = 0; q < M; q++) { G[a * n1 + b] += A[q * n1 + a] * A[q * n1 + b]; }
      for (int a = 0; a < n1; a++) { Ginv[a * n1 + a] = 1.0; }
      for (int c = 0; c < n1; c++)
      {
         int piv = c;
         for (int r = c + 1; r < n1; r++) { if (fabs(G[r * n1 + c]) > fabs(G[piv * n1 + c])) { piv = r; } }
         for (int k = 0; k < n1; k++) { std::swap(G[c * n1 + k], G[piv * n1 + k]); std::swap(Ginv[c * n1 + k], Ginv[piv * n1 + k]); }
         const real_t d = G[c * n1 + c];
         if (fabs(d) < 1e-300) { MFEM_ABORT("abel: ABEL_IC=l2 needs at least ABEL_IC_ORDER + 1 samples per element side"); }
         for (int k = 0; k < n1; k++) { G[c * n1 + k] /= d; Ginv[c * n1 + k] /= d; }
         for (int r = 0; r < n1; r++)
         {
            if (r == c) { continue; }
            const real_t g = G[r * n1 + c];
            for (int k = 0; k < n1; k++) { G[r * n1 + k] -= g * G[c * n1 + k]; Ginv[r * n1 + k] -= g * Ginv[c * n1 + k]; }
         }
      }
      B.assign(n1 * M, 0.0);
      for (int a = 0; a < n1; a++) for (int q = 0; q < M; q++) for (int b = 0; b < n1; b++) { B[a * M + q] += Ginv[a * n1 + b] * A[q * n1 + b]; }
   }
   if (!f.done)
   {
      const int n1 = P + 1;
      std::vector<real_t> R(M * M), E(M * M);             // samples, [y][x]
      real_t rlo = 1e300, rhi = -1e300, elo = 1e300, ehi = -1e300;
      // The limiter's range: the samples of the element and of one suite cell around it (periodic box). The element's
      // own samples alone would clip every smooth slope (its corner lies beyond its outermost sample).
      for (int q = -m; q < M + m; q++) for (int r = -m; r < M + m; r++)
      {
         real_t px = x0 + (r + 0.5) / M * w, py = y0 + (q + 0.5) / M * w;
         px -= floor(px); py -= floor(py);
         const abel::Prim pr = abel::PrimAt(s, px, py, 0.0);
         const real_t e = pr.p / (s.gamma - 1.0);
         if (q >= 0 && q < M && r >= 0 && r < M) { R[q * M + r] = pr.rho; E[q * M + r] = e; }
         rlo = fmin(rlo, pr.rho); rhi = fmax(rhi, pr.rho); elo = fmin(elo, e); ehi = fmax(ehi, e);
      }
      auto fit = [&](const std::vector<real_t> &F, std::vector<real_t> &c)
      {
         std::vector<real_t> T(n1 * M, 0.0);              // T[a][q] = sum_r B[a][r] F[q][r]  (x direction)
         for (int a = 0; a < n1; a++) for (int q = 0; q < M; q++) for (int r = 0; r < M; r++) { T[a * M + q] += B[a * M + r] * F[q * M + r]; }
         c.assign(n1 * n1, 0.0);                          // c[b][a] = sum_q B[b][q] T[a][q]  (y direction)
         for (int b = 0; b < n1; b++) for (int a = 0; a < n1; a++) for (int q = 0; q < M; q++) { c[b * n1 + a] += B[b * M + q] * T[a * M + q]; }
      };
      fit(R, f.rho); fit(E, f.E);
      abel_limit(f.rho, P, 2 * M, rlo, rhi); abel_limit(f.E, P, 2 * M, elo, ehi);
      f.done = true;
   }
   return f;
}

// Density: the value of the suite's cell that holds the point (ABEL_POINT_VALUES=1: the point value). MHD-ALE
// integrates this function over each element with a symmetric Gauss rule of an even number of points, which gives
// each of the element's four cells a quarter of the weight: the element then holds the mass of its four cells.
// With point values the trials started with up to 0.3 per cent more or less mass than the suite's state at 64^2.
static real_t abel_rho0(const Vector &x)
{
   static const bool point = std::getenv("ABEL_POINT_VALUES") != nullptr;
   if (point) { return abel::PrimAt(AbelSpec(), x(0), x(1), 0.0).rho; }
   if (abel_ic_l2()) { real_t xi, eta; int P; const AbelElemFit &f = abel_fit(x, xi, eta, P); return abel_poly(f.rho, P, xi, eta); }
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
   if (abel_ic_l2())
   {
      real_t xi, eta; int P; const AbelElemFit &f = abel_fit(x, xi, eta, P);
      return abel_poly(f.E, P, xi, eta) / abel_poly(f.rho, P, xi, eta);
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
      printf("abel: problem %d, box %s, gamma %.17g, mean B (%g, %g), viscosity %d, initial state %s\n", s.problem, b.c_str(), s.gamma, bm.x, bm.y,
             (int) pd->visc, abel_ic_l2() ? "limited L2 projection per element (ABEL_IC=l2)" : "cell values");
   }
   return pd;
}

}
