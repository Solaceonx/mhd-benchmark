// Shared initial conditions for the GLM/CT comparison suite, for C++ codes (AthenaK, Athena++).
//
// A line-by-line port of sherlock_mhd/src/problems.jl, which remains the source of truth.
// Parameters come from a spec file written by tools/write_spec.jl, so this header never
// re-derives config defaults. tools/check_external_ic.py compares each code's t=0 state
// against Julia's. Coordinates are the periodic unit cube [0,1)^3.
#ifndef ABEL_PROBLEMS_HPP_
#define ABEL_PROBLEMS_HPP_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace abel {

enum Problem {UNIFORM, ENTROPY, ALFVEN, ALFVEN_OBLIQUE, BRIO_WU, BRIO_WU_ROTATED, ORSZAG_TANG,
              ORSZAG_TANG_3D, ROTOR, CURRENT_SHEET, FIELD_LOOP, FIELD_LOOP_3D, BLAST, DECAY,
              KH_UNIFORM, KH_LOCALIZED, GEM};
constexpr int kMaxModes = 32;
constexpr double kPi = 3.14159265358979323846;

// Everything the kernels need; trivially copyable so it can be captured by device lambdas.
struct Spec {
  int problem = UNIFORM, n = 0, sub = 1, planar = 1, dim = 3, nmodes = 0;
  double gamma = 5.0/3.0, t = 0.0;
  double vz = 0.0, u0 = 0.0, b0 = 0.0, bx = 0.0, epsilon = 0.0;
  double beta = 2e5, b_peak = .01, b_width = .1, b0_vector[3] = {.1, 0.0, 0.0};
  double gem_lx = 25.6, gem_lambda = .5, gem_psi0 = .1, gem_nbg = .2;   // GEM reconnection
  double mean[3] = {0, 0, 0}, scale = 1.0;           // decay momentum normalization
  int kv[kMaxModes][3] = {};
  double amp[kMaxModes][3] = {}, phase[kMaxModes] = {};
};

struct Prim {double rho, vx, vy, vz, p;};
struct Vec {double x, y, z;};

inline int ProblemId(const std::string &name) {
  const char *names[] = {"uniform", "entropy", "alfven", "alfven_oblique", "brio_wu",
    "brio_wu_rotated", "orszag_tang", "orszag_tang_3d", "rotor", "current_sheet",
    "field_loop", "field_loop_3d", "blast", "decay", "kh_uniform", "kh_localized", "gem"};
  for (int i = 0; i < 17; ++i) if (name == names[i]) return i;
  std::fprintf(stderr, "abel: unknown problem '%s'\n", name.c_str());
  std::exit(EXIT_FAILURE);
}

// Spec file: one "key value..." per line; "mode kx ky kz ax ay az phase" per decay mode.
inline Spec ReadSpec(const std::string &path) {
  std::ifstream in(path);
  if (!in) {std::fprintf(stderr, "abel: cannot open spec %s\n", path.c_str()); std::exit(1);}
  Spec s; std::string line;
  while (std::getline(in, line)) {
    std::istringstream w(line); std::string key; w >> key;
    if (key.empty() || key[0] == '#') continue;
    if (key == "problem") {std::string v; w >> v; s.problem = ProblemId(v);}
    else if (key == "n") w >> s.n;
    else if (key == "sub") w >> s.sub;
    else if (key == "planar") w >> s.planar;
    else if (key == "dim") w >> s.dim;
    else if (key == "gamma") w >> s.gamma;
    else if (key == "t") w >> s.t;
    else if (key == "vz") w >> s.vz;
    else if (key == "u0") w >> s.u0;
    else if (key == "b0") w >> s.b0;
    else if (key == "bx") w >> s.bx;
    else if (key == "epsilon") w >> s.epsilon;
    else if (key == "beta") w >> s.beta;
    else if (key == "b_peak") w >> s.b_peak;
    else if (key == "b_width") w >> s.b_width;
    else if (key == "gem_lx") w >> s.gem_lx;
    else if (key == "gem_lambda") w >> s.gem_lambda;
    else if (key == "gem_psi0") w >> s.gem_psi0;
    else if (key == "gem_nbg") w >> s.gem_nbg;
    else if (key == "b0_vector") w >> s.b0_vector[0] >> s.b0_vector[1] >> s.b0_vector[2];
    else if (key == "mean") w >> s.mean[0] >> s.mean[1] >> s.mean[2];
    else if (key == "scale") w >> s.scale;
    else if (key == "mode") {
      if (s.nmodes >= kMaxModes) {std::fprintf(stderr, "abel: too many modes\n"); std::exit(1);}
      int m = s.nmodes++;
      w >> s.kv[m][0] >> s.kv[m][1] >> s.kv[m][2] >> s.amp[m][0] >> s.amp[m][1] >> s.amp[m][2] >> s.phase[m];
    }
  }
  if (s.n <= 0) {std::fprintf(stderr, "abel: spec %s has no n\n", path.c_str()); std::exit(1);}
  return s;
}

KOKKOS_INLINE_FUNCTION double Sinc(double x) {
  return x == 0.0 ? 1.0 : std::sin(kPi*x)/(kPi*x);
}
KOKKOS_INLINE_FUNCTION double Mod1(double x) {return x - std::floor(x);}
// Periodic primitive of a +1/-1 square wave: +1 on [0,a) and [b,1), -1 on [a,b).
KOKKOS_INLINE_FUNCTION double SquareIntegral(double x, double a, double b) {
  x = Mod1(x);
  return x < a ? x : (x < b ? 2*a - x : x + 2*a - 2*b);
}
// GEM double Harris sheet (problems.jl "gem"): sheets at y = 1/4, 3/4, lengths scaled by 1/gem_lx.
KOKKOS_INLINE_FUNCTION double GemSheets(const Spec &s, double y) {
  double lam = s.gem_lambda/s.gem_lx, a = 1/std::cosh((y - .25)/lam), b = 1/std::cosh((y - .75)/lam);
  return a*a + b*b;
}
KOKKOS_INLINE_FUNCTION double LnCosh(double u) {
  return std::fabs(u) + std::log1p(std::exp(-2*std::fabs(u))) - std::log(2.0);
}
KOKKOS_INLINE_FUNCTION Vec LoopVelocity(const Spec &s) {
  return s.problem == FIELD_LOOP_3D ? Vec{1.0, 1.0, 2.0} : Vec{2.0, 1.0, s.vz};
}

KOKKOS_INLINE_FUNCTION Prim PrimAt(const Spec &s, double x, double y, double z) {
  const double t = s.t;
  switch (s.problem) {
    case UNIFORM: return {1.0, .2, -.1, .05, 1.0};
    case ENTROPY: return {1 + .2*Sinc(1.0/s.n)*Sinc(1.0/s.n)*std::sin(2*kPi*(x + y - t)), .5, .5, 0.0, 1.0};
    case ALFVEN: {
      double ph = 2*kPi*(x - t);
      return {1.0, 0.0, -.1*std::cos(ph), -.1*std::sin(ph), .1};
    }
    case ALFVEN_OBLIQUE: {
      double s5 = std::sqrt(5.0), tx = -2/s5, ty = 1/s5, ph = 2*kPi*(x + 2*y - s5*t);
      return {1.0, -.1*std::cos(ph)*tx, -.1*std::cos(ph)*ty, -.1*std::sin(ph), .1};
    }
    case BRIO_WU: case BRIO_WU_ROTATED: {
      double xi = s.problem == BRIO_WU_ROTATED ? Mod1(x + y) : x;
      return xi < .5 ? Prim{1.0, 0.0, 0.0, 0.0, 1.0} : Prim{.125, 0.0, 0.0, 0.0, .1};
    }
    case ORSZAG_TANG: case ORSZAG_TANG_3D: {
      double e = s.problem == ORSZAG_TANG_3D ? s.epsilon : 0.0, f = 1 + e*std::sin(2*kPi*z);
      return {25/(36*kPi), -f*std::sin(2*kPi*y), f*std::sin(2*kPi*x), e*std::sin(2*kPi*z), 5/(12*kPi)};
    }
    case ROTOR: {
      const double r0 = .1, r1 = .115;
      double r = std::hypot(x - .5, y - .5), f = (r1 - r)/(r1 - r0);
      if (r <= r0) return {10.0, -s.u0*(y - .5)/r0, s.u0*(x - .5)/r0, 0.0, 1.0};
      if (r < r1) return {1 + 9*f, -f*s.u0*(y - .5)/r, f*s.u0*(x - .5)/r, 0.0, 1.0};
      return {1.0, 0.0, 0.0, 0.0, 1.0};
    }
    case CURRENT_SHEET: return {1.0, s.u0*std::sin(2*kPi*y), 0.0, 0.0, .1};   // Gardiner & Stone (2005)
    case FIELD_LOOP: case FIELD_LOOP_3D: {
      Vec v = LoopVelocity(s); return {1.0, v.x, v.y, v.z, 1.0};
    }
    case BLAST: {
      double dz = s.dim == 3 ? (z - .5)*(z - .5) : 0.0;
      double r = std::sqrt((x - .5)*(x - .5) + (y - .5)*(y - .5) + dz);
      return {1.0, 0.0, 0.0, 0.0, r < .1 ? 1000.0 : .1};
    }
    case DECAY: {
      double v[3] = {0, 0, 0};
      for (int m = 0; m < s.nmodes; ++m) {
        double a = std::sin(2*kPi*(s.kv[m][0]*x + s.kv[m][1]*y + s.kv[m][2]*z) + s.phase[m]);
        for (int d = 0; d < 3; ++d) v[d] += s.amp[m][d]*a;
      }
      return {1.0, v[0], v[1], v[2], 1/s.gamma};
    }
    case KH_UNIFORM: case KH_LOCALIZED: {
      // Tomida et al. (2026): p = 10, v0 = 1, a = 0.05, perturbation 0.01 sin(2 pi x), sigma = 0.2.
      const double a = .05, sig = .2, v0 = 1.0;
      double vx = v0*(std::tanh((y - .25)/a) - std::tanh((y - .75)/a) - 1);
      double vy = .01*std::sin(2*kPi*x)*(std::exp(-((y - .25)/sig)*((y - .25)/sig)) + std::exp(-((y - .75)/sig)*((y - .75)/sig)));
      return {1.0, vx, vy, 0.0, 10.0};
    }
    case GEM: {
      double n = GemSheets(s, y) + s.gem_nbg;
      return {n, 0.0, 0.0, 0.0, .5*n};
    }
  }
  return {1.0, 0.0, 0.0, 0.0, 1.0};
}

// Periodic part of the vector potential; the uniform part is Uniform().
KOKKOS_INLINE_FUNCTION Vec PotentialAt(const Spec &s, double x, double y, double z) {
  const double t = s.t;
  switch (s.problem) {
    case ALFVEN: {
      double ph = 2*kPi*(x - t);
      return {0.0, -.1*std::cos(ph)/(2*kPi), -.1*std::sin(ph)/(2*kPi)};
    }
    case ALFVEN_OBLIQUE: {
      double s5 = std::sqrt(5.0), tx = -2/s5, ty = 1/s5, ph = 2*kPi*(x + 2*y - s5*t);
      return {-.1*std::cos(ph)*tx/(2*s5*kPi), -.1*std::cos(ph)*ty/(2*s5*kPi), -.1*std::sin(ph)/(2*kPi*s5)};
    }
    case BRIO_WU: return {0.0, 0.0, -SquareIntegral(x, .5, 1.0)};
    case BRIO_WU_ROTATED: return {0.0, 0.0, -SquareIntegral(Mod1(x + y), .5, 1.0)/std::sqrt(2.0)};
    case ORSZAG_TANG: case ORSZAG_TANG_3D: {
      double b = 1/std::sqrt(4*kPi);
      return {0.0, 0.0, b*(std::cos(2*kPi*y)/(2*kPi) + std::cos(4*kPi*x)/(4*kPi))};
    }
    case CURRENT_SHEET: return {0.0, 0.0, -SquareIntegral(x, .25, .75)};
    case FIELD_LOOP: return {0.0, 0.0, 1e-3*std::fmax(.3 - std::hypot(x - .5, y - .5), 0.0)};
    case GEM: {
      double lam = s.gem_lambda/s.gem_lx, psi0 = s.gem_psi0/s.gem_lx;
      return {0.0, 0.0, lam*LnCosh((y - .25)/lam) - lam*LnCosh((y - .75)/lam) - y
                        - psi0*std::cos(2*kPi*x)*std::cos(2*kPi*(y - .25))};   // X-points at x = 1/2
    }
    case KH_LOCALIZED: {
      double b = s.b_peak, w = s.b_width, f = -2*b*w*y;
      for (int sh = -1; sh <= 1; ++sh) f += b*w*(std::tanh((y - .25 + sh)/w) - std::tanh((sh - .25)/w));
      return {0.0, 0.0, f};
    }
    case FIELD_LOOP_3D: {
      const double ax = -1/std::sqrt(2.0), az = 1/std::sqrt(2.0);
      double d = 1e300;
      for (int sx = -1; sx <= 1; ++sx) for (int sy = -1; sy <= 1; ++sy) for (int sz = -1; sz <= 1; ++sz) {
        double v1 = x - .5 + sx, v2 = y - .5 + sy, v3 = z - .5 + sz, p = v1*ax + v3*az;
        d = std::fmin(d, std::sqrt(std::fmax(v1*v1 + v2*v2 + v3*v3 - p*p, 0.0)));
      }
      double a = 1e-3*std::fmax(.3 - d, 0.0);
      return {a*ax, 0.0, a*az};
    }
    default: return {0.0, 0.0, 0.0};
  }
}

KOKKOS_INLINE_FUNCTION Vec Uniform(const Spec &s) {
  switch (s.problem) {
    case UNIFORM: case ENTROPY: return {.3, .2, 0.0};
    case ALFVEN: return {1.0, 0.0, 0.0};
    case ALFVEN_OBLIQUE: return {1/std::sqrt(5.0), 2/std::sqrt(5.0), 0.0};
    case BRIO_WU: return {.75, 0.0, 0.0};
    case BRIO_WU_ROTATED: return {.75/std::sqrt(2.0), .75/std::sqrt(2.0), 0.0};
    case ROTOR: return {s.b0/std::sqrt(4*kPi), 0.0, 0.0};
    case BLAST: return {s.bx/std::sqrt(4*kPi), 0.0, 0.0};
    case DECAY: return {s.b0_vector[0], s.b0_vector[1], s.b0_vector[2]};
    case KH_UNIFORM: return {std::sqrt(2*10.0/s.beta), 0.0, 0.0};   // p = 10, beta = 2e5: Bx = 0.01
    case KH_LOCALIZED: return {2*s.b_peak*s.b_width, 0.0, 0.0};
    default: return {0.0, 0.0, 0.0};
  }
}

// Lower-face fields of cell (i,j,k) (0-based global indices), identical to
// faces_from_potential in problems.jl: Stokes on each face of edge-centred A samples.
KOKKOS_INLINE_FUNCTION double FaceB(const Spec &s, int dir, int i, int j, int k) {
  const double h = 1.0/s.n;
  auto c = [h](int q) {return (q + .5)*h;};
  auto f = [h](int q) {return q*h;};
  Vec b0 = Uniform(s);
  if (dir == 0) {        // (Az(i,j+1) - Az(i,j)) - (Ay(k+1) - Ay(k)), at face x_i
    double dz = PotentialAt(s, f(i), f(j + 1), c(k)).z - PotentialAt(s, f(i), f(j), c(k)).z;
    double dy = PotentialAt(s, f(i), c(j), f(k + 1)).y - PotentialAt(s, f(i), c(j), f(k)).y;
    return (dz - dy)/h + b0.x;
  } else if (dir == 1) { // (Ax(k+1) - Ax(k)) - (Az(i+1) - Az(i)), at face y_j
    double dx = PotentialAt(s, c(i), f(j), f(k + 1)).x - PotentialAt(s, c(i), f(j), f(k)).x;
    double dz = PotentialAt(s, f(i + 1), f(j), c(k)).z - PotentialAt(s, f(i), f(j), c(k)).z;
    return (dx - dz)/h + b0.y;
  }
  // (Ay(i+1) - Ay(i)) - (Ax(j+1) - Ax(j)), at face z_k
  double dy = PotentialAt(s, f(i + 1), c(j), f(k)).y - PotentialAt(s, f(i), c(j), f(k)).y;
  double dx = PotentialAt(s, c(i), f(j + 1), f(k)).x - PotentialAt(s, c(i), f(j), f(k)).x;
  return (dy - dx)/h + b0.z;
}

// Cell averages over sub^2 (planar) or sub^3 samples: density, momentum, thermal + kinetic energy.
struct Cell {double rho, m[3], ehydro;};
KOKKOS_INLINE_FUNCTION Cell CellAverage(const Spec &s, int i, int j, int k) {
  const double h = 1.0/s.n; const int m = s.sub;
  Cell out{0.0, {0.0, 0.0, 0.0}, 0.0};
  double thermal = 0.0, kinetic = 0.0; int count = 0;
  for (int a = 0; a < (s.planar ? 1 : m); ++a) for (int b = 0; b < m; ++b) for (int c = 0; c < m; ++c) {
    double oz = s.planar ? 0.0 : (a + .5)/m - .5, oy = (b + .5)/m - .5, ox = (c + .5)/m - .5;
    Prim q = PrimAt(s, (i + .5 + ox)*h, (j + .5 + oy)*h, (k + .5 + oz)*h);
    out.rho += q.rho; out.m[0] += q.rho*q.vx; out.m[1] += q.rho*q.vy; out.m[2] += q.rho*q.vz;
    thermal += q.p/(s.gamma - 1); kinetic += q.rho*(q.vx*q.vx + q.vy*q.vy + q.vz*q.vz)/2;
    ++count;
  }
  out.rho /= count; for (int d = 0; d < 3; ++d) out.m[d] /= count;
  thermal /= count; kinetic /= count;
  if (s.problem == DECAY) {
    for (int d = 0; d < 3; ++d) out.m[d] = (out.m[d] - s.mean[d])*s.scale;
    kinetic = (out.m[0]*out.m[0] + out.m[1]*out.m[1] + out.m[2]*out.m[2])/(2*out.rho);
  }
  out.ehydro = thermal + kinetic;
  return out;
}

}  // namespace abel
#endif  // ABEL_PROBLEMS_HPP_
