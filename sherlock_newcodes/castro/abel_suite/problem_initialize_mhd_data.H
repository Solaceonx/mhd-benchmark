#ifndef problem_initialize_mhd_data_H
#define problem_initialize_mhd_data_H

#include "abel_spec.H"

// Face fields from the vector potential: Stokes on each face of edge-centred A, plus the mean field, as abel::FaceB and
// problems.jl's faces_from_potential: div B = 0 to round-off. B(i,j,k) of direction ibdir is the field on the lower face
// of cell (i,j,k), as Castro's face arrays are indexed.
//
// The potential is evaluated through one function that is never inlined. Inlined (abel::FaceB), the GPU compiler may
// contract the two potential evaluations of a face differently, so that A_z(x_i+1, y) - A_z(x_i, y) is a few ulp instead
// of 0 where A depends on y alone; Castro's start-up check (Castro::check_div_B: |div B| <= 1e-10 |B| in every cell) then
// stops kh_localized, whose field falls to ~1e-6 of its peak between the layers (production 52860617, tasks 3 and 6,
// 2026-11-18). One compiled body gives bit-identical values for equal arguments: the same face fields to round-off, with
// div B exactly 0 for kh_localized.
static AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE
abel::Vec abel_potential (const abel::Spec& s, double x, double y, double z)
{
    return abel::PotentialAt(s, x, y, z);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
double abel_face_b (const abel::Spec& s, int dir, int i, int j, int k)
{
    const double h = 1.0/s.n;
    auto c = [h](int q) {return (q + .5)*h;};
    auto f = [h](int q) {return q*h;};
    const abel::Vec b0 = abel::Uniform(s);
    if (dir == 0) {
        double dz = abel_potential(s, f(i), f(j + 1), c(k)).z - abel_potential(s, f(i), f(j), c(k)).z;
        double dy = abel_potential(s, f(i), c(j), f(k + 1)).y - abel_potential(s, f(i), c(j), f(k)).y;
        return (dz - dy)/h + b0.x;
    } else if (dir == 1) {
        double dx = abel_potential(s, c(i), f(j), f(k + 1)).x - abel_potential(s, c(i), f(j), f(k)).x;
        double dz = abel_potential(s, f(i + 1), f(j), c(k)).z - abel_potential(s, f(i), f(j), c(k)).z;
        return (dx - dz)/h + b0.y;
    }
    double dy = abel_potential(s, f(i + 1), c(j), f(k)).y - abel_potential(s, f(i), c(j), f(k)).y;
    double dx = abel_potential(s, c(i), f(j + 1), f(k)).x - abel_potential(s, c(i), f(j), f(k)).x;
    return (dy - dx)/h + b0.z;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_mhd_data (int i, int j, int k,
                                  Array4<Real> const& B, const int ibdir,
                                  const GeometryData& geomdata)
{
    amrex::ignore_unused(geomdata);
    B(i,j,k) = abel_face_b(*abel_spec, ibdir, i, j, k);
}
#endif
