#ifndef problem_initialize_mhd_data_H
#define problem_initialize_mhd_data_H

#include "abel_spec.H"

// Face fields from the vector potential (abel::FaceB: Stokes on each face of edge-centred A, plus the mean field),
// identical to problems.jl's faces_from_potential: div B = 0 to round-off. B(i,j,k) of direction ibdir is the field on
// the lower face of cell (i,j,k), as Castro's face arrays are indexed.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_mhd_data (int i, int j, int k,
                                  Array4<Real> const& B, const int ibdir,
                                  const GeometryData& geomdata)
{
    amrex::ignore_unused(geomdata);
    B(i,j,k) = abel::FaceB(*abel_spec, ibdir, i, j, k);
}
#endif
