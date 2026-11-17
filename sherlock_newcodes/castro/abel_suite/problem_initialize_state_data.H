#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>
#include "abel_spec.H"

// The suite's cell averages (abel::CellAverage: density, momentum, thermal + kinetic energy over sub^2 or sub^3 samples),
// as AthenaK, Athena++ and Cholla get them. Castro adds the magnetic energy to UEDEN itself after this.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k,
                                    Array4<Real> const& state,
                                    const GeometryData& geomdata)
{
    amrex::ignore_unused(geomdata);
    const abel::Spec& s = *abel_spec;
    const abel::Cell c = abel::CellAverage(s, i, j, s.planar ? 0 : k);
    const Real kin = 0.5_rt * (c.m[0] * c.m[0] + c.m[1] * c.m[1] + c.m[2] * c.m[2]) / c.rho;

    state(i,j,k,URHO) = c.rho;
    state(i,j,k,UMX) = c.m[0];
    state(i,j,k,UMY) = c.m[1];
    state(i,j,k,UMZ) = c.m[2];
    state(i,j,k,UEDEN) = c.ehydro;
    state(i,j,k,UEINT) = c.ehydro - kin;

    eos_t eos_state;
    eos_state.rho = c.rho;
    eos_state.e = (c.ehydro - kin) / c.rho;
    eos_state.T = 1.0_rt;
    for (int n = 0; n < NumSpec; n++) { eos_state.xn[n] = 0.0_rt; }
    eos_state.xn[0] = 1.0_rt;
    eos(eos_input_re, eos_state);
    state(i,j,k,UTEMP) = eos_state.T;

    for (int n = 0; n < NumSpec; n++) { state(i,j,k,UFS+n) = 0.0_rt; }
    state(i,j,k,UFS) = c.rho;
}

#endif
