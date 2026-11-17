// The comparison suite's tests as a Nyx problem (ours). Initial state: the suite's cell averages (abel::CellAverage:
// density, momentum, thermal + kinetic energy over sub^2 or sub^3 samples), from the spec file named by ABEL_SPEC, as
// AthenaK, Athena++, Cholla and Castro get them. Nyx is a hydrodynamics code: only tests in which the magnetic field is
// uniform (it then exerts no force: uniform, entropy), zero (blast_*_b0) or negligible (kh_uniform, beta = 2e5) are run.
#include "Nyx.H"
#include "Prob.H"
#include "abel_spec.H"

void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& /*prob_param*/)
{
    if (abel_spec == nullptr) { abel_read_spec(); }
}

#ifndef NO_HYDRO
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata_state(const int i,
                         const int j,
                         const int k,
                         amrex::Array4<amrex::Real> const& state,
                         amrex::GeometryData const& /*geomdata*/,
                         const amrex::GpuArray<amrex::Real,max_prob_param>& /*prob_param*/)
{
    const abel::Spec& s = *abel_spec;
    const abel::Cell c = abel::CellAverage(s, i, j, s.planar ? 0 : k);
    const amrex::Real kin = 0.5 * (c.m[0] * c.m[0] + c.m[1] * c.m[1] + c.m[2] * c.m[2]) / c.rho;
    state(i, j, k, Density_comp) = c.rho;
    state(i, j, k, Xmom_comp) = c.m[0];
    state(i, j, k, Ymom_comp) = c.m[1];
    state(i, j, k, Zmom_comp) = c.m[2];
    state(i, j, k, Eden_comp) = c.ehydro;
    state(i, j, k, Eint_comp) = c.ehydro - kin;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata(const int i,
                   const int j,
                   const int k,
                   amrex::Array4<amrex::Real> const& state,
                   amrex::Array4<amrex::Real> const& diag_eos,
                   amrex::GeometryData const& geomdata,
                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
{
    prob_initdata_state(i, j, k, state, geomdata, prob_param);
    const int JH = 1;
    const int JHe = 1;
    const amrex::Real h_species_in = prob_param[h_species_comp];
    const amrex::Real gamma_minus_1_in = prob_param[gamma_comp] - 1.0;
    const amrex::Real a = 1.0 / (prob_param[z_in_comp] + 1.0);
    const amrex::Real eint = state(i, j, k, Eint_comp) / state(i, j, k, Density_comp);
    nyx_eos_T_given_Re_device(NULL, gamma_minus_1_in, h_species_in, JH, JHe,
                              &diag_eos(i, j, k, Temp_comp), &diag_eos(i, j, k, Ne_comp),
                              state(i, j, k, Density_comp), eint, a);
    if (diag_eos.nComp() > 2) { diag_eos(i, j, k, Zhi_comp) = 7.5; }
}

void prob_initdata_on_box(const Box& bx,
                          Array4<amrex::Real> const& state,
                          Array4<amrex::Real> const& diag_eos,
                          GeometryData const& geomdata,
                          const GpuArray<Real,max_prob_param>& prob_param)
{
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        prob_initdata(i, j, k, state, diag_eos, geomdata, prob_param);
    });
}

void prob_initdata_state_on_box(const Box& bx,
                                Array4<amrex::Real> const& state,
                                GeometryData const& geomdata,
                                const GpuArray<Real,max_prob_param>& prob_param)
{
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        prob_initdata_state(i, j, k, state, geomdata, prob_param);
    });
}
#endif
