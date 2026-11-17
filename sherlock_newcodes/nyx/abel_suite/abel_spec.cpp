#include <cstdlib>
#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include "abel_spec.H"

AMREX_GPU_MANAGED abel::Spec* abel_spec = nullptr;

void abel_read_spec ()
{
    const char* path = std::getenv("ABEL_SPEC");
    if (path == nullptr) { amrex::Abort("abel: set ABEL_SPEC to the spec file written by tools/write_spec.jl"); }
    const abel::Spec s = abel::ReadSpec(path);
    amrex::ParmParse pp("amr");
    amrex::Vector<int> n_cell;
    pp.getarr("n_cell", n_cell);
    if (n_cell[0] != s.n || n_cell[1] != s.n || (!s.planar && n_cell[2] != s.n)) {
        amrex::Abort("abel: amr.n_cell does not match the spec's N (planar tests: N N slab; 3-D: N N N)");
    }
    if (abel_spec == nullptr) {
        void* p = amrex::The_Managed_Arena()->alloc(sizeof(abel::Spec));
        abel_spec = new (p) abel::Spec(s);
    } else {
        *abel_spec = s;
    }
    amrex::Print() << "abel: problem " << s.problem << ", N " << s.n << ", sub " << s.sub << ", planar " << s.planar
                   << ", gamma " << s.gamma << "\n";
}
