#include "Nyx.H"
#include "Prob.H"

// No refinement (amr.max_level = 0): the tags are never used.
void prob_errtags_default(amrex::Vector<amrex::AMRErrorTag>& /*errtags*/)
{
}
