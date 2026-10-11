// Sampling of the MHD-ALE state on the comparison suite's uniform grid (ours; see abel_dump.cpp).
#ifndef ABEL_DUMP_HPP
#define ABEL_DUMP_HPP
#include "mfem.hpp"

namespace mfem
{
void AbelDump(int frame, real_t t, int steps, ParMesh &pmesh, ParGridFunction &rho_gf, ParGridFunction &v_gf,
              ParGridFunction &e_gf, ParGridFunction &B_gf, const ParGridFunction &gamma_gf, real_t mu,
              const char *outputdir);
}
#endif
