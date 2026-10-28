// Why the tests with a mean field stop at 256^2 and 512^2: is the divergence of a uniform field, projected onto
// MHD-ALE's magnetic space, below the 1e-11 x its norm that DivFreeProject (tools.cpp) needs to keep the projection,
// or does it fall to the mixed solve (singular on a fully periodic mesh)? The mesh as MHD.cpp makes it for the suite:
// the unit square, NE x NE quadrilaterals (N = 2 NE), periodic in x and y, Q2 discontinuous nodes; RT of order 2
// (-oa 3). Three meshes: as made, translated as the comoving rezone (-crz) places it after a remap, and deformed by a
// smooth periodic displacement (det J 0.8-1.2, as the logs show before a remap). Nothing of MHD-ALE is used.
//   srun -n 4 divfree_check
#include "mfem.hpp"
#include <cmath>
#include <cstdio>
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int order_v = 2, order_A = 3;
   const real_t tol = 1e-11;                     // DivFreeProject's shortcut
   if (Mpi::Root()) { printf("%5s %5s %-10s %-11s %12s %12s %12s  %s\n", "N", "NE", "mesh", "field", "div L2", "L2 norm", "ratio", "DivFreeProject"); }
   for (int ne : {16, 32, 64, 128, 256})
   {
      Mesh serial = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, true, 1.0, 1.0);
      std::vector<Vector> t = {Vector({1.0, 0.0}), Vector({0.0, 1.0})};
      Mesh periodic = Mesh::MakePeriodic(serial, serial.CreatePeriodicVertexMapping(t));
      periodic.SetCurvature(order_v, true, -1, Ordering::byNODES);
      ParMesh pmesh(MPI_COMM_WORLD, periodic);
      RT_FECollection rt(order_A-1, 2);
      ParFiniteElementSpace fes(&pmesh, &rt);
      ParGridFunction &nodes = *dynamic_cast<ParGridFunction *>(pmesh.GetNodes());
      const Vector made(nodes);
      const int n = nodes.ParFESpace()->GetNDofs();
      for (const char *mesh : {"as made", "translated", "deformed"})
      {
         nodes = made;
         for (int i = 0; i < n; ++i)
         {
            real_t &x = nodes(i), &y = nodes(n+i);
            if (mesh[0] == 't') { x += 0.0017051947; y += 0.0003129; }
            if (mesh[0] == 'd')
            {
               const real_t x0 = x, y0 = y, a = 0.015, k = 2*M_PI;
               x += a*std::sin(k*x0)*std::cos(k*y0);
               y += a*std::cos(k*x0)*std::sin(k*y0);
            }
         }
         pmesh.NodesUpdated();
         for (const char *field : {"(0.75,0)", "(0.6,0.8)"})
         {
            Vector b0 = field[3] == '7' ? Vector({0.75, 0.0}) : Vector({0.6, 0.8});
            VectorConstantCoefficient b0c(b0);
            ParGridFunction b(&fes);
            b.ProjectCoefficient(b0c);
            ConstantCoefficient zero(0.0);
            Vector z(2); z = 0.0;
            VectorConstantCoefficient zc(z);
            const real_t div = b.ComputeDivError(&zero), norm = b.ComputeL2Error(zc);
            if (Mpi::Root())
            {
               printf("%5d %5d %-10s %-11s %12.3e %12.3e %12.3e  %s\n", 2*ne, ne, mesh, field, div, norm, div/norm,
                      div <= tol*norm ? "keeps the projection" : "FALLS TO THE MIXED SOLVE");
            }
         }
      }
   }
   return 0;
}
