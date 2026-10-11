// The MHD-ALE state as cell averages on the comparison suite's uniform N x N grid of the unit box (ours).
//
// MHD-ALE's mesh moves with the gas and its fields are polynomials on curved elements, so the state is sampled where
// the suite's cells are: every cell holds S x S sample points on a regular sub-grid (ABEL_N, ABEL_SUB; S = 4 unless
// set), each point is located in the element that contains it by inverting the element map, the fields are evaluated
// there, and a cell's value is the mean of its samples. Density, momentum, total energy and B are averaged (the
// conserved quantities), not the primitives. Elements that have drifted out of the unit box are found by their own
// coordinates, so a periodic box needs no special treatment: a sample at x + k is the sample at x.
//
// Written per frame to <outputdir>/abel_frame_NNN.bin: N*N*9 doubles (rho, mx, my, mz, E, Bx, By, Bz, psi), x
// fastest, then y, then the field (the suite's layout); mz, Bz and psi are zero in a 2D run. abel_frame_NNN.txt
// holds the time, the step count and how the sampling went: a sample found in no element is left out of its
// cell's mean and counted, one found in two elements (on an edge) counts twice.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "abel_dump.hpp"

namespace mfem
{

void AbelDump(int frame, real_t t, int steps, ParMesh &pmesh, ParGridFunction &rho_gf, ParGridFunction &v_gf,
              ParGridFunction &e_gf, ParGridFunction &B_gf, const ParGridFunction &gamma_gf, real_t mu,
              const char *outputdir)
{
   const char *env_n = std::getenv("ABEL_N");
   if (!env_n) { return; }                               // no grid named: nothing to write
   const int N = std::atoi(env_n);
   const char *env_s = std::getenv("ABEL_SUB");
   const int S = env_s ? std::atoi(env_s) : 4;
   MFEM_VERIFY(N > 0 && S > 0 && pmesh.Dimension() == 2, "abel dump: needs ABEL_N > 0 and a 2D mesh");
   const int M = N * S;                                  // sample points per direction
   const int nf = 9;
   std::vector<double> sum((size_t) N * N * nf, 0.0), cnt((size_t) N * N, 0.0);
   double tried = 0.0, unconverged = 0.0;

   Vector pt(2), vv(2), bb(2);
   IntegrationPoint ip;
   for (int el = 0; el < pmesh.GetNE(); el++)
   {
      ElementTransformation *T = pmesh.GetElementTransformation(el);
      IsoparametricTransformation *iso = dynamic_cast<IsoparametricTransformation *>(T);
      MFEM_VERIFY(iso, "abel dump: element transformation is not isoparametric");
      const DenseMatrix &pm = iso->GetPointMat();        // node coordinates, one column per node
      double lo[2] = {1e300, 1e300}, hi[2] = {-1e300, -1e300};
      for (int k = 0; k < pm.Width(); k++) for (int d = 0; d < 2; d++)
      {
         lo[d] = fmin(lo[d], pm(d, k)); hi[d] = fmax(hi[d], pm(d, k));
      }
      // A curved edge can bulge past its nodes: widen the box of the nodes by a tenth of its size.
      for (int d = 0; d < 2; d++) { const double w = 0.1 * (hi[d] - lo[d]) + 1e-12; lo[d] -= w; hi[d] += w; }
      const long i0 = (long) ceil(lo[0] * M - 0.5), i1 = (long) floor(hi[0] * M - 0.5);
      const long j0 = (long) ceil(lo[1] * M - 0.5), j1 = (long) floor(hi[1] * M - 0.5);
      const double gamma = gamma_gf(el);
      for (long j = j0; j <= j1; j++) for (long i = i0; i <= i1; i++)
      {
         pt(0) = (i + 0.5) / M; pt(1) = (j + 0.5) / M;
         tried += 1.0;
         const int res = T->TransformBack(pt, ip);
         if (res != InverseElementTransformation::Inside)
         {
            if (res != InverseElementTransformation::Outside) { unconverged += 1.0; }
            continue;
         }
         T->SetIntPoint(&ip);
         const double rho = rho_gf.GetValue(*T, ip);
         const double e = e_gf.GetValue(*T, ip);
         v_gf.GetVectorValue(*T, ip, vv);
         B_gf.GetVectorValue(*T, ip, bb);
         const long ci = (((i % M) + M) % M) / S, cj = (((j % M) + M) % M) / S;
         const size_t c = (size_t) ci + (size_t) N * (size_t) cj;
         const double b2 = bb(0) * bb(0) + bb(1) * bb(1), v2 = vv(0) * vv(0) + vv(1) * vv(1);
         sum[c + 0 * (size_t) N * N] += rho;
         sum[c + 1 * (size_t) N * N] += rho * vv(0);
         sum[c + 2 * (size_t) N * N] += rho * vv(1);
         sum[c + 4 * (size_t) N * N] += rho * e + 0.5 * rho * v2 + 0.5 * b2 / mu;
         sum[c + 5 * (size_t) N * N] += bb(0) / sqrt(mu);
         sum[c + 6 * (size_t) N * N] += bb(1) / sqrt(mu);
         cnt[c] += 1.0;
         (void) gamma;
      }
   }

   MPI_Comm comm = pmesh.GetComm();
   int rank; MPI_Comm_rank(comm, &rank);
   std::vector<double> gsum(rank == 0 ? sum.size() : 1), gcnt(rank == 0 ? cnt.size() : 1);
   double loc[2] = {tried, unconverged}, glb[2] = {0.0, 0.0};
   MPI_Reduce(sum.data(), gsum.data(), (int) sum.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
   MPI_Reduce(cnt.data(), gcnt.data(), (int) cnt.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
   MPI_Reduce(loc, glb, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
   if (rank != 0) { return; }

   double cmin = 1e300, cmax = 0.0, missing = 0.0, twice = 0.0, total[nf] = {0.0};
   long empty = 0;
   for (size_t c = 0; c < (size_t) N * N; c++)
   {
      cmin = fmin(cmin, gcnt[c]); cmax = fmax(cmax, gcnt[c]);
      if (gcnt[c] < S * S) { missing += S * S - gcnt[c]; }
      if (gcnt[c] > S * S) { twice += gcnt[c] - S * S; }
      if (gcnt[c] == 0.0) { empty++; continue; }
      for (int f = 0; f < nf; f++) { gsum[c + f * (size_t) N * N] /= gcnt[c]; total[f] += gsum[c + f * (size_t) N * N]; }
   }
   char name[1024];
   snprintf(name, sizeof(name), "%s/abel_frame_%03d.bin", outputdir, frame);
   FILE *f = fopen(name, "wb");
   MFEM_VERIFY(f, "abel dump: cannot write " << name);
   fwrite(gsum.data(), sizeof(double), gsum.size(), f);
   fclose(f);
   snprintf(name, sizeof(name), "%s/abel_frame_%03d.txt", outputdir, frame);
   f = fopen(name, "w");
   MFEM_VERIFY(f, "abel dump: cannot write " << name);
   fprintf(f, "frame %d\ntime %.17g\nsteps %d\nn %d\nsub %d\nsamples_per_cell_min %g\nsamples_per_cell_max %g\n"
              "samples_missing %g\nsamples_twice %g\ncells_empty %ld\ninversions_tried %g\ninversions_unconverged %g\n",
           frame, (double) t, steps, N, S, cmin, cmax, missing, twice, empty, glb[0], glb[1]);
   const char *names[nf] = {"rho", "mx", "my", "mz", "E", "Bx", "By", "Bz", "psi"};
   for (int k = 0; k < nf; k++) { fprintf(f, "mean_%s %.17g\n", names[k], total[k] / ((double) N * N)); }
   fclose(f);
   printf("abel dump: frame %d at t = %.17g, step %d: samples per cell %g to %g, %g missing, %ld empty cells\n",
          frame, (double) t, steps, cmin, cmax, missing, empty);
}

}
