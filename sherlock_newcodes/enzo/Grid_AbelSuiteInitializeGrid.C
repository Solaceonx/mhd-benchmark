/***********************************************************************
/
/  GRID CLASS (INITIALIZE THE COMPARISON SUITE'S TESTS; abel)
/
/  PURPOSE:  The starting state of the spec file in $ABEL_SPEC (tools/write_spec.jl), from the suite's shared
/            port abel_problems.hpp, as AthenaK's abel_suite.cpp builds it: face fields of the vector potential
/            (MagneticField, the lower face of cell i at index i), the cell-centred field as their mean
/            (CenterMagneticField), and cell averages of rho, momentum and thermal + kinetic energy. As
/            Grid_MHDOrszagTangInitGrid.C, MHD_RK (Dedner) starts from the same face fields, turned on for the
/            set-up only. Every cell, ghosts included, from the periodic global index of its centre.
/
************************************************************************/
#include <cmath>
#include <cstdlib>

#include "preincludes.h"
#include "macros_and_parameters.h"
#include "typedefs.h"
#include "global_data.h"
#include "Fluxes.h"
#include "GridList.h"
#include "ExternalBoundary.h"
#include "fortran.def"
#include "Grid.h"
#include "abel_problems.hpp"

int FindField(int field, int farray[], int numfields);

int grid::AbelSuiteInitializeGrid()
{
  NumberOfBaryonFields = 0;
  FieldType[NumberOfBaryonFields++] = Density;
  if (DualEnergyFormalism) FieldType[NumberOfBaryonFields++] = InternalEnergy;
  if (EquationOfState == 0) FieldType[NumberOfBaryonFields++] = TotalEnergy;
  FieldType[NumberOfBaryonFields++] = Velocity1;
  FieldType[NumberOfBaryonFields++] = Velocity2;
  FieldType[NumberOfBaryonFields++] = Velocity3;
  if (UseMHD) {
    FieldType[NumberOfBaryonFields++] = Bfield1;
    FieldType[NumberOfBaryonFields++] = Bfield2;
    FieldType[NumberOfBaryonFields++] = Bfield3;
  }
  if (HydroMethod == MHD_RK) FieldType[NumberOfBaryonFields++] = PhiField;
  if (ProcessorNumber != MyProcessorNumber) return SUCCESS;
  if (!UseMHD) ENZO_FAIL("abel: the suite needs an MHD solver (HydroMethod 4 or 6)");

  const char *path = getenv("ABEL_SPEC");
  if (path == NULL) ENZO_FAIL("abel: ABEL_SPEC is not set");
  const abel::Spec spec = abel::ReadSpec(path);
  const int n = spec.n;
  if (fabs(Gamma - spec.gamma) > 1e-12) ENZO_FAIL("abel: Gamma differs from the spec's");

  if (HydroMethod == MHD_RK) {
    UseMHDCT = TRUE;
    MHD_SetupDims();
  }
  this->AllocateGrids();

  double dx[3];
  for (int d = 0; d < 3; d++)
    dx[d] = d < GridRank ? (GridRightEdge[d] - GridLeftEdge[d])/(GridDimension[d] - 2*NumberOfGhostZones) : 1.0;
  // periodic global index of local cell i along d (its centre); the lower face of that cell has the same index
  auto cell = [&](int d, int i) {
    if (d >= GridRank || (d == 2 && spec.planar)) return 0;
    const double x = GridLeftEdge[d] + (i - GridStartIndex[d] + 0.5)*dx[d] - DomainLeftEdge[d];
    const int q = static_cast<int>(std::floor(x*n));
    return ((q % n) + n) % n;
  };

  for (int f = 0; f < 3; f++)
    for (int k = 0; k < MagneticDims[f][2]; k++)
      for (int j = 0; j < MagneticDims[f][1]; j++)
        for (int i = 0; i < MagneticDims[f][0]; i++) {
          const int index = i + MagneticDims[f][0]*(j + MagneticDims[f][1]*k);
          MagneticField[f][index] = abel::FaceB(spec, f, cell(0, i), cell(1, j), cell(2, k));
        }
  if (this->CenterMagneticField() == FAIL) ENZO_FAIL("abel: CenterMagneticField failed");

  int DensNum, GENum, TENum, Vel1Num, Vel2Num, Vel3Num, B1Num, B2Num, B3Num;
  IdentifyPhysicalQuantities(DensNum, GENum, Vel1Num, Vel2Num, Vel3Num, TENum, B1Num, B2Num, B3Num);
  const int PhiNum = HydroMethod == MHD_RK ? FindField(PhiField, FieldType, NumberOfBaryonFields) : -1;
  for (int k = 0; k < GridDimension[2]; k++)
    for (int j = 0; j < GridDimension[1]; j++)
      for (int i = 0; i < GridDimension[0]; i++) {
        const int index = i + GridDimension[0]*(j + GridDimension[1]*k);
        const abel::Cell c = abel::CellAverage(spec, cell(0, i), cell(1, j), cell(2, k));
        const double bx = BaryonField[B1Num][index], by = BaryonField[B2Num][index], bz = BaryonField[B3Num][index];
        const double kinetic = 0.5*(c.m[0]*c.m[0] + c.m[1]*c.m[1] + c.m[2]*c.m[2])/c.rho;
        BaryonField[DensNum][index] = c.rho;
        BaryonField[Vel1Num][index] = c.m[0]/c.rho;
        BaryonField[Vel2Num][index] = c.m[1]/c.rho;
        BaryonField[Vel3Num][index] = c.m[2]/c.rho;
        if (EquationOfState == 0) BaryonField[TENum][index] = (c.ehydro + 0.5*(bx*bx + by*by + bz*bz))/c.rho;
        if (DualEnergyFormalism) BaryonField[GENum][index] = (c.ehydro - kinetic)/c.rho;
        if (PhiNum >= 0) BaryonField[PhiNum][index] = 0.0;
      }

  if (HydroMethod == MHD_RK) {
    UseMHDCT = FALSE;
    for (int f = 0; f < 3; f++) {
      delete [] MagneticField[f];
      delete [] ElectricField[f];
      MagneticField[f] = NULL;
      ElectricField[f] = NULL;
    }
  }
  return SUCCESS;
}
