/***********************************************************************
/
/  INITIALIZE THE COMPARISON SUITE'S TESTS (abel; ProblemType = 777)
/
/  PURPOSE:  The starting state of the spec file in $ABEL_SPEC (tools/write_spec.jl) on the periodic unit
/            square or cube, from the suite's shared port abel_problems.hpp (grid::AbelSuiteInitializeGrid).
/            Field labels as MHDOrszagTangInit.C, for both MHD solvers (MHD_RK: Dedner; MHD_Li: CT).
/
************************************************************************/
#include "preincludes.h"
#include "macros_and_parameters.h"
#include "typedefs.h"
#include "global_data.h"
#include "Fluxes.h"
#include "GridList.h"
#include "ExternalBoundary.h"
#include "Grid.h"
#include "Hierarchy.h"
#include "TopGridData.h"

int MHDCTSetupFieldLabels();

int AbelSuiteInitialize(FILE *fptr, FILE *Outfptr, HierarchyEntry &TopGrid, TopGridData &MetaData)
{
  int count = 0;
  DataLabel[count++] = (char*) "Density";
  if (EquationOfState == 0) DataLabel[count++] = (char*) "TotalEnergy";
  if (DualEnergyFormalism) DataLabel[count++] = (char*) "GasEnergy";
  DataLabel[count++] = (char*) "x-velocity";
  DataLabel[count++] = (char*) "y-velocity";
  DataLabel[count++] = (char*) "z-velocity";
  if (UseMHD) {
    DataLabel[count++] = (char*) "Bx";
    DataLabel[count++] = (char*) "By";
    DataLabel[count++] = (char*) "Bz";
  }
  if (HydroMethod == MHD_RK) DataLabel[count++] = (char*) "Phi";
  for (int i = 0; i < count; i++) DataUnits[i] = NULL;
  if (UseMHDCT) MHDCTSetupFieldLabels();
  if (TopGrid.GridData->AbelSuiteInitializeGrid() == FAIL) ENZO_FAIL("abel: AbelSuiteInitializeGrid failed");
  return SUCCESS;
}
