"""Prepare an Enzo checkout for the comparison suite (idempotent).

    python3 patch_enzo.py ENZO_DIR ABEL_PROBLEMS_HPP

Adds ProblemType = 777 (no other problem uses it): AbelSuiteInitialize.C and Grid_AbelSuiteInitializeGrid.C, with
the suite's abel_problems.hpp, in src/enzo/, registered in sources.cmake, Grid.h and InitializeNew.C. The starting
state is that of the spec file in $ABEL_SPEC (tools/write_spec.jl).

Steps land on dump times, as every other code of the suite: Enzo shortens a step that would pass a dump time to
1.0001 times the remaining interval (EvolveHierarchy.C) and dumps at the first time at or past it (CheckForOutput.C),
so its frames were up to 1e-4 of a step late, more than the driver's 1e-6 of an output interval. Now the step ends
on the dump time and the dump is written within round-off of it. Nothing in the solvers changes.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def edit(path, old, new, done):
    text = path.read_text()
    if done in text:return
    if text.count(old) != 1:sys.exit(f"patch_enzo.py: {path.name}: expected this text once, found it {text.count(old)} times:\n{old}")
    path.write_text(text.replace(old, new))


def main(root, header):
    src = Path(root)/"src/enzo"
    for f in ("AbelSuiteInitialize.C", "Grid_AbelSuiteInitializeGrid.C"):shutil.copy(HERE/f, src/f)
    shutil.copy(header, src/"abel_problems.hpp")
    edit(src/"sources.cmake", "    MHDOrszagTangInit.C\n", "    MHDOrszagTangInit.C\n    AbelSuiteInitialize.C\n    Grid_AbelSuiteInitializeGrid.C\n", "AbelSuiteInitialize.C")
    edit(src/"Grid.h", "  int MHDOrszagTangInitGrid(float Density,float Pressure, float V0, float B0 );\n",
         "  int MHDOrszagTangInitGrid(float Density,float Pressure, float V0, float B0 );\n"
         "  int AbelSuiteInitializeGrid();  // abel: the comparison suite (ProblemType 777)\n", "AbelSuiteInitializeGrid")
    edit(src/"InitializeNew.C", "int MHDOrszagTangInit(FILE *fptr, FILE *Outfptr, HierarchyEntry &TopGrid,\n",
         "int AbelSuiteInitialize(FILE *fptr, FILE *Outfptr, HierarchyEntry &TopGrid, TopGridData &MetaData);  // abel\n"
         "int MHDOrszagTangInit(FILE *fptr, FILE *Outfptr, HierarchyEntry &TopGrid,\n", "int AbelSuiteInitialize(")
    edit(src/"InitializeNew.C", "  if (ProblemType == 104) \n    ret = MHDLoopInit(fptr, Outfptr, TopGrid, MetaData, Exterior);\n",
         "  if (ProblemType == 104) \n    ret = MHDLoopInit(fptr, Outfptr, TopGrid, MetaData, Exterior);\n"
         "  /* 777) abel: the comparison suite */\n  if (ProblemType == 777)\n"
         "    ret = AbelSuiteInitialize(fptr, Outfptr, TopGrid, MetaData);\n", "ProblemType == 777")
    edit(src/"EvolveHierarchy.C", "        dt = min(1.0001*(MetaData.TimeLastDataDump + MetaData.dtDataDump -\n              MetaData.Time), dt);\n",
         "        // abel: end the step on the dump time (was 1.0001 times the remaining interval)\n"
         "        dt = min(1.0*(MetaData.TimeLastDataDump + MetaData.dtDataDump -\n              MetaData.Time), dt);\n", "abel: end the step on the dump time")
    edit(src/"CheckForOutput.C", "  if (MetaData.Time >= MetaData.TimeLastDataDump + MetaData.dtDataDump\n      && MetaData.dtDataDump > 0.0) {\n",
         "  // abel: a step that ended on the dump time, within round-off\n"
         "  if (MetaData.Time >= MetaData.TimeLastDataDump + MetaData.dtDataDump - 1e-10*MetaData.dtDataDump\n"
         "      && MetaData.dtDataDump > 0.0) {\n", "a step that ended on the dump time")
    edit(src/"InitializeNew.C", "    MetaData.TimeLastDataDump = MetaData.Time - MetaData.dtDataDump*1.00001;\n",
         "    // abel: the first dump due exactly now (the factor 1.00001 put every later dump 1e-5 of an interval early)\n"
         "    MetaData.TimeLastDataDump = MetaData.Time - MetaData.dtDataDump;\n", "the first dump due exactly now")
    print(f"patched {root}")


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
