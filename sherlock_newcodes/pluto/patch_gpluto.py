"""Prepare a copy of gPLUTO for the comparison suite (idempotent).

    python3 patch_gpluto.py GPLUTO_DIR

1. Steps land on output times, as every other code of the suite (AthenaK's patch_athenak.py, our GPU driver,
   RAMSES, Athena++): gPLUTO shortens only the last step (to tstop) and writes an output BEFORE the step that would
   cross an output time, so its frames were at slightly earlier times. Now a step that would pass the next output
   time of an output written at a time cadence is shortened to end on it, the time is set to that output time
   exactly after the step, and the output is written when the current time is an output time. Nothing else changes.
2. Config/linux.sherlock.acc.defs: the OpenACC MPI build (linux.mpicxx.acc.defs) for the H100 (cc90), with NVHPC's
   compilers; its HDF5 is off (our outputs are .dbl).
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def edit(path, old, new, done):
    text = path.read_text()
    if done in text:return
    if text.count(old) != 1:sys.exit(f"patch_gpluto.py: {path.name}: expected this text once, found it {text.count(old)} times:\n{old}")
    path.write_text(text.replace(old, new))


def main(root):
    root = Path(root);main_cpp = root/"Src/main.cpp"
    edit(main_cpp, "    if (g_stepNumber == cmd_line.maxsteps && cmd_line.maxsteps >= 0) {\n",
         "    // abel: nor past the next output time of an output written at a time cadence\n"
         "    if (!last_step) {\n"
         "      for (int n_ = 0; n_ < MAX_OUTPUT_TYPES; n_++) {\n"
         "        const double dto = runtime.output[n_].dt;\n"
         "        if (dto <= 0.0) continue;\n"
         "        const double tn = (floor(Dts.time/dto + 1.e-9) + 1.0)*dto;\n"
         "        if (Dts.time + Dts.dt > tn) Dts.dt = tn - Dts.time;\n"
         "      }\n"
         "    }\n"
         "    if (g_stepNumber == cmd_line.maxsteps && cmd_line.maxsteps >= 0) {\n", "abel: nor past the next output time")
    edit(main_cpp, "    Dts.time             += Dts.dt;\n",
         "    Dts.time             += Dts.dt;\n"
         "    // abel: a step shortened to land on an output time ends on it exactly\n"
         "    for (int n_ = 0; n_ < MAX_OUTPUT_TYPES; n_++) {\n"
         "      const double dto = runtime.output[n_].dt;\n"
         "      if (dto <= 0.0) continue;\n"
         "      const double tn = floor(Dts.time/dto + 0.5)*dto;\n"
         "      if (fabs(Dts.time - tn) < 1.e-9*dto) Dts.time = tn;\n"
         "    }\n", "a step shortened to land on an output time")
    edit(main_cpp, "      check_dt = (int) ((t + dt)/output->dt) - (int)(t/output->dt);\n",
         "      // abel: write when the current time is an output time (the steps land on them)\n"
         "      check_dt = fabs(t/output->dt - floor(t/output->dt + 0.5)) < 1.e-9;\n", "write when the current time is an output time")
    shutil.copy(HERE/"linux.sherlock.acc.defs", root/"Config/linux.sherlock.acc.defs")
    print(f"patched {root}")


if __name__ == "__main__":
    if len(sys.argv) != 2:sys.exit(__doc__)
    main(sys.argv[1])
