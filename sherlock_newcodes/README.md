# Five more codes: AthenaPK, Cholla, gPLUTO, Enzo (and Phantom, next)

lyxia, 2026-10-29: "do all of it and send it on sherlock". Every code runs the suite's tests from the suite's own
starting state (`abel_problems.hpp`, as AthenaK and Athena++) through the sweep's driver (`newcodes_run.py`, importing
`sherlock_sweep/codes/external.py`), so records, frames and measurements are those of every other code.

| code | licence | where | set-ups (`newcodes_run.py` VARIANTS) |
|---|---|---|---|
| AthenaPK (Parthenon, Kokkos) `6d31f708` | BSD-3 | one H100 | `matched` PLM/HLLE/VL2, `best` PPM/HLLD/RK3; GLM cleaning, c_h from the step |
| Cholla `191e59b0` | MIT | one H100 | `plmc`, `ppmc`: HLLD, Van Leer, CT (its only MHD integrator and solver); planar tests on N x N x 8 |
| gPLUTO `870aa807` | none stated ("freely distributed") | one H100 (OpenACC) | `ct_matched`, `ct_best`, `glm_matched`, `glm_best`: LINEAR/RK2/HLL or PARABOLIC/RK3/HLLD, CT or GLM |
| Enzo `45f4c64b` | NCSA/Illinois | CPUs | `dedner` (HydroMethod 4: Wang & Abel, HLL, PLM, RK2), `ct` (HydroMethod 6: Collins et al., HLLD, PLM) |
| Phantom `f535c074` | GPL-3 (+ keep the name) | CPUs | SPH with cleaning; set-up and particle-to-grid not written yet |

gPLUTO has no licence file: its source stays out of this repository, as MHD-ALE's; our changes are applied at build
time. Our changes to each code (patch_*.py; every edit anchored, failing loudly):

- all: the suite's starting state (AthenaPK problem `abel_suite`, Cholla `init=Abel`, gPLUTO `init.cpp`, Enzo
  ProblemType 777), from `$ABEL_SPEC` (tools/write_spec.jl) or the input file;
- steps that land on output times, as every other code of the suite: Parthenon (AthenaPK) clipped only at tlim,
  gPLUTO wrote before the step crossing an output time, Enzo overshot by 1.0001; Cholla already lands.
  Nothing in any solver changes.

## Run it

Send (Mac, project directory), then on Sherlock: fetch, build job (CPU node, 4 h), trial job (one H100, after the build):

    COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - sherlock_newcodes | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_newcodes/submit.sh'

Status: `bash -l /scratch/users/lyxia/mhd-study/sherlock_newcodes/status.sh`. The trial checks every set-up's starting
state against the suite's (4 tests) and Orszag-Tang at 64^2 against AthenaK; production is sized from its timings.
