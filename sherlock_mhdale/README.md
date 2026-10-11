# MHD-ALE in the comparison suite

MHD-ALE is "a structure-preserving, helicity-conserving high-order ALE finite element method for compressible ideal
MHD" on MFEM: the mesh moves with the gas and is remapped from time to time; B is the curl of a vector potential in a
Raviart-Thomas space, so its divergence is zero by construction. MPI on CPUs.

We run **T. Abel's fork**, github.com/yipihey/MHD-ALE at commit `2e900cd4` (2026-10-07), of the original
github.com/ruijie-xi/MHD-ALE (`0e0564ed`). The fork adds what the suite needs: remap in a box periodic in x as well
(`-mst 1`, `-crz`), and a correction that keeps the stretching of a uniform background field across remaps (`-pmf`),
with its own regression suite (`tests/regression.py`). MFEM is the fork its README names
(github.com/ruijie-xi/mfem-develop, `842fac6a`), gslib commit `375eda25`; hypre 2.31.0 and METIS 4.0.3 are our
choice (the README uses system packages). No licence file in either repository.

Nothing of MHD-ALE is in this directory. `fetch.sh` gets it at the pinned commit; `prepare_patch.py` copies it and
adds the suite to the copy (problem 100), without changing the solver.

## What it can run of the suite

In 2D the code has no v_z and no B_z (scalar potential A_z, two velocity components; the fork's own periodic test
checks for this). Of the 17 planar tests, `uniform`, `alfven`, `alfven_oblique` and `field_loop` need one of them and
are not run; the other 13 are (`options.sh`), all in the suite's periodic unit box.

Read in the source, not changed, to ask about: the wave speed for the timestep and the viscosity is
`sqrt(c_s^2 + |B|/sqrt(mu rho))` (`MHD_solver.hpp:133`), the Alfven speed where its square belongs, which is too
small where the field is strong (the magnetised blasts).

## What we add (ours)

- `testcase_Abel.cpp`: the suite's starting state, from `abel_problems.hpp` of the sweep bundle (the functions
  AthenaK and Athena++ use) and the spec file of `tools/write_spec.jl` (`ABEL_SPEC`); mu = 1; a test with v_z or
  B_z is refused. `ABEL_BOX`: `periodic` (used), `xwalls`, `walls`.
- `abel_dump.cpp`: the state as cell averages on the suite's N x N grid (`ABEL_N`), by locating S x S sample points
  per cell (`ABEL_SUB`, 4) in the moving curved elements and averaging the conserved quantities.
- `prepare_patch.py`, edits to `MHD.cpp`: the state is written at t = 0 and at the output times; the step before an
  output time is shortened to end on it, and the step size is restored afterwards.
- `make_mesh.py`: the unit square as N/2 x N/2 elements (velocity of degree 2: N nodes per side).
- `options.sh`: the run options (the fork's own: degrees 2/1/1/3, SSP RK3, CFL 0.5, remap every 10 steps,
  `-mst 1 -crz -pmf`), and the list of tests.
- `mhdale_run.py`: MHD-ALE in the sweep's driver (`codes/external.py`, imported): the run directories
  (`$SCRATCH/mhd-results-sweep/<test>-mhdale-ale-run-N<n>-s42/`), records and measurements are those of the other codes.

## Running it

`check.sh` (on a login node) fetches the sources and submits two jobs:

1. `trial.sbatch`: builds everything; runs the fork's regression suite; compares the starting state of each of the
   13 tests at 64^2 with the suite's own (`ic_check.py`); takes 30 steps with remaps on two ranks for four tests.
   It ends cleanly only if all of that is clean.
2. `mhdale.sbatch`, to start only if the trial passed: 13 array tasks, one per test, each running that test's grids
   (32^2 to 512^2) in turn on up to 16 ranks. MHD-ALE has no restart, so a run that does not fit in 48 hours ends
   its task there.

On the Mac:

    cd /Users/lyxia/Projects/research/abel_mhd && COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - sherlock_mhdale | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_mhdale/check.sh'

and later, as often as wanted:

    ssh lyxia@login.sherlock.stanford.edu 'bash -l /scratch/users/lyxia/mhd-study/sherlock_mhdale/status.sh'

## State of testing

**The first trial ran (job 46917547, 2026-10-07, the original code `0e0564ed` with our additions):** everything built,
our two C++ files included. With walls the rotor's starting state matched the suite's (L1/max 0.001 to 0.002,
mass and energy within 0.3 per cent, every cell sampled 16 times), 30 steps with two remaps ran on two ranks
(0.44 s a step at 64^2), and the four tests with v_z or B_z were refused as intended. Two things came out of it:

- The blast started with 3.5 per cent less energy than the suite's state at 64^2: the code samples the energy at
  nodes, which misplaces the energy of a jump. `testcase_Abel.cpp` now returns the value of the suite's cell that
  holds the node and likewise for the density (`ABEL_POINT_VALUES=1` for point values).
- In the periodic box the original code ended with a segmentation fault at start-up. The fork's `tools.cpp` names a
  cause it removes: the projection of the mean field onto a fully periodic mesh was a singular solve.

**The fork ran (job 46932609, 2026-10-07):** it builds; its regression suite passes ("All 104 checks passed");
the periodic box starts and the starting states of 9 of the 13 tests agree with the suite's at 64^2 (the three
blasts and the rotated Brio-Wu were off in energy by 0.6 to 3.5 per cent, the jump again: fixed as above, not yet
rerun); 25 to 29 steps with two remaps ran on two ranks for Orszag-Tang, rotor, Brio-Wu and the field loop, with
MHD-ALE's own divergence error of B at 1e-13 to 1e-12. Those took 0.6 to 2.6 s a step at 64^2 on two ranks, with
2 to 6 of 31 attempted steps repeated. That trial was marked not clean for the energies and, wrongly, for the step
count (the step limit counts repeated attempts; the trial's test now allows for it), so no production run started.

`ic_check.py` gives OK, CHECK (beyond the tight tolerances; the trial goes on) or BAD (a gross difference; the trial
stops). The runner and the production job have only been run here, with stand-in executables.
