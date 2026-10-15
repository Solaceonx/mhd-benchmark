# Driven turbulence: the pilot

Hydrodynamic turbulence in a periodic box, stirred continuously until it is statistically steady, run by two codes
that share one definition of the stirring: Tom Abel's GPU kernels and RAMSES. The pilot is small on purpose (two Mach
numbers, two grids). It is there to show that the driving, the steady-state check and the measurements work on two
codes before the full matrix (Mach 20 to 50, subsonic, compressive driving, more codes) is run.

Nothing here changes the sweep bundle, Tom's file or the RAMSES source tree. Results go to
`$SCRATCH/mhd-results-turb`.

**Read first: RAMSES's driving is not solenoidal as released.** The pilot found a fault in the Helmholtz projection
of RAMSES's turbulence module; with `comp_frac = 0` a fifth of the power of its force is compressive. RAMSES therefore
runs twice here, as released (`ramses-*`) and with the three lines corrected (`ramsesfix-*`). Only `ramsesfix-*`
shares its driving with Tom's kernels. See "The fault in RAMSES's projection" below.

## How to run it

On a Sherlock login node (the Mac command in the conversation unpacks the bundle and runs this):

    bash -l sherlock_turb/check.sh      # build, 32^3 trial, submit; safe to run again
    bash -l sherlock_turb/status.sh     # queue, driving strengths, table of every run, logs that report trouble

`check.sh` builds RAMSES with its driving module, runs the whole RAMSES chain once on a 32^3 box, and submits:

1. `turb-cal` (16 cores, under an hour): finds the driving strength for each target Mach number.
2. `turb-gpu` (one H100, under an hour of running): Tom's kernels, after the calibration.
3. eight `turb-ramses-*` jobs (RAMSES as released) and eight `turb-ramsesfix-*` jobs (projection corrected), after
   the calibration: 128^3 on 32 cores (15 to 20 minutes each), 256^3 on 64 cores (one to two hours each; time limits
   are set far above that).

Run again, `check.sh` submits only what is neither finished nor queued, and a RAMSES run that was cut short continues
from its last output. Each set of eight RAMSES runs took about 430 core-hours, all GPU runs 18 minutes. While a
256^3 RAMSES run is going it holds up to 50 GB of its own output, which is deleted when the run has been measured;
what stays is about 0.6 GB per 256^3 run.

## The set-up

Units: box length 1, mean density 1, sound speed 1. Start: uniform gas at rest.

**Nearly isothermal gas.** Tom's kernels have no isothermal equation of state, so every code runs an ideal gas with
gamma - 1 = 1e-5 (exactly the single-precision number nearest to 1.00001, the same in both codes). The internal
energy is then 1e5 times the pressure, and all the heat a Mach 10 run dissipates in ten driving times warms the gas
by under one per cent. No cooling step is needed, and because nothing removes energy the total energy rises at
exactly the rate the driving puts it in: that slope is the dissipation rate of a steady run, measured the same way
in every code. `cs_end` in the table shows how isothermal each run stayed.

One consequence: Tom's Float16 kernels cannot run this gas. They store fluxes in half precision, whose largest
number is 65504, and the energy flux here is about 1e5 times the velocity. This is the overflow already reported to
him for Mach 9 and the blasts, met earlier. `hydro16` is in the list so that its trial shows it; if it stops there
it is left out of the runs. Including the Float16 kernels would need either a cooling step each timestep with
gamma = 1.01, or his fix of storing fluxes in scaled units.

**Driving.** An acceleration field that changes smoothly and randomly in time (an Ornstein-Uhlenbeck process with
memory time T), made of Fourier modes with 1 < |k| < 3 only, amplitude 1 - (k - 2)^2: the stirring acts on the
largest scales and leaves the rest to the code. T = L/(2 M c_s), the usual turbulent crossing time for the target
Mach number M. Solenoidal in the pilot (`comp_frac = 0`); `comp_frac = 1` gives compressive driving.

Both codes already contain this driving: RAMSES in its `turb/` module, Tom's kernels as a port of it. They are made
equal in three ways, none of which changes either code:

- the same mode amplitudes (RAMSES: `forcing_power_spectrum='parabolic'`; Tom's: the driver fills his amplitude
  array with the same formula in place of his default 1/k up to k = 32);
- the same strength at every moment: the field is scaled to an rms acceleration of exactly sqrt(3) `a_rms` after
  each step of the process (RAMSES: `turb_exact_rms=.true.`; Tom's: the driver does it). Without this each code
  normalises in its own way, and RAMSES's own normalisation is an empirical fit;
- no ramp: Tom's kernels ramp the force up over the first T, RAMSES does not, so his kernels are given a time past
  the ramp.

What cannot be made equal is the random pattern itself (different random number generators), so two codes are
compared through time averages, and the GPU runs use two seeds to show how much two patterns differ.

A fourth difference was not in the plan and was found by the pilot: the direction of the force, next section.

## The fault in RAMSES's projection

`find_unitk` in `turb/turb_force_utils.f90` gives the direction of a mode's wave vector to the Helmholtz projection.
For the negative half of each axis it uses `TURB_GS - i`, the absolute value of the wave number, where the direction
needs the signed one, `i - TURB_GS` (the same file's power spectrum routine uses the signed form, where the sign does
not matter). For a mode whose components all have one sign the result is the true direction or its opposite, which
project alike. For a mode with components of different sign it is another direction, and "removing the compressive
part" removes the wrong part. Same lines in RAMSES at the sweep's pinned commit (5f0ed46, 2026-09-24), on its `dev`
branch on 2026-10-10, and in RAMSES II (`turb/turb_commons.f90`, aafd2ab). Tom's port uses signed wave numbers.

Measured on a laptop from the force fields RAMSES saves in its outputs (65 fields, `comp_frac = 0`):

- 20.0 % of the force's power lies along the true wave vector, i.e. is compressive; along the vector `find_unitk`
  uses, 5e-32;
- all of it is in the 48 of the 86 driven modes whose components differ in sign: they hold 54 % of the power, 37 % of
  which is compressive; in the other 38 modes, none;
- spectrum and memory time are as intended and equal to Tom's (so nothing else differs);
- with the three lines corrected the compressive share is 5e-32, and the same runs (32^3, three seeds, same strength)
  reach a Mach number 4 to 8 % higher and take up 13 to 26 % more power.

That is the size and sign of the gap between the two code families in the first Sherlock runs: Tom's kernels,
which are truly solenoidal, sat 4 to 5 % higher in Mach number and 13 to 22 % higher in power than released RAMSES,
while within each family (two solver settings, two kernels) the power agreed to 2 %.

On Sherlock (2026-10-10) the corrected runs closed that gap. Time averages from 2 T, same driving strength:

| | Mach reached, 128^3 / 256^3 | power, 128^3 / 256^3 |
|---|---|---|
| target 3: `ramses-matched` (released) | 3.07 / 3.07 | 29.4 / 28.8 |
| target 3: `ramsesfix-matched` | 3.16 / 3.24 | 34.5 / 35.3 |
| target 3: `gpu-hydro`, two seeds | 3.17, 3.18 / 3.24, 3.25 | 32.7, 34.5 / 33.4, 34.2 |
| target 10: `ramses-matched` (released) | 9.93 / 9.83 | 1020 / 1030 |
| target 10: `ramsesfix-matched` | 10.10 / 10.20 | 1210 / 1190 |
| target 10: `gpu-hydro`, two seeds | 10.14, 10.19 / 10.11, 10.25 | 1200, 1240 / 1200, 1240 |

Corrected RAMSES and Tom's Euler kernel now agree within 1 % in Mach number and within the difference of two seeds in
power; their effective Reynolds numbers agree within 5 % on both grids at both Mach numbers (the same scheme in two
codes). The correction itself moved RAMSES by +2 to +6 % in Mach number and +16 to +26 % in power.

What this bundle does about it:

- `build_ramses.sh fixed` compiles a copy of that one file with the three lines corrected, through RAMSES's own
  `PATCH` mechanism, into `bin-turb-fixed`; it refuses unless exactly three lines change. Runs: `ramsesfix-*`.
- The released runs stay, as the record of what RAMSES's driving does today.
- As a second check, `gpu-hydro_ramsesproj-*` (128^3 only) drives Tom's Euler kernel with RAMSES's released
  projection (his unit vectors replaced by absolute values; compressive share 0.22 in `test_local.jl`). If the
  projection is the whole story, these land on `ramses-matched-*` and `gpu-hydro-*` on `ramsesfix-matched-*`.
- The driving strengths in `forcing.toml` were found with released RAMSES and are kept: they are only the common
  number every run uses. With a solenoidal force the same strength gives about Mach 3.2 and 10.2.

It also matters beyond this pilot: any RAMSES or RAMSES II run that sets `comp_frac` gets a mixture different from the
one asked for, so a study of the driving's compressive fraction needs the correction.

**Driving strength.** `a_rms` for each target Mach number is found once (RAMSES "matched", 64^3, average of the Mach
number over 2 to 6 T, repeated until within 5 %) and written to `forcing.toml`. Every code and grid then uses that
number unchanged. So a run is *named* by its target Mach number, and the Mach number it actually reaches is a
result: a more diffusive scheme reaches a lower one from the same stirring.

**Length.** 10 T. Averages are taken from 2 T on, the usual choice; whether that is steady is checked, not assumed.

## The codes and their settings

| name | what runs |
|---|---|
| `gpu-hydro` | Tom's Euler kernel `step_hydro!`, Float32 tile: minmod slopes, Hancock predictor, HLL |
| `gpu-glm` | his original MHD kernel `step_tiled!` with zero field, as the sweep runs it: MonCen slopes, Hancock, LLF, GLM cleaning. The cleaning speed c_h enters its LLF wave speed, so it is diffusive even without a field |
| `gpu-hydro16` | `step_hydro!` with its default Float16 tile (expected to stop, see above) |
| `gpu-hydro_ramsesproj` | `gpu-hydro` driven with RAMSES's released projection: a check, 128^3 only |
| `ramses-matched` | RAMSES hydro solver: minmod slopes, HLL, Courant 0.5, i.e. the scheme of `gpu-hydro` as far as RAMSES has it; driving as released |
| `ramses-best` | RAMSES hydro solver: MonCen slopes, HLLC, Courant 0.8; driving as released |
| `ramsesfix-matched`, `ramsesfix-best` | the same two with the projection corrected: the runs to compare with `gpu-*` |

Tom's kernels and RAMSES use the same timestep rule (dx over the largest sum over directions of |v| + c, times the
Courant number), and `gpu-*` and `ramses-matched` the same Courant number, 0.5. The GPU state is single precision,
RAMSES double.

## What is measured

One implementation, `measure.jl`, for every code: the GPU driver calls it on its own state, `turb_measure.jl` on the
frames RAMSES writes. It uses the sweep's reductions and its spectra code.

- `series.csv`: energies, sound speed, rms Mach number (volume- and mass-weighted), density extremes and spread,
  spread of ln(rho), enstrophy and squared divergence (plain and density-weighted), cells at the density floor.
  GPU runs: 40 lines per T. RAMSES: one line per frame, 4 per T (its log does not print the kinetic energy).
- `spectra.csv` and `pdf.csv`, 4 per T: the sweep's spectra (velocity, solenoidal and compressive parts,
  density-weighted velocities, density) and the distribution of ln(rho) by volume and by mass.
- `slice_###.bin` once per T and `state_final.bin`: midplane and final state in the sweep's layout.
- `run.toml`: settings, status, steps, speed. A run in which the solver stops is recorded as failed with the time
  it reached and why; it is a result, not an error of the pipeline.

`summarize.py` (run by `status.sh`) turns these into the table and into `summary.csv`. Its header explains every
column: achieved Mach number, the steady-state check (second half of the window against the first), density spread
and the `b` parameter of the literature, dissipation rate, an effective Reynolds number, how isothermal the run
stayed.

A time in a GPU run's `series.csv` is the first step at or past the nominal time (late by less than one step); only
a change of the forcing field cuts a step short, as in Tom's own `run_turb`.

## What was tested, and what was not

Tested on a laptop (RAMSES built there without MPI at the sweep's pinned commit; `test_local.jl` for the rest):

- the forcing as this bundle sets it up in Tom's code: rms exactly sqrt(3) `a_rms`, all power at 1 < |k| < 3,
  compressive share 0, 1 and 1/3 for `comp_frac` 0, 1 and 0.5, correlation exp(-1) after one T;
- every measurement against the sweep's `Lean.diagnostics` or a direct calculation;
- RAMSES: its log reports the force at exactly the requested strength; build script, namelist, the reader of its
  hydro output, conversion, measurement, the search for the driving strength (two tries per Mach number at 32^3),
  and a run stopped and continued, which ends bit for bit like an uninterrupted one;
- the GPU driver's whole loop with Tom's CPU path (`--backend cpu`) in place of the GPU kernels. With the driving
  strength RAMSES found for Mach 10 at 32^3, his CPU path reached Mach 9.97 and RAMSES "best" 10.10: the two
  drivings are equivalent in practice, not only on paper;
- `check.sh`, `status.sh` and both job scripts against stand-ins for the Slurm commands;
- `test_gpu_compile.jl`: every function the bundle runs on the GPU, compiled for the NVIDIA target with CUDA.jl's own
  compiler set-up (this needs no GPU): the 15 reductions of the measurements and the timestep scan inside CUDA.jl's
  reduction kernel, the velocity-gradient kernel, and Tom's three kernel launches. Added after the first GPU job
  (47120899) stopped in its trial: the kernels had run their first steps, but one measurement was written as a named
  function with a variable number of arguments, which Julia runs on a CPU and cannot compile for a GPU. The check
  reproduces that error on a laptop and passes with the function written out.

First results on Sherlock (2026-10-09): the calibration hit both targets at its first try (a_rms 15 and 165 gave Mach
3.00 and 9.98 at 64^3); the RAMSES runs under MPI are steady from 2 T, within 2 % of their target Mach number, and
isothermal to 0.02 % (Mach 3) and 0.24 % (Mach 10); RAMSES does 0.4 to 0.6 million cell updates per second and rank.

The second GPU job (47130757, 2026-10-10) ran all sixteen runs of `hydro` and `glm` to the end, steady, in 18
minutes: 3.5 to 4.3 billion cell updates per second for the Euler kernel and 1.4 billion for the MHD kernel with the
force on. `hydro16` stopped in its trial at step 9 with a non-finite state, the half-precision overflow expected.

## Higher Mach numbers

`TURB_CONFIG=config_highmach.toml bash -l sherlock_turb/check.sh` runs the same pilot at Mach 30 and 50 (128^3 and
256^3, Tom's Euler and MHD kernels with two seeds, RAMSES HLL and HLLC). It differs from the first set in three ways:

- gamma - 1 = 1e-6, ten times smaller. The heat a run dissipates grows as Mach^2: with 1e-5 the gas of a Mach 50 run
  would warm by about 14 % in ten driving times, with 1e-6 by about 1.4 %.
- RAMSES runs with the corrected projection only, and the driving strengths for these Mach numbers are found with it
  (`[calibration] projection = "fixed"`). They are added to the same `forcing.toml`.
- No Float16 kernel and no check runs. The set has its own job names (`turb-cal-highmach`, `turb-gpu-highmach`) and
  its own trial directory, so it can be submitted while jobs of the first set are still queued.

The density floor stays at 1e-6 of the mean, as in the first set. At these Mach numbers the lowest densities reach
it; the `floor` column of the table says in how many cells. Tom's kernels reset a cell at the floor (density at the
floor, no momentum), RAMSES only limits the density where it divides by it, so this is one place where the codes may
part.

First run on Sherlock (2026-10-11): RAMSES finished, and the measurement then stopped on its output, because RAMSES
had let the density fall below zero in some cells (at 128^3: a few hundred to about two thousand of two million, the
lowest around -1e-7), and the spectra take a root of the density. RAMSES floors the density only where it forms
velocities, not in the state it keeps. The measurements now treat a cell below the floor as both codes do, at the
floor of 1e-6, count the cells at or below zero in a column of their own (`negative_density_cells`), and the table
lists the runs that have any. Mass and the density extremes are of the density as it is. Runs whose RAMSES output
was already converted are only measured again when their job runs again.

Tried on a laptop at 32^3, to 4 T: both codes run this gas steadily at both Mach numbers. With one driving strength
each, RAMSES HLLC reached Mach 29.4 and 48.6 and Tom's CPU path (Float32) 29.0 and 48.1; the sound speed rose by
0.08 % and 0.2 %; Tom's path had about 20 cells at the floor, RAMSES none.

## Files

`config.toml` the pilot, `config_highmach.toml` Mach 30 and 50; `check.sh`, `status.sh`, `trial.sh`; `build_ramses.sh`; `ramses_turb.py` (calibrate, run,
plan) and `ramses.sbatch`; `turb_drive.jl` and `gpu.sbatch`; `measure.jl`, `turb_measure.jl`; `summarize.py`;
`test_local.jl`, `test_gpu_compile.jl`; `trial_config.toml`.

## Not in the pilot

AthenaK and Athena++ (both can drive; Athena++ needs a rebuild with FFT), RAMSES II (waiting for the performance
pull request), higher and subsonic Mach numbers, compressive driving, 512^3, magnetic fields. The MHD kernels need
only a mean field in the initial state and a few more columns in `measure.jl`.

## Movie

The movie of the Turbulence page needs the density more often than the pilot saved it (a slice per driving time).
`movie.sh` repeats four runs of the pilot (Tom's Euler kernel, 256³, seed 1, Mach 3, 10, 30 and 50) with the midplane
density written 20 times per driving time (`movie_per_T` in `config_movie.toml` and `config_movie_highmach.toml`),
into `$SCRATCH/mhd-results-turb-movie`, so the pilot's results are not touched. About four minutes a run on an H100.

On the Mac, to send the bundle and submit:

```bash
cd ~/Projects/research/abel_mhd && COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - sherlock_turb | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && mkdir -p turb_update && tar -xzf - -C turb_update && mv -f turb_update/sherlock_turb/* sherlock_turb/ && rm -rf turb_update && bash -l sherlock_turb/movie.sh'
```

When both jobs have ended, on the Mac (about 210 MB):

```bash
mkdir -p ~/Projects/research/abel_mhd/turb_movie && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-results-turb-movie && find . -maxdepth 2 -path "./gpu-hydro-*" \( -name "rho_*.bin" -o -name run.toml \) | tar -cf - -T -' | tar -xf - -C ~/Projects/research/abel_mhd/turb_movie
```

Then `.venv-movie/bin/python scripts/make_turb_movie.py` draws a synchronized compilation,
`progress/media/tests/driven_turbulence-density.mp4`, and individual `driven_M3-density.mp4`,
`driven_M10-density.mp4`, `driven_M30-density.mp4` and `driven_M50-density.mp4` movies beside it.
Each individual movie is 1440 × 1600, at 20 fps, with 201 real snapshots and brief endpoint holds.
Rendering checks completion, consecutive frame numbers, file sizes and finite positive density first.
Colour limits stay fixed over time; no temporal interpolation is applied. PNG posters and JSON provenance accompany each MP4.
Use `--mach 30 50` to render only those Mach numbers.
Run `.venv-movie/bin/python scripts/build_turbulence.py --page-only` and
`.venv-movie/bin/python scripts/build_highmach_results.py --page-only` to add the players.
After rendering and checking the published movies, `turb_movie/` can be deleted.

