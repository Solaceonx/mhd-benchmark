# GIZMO in the comparison suite

GIZMO (github.com/pfhopkins/gizmo, pinned at `519be7d`, 2026-10-04, GPL; the Bitbucket copy is an archive): mesh-free
finite-volume Godunov methods (Hopkins 2015), MHD with Dedner cleaning plus the constrained-gradient correction
(Hopkins & Raives 2016; Hopkins 2016). A particle code next to the grids of the other codes and AREPO's moving mesh.

## Set-ups

| Label | Method |
|---|---|
| `gizmo_mfm` | meshless finite mass (GIZMO's default): particles move with the gas and keep their mass |
| `gizmo_mfv` | meshless finite volume: particles move with the gas, mass flows between them |

Both: `MAGNETIC`, `MHD_CONSTRAINED_GRADIENT=1` (GIZMO's recommended setting), cubic-spline kernel with 20 neighbours in
2D (GIZMO's own test problems), `FORCE_EQUAL_TIMESTEPS` (one timestep, as the other codes), double precision. The
Courant factor is GIZMO's own for MHD, 0.2: it can be changed only in GIZMO's `DEVELOPER_MODE`, which stays off.

## What runs

The sweep's 17 planar tests at each test's own grids (32² or 64² to 512²), both set-ups: 164 runs on CPUs, as AREPO.
The 3D tests are not in this pass (512³ is 134 million particles). One array task per test and set-up; task 0 is the
pilot: it builds the six executables (two methods × the three adiabatic indices of the planar tests: GIZMO fixes γ at
compile time) and runs Orszag–Tang at 32² and 64² with both methods; the 34 tasks start only if all four pilot runs
finish.

`gizmo_run.py` plugs GIZMO into the sweep's external-code driver (`sherlock_sweep/codes/external.py`, imported, not
changed), as `sherlock_arepo/arepo_run.py` does AREPO: run directories
`$SCRATCH/mhd-results-sweep/<test>-gizmo-<method>-run-N<n>-s42/`, the sweep's records (`run.toml`, `diagnostics.csv`,
`state_###.bin`) and measurements. Extra per run: `gizmo.log`, `params.txt` and `gizmo_divergence.csv` (GIZMO's own
measure of div B on its particles at every output). Specific to GIZMO, in `gizmo_run.py`:

- Initial conditions: one particle per cell centre of the grid, with the cell's mass, from the sweep's own Float32
  initial state. GIZMO works with μ₀ = 1 as we do; with `UnitMagneticField_in_gauss` = √(4π), its default for cgs
  code units, B goes in and out unchanged.
- Output times: GIZMO's steps are power-of-two fractions of TimeMax − TimeBegin; TimeMax is a power of two times the
  output interval, so every output time is a synchronisation point, and the run ends through GIZMO's stop file once
  the last snapshot is written.
- Snapshots to the grid: the particles move, so each grid cell takes the mean of the particles nearest to 4 × 4 points
  in it (as AREPO's moving mesh; for GIZMO's kernel-weighted volumes an approximation at the particle scale).
- A run GIZMO stops itself (`ENDRUN issued ...`) is recorded as failed with GIZMO's reason, not as a job being killed.

## State of testing

On the Mac, without GIZMO (it needs MPI, GSL and HDF5, which the Mac lacks), with a stand-in for its executable that
reads the parameter file and initial conditions and writes GIZMO-format snapshots:

- through the sweep's driver, initial conditions → snapshots → our state is exact: every field equal at every output,
  all 21 output times of the field loop, the stop file ends the run after the last snapshot;
- particles moved off the cell centres: the grid still holds the total mass exactly;
- a run GIZMO stops itself: recorded as failed, with the time, the function and the reason;
- every parameter tag is one GIZMO's build accepts (the `DEVELOPER_MODE` ones are left out: GIZMO stops at a tag its
  build does not know); GIZMO's own configuration step accepts all six configurations, and its declarations
  preprocess with them without any of its compile-time consistency errors.

**Not tested:** GIZMO itself. The pilot builds and runs it first.

## Commands

On the Mac, in the project directory (sends the bundle; `submit.sh` gets the source and submits the pilot and,
depending on it, the 34 tasks; it refuses while a `gizmo` job is still queued):

```bash
cd /Users/lyxia/Projects/research/abel_mhd && COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - sherlock_gizmo | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_gizmo/submit.sh'
```

On Sherlock, any time:

```bash
bash -l /scratch/users/lyxia/mhd-study/sherlock_gizmo/status.sh
```

If the pilot fails, the 34 tasks wait on it forever (`squeue` shows `DependencyNeverSatisfied`): cancel them with
`scancel -n gizmo` and send the status output. One test again by hand: `GIZMO_ONLY="field_loop mfm 64" sbatch --array=0
gizmo.sbatch` (with `GIZMO_RANKS`, `MHD_RESULTS`). Python's h5py and scipy come from the environment
`$SCRATCH/mhd-venv-arepo` that the AREPO runs made (made here if missing). Overrides: `GIZMO_COMPILER_MODULES`
(default `gcc/12.4.0 openmpi/4.1.6`), `GIZMO_MODULES` (default `gsl/2.7 hdf5/1.14.4`), `GSL_PREFIX`, `HDF5_PREFIX`,
`GIZMO_MEMORY_MB_PER_RANK`, `GIZMO_VENV`.

Fetch to the Mac when the tasks are done:

```bash
cd /Users/lyxia/Projects/research/abel_mhd && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash /scratch/users/lyxia/mhd-study/sherlock_gizmo/tools/pack_for_mac.sh' | tar -xzf - -C sweep_results
```
