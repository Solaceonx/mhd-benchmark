# GIZMO in the comparison suite

GIZMO (github.com/pfhopkins/gizmo, pinned at `519be7d`, 2026-10-04, GPL; the Bitbucket copy is an archive): mesh-free
finite-volume Godunov methods (Hopkins 2015), MHD with Dedner cleaning plus the constrained-gradient correction
(Hopkins & Raives 2016; Hopkins 2016). A particle code next to the grids of the other codes and AREPO's moving mesh.

## Set-up

| Label | Method |
|---|---|
| `gizmo_mfm` | meshless finite mass (GIZMO's default): particles move with the gas and keep their mass |

`MAGNETIC`, `MHD_CONSTRAINED_GRADIENT=1` (GIZMO's recommended setting), cubic-spline kernel with 20 neighbours in 2D
(GIZMO's guidance: 14–20), `FORCE_EQUAL_TIMESTEPS` (one timestep, as the other codes), double precision,
`OUTPUT_BFIELD_DIVCLEAN_INFO` (its divergence of B in the snapshots; output only). The Courant factor is GIZMO's own for
MHD, 0.2: it can be changed only in GIZMO's `DEVELOPER_MODE`, which stays off.

GIZMO's other methods were built and run on the Mac at this commit and left out (`build.sh` lists the findings):
meshless finite volume and the regular grid built on it have NaN gas velocities from the very first output and GIZMO
then removes every particle (2D and 3D, 1 to 4 ranks, with and without MHD, any mesh motion, gravity on or off; the grid
with the constrained-gradient option does not even compile); pressure-SPH runs but gains 60 % of its total energy on
Orszag–Tang at 64² by t = 0.5 (50 % with the quintic kernel and 40 neighbours), where MFM loses about 1 % (its cleaning removes some, as GLM does). These are
worth reporting to GIZMO's maintainers.

## What runs

The sweep's 17 planar tests at each test's own grids (32² or 64² to 512²): 82 runs on CPUs, as AREPO's per mesh. The 3D
tests are not in this pass (512³ is 134 million particles). One array task per test; task 0 is the pilot: it builds the
three executables (the three adiabatic indices of the planar tests: GIZMO fixes γ at compile time) and runs
Orszag–Tang at 32² and 64²; the 17 tasks start only if both pilot runs finish.

`gizmo_run.py` plugs GIZMO into the sweep's external-code driver (`sherlock_sweep/codes/external.py`, imported, not
changed), as `sherlock_arepo/arepo_run.py` does AREPO: run directories
`$SCRATCH/mhd-results-sweep/<test>-gizmo-mfm-run-N<n>-s42/`, the sweep's records (`run.toml`, `diagnostics.csv`,
`state_###.bin`) and measurements. Extra per run: `gizmo.log`, `params.txt` and `gizmo_divergence.csv` (GIZMO's own
measure of div B on its particles at every output). Specific to GIZMO, in `gizmo_run.py`:

- Initial conditions: one particle per cell centre of the grid, with the cell's mass, from the sweep's own Float32
  initial state. GIZMO works with μ₀ = 1 as we do; with `UnitMagneticField_in_gauss` = √(4π), its default for cgs
  code units, B goes in and out unchanged.
- Output times: GIZMO's steps are power-of-two fractions of TimeMax − TimeBegin; TimeMax is a power of two times the
  output interval, so every output time is a synchronisation point, and the run ends through GIZMO's stop file once
  the last snapshot is written.
- Snapshots to the grid: each particle's mass, momentum, total energy and field times volume are deposited
  cloud-in-cell and divided by the cell volume, so the grid holds exactly the particles' mass, momentum, energy and
  magnetic flux; a particle still at a cell centre lands wholly in its cell.
- A run GIZMO stops itself (`ENDRUN issued ...`) is recorded as failed with GIZMO's reason, not as a job being killed.

## Tested on the Mac (2026-10-23)

GIZMO itself, built with this bundle's `build.sh` (Homebrew open-mpi, gsl, hdf5; 2 MPI ranks), through the sweep's
driver:

| Run | GIZMO MFM | for comparison (the sweep's records) |
|---|---|---|
| Orszag–Tang 64², energies at t = 0.5: kinetic / magnetic / thermal | 0.0427 / 0.0582 / 0.2454 | AthenaK PPM/HLLD 0.0442 / 0.0595 / 0.2455; RAMSES HLL 0.0440 / 0.0550 / 0.2502 |
| the same, mass on the grid end / start − 1 | 7 × 10⁻¹¹ | |
| field loop 32², relative error of B at t = 2 | 0.30 | RAMSES HLL 0.44, GLM f16 fixed 0.42, AthenaK PLM/HLLE 0.60 |

The pilot's own runs (Orszag–Tang 32² and 64²) finish and pass its check. With a stand-in executable that writes
GIZMO-format snapshots: initial conditions → snapshots → our state is exact at all 21 outputs; particles moved off the
centres keep the grid's mass to 2 × 10⁻¹⁰; a run GIZMO stops itself is recorded as failed with the time, the function
and the reason. Every parameter tag is one this build accepts (GIZMO stops at a tag it does not know).

**Not tested:** the build on Sherlock's compilers and libraries (gcc 12, openmpi 4.1, HDF5 1.14 there; clang, openmpi
5, HDF5 2.2 on the Mac). The pilot does that first.

## Commands

On the Mac, in the project directory (sends the bundle; `submit.sh` gets the source and submits the pilot and,
depending on it, the 17 tasks; it refuses while a `gizmo` job is still queued):

```bash
cd /Users/lyxia/Projects/research/abel_mhd && COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - sherlock_gizmo | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_gizmo/submit.sh'
```

On Sherlock, any time:

```bash
bash -l /scratch/users/lyxia/mhd-study/sherlock_gizmo/status.sh
```

If the pilot fails, the 17 tasks wait on it forever (`squeue` shows `DependencyNeverSatisfied`): cancel them with
`scancel -n gizmo` and send the status output. One test again by hand: `GIZMO_ONLY="field_loop mfm 64" sbatch --array=0
gizmo.sbatch` (with `GIZMO_RANKS`, `MHD_RESULTS`). Python's h5py and scipy come from the environment
`$HOME/.venvs/mhd-h5py`, shared with the AREPO runs (made here if missing; environments stay off `$SCRATCH`, Sherlock's rule 8 for automated tools). Overrides: `GIZMO_COMPILER_MODULES`
(default `gcc/12.4.0 openmpi/4.1.6`), `GIZMO_MODULES` (default `gsl/2.7 hdf5/1.14.4`), `GSL_PREFIX`, `HDF5_PREFIX`,
`GIZMO_MEMORY_MB_PER_RANK`, `GIZMO_VENV`.

Fetch to the Mac when the tasks are done:

```bash
cd /Users/lyxia/Projects/research/abel_mhd && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash /scratch/users/lyxia/mhd-study/sherlock_gizmo/tools/pack_for_mac.sh' | tar -xzf - -C sweep_results
```
