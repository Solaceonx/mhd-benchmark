# AREPO in the comparison suite

The public AREPO release (arepo-code.org; pinned at commit `351aa811`, 2026-08-24, GPL v3): finite volumes on a
Voronoi mesh, second order, HLLD, and Powell's eight-wave scheme for the divergence of B, the method the AREPO runs of
Ugalino et al. (2026) use. It adds a third way of handling div B next to constrained transport and GLM cleaning, and
a moving mesh next to fixed grids.

## Set-ups

| Label | Mesh |
|---|---|
| `arepo_static` | the Cartesian grid of the other codes, not moving (`VORONOI_STATIC_MESH`) |
| `arepo_moving` | the same initial mesh, moving with the gas (AREPO's normal mode, with its mesh regularisation) |

Both: `MHD`, `MHD_POWELL`, `RIEMANN_HLLD`, one global timestep, Courant factor 0.3 (AREPO's examples), double precision.

## What runs

The sweep's 17 planar tests at each test's own grids (32² or 64² to 512²), both set-ups: 164 runs on CPUs. The 3D
tests are not in this pass.

`arepo_run.py` plugs AREPO into the sweep's external-code driver (`sherlock_sweep/codes/external.py`, imported, not
changed), so the run directories (`$SCRATCH/mhd-results-sweep/<test>-arepo-<mesh>-run-N<n>-s42/`), the records
(`run.toml`, `diagnostics.csv`, `state_###.bin`) and the measurements are those of RAMSES, AthenaK and Athena++.
Extra per run: `arepo.log`, `param.txt`, and `arepo_divergence.csv` (the divergence AREPO itself reports on its own
mesh at every output).

Two things specific to AREPO, both handled in `arepo_run.py` (its header explains them): the snapshots are made to
land exactly on the sweep's output times, and a moving mesh is averaged onto the grid by sampling.

## State of testing

Checked on the laptop against a stand-in for the AREPO binary: initial conditions, parameter file, stopping after the
last output, reading snapshots (exact round trip on the static mesh), the moving-mesh averaging, the sweep's
post-processing, clean-up. **Not checked: AREPO itself.** It needs MPI, GSL, GMP and HDF5, which the laptop lacks, so
the build and the first real runs happen in the pilot task on Sherlock, and the full array starts only if the pilot's
four runs complete.

## Commands

On the Mac:

    cd /Users/lyxia/Projects/research/abel_mhd && scp sherlock_arepo.tar.gz sherlock_arepo.tar.gz.sha256 lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/

On Sherlock:

    cd /scratch/users/lyxia/mhd-study && sha256sum -c sherlock_arepo.tar.gz.sha256 && tar -xzf sherlock_arepo.tar.gz && cd sherlock_arepo && bash fetch.sh && mkdir -p logs && p=$(sbatch --parsable --array=0 arepo.sbatch) && sbatch --dependency=afterok:$p --array=1-34 arepo.sbatch

After the pilot (a few minutes once it starts):

    cat /scratch/users/lyxia/mhd-study/sherlock_arepo/logs/pilot_summary.txt; tail -n 25 /scratch/users/lyxia/mhd-study/sherlock_arepo/logs/arepo-*_0.out

If the pilot fails the array never starts (`squeue` shows it waiting on the dependency); cancel it with
`scancel -n arepo` and send the output above.

Python's h5py and scipy come from pip in `$HOME/.venvs/mhd-h5py` (environments stay off `$SCRATCH`: Sherlock's rule 8 for automated tools) (Sherlock's `py-h5py` module does not import with
the numpy 2 module the sweep uses).

Overrides: `AREPO_COMPILER_MODULES` (default `gcc/12.4.0 openmpi/4.1.6`, as for RAMSES), `AREPO_MODULES`
(default `gsl/2.7 hdf5/1.14.4 gmp/6.2.1`), `AREPO_VENV`, `GSL_PREFIX`, `HDF5_PREFIX`, `GMP_PREFIX`, `SWEEP_BUNDLE`, `AREPO_MEMORY_MB_PER_RANK`.

## First results and the four static-mesh failures (2026-10-06)

Job 46813279 (pilot) and 46813282 (array): 156 of 164 runs complete. Tasks 9, 11, 17 and 21 (static mesh:
`field_loop`, `field_loop_novz`, `orszag_tang`, `current_sheet`) stopped in their 256² run on 8 ranks: AREPO ended
itself with "time-step of integer size 1 not allowed" (`timestep.c:740`) at t = 0.84, 0.88, 0.42 and 1.92, its
timestep having fallen to the smallest its time line holds (about 1e-8). The same tests ran to the end on the moving
mesh at 256² and 512², and on the static mesh at 128². Why the timestep collapses is not known yet.

- `arepo_run.py` now records a run AREPO ends itself as **failed**, with the time reached and AREPO's message
  (`arepo_stopped`, `arepo_stopped_at_time`, `arepo_last_timestep` in `run.toml`); before, srun's report of the abort
  made the sweep's driver treat it as a job being stopped, record nothing, and skip the remaining grids.
- `bash -l rerun_failed.sh` runs those four tasks again (256² is recorded as failed, then 512² runs) and two check
  runs of Orszag–Tang 256² on the static mesh in `$SCRATCH/mhd-results-arepo-checks/`: on one rank (`ranks1`), and
  on 8 ranks with Courant factor 0.15 (`courant015`). Options of `arepo.sbatch` for such runs: `AREPO_ONLY="test mesh
  grid…"`, `AREPO_RANKS`, `AREPO_COURANT`, `MHD_RESULTS`.
- `bash -l status.sh` shows the jobs, the failed tasks' logs, the runs without a finished record, and the checks.

## Outcome of the reruns and checks (job 46881897 and the two check runs, 2026-10-07)

All 164 runs are recorded: 157 complete, 7 failed. The seven are all on the static mesh, and in each AREPO stopped
itself with "time-step of integer size 1 not allowed":

| Run | stopped at t = | of |
|---|---|---|
| `orszag_tang` 256², 512² | 0.422, 0.293 | 0.5 |
| `field_loop` 256², 512² | 0.839, 0.556 | 2.0 |
| `field_loop_novz` 256², 512² | 0.883, 0.398 | 2.0 |
| `current_sheet` 256² | 1.923 | 2.5 (512² ran to the end) |

The two checks of Orszag–Tang 256² on the static mesh: on **one rank** it stops at the same sync point (4470,
t = 0.4221); with **Courant factor 0.15** on 8 ranks it stops a little later (t = 0.454). So the collapse does not
come from running in parallel and is not removed by a smaller timestep. In the field-loop runs the magnetic energy
at the last output is of order 1e-2 (256², t = 0.8: 2.6e-2), against a starting value near 1.4e-7, with AREPO's
own divergence measure near 0.4: the field itself grows until the timestep collapses. The same tests on the moving
mesh ran to the end at every grid. Static mesh with Powell cleaning is not how AREPO is normally run (its MHD
papers use the moving mesh); whether this instability is expected there is a question for an AREPO developer.
