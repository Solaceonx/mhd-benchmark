# Running on Sherlock

Every run set is a self-contained directory (a bundle) that is copied to
`/scratch/users/<user>/mhd-study/` and run there. Results go to `$SCRATCH/mhd-results-<name>`; built codes to
`$MHD_CODES`. Each bundle's README has its exact commands.

| Bundle | Runs | Results directory |
|---|---|---|
| `sherlock_mhd/` | First campaign: 21 tests, the LLF kernel and the CT prototype against AthenaK | |
| `sherlock_rerun/` | Four tests rerun with corrected set-ups (current sheet, three Kelvin–Helmholtz) | |
| `sherlock_movies/` | The same tests with dense output, for movies | |
| `sherlock_gem/` | GEM reconnection | |
| `marlowe_mhd/` | The first campaign prepared for the Marlowe cluster | |
| `sherlock_sweep/` | The comprehensive sweep (`SWEEP_RUN.md`, `VALIDATION.md`) | `mhd-results-sweep` |
| `sherlock_arepo/` | AREPO, plugged into the sweep's driver | `mhd-results-sweep` |
| `sherlock_ramses2/` | RAMSES II speed test | `mhd-results-ramses2` |
| `sherlock_mhdale/` | MHD-ALE trial and 13-test chain | |
| `sherlock_turb/` | Driven turbulence | `mhd-results-turb` |
| `sherlock_checks/` | Courant-number and single-precision checks of Tom's kernels | `mhd-results-checks` |

## Sending, checking, fetching

Tom's kernels first, since the bundles do not carry them in this repository: `bash scripts/fetch_solver.sh`.

Send a bundle and start its check in one command from the laptop:

```bash
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - sherlock_turb | ssh -o ServerAliveInterval=30 USER@login.sherlock.stanford.edu 'cd /scratch/users/USER/mhd-study && tar -xzf - && bash -l sherlock_turb/check.sh'
```

Each bundle has a `check.sh` (build, a small trial, then submit) and a `status.sh` (queue, logs, a table of results).
While jobs run, replace files by extracting into a side directory and moving them into place, never by overwriting a
script that may be executing. Small result files come back with a `find … | tar` over ssh (the command for the
turbulence runs is printed on the notebook's Turbulence page; the sweep's is `scripts/sweep_fetch.py`).

## The sweep's coordinator

`sherlock_sweep/tools/sweep.py` runs as one long job that wakes every five minutes and submits work in waves below
the per-user job limit, after two gates (builds, and the resume checks) have passed. One Slurm job per unit, one job
per run directory (a lock file), and run directories are named by content so a requeued job resumes its own work.
Subcommands used most: `status`, `runs --json`, `audit`, `redo`, `skip`, `dedupe`. `VALIDATION.md` lists every fault
found in the harness on the way and its fix (ten patches).

## Scheduling

- H100s: partitions `gpu,owners` with requeue. Owner nodes preempt, so everything must resume.
- About 100 queued tasks per user; the coordinator keeps under 90.
- Large CPU jobs on `owners` were preempted 7 to 24 times each at 512³; restart points every 20 minutes made that
  survivable.

## Pitfalls

- Never resubmit a job that is still in `squeue`: Slurm requeues preempted jobs itself, and two jobs in one run
  directory corrupt it.
- `sbatch` passes the submitting shell's environment on; set what a job needs inside the job.
- The system `git` is old (no `-C`); glibc is 2.17, so `pip` needs `--only-binary=:all:` and pinned versions.
- The `py-h5py` module does not import with the `py-numpy` 2.2 module; use a virtual environment.
- Loading `hdf5/1.14.4` swaps Open MPI 4.1.6 for 5.0.5.
- nvfortran 25.1 has an internal compiler error on one RAMSES II file at `-O4`.
- Try every build on a login or dev node before queueing for a GPU: the H100 queue can take a day.
- A job killed from outside (preemption, a lost Lustre client, `scontrol requeue`) must not be recorded as a failed
  run; the driver tells the two apart since patch 10.
- `$SCRATCH` is purged 90 days after a file's last change. See [data](data.md).
- GPU code is compiled on the laptop first, without a GPU, by targeting PTX through `CUDA.GPUCompiler`
  (`sherlock_turb/test_gpu_compile.jl`). It catches the errors that otherwise cost a queue wait.
