# RAMSES II on one H100: speed, with validation data

Suggested by Eric Moseley (2026-10-05). RAMSES II (github.com/ramses-organisation/ramses2, pinned at commit
`aafd2abb`, 2026-09-26) has a CUDA Fortran GPU backend whose MHD is constrained transport in double precision.
This bundle times it on the problem of the sweep's throughput test, 3D Orszag–Tang to t = 0.05, at 64³–512³.

## What runs

- Builds: `nsubgrid` = 1 and 2 (the GPU tile size, hard-coded in `gpu/gpu_utils.cuf`, so each value is a build; the
  MHD kernel has launch settings for 1 and 2 only) × double and single precision (`NPRE` = 8, 4). Four builds, each in
  its own copy of the source under `$SCRATCH/mhd-codes-sweep/ramses2-build/`.
- Runs: each build × two solver settings (the sweep's RAMSES "matched": HLL, Courant 0.5; "best": HLLD, Courant 0.8;
  MonCen slopes) × 64³, 128³, 256³, 512³. No output files, except one extra 256³ run that keeps the code's own
  snapshots at t = 0 and the end (`ot_ns1_p8/hlld_L8_dump/output_0000{1,2}`, about 3 GB on scratch) so the fields can
  be checked later.
- A build or run that fails is reported in the summary and the job goes on.

## What the patch changes (`prepare_patch.py`)

Upstream Orszag–Tang (`INIT=OT`) is 2D only and does not compile at the pinned commit (`pi` declared twice). The patch
copies three upstream files with small edits: the 3D problem of our test (velocities modulated along z, ε = 0.2; only
A_z non-zero), the duplicate declaration removed, and the energies and time printed to full precision instead of
three digits. Nothing in the solver is touched. Dry-run on the laptop's CPU at 64³: energies at t = 0.05 of
0.114365 / 0.039341 / 0.199893 (kinetic / magnetic / thermal), between the sweep's GLM HLLD and AthenaK PPM/HLLD.

## What comes back (`$SCRATCH/mhd-results-ramses2/`)

- `summary.txt`, `summary.json`: per run, cell-updates per second (median step), seconds to t = 0.05, peak GPU memory,
  the three energies at t = 0.05 (interpolated: the code does not shorten its last step), largest mass and energy drift.
- `ramses2_speed_records.tar.gz`: every `run.log`, namelist, GPU memory trace, `BUILD_INFO.txt` (commit, make options,
  compiler, GPU), `build.log` and `provenance.txt` (node, driver, modules). Small; copy it back to keep.

## Commands

On the Mac:

    cd /Users/lyxia/Projects/research/abel_mhd && scp sherlock_ramses2.tar.gz sherlock_ramses2.tar.gz.sha256 lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/

On Sherlock:

    cd /scratch/users/lyxia/mhd-study && sha256sum -c sherlock_ramses2.tar.gz.sha256 && tar -xzf sherlock_ramses2.tar.gz && cd sherlock_ramses2 && bash fetch.sh && bash -l check_build.sh

`check_build.sh` first builds the code without a GPU (on a node of the `dev` partition if one is free, else on the
login node; a few minutes) and submits `speed.sbatch` only if that works, so that a build problem does not cost a
wait in the H100 queue. Or, in one step from the Mac once the source is on Sherlock:

    cd /Users/lyxia/Projects/research/abel_mhd && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_ramses2/check_build.sh' < sherlock_ramses2.tar.gz

**Compiler.** nvfortran 25.1 stops with an internal compiler error on `cooling/cooling_module.f90` at the Makefile's
`-O4` (job 46663995, 2026-10-06; the file is not used by these runs). `build.sh` builds an object that fails this way
at the highest level that works (`-O3` down to `-O0`) and lists it in `BUILD_INFO.txt` and at the end of
`summary.txt`; a GPU kernel among them is flagged, since the speeds would then be lower bounds. If the newest `nvhpc`
module cannot build the code at all, the next two are tried (`compiler.sh`).

When the job has ended (about an hour once it starts):

    cat /scratch/users/lyxia/mhd-results-ramses2/summary.txt; tail -n 30 /scratch/users/lyxia/mhd-study/sherlock_ramses2/logs/ramses2-speed-*.out

Overrides: `RAMSES2_NVHPC_MODULE` (default: the newest `nvhpc` module that builds the code), `RAMSES2_CHECK_ON=login`, `RAMSES2_CUDA_ARCH` (default: from the GPU),
`RAMSES2_RESULTS`, `MHD_CODES`.
