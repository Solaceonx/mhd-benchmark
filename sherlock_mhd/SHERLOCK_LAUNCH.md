# Sherlock storage and first launch — lyxia

Reviewed 2026-09-25 against `PROJECT_SUMMARY.md`, current job scripts, and the
storage paths supplied from OnDemand. This is a local preparation record: no
remote files, quotas, installed modules, GPU allocations or job results have
been inspected in this review.

## Your storage map

| Variable | Path shown in your OnDemand account | Use for this study |
|---|---|---|
| `$HOME` | `/home/users/lyxia` | Small source copy, configuration, personal Julia installation/depot |
| `$GROUP_HOME` | `/home/groups/tabel` | Shared software and selected durable results, following lab conventions |
| `$SCRATCH` | `/scratch/users/lyxia` | Working copy, builds, simulations, temporary return archives |
| `$GROUP_SCRATCH` | `/scratch/groups/tabel` | Shared working data if the lab needs it; unused by default |

Your current `/home/users/lyxia/ondemand` view is a directory inside `$HOME`.
The presence of its `data/` directory does not mean it contains our simulations.
Choose **Files → SCRATCH** to browse the working area. Use **Clusters → Sherlock
Shell Access** for the commands below; that opens a login shell, not a GPU allocation.

Sherlock recommends scratch for job I/O. Personal home has a published 15 GB
quota; the Julia depot can consume a substantial part of it. Scratch has no
backup and a 90-day content-modification purge policy; group scratch uses the
same policy. Download completed results promptly. Check actual usage with
`sh_quota`, rather than assuming the documented limits equal available space.
Sources: [filesystems](https://www.sherlock.stanford.edu/docs/storage/filesystems/),
[OnDemand](https://www.sherlock.stanford.edu/docs/user-guide/ondemand/).

## Locations used by the existing scripts

```text
/home/users/lyxia/
  sherlock_mhd/                      small uploaded source copy (optional)
  .julia-mhd/                       default Julia package depot
/scratch/users/lyxia/
  mhd-study/sherlock_mhd/            submit jobs from here
    configs/  jobs/  src/  codes/   inputs and adapters
    logs/                           Slurm stdout/stderr
    returns/                        .tar.gz plus .sha256 to download
  mhd-codes/                        pinned AthenaK/RAMSES/Athena++ sources and builds
  mhd-results/                      raw per-job results
```

These are the current defaults, not directories verified to exist remotely.
`MHD_CODES`, `MHD_RESULTS`, and `MHD_JULIA_DEPOT` override their respective
locations. `returns/` and `logs/` are under the submitted bundle, so submit from
the scratch copy. Keep the Mac source/archive as the durable copy of scratch
code and build inputs. No scripts write into the group's directories by default.

## Upload and inspect the environment

Upload `sherlock_mhd.tar.gz` and its checksum through OnDemand to
`/scratch/users/lyxia/mhd-study` (create that directory first). Open a shell:

```bash
cd "$SCRATCH/mhd-study"
sha256sum -c sherlock_mhd.tar.gz.sha256
# First installation only; use a new directory for later releases/active runs.
tar -xzf sherlock_mhd.tar.gz
cd sherlock_mhd
sh_quota
module spider julia
module spider python
module spider cuda
module spider gcc
module spider cmake
module spider openmpi
node_feat -p gpu
```

Choose Julia **1.12.x** (lockfile: 1.12.5), Python **3.11 or newer** with NumPy,
and compatible CUDA/GCC/CMake/MPI modules. Load the chosen Python module in the
submission shell: `submit.sh` uses Python before the job's module setup runs.
The defaults in the scripts are placeholders until checked on Sherlock.

Set the appropriate overrides in that shell: `JULIA_MODULE` (or `JULIA_BIN`),
`MHD_PYTHON_MODULES`, `MHD_GPU_BUILD_MODULES`, `MHD_GPU_RUN_MODULES`, and
`MHD_CPU_BUILD_MODULES`. Preserve those choices for later submissions; the
GPU runtime modules must be compatible with the modules used to build AthenaK.
If no Julia 1.12 module exists, a separate Linux installation is needed.

Pick one available GPU feature and set `MHD_GPU_FEATURE` to its actual name.
The current architecture detector supports compute capabilities 7.0, 7.5,
8.0, 8.6, 8.9 and 9.0. A different architecture needs a build-script update.
Use the same feature for AthenaK's build and all GPU comparisons.

## Staged first run

For a single submission command that schedules these gates and the full matrix
behind them, see `tools/submit_campaign.sh` in the main README. The stages below
remain useful for inspecting or rerunning an individual job.

These commands submit jobs; run each stage only after checking the preceding
stage's result. Nothing automatically submits the whole matrix.

```bash
bash tools/submit.sh setup
# After setup succeeds:
bash tools/submit.sh smoke --constraint="$MHD_GPU_FEATURE"
```

Inspect `logs/`, `returns/`, `exit_code.txt`, and per-test `suite.toml` files in
`$SCRATCH/mhd-results/smoke-JOBID/data/`. The smoke job tests the Julia GPU
driver's 25 configs at N=32. It does not test AthenaK or the MPI codes, and does
not exercise the production GLM kernel's N-divisible-by-6 tile specialization.
Initial compilation may need a longer Slurm limit; retain logs if it times out.

Then fetch and build the independent codes:

```bash
bash tools/fetch_codes.sh
bash tools/submit.sh build-gpu-codes --constraint="$MHD_GPU_FEATURE"
bash tools/submit.sh build-cpu-codes
```

After those builds and the Julia smoke pass, start a bounded comparison:

```bash
bash tools/submit.sh orszag_tang --array=1 --constraint="$MHD_GPU_FEATURE"
bash tools/submit.sh orszag_tang --code athenak:matched --array=1 --constraint="$MHD_GPU_FEATURE"
bash tools/submit.sh orszag_tang --code athenak:best --array=1 --constraint="$MHD_GPU_FEATURE"
```

These select N=64 for both drivers (planar physics in cubic storage). Validate
the outputs before high-resolution references or the full matrix. RAMSES and
Athena++ need their own successful small MPI runs before the larger CPU jobs.
The Julia smoke pass alone cannot establish multi-code readiness.

The comparison currently uses FP32 GLM/CT against FP64 AthenaK. Report that
qualification, and compare accuracy against GPU cost without mixing CPU timings.

## Download to the Mac

OnDemand: **Files → SCRATCH → mhd-study → sherlock_mhd → returns**. Download
each result archive and checksum. Or, from the Mac repository root:

```bash
mkdir -p scripts/out/sherlock_downloads
rsync -av lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/sherlock_mhd/returns/ scripts/out/sherlock_downloads/
```

Follow `README.md` for checksum verification, extraction, local analysis and
notebook rebuilding. A Slurm timeout can leave only raw files in `mhd-results`;
after the job stops, use `tools/pack_results.sh` on that job directory. Keep
raw files until the downloaded archive is verified. Packaging temporarily keeps
both the raw data and a compressed copy, so budget space for both.
