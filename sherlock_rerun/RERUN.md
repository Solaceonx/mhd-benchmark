# Sherlock four-test rerun (2026-09-28)

The first Sherlock campaign (`sherlock_mhd`, ledger `campaigns/20260927T173407Z.tsv`)
ran four tests with setups that the 2026-09-28 literature check corrected. This
bundle reruns only those four, GPU-only like the first campaign, on the same
`GPU_SKU:H100_SXM5` nodes so the results sit on the same hardware as the other 21 tests.

| Test | First campaign | This rerun |
|---|---|---|
| `current_sheet` | vx = 0.1 sin 2πy on [0,2] (FLASH guide) | vx = 0.1 sin πy on [0,2] (Gardiner & Stone 2005, RAMSES) |
| `kh_uniform` | p = 1, v = ±0.5, 0.01 sin 4πx with σ = 0.1, Bx = 0.003 | Tomida et al. (2026) III.1: p = 10, v = ±1, 0.01 sin 2πx with σ = 0.2, Bx = 0.01 |
| `kh_localized` | as above, band width 0.1 | as above, sech² band with Tomida's FWHM (w = 0.0945) |
| `kh_localized_dtjump` | jump at t = 1.5, `glm` + `ct` | Tomida III.3: jump at t = 3.0, run to 3.2; `glm`, `glm_hll16`, `glm_constch`, `ct` |

The science files (`src/problems.jl`, `src/run_suite.jl`,
`codes/common/abel_problems.hpp`, `codes/external.py`, the four configs) are
byte-identical to `marlowe_mhd`, except `kh_localized_dtjump.toml`, which adds
`slurm_time = "02:00:00"` (its pessimistic task estimate reached the 1 h default).
New outputs in these runs: `psi_max`, upper-layer leakage (`leak_band_bmax`,
`leak_band_brms`), per-step `dense_diagnostics.csv` around the timestep drop, and
device memory.

**Kept apart from the first campaign:** AthenaK is rebuilt with the corrected
header into `$SCRATCH/mhd-codes-rerun`, results go to `$SCRATCH/mhd-results-rerun`,
and every job is named `rerun-*`. `tools/submit_campaign.sh` refuses to run if
`MHD_CODES` or `MHD_RESULTS` points at the first campaign's directories.

**Partitions and preemption:** GPU jobs are submitted to `gpu,owners` with
`--requeue` and the H100 SXM5 feature, so they also start on idle owner nodes.
An owner can preempt them; Slurm then requeues the job and it restarts from t=0,
writing under `$MHD_RESULTS/restartN/` (`jobs/common.sh`). The interrupted attempt's
partial directory is kept but never packed into `returns/`. Override the partitions
with `--gpu-partition=gpu` or `MHD_GPU_PARTITION=gpu`.

**Scope and cost:** 4 gates (smoke of the four tests at 32³, AthenaK build,
current sheet N=64 pilots for GLM/CT and AthenaK) and 12 matrix commands: GPU
driver ×4, AthenaK matched/best ×3 tests, AthenaK references for both KH tests.
`julia tools/estimate_cost.jl` gives 4 GPU-h [2–8] plus under 1 GPU-h of gates.
RAMSES and Athena++ are not submitted (`--gpu-only`), as in the first campaign. If
CPU baselines are run later, the KH tests at N=128 need a longer RAMSES time limit
(≈4 h central, 6.7 h pessimistic, against the configs' 3 h).

## Commands

From the Mac (the tarball is built without macOS `._*` files):

```
cd /Users/lyxia/Projects/research/abel_mhd
scp sherlock_rerun.tar.gz sherlock_rerun.tar.gz.sha256 lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/
```

On a Sherlock login node:

```
cd /scratch/users/lyxia/mhd-study && sha256sum -c sherlock_rerun.tar.gz.sha256 && tar -xzf sherlock_rerun.tar.gz && cd sherlock_rerun && sha256sum -c --quiet BUNDLE_SHA256SUMS && echo BUNDLE_OK
```

Optional: cancel the first campaign's still-queued jobs for these four tests
(they would run the superseded setups). Review the list first; the pattern
cannot match `rerun-*` jobs.

```
squeue -u "$USER" -h -o '%F %j' | awk '$2 ~ /^mhd-(current_sheet|kh_uniform|kh_localized)/' | sort -u
squeue -u "$USER" -h -o '%F %j' | awk '$2 ~ /^mhd-(current_sheet|kh_uniform|kh_localized)/ {print $1}' | sort -u | xargs -r scancel
```

The first campaign's coordinator still has `kh_uniform` AthenaK jobs to submit;
run the same two lines again after it reports all commands submitted.

Submit from the login node (it fetches the pinned sources into
`$SCRATCH/mhd-codes-rerun`, then submits the gates and the matrix):

```
unset MHD_CODES MHD_RESULTS && bash tools/submit_campaign.sh --gpu-only
```

If it stops with `QOSMaxSubmitJobPerUserLimit`, hand the rest to the coordinator,
which retries every 5 minutes (use the ledger path it printed):

```
MHD_CAMPAIGN_LEDGER="$(ls -t campaigns/*.tsv | head -1)" sbatch jobs/20_auto_gpu_campaign.sbatch
```

Watch it:

```
squeue -u "$USER" -r -o '%.18i %.40j %.8T %.10M %R' | grep -E 'JOBID|rerun-'
```

Each job packs its own archive into `returns/` (with a `.sha256`), as in the
first campaign. Download that folder when the jobs finish and analyse these four
tests with `MHD_ANALYSIS_BUNDLE=sherlock_rerun`.
