# Checks: Courant number and single precision

Two things the sweep left open about Tom Abel's kernels, on one H100 in about two to four hours:

1. **Courant number.** The sweep ran the kernels at 0.25 in the solver's own rule (the sum over the three
   directions), the solver's default is 0.7, and the other codes run at 0.3 against the fastest single direction.
   Here the LLF, production and HLLD kernels run at **0.4 and 0.7**.
2. **Half precision.** The production kernel stores its tile in Float16 and stops converging past 64 to 128 cells.
   `glm_hll32` and `glm_ppm32` are the same kernels with a Float32 tile, at 0.25 (to set beside the sweep), 0.4
   and 0.7.

## What runs

`python3 checks.py plan` lists it: 91 run directories, 279 runs.

| Test | Grids | Why |
|---|---|---|
| `entropy`, `alfven` | 32 to 256 (entropy to 512 for `glm_hll32`) | error against the exact answer, order of convergence |
| `brio_wu`, `orszag_tang` | 32 to 256 | shocks |
| `blast_2d_b100` | 64 to 256 | where the half-precision kernels stop |
| `kh_localized` | 128 (256 for `glm_hll32` at 0.25) | leakage |
| `decay_supersonic` | 64³, 128³ (two seeds), 256³ | Mach 9, where the half-precision kernels stop |
| `orszag_tang_3d` | 128³, 256³; 512³ for LLF and `glm_hll32` at 0.4, LLF at 0.7 | time to solution |
| `throughput_ot3d` | 96³ to 512³, the two new kernels | speed of a Float32 tile, 4-cell and 6-cell |

Planar tests stop at 256 because the kernels are 3D: a planar test on N² costs N³ cells a step. Every setting other
than the Courant number, the kernels and the grids is the sweep's (`configs/` is written from the sweep's configs by
`python3 checks.py configs`). One timing repeat.

## How the Float32 variants are made

Nothing in Tom's source is edited. His `step_plm!` lists the (tile, precision, Riemann solver) triples he uses; the
kernels behind it, `integrator_plm!` and `integrator_plm_ppm!`, are generic in all three. Our driver
(`sherlock_sweep/src/run_suite.jl`, `plm_unlisted!`) calls the same kernels for a triple `step_plm!` does not list,
with `step_plm!`'s lines before the launch copied. The **control** `glm_hlld_x` calls a listed triple (HLLD, Float32)
through that same path: the trial requires it to reproduce `glm_hlld` digit for digit, or nothing runs.

The driver also accepted Courant numbers only up to 0.5; that guard is now 1, the limit of the sum rule.

## Run it

On the Mac, in the project directory (the driver source travels as `sherlock_checks/src`, so the finished sweep's
bundle on Sherlock is not touched):

```bash
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - -s ',^sherlock_sweep/src,sherlock_checks/src,' sherlock_checks sherlock_sweep/src | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_checks/check.sh'
```

On Sherlock, any time:

```bash
bash -l /scratch/users/lyxia/mhd-study/sherlock_checks/status.sh
```

Results: `$SCRATCH/mhd-results-checks/<test>-c<025|040|070>-N<n>-s<seed>/data/<kernel>_r1/`, the sweep's format.
The job first runs a trial (Orszag–Tang to t = 0.05 on 32³ and 48³, the second for the 6-cell tile), then the runs,
cheapest first. Requeued, it keeps what is finished. A kernel that fails its trial is left out and reported.

## Tested before sending

- `julia --project=../sherlock_sweep/env test_gpu_compile.jl` on a laptop: the new triples compile for the NVIDIA
  target (with both tile sizes), every configuration passes the driver's checks, every planned run is one its
  configuration allows.
- The job, submit and status scripts against stand-ins for Slurm and the driver; the trial verdict on identical,
  altered and missing controls; the table on real sweep records.
- **Not tested:** the new kernels on a GPU. That is what the trial is for.
