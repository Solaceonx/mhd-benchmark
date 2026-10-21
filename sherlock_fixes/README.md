# The sweep-fixes branch of Tom Abel's kernels, on Sherlock

Tom asked for the problems the sweep found in his kernels to be fixed on a branch and tested. The branch is
`sweep-fixes` of his repository (github.com/yipihey/GLMMHDTurb), made in the clone `GLMMHDTurb/` (not committed here:
`prepare.sh` copies it into `vendor/`, git-ignored). Four commits on top of `15ce37d`:

| Commit | Fix | Evidence before the GPU |
|---|---|---|
| HLLD guard | `abs(denK) < 1f-12` never triggers in Float32; now `abs(denK) < 1f-4*pts` (star total pressure) | CPU emulation: current sheet 32–128 and GEM 128 run to the end (they stopped after 6–17 steps and at t = 0.09) |
| c_h from the CFL step | every `step_*` takes `dt_cfl`; a step shortened to land on an output keeps its CFL step's c_h | emulation: the inflated c_h killed runs right after an output |
| CT prototype | full limited slope (`pslope` returned half), predictor with the fluxes of all three directions (`hanc3d`) | emulation: second order, ~90× less error at 128² |
| Float16 tile, cell-centred CT | `anchored`: Float16 differences from tile anchors, fluxes as dt/dx·F in head + tail; `fct`: flux-CT (Tóth 2000) in place of GLM | emulation: entropy wave second order again (256²: 6.0e-5 against 2.4e-3); leak into the field-free vortex 3.2× the reference against 111× |

The Courant number needs no code change (`sherlock_checks/` measures 0.4 and 0.7).

## Tested before sending (on the Mac)

- `scripts/solver_checks/fix_branch.jl` runs the branch's kernels on the CPU, as written, through
  `scripts/solver_checks/kernel_cpu.jl` (a stand-in for CUDA's thread intrinsics):
  - with every fix off, the branch reproduces the original kernels bit for bit (PLM f16 with both tiles, PPM, HLLD with
    the old guard put back, `step_tiled!`, and the driver's own launch of a Float32 tile);
  - each fix in 3D on z-invariant data agrees with the planar emulation the README of `scripts/solver_checks/` reports
    (to 1e-7, or below the Float16 rounding itself where a value is stored in Float16), so those tables are the branch's;
  - anchored storage conserves mass across tile edges; cell-centred CT keeps the corner divergence of B at 3e-7 of |B|
    on a 3D state (GLM, same kernel: 1.8e-3); the B = 100/√4π blast at 32³ runs to the end with anchored storage and
    stops after 17 steps with the original.
- `julia --project=../sherlock_sweep/env test_gpu_compile.jl` compiles every new kernel for the NVIDIA target and checks
  the configurations and the run list against the driver.
- **Not tested:** anything on a GPU, and the speed. That is what this job is for.

## What runs

`python3 fixes.py plan ../progress/sweep_status/export_latest.json` lists it with the GPU time each run should take
from the sweep's rates: 38 run directories, 189 runs, about 5 GPU hours. Every setting is the sweep's (Courant 0.25,
one repeat), so each result sits beside the sweep's record of the original kernel. The driver's methods
(`sherlock_sweep/src/run_suite.jl`):

| Method | Kernel | Set beside |
|---|---|---|
| `glm_fix` | `step_tiled!` LLF, c_h from the CFL step | `glm` |
| `glm_fix_hll16` | production kernel, anchored Float16 | `glm_hll16` |
| `glm_fix_hlld` | Float32 tile, HLLD with the new guard | `glm_hlld` |
| `glm_fix_ppm` | PPM, anchored Float16 | `glm_ppm` |
| `glm_fct16` | production kernel, anchored Float16, cell-centred CT | `glm_hll16` |
| `glm_fct32` | Float32 tile, HLL, cell-centred CT | `glm_hlld` |
| `ct_fix` | CT prototype, full slope, predictor in all directions | `ct` |

`glm`, `glm_hll16` and `glm_ppm` also run in the speed test in the same job: on the branch they are the original
kernels bit for bit. With the branch in `vendor/`, `glm_hlld`, `glm_hlld_x` and `ct` refuse to run (the branch
changed what they run).

| Tests | Grids | For |
|---|---|---|
| `entropy`, `alfven` | 32–256 (entropy 512 for the two Float16 kernels) | convergence of the Float16 kernels and the CT prototype |
| `blast_2d_b50`, `blast_2d_b100`, `blast_3d_b100` | 64–256 (3D: 64, 128) | Float16 range, c_h on shortened steps, HLLD |
| `current_sheet`, `gem` | 64–256, 128–256 | the HLLD guard |
| `decay_supersonic` | 64³–256³ | Mach 9 |
| `kh_localized`, `field_loop` | 128–256, 64–256 | leakage; cell-centred CT |
| `orszag_tang`, `orszag_tang_3d` | 128, 256 | nothing may get worse |
| `throughput_ot3d` | 128³–512³ | speed, originals in the same job |

## Run it

On the Mac, in the project directory: put the branch into `vendor/`, compile-check, send (the driver source travels as
`sherlock_fixes/src`, so the sweep's bundle on Sherlock is not touched):

```bash
bash sherlock_fixes/prepare.sh && (cd sherlock_fixes && julia --project=../sherlock_sweep/env test_gpu_compile.jl)
```

```bash
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - -s ',^sherlock_sweep/src,sherlock_fixes/src,' sherlock_fixes sherlock_sweep/src | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_fixes/fix.sh'
```

On Sherlock, any time:

```bash
bash -l /scratch/users/lyxia/mhd-study/sherlock_fixes/status.sh
```

Results: `$SCRATCH/mhd-results-fixes/<test>-N<n>-s<seed>/data/<kernel>_r1/`, the sweep's format, and `summary.txt`
(each fixed kernel beside the original). The job first runs a trial (Orszag–Tang to t = 0.05 on 32³ and 48³ with
every new kernel); a kernel that fails it is left out and reported. Requeued, it keeps what is finished.

## Checking the fixes (job `mhd-verify`, 2026-10-19)

The runs above measure the fixes; these check that they are right. `python3 fixes.py --plan verify plan`: 32 run
directories, 155 runs, about 3–4 GPU hours, results in `$SCRATCH/mhd-results-verify`.

| Runs | Checks |
|---|---|
| `glm`, `glm_hll16`, `glm_ppm` on the branch: entropy and Orszag–Tang at 128 | with the fixes off the branch is Tom's code: `control.txt` compares every cell of `diagnostics.csv` with the sweep's run |
| entropy, oblique Alfvén at 48, 96, 192, 384 | the 6-cell tile (grids divisible by 6); every convergence run before used the 4-cell tile |
| oblique Alfvén 32–256, field loop without v_z 64–256: `ct_fix`, `glm_fct16/32` | the CT predictor and cell-centred CT on a wave oblique to the grid |
| Brio–Wu, Brio–Wu at 45°, rotor at 128, 256: every fixed kernel | shocks are no worse: `scripts/fix_accuracy.py` scores them on the Mac against the AthenaK 4096² reference |
| 3D blast with B = 50/√4π, decay from Mach 2 (3 seeds; 256³ one) | robustness in 3D |

Driven turbulence is not rerun: the pilot uses `step_hydro!`, which the branch does not change.

On the Mac, in the project directory (sends the plan; leaves `src/` and `vendor/` alone while a `mhd-fixes` job runs):

```bash
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - -s ',^sherlock_sweep/src,sherlock_fixes/src,' sherlock_fixes sherlock_sweep/src | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && if squeue -h -u lyxia -n mhd-fixes | grep -q .; then tar -xzf - --exclude=sherlock_fixes/src --exclude=sherlock_fixes/vendor; else tar -xzf -; fi && bash -l sherlock_fixes/verify.sh'
```

On Sherlock: `bash -l /scratch/users/lyxia/mhd-study/sherlock_fixes/status.sh verify`.

Fetch both result sets to the Mac (tables, records, last states of the planar shock, leak and field-loop runs), then score:

```bash
mkdir -p fix_results && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash /scratch/users/lyxia/mhd-study/sherlock_fixes/tools/pack_for_mac.sh' | tar -xzf - -C fix_results && python3 scripts/fix_accuracy.py
```

## The full rerun (2026-10-21)

lyxia and Tom keep Float16, so the fixed kernels replace the originals in the comparison: every sweep test (26, not the
resume gates) at every grid and seed the sweep used, one repeat (the sweep's repeats on `throughput_ot3d`), one run
directory per kernel, `$SCRATCH/mhd-results-rerun/<test>-N<n>-s<seed>-<kernel>/`.

Before it, the CT prototype got its last fix: edge EMFs upwinded by the mass flux (Gardiner & Stone 2005), with the
reference EMF from the predictor's half-step velocity and the start-of-step field (branch commit `7256715`). On the
CPU emulation (`scripts/solver_checks/fieldloop_emf.jl`) the field loop goes 0.26, 0.16 at 64², 128² (was 0.60,
0.54; GLM 0.26, 0.16), the Alfvén wave stays second order (as GLM with HLL), smooth advection converges as GLM. The
cell-centred CT option keeps its plain average: upwinding did not change its field loop (0.25 against 0.26 at 128²).

| Job | What | GPU hours |
|---|---|---|
| `emfcheck.sh` → `mhd-emfcheck` | `ct_fix` on field loops, Alfvén waves, entropy, Kelvin–Helmholtz leak, shocks (64–256) | ~2 |
| `rerun.sh glm` → `mhd-rerun-glm` (9 array tasks) | `glm_fix_hll16`, `glm_fct16`, `glm_fix_ppm`, `glm_fix_hlld` everywhere | ~69 |
| `rerun.sh ct` → `mhd-rerun-ct` (9 array tasks) | `ct_fix` everywhere, after `mhd-emfcheck` passes | ~69 |

`python3 fixes.py --plan rerun configs ../progress/sweep_status/export_latest.json` writes `configs_rerun/` and the
expected seconds of each run (`estimates.json`, from the sweep's rates and the measured slowdowns); `rerun.sh` splits
a group into tasks of about 8 hours (`RERUN_HOURS`), longest runs first onto the least loaded task. A task that runs
out of time: run `rerun.sh` again, it keeps finished runs and resumes interrupted ones.

On the Mac (sends the bundle; leaves `src/` and `vendor/` alone if any `mhd-` job is in the queue), then submits the
check and the GLM rerun:

```bash
cd /Users/lyxia/Projects/research/abel_mhd && bash sherlock_fixes/prepare.sh && COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - -s ',^sherlock_sweep/src,sherlock_fixes/src,' sherlock_fixes sherlock_sweep/src | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && if squeue -h -u lyxia | grep -q mhd-; then tar -xzf - --exclude=sherlock_fixes/src --exclude=sherlock_fixes/vendor; else tar -xzf -; fi && bash -l sherlock_fixes/emfcheck.sh && bash -l sherlock_fixes/rerun.sh glm'
```

On Sherlock: `bash -l /scratch/users/lyxia/mhd-study/sherlock_fixes/status.sh emfcheck` (or `rerun`); after the check
passes, `bash -l /scratch/users/lyxia/mhd-study/sherlock_fixes/rerun.sh ct`. Fetch with the command above (the
fetch script now includes the check and the rerun).
