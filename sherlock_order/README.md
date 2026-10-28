# Order x solver: what a higher-order reconstruction buys, apart from the Riemann solver

The work-precision study (`scripts/work_precision.py`, "Our figures") compares each code's sweep set-ups, but those
change the reconstruction, the Riemann solver and the time integrator at once (AthenaK and Athena++: PLM/HLLE/RK2 or
VL2 against PPM/HLLD/RK3), or only one of them in one code. Here every code that can change them one at a time gets
all four pairs of reconstruction (PLM, PPM) and solver (HLL-family, HLLD) at one integrator, so each code's gain at a
given error splits into the reconstruction's, the solver's and how the two combine, and the codes can be compared on
each.

| code | PLM, HLL-family | PPM, HLL-family | PLM, HLLD | PPM, HLLD |
|---|---|---|---|---|
| AthenaK, RK3, H100 | `athenak_plm_hlle_rk3` (new) | `athenak_ppm_hlle` (new) | `athenak_plm_hlld` (new) | `athenak_best` (sweep) |
| Athena++, RK3, CPU | `athenapp_plm_hlle_rk3` (new) | `athenapp_ppm_hlle` (new) | `athenapp_plm_hlld` (new) | `athenapp_best` (sweep) |
| Abel et al., Float32 4-cell tile, H100 | `glm_fix_hll32` (new) | `glm_fix_ppm32` (new) | `glm_fix_hlld` (fixes rerun) | `glm_fix_ppm_hlld` (new) |

- AthenaK and Athena++: run-time settings (`order.py`, `VARIANTS`), passed to the sweep's own `codes/external.py`;
  Athena++ uses the sweep's two builds (it fixes the solver at build time). PLM/HLLE at RK3 against the sweep's
  PLM/HLLE ("matched", RK2 or VL2) also gives the integrator's share. Everything else as in the sweep: AthenaK in
  double precision runs the planar tests as z-invariant cubes, Athena++ on the plane with the sweep's rank count for
  each test and grid.
- Abel et al.'s kernels, sweep-fixes branch (`vendor/`, `prepare.sh`), unedited: the sweep's GPU driver
  (`sherlock_sweep/src/run_suite.jl`) calls the PLM kernel with HLL (a triple `step_plm!` lists) and the PPM kernel with
  HLL or HLLD (triples it does not list; the kernels are generic in the solver), all at Float32 with the 4-cell tile of
  `glm_fix_hlld`, with c_h from the CFL step as the other fixed kernels. His Courant rule (0.25 summed over three
  directions) as in the sweep.
- Tests: the work-precision tests except the current sheet (no set-up converges on it): entropy, alfven,
  alfven_oblique, field_loop_novz, brio_wu, brio_wu_rotated, orszag_tang, rotor, blast_2d_b0, blast_2d_b50, kh_uniform,
  gem, field_loop_3d; each config's grids up to 512, seed 42, one repeat. 567 runs.
- Cost (`order.py plan`, from the sweep's records of the nearest set-up): about 29 GPU hours for Abel et al.'s three
  kernels, 37 for AthenaK's three, 620 core-hours for Athena++'s three.

## Run it

On the Mac, in the project directory: put the branch into `vendor/`, write the configs and estimates, compile-check
(Homebrew's Julia needs OpenSSL 3 on the library path since Homebrew linked OpenSSL 4):

```bash
bash sherlock_order/prepare.sh && python3 sherlock_order/order.py configs && (cd sherlock_order && DYLD_LIBRARY_PATH=/opt/homebrew/opt/openssl@3/lib julia --project=../sherlock_sweep/env test_gpu_compile.jl)
```

Send it and submit (the driver source travels as `sherlock_order/src`, so the sweep's bundle on Sherlock is not
touched). `submit.sh` submits three pilots (Orszag–Tang at 32² and 64² with every new set-up) and the arrays, each
waiting on its pilot; it refuses while an `order-*` job is queued:

```bash
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - -s ',^sherlock_sweep/src,sherlock_order/src,' sherlock_order sherlock_sweep/src | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_order/submit.sh'
```

On Sherlock, any time:

```bash
bash -l /scratch/users/lyxia/mhd-study/sherlock_order/status.sh
```

If a pilot fails, its array waits forever (`DependencyNeverSatisfied`): cancel that array (`scancel` with its job id
from `status.sh`) and send the status output. After a task ran out of time: `bash -l submit.sh resume` (no pilots;
finished runs are kept, an interrupted one continues).

## Fetch

On the Mac, into `order_results/` (git-ignored), then the errors and the analysis:

```bash
mkdir -p order_results && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash /scratch/users/lyxia/mhd-study/sherlock_order/pack_for_mac.sh' | tar -xzf - -C order_results && .venv-movie/bin/python scripts/test_diagnostics.py entropy alfven alfven_oblique field_loop_novz brio_wu brio_wu_rotated orszag_tang rotor blast_2d_b0 blast_2d_b50 kh_uniform gem field_loop_3d && python3 scripts/work_precision.py
```

`scripts/sweep_data.py` reads the runs (`ORDER`, `ORDER_RESULTS`); `work_precision.py` prints each code's split
(`GRID`, `factorial`).
