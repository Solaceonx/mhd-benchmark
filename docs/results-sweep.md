# Results of the sweep

Stage 1 ran on Sherlock from 2026-09-30 to 2026-10-05: 1,261 runs in 227 Slurm jobs. 1,149 completed and 112 were
stopped by their own solver, which is a result and not a failure of the job. About 211 GPU-hours and more than 80,000
core-hours. AREPO added 164 runs afterwards (157 completed). Numbers below are computed by the scripts in `scripts/` from the fetched
records; the notebook's Sweep and Papers pages show them run by run.

## Speed, one H100, 3D Orszag–Tang

| Set-up | 10⁹ cell updates per second |
|---|---|
| GLM, LLF | 2.01 |
| GLM, production (half precision, HLL) | 1.99; 3.1 to 3.4 on grids divisible by 6 |
| GLM, HLLD | 1.63 |
| GLM, PPM | 0.58; 1.12 to 1.15 on grids divisible by 6 |
| CT prototype | 1.21 |
| AthenaK, PLM/HLLE/RK2, 512³ | 0.28 to 0.40 |
| AthenaK, PPM4/HLLD/RK3, 512³ | 0.19 to 0.23 |

The science tests all ran on power-of-two grids, so they never saw the production kernel at its best speed.

**Time to solution** on 3D Orszag–Tang at 512³: GLM LLF 1,211 s, production 1,232 s, CT prototype 1,648 s, AthenaK
PLM 1,930 s, AthenaK PPM 3,387 s, Athena++ PLM about 7,650 s on 128 cores. Tom's kernels took 17,302 steps against
AthenaK's 5,264, because the sweep ran them at Courant number 0.25 in a rule that sums the three directions. At
the code's default the GLM times would be about 2.8 times shorter; that was not run.

## Accuracy

- GLM with LLF or HLLD on Float32 tiles is second order up to 256 to 512 cells, and within about 25 % of the same-order
  CT codes on Brio–Wu, Kelvin–Helmholtz, the blasts and GEM. HLLD sits between the matched and best set-ups.
- The three half-precision kernels stop converging past 64 to 128 cells and their error then rises with resolution
  (entropy wave: 1.6 × 10⁻³ rising to 3.2 × 10⁻³, where LLF reaches 3.4 × 10⁻⁵ at 512).
  [Why](issues-found.md#toms-kernels).
- The CT prototype is first order (measured order 0.74 to 0.99) with 50 to 250 times the LLF kernel's error.
  [Why](issues-found.md#toms-kernels).

## Leakage into a field-free region

Median field strength inside the field-free vortex of `kh_localized` at t = 3, relative to the 4096² reference:

| | 128² | 512² |
|---|---|---|
| AthenaK, RAMSES, Athena++ (six set-ups) | 0.9 to 1.8 | 1.0 |
| GLM kernels | 11 to 110 | 8 to 13 |
| CT prototype | 389 | 3.4 |

GLM's leak levels off and does not converge away, and a constant cleaning speed leaks as much. The prototype's is
plain diffusion and converges. At the sudden timestep change the largest cleaning potential jumps 40-fold (LLF) and
35-fold (production), and 1.5-fold with constant c_h: Tomida et al. report about 40. A cell-centred CT update in place
of the cleaning, tried on the CPU, brings the leak to 2.4 times the reference at 256²
(`scripts/solver_checks/`).

## Robustness

Runs finished, of 140 per kernel: LLF 139, CT prototype 140, constant c_h 119, HLLD 118, production 106, PPM 86.
Of the 132 stops, 62 are in the six blast tests, which already ran with a pressure floor. At Mach 9 the LLF kernel
finishes 7 of 8 runs, where AthenaK PPM/HLLD and RAMSES HLL stall and Athena++ PPM/HLLD ends in NaN. AREPO: 82 of 82
on the moving mesh, 75 of 82 on the static one.

## Divergence

The normalised divergence |∇·B| Δx / |B| of the GLM kernels is 10⁻³ to 10⁻². It is the measure Ugalino et al. 2026
use to compare their codes. In the Mach 9 decay the growth of magnetic energy agrees between five codes to about
±20 %.

## What the comparison papers show, redrawn

The Papers page redraws 40 figures of Tomida et al. 2026, Kritsuk et al. 2011, Gardiner & Stone 2005 and 2008, Birn
et al. 2001 and Ugalino et al. 2026 from these runs (`scripts/paper_*.py`). Only Tomida et al. is CC BY, so only its
original figures are shown beside ours.

## Caveats

- Courant number: see above. Every speed-per-accuracy statement inherits it.
- No Float32 run of the production kernel was made, so "half precision is the cause" rests on the CPU checks.
- `decay` starts at Mach 2; nothing here is subsonic.
- The first campaign (pages 01 to 06 of the notebook, `sherlock_mhd/`) ran only the LLF kernel and the prototype.
