# Checks of the solver kernels on the CPU

Written 2026-10-05 to answer "can Tom's code be improved": each weakness the comprehensive sweep found was traced
to a place in the source (`sherlock_sweep/vendor/glmmhd_turb.jl`, `ct_mhd.jl`, byte-identical to `GLMMHDTurb/`) and
tested here with **his own functions, unedited**, on small planar problems.

The functions (`cons2prim`, `prim_slope`, `hancock`, `riemann`, `hll_x`, `hlld_x`, the PPM helpers; `pslope`,
`hanc1d`, `hll` of the CT prototype) are loaded from the vendored files by line range (`vendor_physics.jl`, with a
hash check). `emu.jl` and `ct_driver.jl` re-implement only the loop structure of the GPU kernels on an N × N plane,
with values passing through storage at the same points. Each proposed change is one edited line of his source, or an
option of the emulation.

## How far the emulation can be trusted

It reproduces the sweep's records of the kernels that involve no half precision to three digits, and those of the
half-precision kernels to about 5 % (they depend on rounding details):

| Test (grids) | Sweep on the H100 | Emulation |
|---|---|---|
| Entropy wave, GLM LLF (32–256) | 3.33e-3, 7.89e-4, 2.01e-4, 4.84e-5 | 3.33e-3, 7.89e-4, 2.01e-4, 4.88e-5 |
| Entropy wave, GLM HLLD (32–256) | 3.87e-3, 1.11e-3, 2.91e-4, 7.42e-5 | 3.87e-3, 1.11e-3, 2.91e-4, 7.40e-5 |
| Entropy wave, GLM f16/HLL (32–256) | 3.21e-3, 1.59e-3, 1.95e-3, 2.53e-3 | 3.09e-3, 1.45e-3, 1.98e-3, 2.44e-3 |
| Entropy wave, GLM PPM (32–256) | 1.83e-3, 9.02e-4, 1.55e-3, 1.43e-3 | 1.86e-3, 9.26e-4, 1.56e-3, 1.39e-3 |
| Oblique Alfvén wave, GLM LLF (32–128) | 7.19e-3, 2.05e-3, 5.98e-4 | 7.18e-3, 2.05e-3, 5.98e-4 |
| Entropy wave, CT prototype (32–128) | 6.98e-2, 4.18e-2, 2.30e-2 | 7.00e-2, 4.18e-2, 2.30e-2 |
| Alfvén wave, CT prototype (32–128) | 1.85e-2, 9.66e-3, 4.94e-3 | 1.85e-2, 9.66e-3, 4.94e-3 |
| Field in the field-free vortex, GLM LLF (128, 256) | 4.32e-5, 8.79e-6 | 4.57e-5, 8.77e-6 |

Limits: planar problems only (no 3D run was emulated); nothing here says what a change costs on the GPU.

## Findings

### 1. The CT prototype uses half the slope and a one-direction predictor (`ct_slope.jl`)

`pslope` (ct_mhd.jl line 36) returns `0.5*moncen(...)`, half the limited slope, as the GLM code's `prim_slope` does.
But `hanc1d` (line 75) and the interface states (lines 112–113) multiply it by 0.5 again, where the GLM code uses ±1.
The reconstruction therefore uses half the slope, which is first order. Separately, `hanc1d` advances a cell with the
flux difference of the face's own direction only, so the fluxes through the other faces (half of every edge EMF) are
not centred in time.

| L1 error, N = 32, 64, 128 | Entropy wave | order | Alfvén wave | order |
|---|---|---|---|---|
| as written | 7.00e-2, 4.18e-2, 2.30e-2 | 0.74, 0.87 | 1.85e-2, 9.66e-3, 4.94e-3 | 0.94, 0.97 |
| full slope (one token) | 3.67e-3, 1.32e-3, 5.10e-4 | 1.48, 1.37 | 1.66e-3, 6.00e-4, 2.84e-4 | 1.47, 1.08 |
| full slope, predictor in all directions | 3.24e-3, 8.97e-4, 2.38e-4 | 1.85, 1.92 | 1.33e-3, 2.76e-4, 5.81e-5 | 2.27, 2.25 |

At 128² that is 97 and 85 times less error than as written. For comparison the sweep has, on the Alfvén wave at 128:
GLM LLF 1.9e-4, AthenaK PLM/HLLE 3.7e-4, AthenaK PPM/HLLD 1.4e-4.

### 2. Half-precision storage loses the convergence (`half_precision.jl`, first part)

`integrator_plm!` with `HALF` stores the primitives, the interface states **and the Riemann fluxes** in Float16
(`_spT` into `SP`, `LX` … `RZ`; the fluxes are written back into `LX`, `LY`, `LZ` on lines 830–836). REPORT.md says
the round trip touches only the reconstruction; the fluxes go through it too. A Float16 has 11 significant bits, so
slopes and flux differences of smooth flows are differences of numbers rounded at 5e-4 relative: the error grows with
resolution instead of falling.

Storing the difference from a reference instead (each tile's first cell for cell and interface states, that cell's
own flux for fluxes; every tile decodes a shared face identically, so conservation is kept) restores second order at
the same storage:

| Entropy wave, L1 density error, N = 32, 64, 128, 256 | PLM/HLL | PPM/HLL |
|---|---|---|
| Float16 as the kernel | 3.09e-3, 1.45e-3, 1.98e-3, 2.44e-3 | 1.86e-3, 9.26e-4, 1.56e-3, 1.39e-3 |
| Float32 | 3.23e-3, 8.96e-4, 2.38e-4, 5.89e-5 | 1.64e-3, 3.00e-4, 7.37e-5, 1.89e-5 |
| Float16 of differences from a tile reference | 3.23e-3, 8.88e-4, 2.34e-4, 6.02e-5 | 1.66e-3, 3.09e-4, 7.48e-5, 1.89e-5 |

With it the PPM kernel reaches 1.9e-5 at 256², below every PLM kernel and below AthenaK PPM/HLLD in the sweep (3.1e-5).
Which rounding matters more: with only the fluxes in Float32 the error at 256² is still 3.0e-3; with only the
primitives and interface states in Float32 it is 1.2e-3.

### 3. Half-precision storage overflows (`half_precision.jl`, second part)

The largest Float16 is 65504; a larger flux becomes Inf. Largest fluxes in Float32 runs:

| Run | energy flux | flux of ψ (c_h² B_n*) |
|---|---|---|
| Blast, B = 50/√4π, 64² / 128² | 3.3e4 / 3.9e4 | 5.3e4 / 5.9e4 (and 1e6 to 1e10 on steps shortened to land on an output) |
| Blast, B = 100/√4π, 64² / 128² | 4.1e4 / 4.6e4 | 1.1e5 / 1.3e5 |
| Blast, B = 0, 64² | 3.1e4 | 0 |
| 2D Mach 9 (stand-in for `decay_supersonic`), 128² / 256² | 4.4e4 / 8.5e4 | – |

The Float16 kernel stops exactly where these exceed 65504: both magnetised blasts within the first output interval
(the sweep: every grid), the Mach 9 flow at 256² but not at 128², and not the unmagnetised blast.

- Storing B_n* and multiplying by c_h² after reading it back (one line in `riemann_hll`/`riemann`, one in the update)
  carries both blasts to the end at 32², 64² and 128².
- Taking c_h from the CFL step instead of the actual, possibly shortened, step is enough for B = 50 but not B = 100.
- Storing the fluxes in units of 1/1024 of the grid's largest flux of each component (nine numbers per step) carries
  the Mach 9 flow and the blasts through, and leaves the entropy-wave errors of finding 2 unchanged.

### 4. HLLD's degenerate-case guard never triggers in Float32 (`hlld_tolerance.jl`)

`hlld_x` (line 227): `small = abs(denK) < 1.0f-12`, with `denK = d(S-u)(S-SM) - bn²`. Where the field lies along the
face normal and the Alfvén speed exceeds the sound speed, c_fast = c_Alfvén and `denK` is rounding noise of order
1e-6, far above the tolerance, so the star states are noise. One face of the current sheet (two states equal to 1e-5):
the original returns a flux of B_x of −0.1072 where the exact value is −0.0982, and a momentum flux of +0.0084 where
HLL gives −0.0005.

With `small = abs(denK) < 1.0f-4*pts` (a tolerance relative to the total pressure):

| Run | original | new guard |
|---|---|---|
| Current sheet 32², 64², 128² | stops after 12, 17, 6 steps | reaches t = 2.5 |
| GEM 128² | stops at t = 0.09 | reaches the end |
| Oblique Alfvén wave 32–128 | 7.03e-3, 2.19e-3, 6.20e-4 | 7.03e-3, 2.19e-3, 6.21e-4 |
| Orszag–Tang 128², energies at t = 0.5 | 0.045031, 0.060496 | 0.045031, 0.060496 |

(The step at which the original stops moves by a few steps when the surrounding code is compiled differently, as
expected of rounding noise.) The current sheet at 128² then takes 11,900 steps where LLF takes 4,300: some cells reach high signal speeds, so
that run is not fully healthy. The sweep's other HLLD stops (3D blasts, Mach 9, 3D Orszag–Tang, decay at 512³) are in
strongly magnetised regions too but were not emulated.

### 5. c_h follows 1/Δt (`leakage.jl` variant `ch_cfl`, `half_precision.jl`)

`ch = courant*dx/dt/3*scale` (lines 582, 751, 884) uses the step actually taken. A step shortened to land on an
output makes c_h, and the flux of ψ with it, arbitrarily large (finding 3; the timestep-change test on the Papers
page). When this emulation's time loop allowed leftover steps a hundred times smaller than the driver's, every
solver died right after an output. c_h from the CFL step (the `ctot` maximum already computed in `compute_dt`)
removes it; the sweep driver has this as `glm_ch_at_outputs = "cfl"`, used for the movie runs only.

### 6. The Courant number (`courant.jl`)

The sweep ran at 0.25 in the solver's rule (dt = C dx / max of the **sum** over three directions of |v| + c_fast);
the default in `Params` is 0.7. Entropy-wave error at 128² and Orszag–Tang 128²:

| | C = 0.25 | C = 0.4 | C = 0.7 |
|---|---|---|---|
| steps for the entropy wave | 2800 | 1760 | 1000 |
| LLF | 2.01e-4, finishes | 1.30e-4, finishes | 1.22e-4, finishes |
| HLL | 2.38e-4, finishes | 1.83e-4, finishes | 9.58e-5, pressure ≤ 0 at t = 0.28 |
| HLLD | 2.91e-4, finishes | 2.73e-4, finishes | 2.36e-4, pressure ≤ 0 at t = 0.44 |

So 0.4 costs nothing in accuracy or robustness here and saves 1.6× in steps; 0.7 saves 2.8× and is fine for LLF.

### 7. Leakage into field-free regions: not the damping; cell-centred CT removes most of it (`leakage.jl`, `cell_centred_ct.jl`)

The field that reaches the field-free vortex of the confined-field Kelvin–Helmholtz test (reference 4.1e-7):

| 128², LLF | field | × reference |
|---|---|---|
| GLM as in the sweep (ψ damped over 0.18 box lengths) | 4.57e-5 | 111 |
| no damping | 4.51e-5 | 110 |
| damping over 10, 2.5, 1, 0.33 cells | 4.65e-5, 4.41e-5, 3.81e-5, 2.47e-5 | 113 … 60 |
| `glm_ch_scale = 0.25` (the default in `Params`; the sweep used 1) | 4.70e-5 | 115 |
| cell-centred CT in place of GLM | 1.33e-6 | 3.2 |

| 256², LLF | field | × reference |
|---|---|---|
| GLM as in the sweep | 8.77e-6 | 21 |
| cell-centred CT in place of GLM | 9.77e-7 | 2.4 |

No setting of the cleaning changes the leak by more than a factor of two. Replacing GLM by flux-CT on the
cell-centred field (Tóth 2000, sec. 4.3: one EMF per cell corner, the average of the four face fluxes around it;
the field updated with central differences of it) keeps everything else of the kernel and has no ψ, no c_h and
8 variables instead of 9. On the other tests, next to GLM with the same solver:

| | GLM | cell-centred CT |
|---|---|---|
| Oblique Alfvén wave, LLF, 32–128 | 7.18e-3, 2.05e-3, 5.98e-4 | 6.33e-3, 1.73e-3, 4.67e-4 |
| Field loop 64², energy left at t = 2 (LLF / HLL) | 0.815 / 0.838 | 0.863 / 0.888 |
| Field loop with v_z, rms B_z/B_0 (LLF / HLL) | 5.6e-3 / 7.4e-3 | 1.1e-3 / 2.0e-3 |
| Divergence at cell corners (× dx) | 3.4e-7 | 7.6e-10 |
| Orszag–Tang 128², current sheet 64² and 128², blast B = 100 at 64² and 128² | finish | finish |

The cost is in the kernel layout: the update needs the fluxes one face further out, so a tile needs one more layer
of halo. That was not measured.

## Not examined

- Why the PPM kernel stops on more runs than PLM at 512 (field loops, Kelvin–Helmholtz, Orszag–Tang).
- Any 3D run, and the speed of any of the changes on the GPU.

## Running

Julia 1.12, no packages beyond the standard library. Times on four threads of the laptop:

| Command | Time |
|---|---|
| `julia ct_slope.jl` | 1.5 min (one thread) |
| `julia -t 4 half_precision.jl` | 6 min |
| `julia -t 4 hlld_tolerance.jl` | 2 min |
| `julia -t 4 courant.jl` | 40 s |
| `julia -t 4 leakage.jl 128 original nodamp cell0.1 cell0.4 cell1 cell3 scale0.25 cell_ct` | 3.5 min |
| `julia -t 4 cell_centred_ct.jl` | 1 min |

Grids can be given as arguments to the first four (`julia ct_slope.jl 32 64`) for a quick look.
