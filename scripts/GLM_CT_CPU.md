# Matched 2D CPU GLM–CT comparison

The subsequent [64²–512² spectral study](MHD_SPECTRA.md) uses this same solver.
The runner now saves native CT face arrays at every output frame and accepts an
optional third command-line argument for frame count (default 101).

The current local baseline is a genuine 64 × 64 Orszag–Tang MHD problem on the
MacBook CPU, with four Julia threads. Arrays are (N,N,9), not a cubic grid with
repeated slices. The spectral-gap and hydrodynamic experiments are separate and
are not involved in this comparison.

## What is reused and what changed

`mhd_2d_cpu.jl` provides a shared CPU driver for both magnetic-field methods. It
imports the existing primitive/conserved conversions, ideal-MHD fluxes,
magnetosonic speeds, MonCen limiter and HLL solver from
`GLMMHDTurb/glmmhd_turb.jl`. The CT edge-electric-field average and staggered curl
follow `GLMMHDTurb/ct_mhd.jl` (Balsara–Spicer arithmetic averaging).

The driver adapts the methods to CPU loops and true 2D storage. Both paths use
the same PLM reconstruction and SSP-RK2 time integration. **This is a shared
reference implementation, not a timing run of the original optimized GPU
kernels or a literal port of their Hancock time predictor.** The original
CUDA and SIMD files are unchanged. Although the imported library loads CUDA.jl,
this driver allocates only CPU arrays and launches no CUDA kernels.

GLM evolves cell-centered Bx, By and psi, using the existing GLM normal-field
interface solution. Cleaning speed is the largest directional signal speed
over both states at the beginning of each shared step. Psi damping is
Strang-split, with decay time 0.18 L/ch, following the existing parameterization.
There is no psi energy contribution or magnetic-energy correction in the
total-energy variable, matching the existing GLM formulation.

CT evolves Bx and By on lower x/y faces via the discrete curl of corner Ez.
Hydrodynamic fluxes use the single face-normal B. Cell-centered B is reconstructed
by adjacent-face averaging for pressure recovery and diagnostics. Total energy
is updated conservatively, without a post-update magnetic-energy correction.

This driver currently supports **planar MHD only**: vz = Bz = 0; nonplanar states
are rejected. It is not yet a general 2.5D/3D solver. CT here is the simple
arithmetic-EMF prototype, not an upwind-CT implementation.

## Matched configuration

- Unit periodic square; N = 64; gamma = 5/3; Float32 states.
- Standard Orszag–Tang initial condition: rho = 25/(36 pi), p = 5/(12 pi),
  vx = -sin(2 pi y), vy = sin(2 pi x), Bx = -sin(2 pi y)/sqrt(4 pi),
  By = sin(4 pi x)/sqrt(4 pi). vz, Bz and psi are zero.
- For these particular fields, transverse-only variation makes native CT face
  sampling and cell-centered GLM sampling agree exactly after face averaging.
  Initial density, momentum, pressure, B, and total energy are identical.
- HLL flux, MonCen PLM and SSP-RK2 for both methods. Shared positivity fallback
  zeroes reconstruction slopes where a face density or pressure would fail.
  Raw stage states are checked; no density or pressure repairs are applied.
- A common time step obeys dt <= 0.35 dx/(2 ch), including GLM cleaning speed,
  and lands on each of the 101 output times through t = 0.5.

## Results on this MacBook

Both methods completed 500 common steps. Excluding compilation, the entire
comparison including output and diagnostics took 1.03 seconds. Step calls took
0.364 s for GLM and 0.384 s for CT. These calls include stage validity checks;
they exclude CFL computation, output, and external diagnostics. Invocation order
alternates each step. This is a single small CPU run, **not a robust performance
benchmark or evidence of relative GPU performance**.

| Quantity | GLM | CT |
|---|---:|---:|
| Maximum relative mass error at saved times | 1.29e-8 | 1.91e-8 |
| Maximum relative total-energy error at saved times | 1.73e-8 | 2.20e-8 |
| Final minimum density | 0.0987 | 0.0997 |
| Final minimum pressure | 0.0351 | 0.0350 |
| Final cell-centered divergence RMS | 0.1171 | 0.1673 |
| Final dx × cell-centered divergence RMS / B_rms | 0.00551 | 0.00777 |
| Final native face-divergence maximum | not applicable | 1.47e-4 |

The final mean absolute density difference divided by the CT mean density is
1.387%. This is a difference between solutions, not an accuracy error against a
known solution. Determining accuracy requires reference solutions or convergence
studies.

**The divergence operators matter.** CT's preserved constraint is the difference
of opposite *face* fields. Its curl update cancels analytically; the measured
small residual accumulates from Float32 arithmetic over steps and is amplified
by division by dx. Applying central differences to the *interpolated cell*
fields is a different operator and need not vanish. The movie explicitly shows
that common cell-centered operator for both methods, while labeling native CT
face divergence separately. A larger cell-centered CT diagnostic does not mean
that CT failed its discrete constraint.

## Checks and artifacts

`check_mhd_2d.jl` passes 15 checks: uniform-state preservation, identical OT
initial states, zero initial face divergence, preservation of divergence by a
random discrete curl update, reduction of an imposed GLM divergence error, and
short-run positive pressure/density plus mass/energy conservation. GLM reduces
the test's divergence RMS from 0.0883 to 0.0176 over t = 0.25.

Independent Python checks verify all 202 saved states have finite positive
density/pressure and retain planar symmetry, identical output times, conservation,
and agreement of independently computed divergence with the Julia diagnostics.
These are initial validation checks; this is not a complete MHD shock-test suite.

Outputs in `scripts/out/glm_ct_64/` include:

- `glm_ct_comparison.mp4` and `.gif`: synchronized density, |B|, density difference
  and divergence histories. Scales are fixed and shared between methods.
- `comparison.png`, `diagnostics.png`: final fields and numerical diagnostics.
- `metadata.toml`, `validation.json`, and per-method diagnostic CSVs.
- `glm_000.bin` through `glm_100.bin`, and matching `ct_*.bin`: little-endian
  Float32 conserved states in Julia shape (N,N,9), with x varying fastest.
- `ct_final_bxface.bin`, `ct_final_byface.bin`: final native CT face fields.

## Reproduce

From the repository root, using a fresh output directory:

```sh
julia --project=GLMMHDTurb -t 4 scripts/check_mhd_2d.jl
julia --project=GLMMHDTurb -t 4 scripts/run_glm_ct_2d.jl 64 scripts/out/glm_ct_new
.venv-movie/bin/python scripts/plot_glm_ct_2d.py scripts/out/glm_ct_new
```

The renderer uses the existing movie environment (`requirements-movie.txt`).
The simulation refuses to overwrite an existing nonempty output directory.
Pass a different N for another grid size. These CPU results provide a local
reference while GPU access is being arranged; future GPU comparisons should
match the numerical algorithm before drawing performance or accuracy conclusions.
