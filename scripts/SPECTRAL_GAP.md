# Driven preparation followed by a velocity-gap decay experiment

This is a small, pure-hydrodynamic adaptation of the BOX preparation idea in
[Lane et al. (2021), §§2.1.3–2.2](https://arxiv.org/pdf/2110.14816).
It does not reproduce their magnetized cloud or TURBSPHERE geometry, gravity,
physical forcing timescale, or star formation calculations.

## Model and preparation

The new `isothermal_hydro.jl` follows the existing Orszag–Tang CPU workflow:
SIMD lanes along x, periodic ghost cells, MonCen piecewise-linear reconstruction,
MUSCL–Hancock prediction, and LLF interface fluxes. It evolves four conserved
fields: density and three momentum components. The equation of state is exactly
P = rho cs² with cs = 1. There is no total-energy equation, magnetic field,
gravity, explicit viscosity, or pressure reset. This differs from the earlier
gamma = 1.01 demonstration: heat from dissipation is implicitly removed, not
retained. Nonpositive trial states would be rejected and retried; none occurred
in this experiment. Face reconstruction falls back to first order in lanes
where its density would otherwise be nonpositive.

The unit periodic box has 64³ cells, runs on four CPU threads, and starts from
uniform density at rest. A solenoidal Gaussian acceleration field is generated
in Fourier space with 1 ≤ |k| ≤ 4, a Gaussian amplitude envelope centered at
k = 2 with width 0.65, and seed 20260923. Wavenumbers are in units of 2π/L.
The real field is normalized by its ensemble rms, not realization by realization.
OU updates use exp(-dt/tau), with dt = 0.005 and tau = 0.05. The mass-weighted
mean acceleration is subtracted to avoid bulk acceleration; this changes only
the spatially constant mode and leaves the forcing solenoidal.

The forcing amplitude is calibrated during the first four nominal box crossing
times (each 0.1 code time) to approach Mach 10, then held fixed. Preparation ends
when, over a four-crossing-time window, the mean Mach is 8–12 and comparing its
two halves gives less than 25% kinetic-energy drift, 15% drift in the standard
deviation of log density, and 0.25 L1 change in the normalized velocity spectrum.
These are practical stationarity checks, not proof of a resolved inertial range.

The run passed at eight nominal crossing times. Over the last four:

| Diagnostic | Measured result |
|---|---:|
| Mean Mach | 10.114 |
| Kinetic-energy mean drift | 1.69% |
| Log-density standard-deviation drift | 1.63% |
| Normalized velocity-spectrum L1 drift | 0.0247 |

## Branching and spectra

The prepared snapshot has Mach 10.341. The unchanged branch retains it exactly.
For the gap branch, Fourier velocity coefficients with 4 ≤ |k| < 7 are set to
zero. A spatially uniform velocity correction restores the original momentum.
Density is unchanged, so pressure is unchanged. Remaining nonzero modes are
not rescaled. Both branches have no forcing thereafter.

Removing these modes eliminates 13.69% of the nonzero-mode velocity variance
and 15.76% of the mass-weighted turbulent kinetic energy. The gap branch starts
at Mach 9.491. This energy difference is part of the perturbation, so differences
in subsequent decay cannot be attributed to the gap's location alone.

The velocity spectrum is the shell sum of |FFT(v - v_bulk)|² / (2 N⁶).
The kinetic spectrum uses sqrt(rho) (v - v_bulk), whose shell sum including k = 0
matches the mean turbulent kinetic energy by Parseval's identity. Shells cover
[k, k+1). The velocity variance fraction above excludes the k = 0 bin. The
mass-weighted bulk velocity and volume-weighted mean velocity need not coincide.

The gap is imposed on velocity, not on the density-weighted field. Multiplying
by sqrt(rho) mixes Fourier modes, so the kinetic spectrum retains power in the
selected band even at the initial instant. The movie shows both spectra.

Times in both decay branches use the **same** initial control crossing time,
L / v_rms,control = 0.0967007. Each branch runs for three such crossing times,
with 101 equally spaced density frames and spectra.

## What happened

The initially empty velocity band regains power rapidly. Relative to the
unchanged branch's band power at the same time, the gap branch first exceeds:

| Relative band power | First saved time, in control crossing times |
|---|---:|
| 25% | 0.06 |
| 50% | 0.09 |
| 90% | 0.27 |

These are sampled threshold crossings, not interpolated measurements. The
reference branch's power is also decaying. At t = 3 crossings, Mach numbers are
1.651 (unchanged) and 1.496 (gap), while band powers are almost equal.

The 64³ run took 77.2 seconds for preparation and both decays, excluding initial
compilation and rendering. Preparation took 56.6 seconds; the two decays took
10.5 and 9.4 seconds. No trial steps were rejected. Maximum errors in mean mass
and mean momentum were 1.02e-8 and 9.02e-8 in code units. Total energy is not a
conserved variable in the isothermal model.

The selected wavelengths have only about 9–16 cells. This is an exploratory
demonstration: numerical dissipation and the disturbed density–velocity
correlations affect the response. Recovery of band power is not a measurement
of spectral energy flux or proof of a particular cascade mechanism. A physical
recovery timescale needs higher-resolution runs, matched forcing realizations,
and multiple prepared snapshots.

## Validation

`check_isothermal.jl` checks exact preservation of a uniform moving state and
propagation of a small-amplitude acoustic wave. Mean density errors decrease
from 4.95e-6 at 32³ to 1.11e-6 at 64³. The experiment checks finite positive
density, mass/momentum conservation, and the kinetic-spectrum Parseval sum.

Independent Python FFTs of saved checkpoints verify identical density,
1.80e-17 residual velocity-band power relative to its original value, only
2.55e-8 relative change in unfiltered nonzero velocity modes, and 5.81e-10 change
in mean momentum. Details are in `out/spectral_gap_64/validation.json`.

## Reproduce and inspect

From the repository root:

```sh
julia --project=GLMMHDTurb -t 4 scripts/check_isothermal.jl
julia --project=GLMMHDTurb -t 4 scripts/run_spectral_gap.jl 64 scripts/out/spectral_gap_new
.venv-movie/bin/python scripts/make_spectral_gap_movie.py scripts/out/spectral_gap_new
```

Use a new output directory; existing results are never overwritten by the
simulation. The renderer uses the environment described in `DECAY.md` and
`requirements-movie.txt`. The same forcing seed at a different N currently
produces a different random realization because noise is drawn on the full
grid; it is not a matched-realization resolution test.

Outputs in `out/spectral_gap_64/`:

- `spectral_gap_comparison.mp4` and `.gif`: density, spectra, kinetic energy,
  and velocity-band power, all compared at common physical times.
- `initial_comparison.png`, `preview.png`: stills.
- `spectra_vs_time.png`, `gap_recovery.png`, `preparation.png`: full history,
  early gap recovery, and preparation checks.
- `metadata.toml`, `preparation.csv`, per-branch diagnostics and spectral CSVs.
- `prepared_state.bin`, `gap_initial_state.bin`, and each final state: full 3D
  conserved fields, Float32 little-endian, Julia shape (N,N,N,4), x varying fastest.
- Per-branch middle-plane density binaries, shape (N,N), x varying fastest.
