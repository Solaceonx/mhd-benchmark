# CPU comparison audit and exact-reference validation

The notebook now includes movies for every physics page. Orszag–Tang has
24 matched movies: GLM, CT and Athena at 64², 128², 256² and 512², for density
and magnetic magnitude. Grid/field selection changes all three movies together.
The Athena movies use the dense **512²** trajectory. Their provenance is separate
from the reference used for quantitative error tables. All methods share actual
output times, color limits and playback timing; no intermediate flow states are
invented. Changing grid or field pauses and resets all players.

Hydro decay and prepared decay retain their existing videos. The acoustic check
now has a new 21-output simulation and a linked movie of its linear analytical
reference. CT is explicitly inapplicable to these hydro-only experiments.

## Exact magnetized entropy wave

The unchanged planar GLM/CT driver evolves a periodic unit square with

```
rho(x,y,t) = 1 + 0.2 sin(2 pi (x+y-t))
v = (0.5, 0.5, 0), p = 1, B = (0.3, 0.2, 0)
```

This is an exact finite-amplitude solution: constant velocity advects density,
and constant pressure and magnetic field have zero force gradients. The
conserved momentum and energy are affine in density. The cell-average density
therefore replaces 0.2 by `0.2*sinc(1/N)^2`; initialization and the reference use
the same averages. This avoids an initialization sampling discrepancy.

Both methods run at 16², 32², 64² and 128², to one period t=1, using common CFL
steps, 41 output times, Float32 evolution and a Float64 analytical reference.
The pressure-energy conversion uses the solver's Float32 gamma consistently.

| N | GLM density mean absolute error | CT density mean absolute error |
|---:|---:|---:|
| 16 | 1.896602e-2 | 1.896591e-2 |
| 32 | 3.894079e-3 | 3.894148e-3 |
| 64 | 1.189467e-3 | 1.189329e-3 |
| 128 | 3.292124e-4 | 3.293870e-4 |

Successive observed L1 orders are approximately 2.28, 1.71 and 1.85 for both
methods. These describe this particular test, limiter and resolution range.
They are not a claim of universal convergence order. Positivity is checked at
every RK stage; all saved mass/energy drifts are below 1e-5. Pressure and B
remain constant within 1e-4. The 64² states and exact reference have three linked
movies on `progress/entropy-wave.html`.

Because B stays uniform, this checks multidimensional entropy transport, not
magnetic-wave propagation or divergence handling under evolving field structure.
Smooth waves are a standard convergence diagnostic, complementary to shocks;
see the [Athena wave-test discussion](https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html).
Our finite-amplitude entropy setup is not a reproduction of its linear-wave suite.

## Finer independent reference completed

Athena++ 1024² completed to t=0.5; the active notebook now uses it for norms
and 512² for reference sensitivity. All four local grids have a finer reference.
See [the updated errors and sensitivity table](MHD_SPECTRA.md). No simulations
from this pass remain running.

## Performance: still an open measurement

The old run times are single exploratory samples. They exclude compilation but
have different measurement scopes, share a timestep, and some overlapped other
work. Do not interpret them as reliable GLM/CT speedups or Athena comparisons.

`benchmark_cpu.jl` is prepared for **five repetitions** at 64², 128² and 256².
It first computes a common CFL schedule using both methods, warms each full
replay, then measures each method separately from a fresh state, alternating
method order. Every repetition must exactly reproduce its untimed final state.
It leaves garbage collection enabled, collecting before each sample.

Its scope is compute-only: fluxes, RK updates, damping/face updates and stage
validity checks. It excludes schedule generation, initialization, diagnostics,
disk output and compilation. It is neither end-to-end time nor cost at equal
accuracy. OS activity, thermal state and clock frequency remain uncontrolled.
One- and four-thread sessions should be run sequentially on an otherwise quiet
machine, reporting median, spread and cell updates/s. **These repetitions have
not been run**; the user requested finishing the existing simulations first.

## What still improves the study

- Reference sensitivity at each local grid, including 512², using a finer Athena
  solution. Finite-reference differences remain estimates, not exact error bounds.
- Smooth magnetosonic waves and magnetic-loop advection to exercise evolving B.
- Shock tubes and rotor tests for robustness; these may require additional boundary
  conditions and the current solver's planar restriction must be respected.
- Timestep and precision sensitivity, and consistently averaged OT initial data.
- Controlled repeated timing, followed by time-to-solution at matched error.
- Additional physics tests before claiming a general GLM-versus-CT ranking.

The local methods share HLL/PLM/RK2; Athena uses HLLD/PPM/RK3 and double precision.
Comparisons assess these complete implementations. They do not isolate every
algorithmic difference or establish CT as universally superior.

## Reproduce in fresh directories

```sh
julia --project=GLMMHDTurb -t 4 scripts/run_entropy_wave.jl scripts/out/entropy_wave_new
julia --project=GLMMHDTurb -t 4 scripts/run_acoustic_movie.jl scripts/out/acoustic_movie_new

# Render the canonical saved runs (no simulation):
.venv-movie/bin/python scripts/render_validation_movies.py
.venv-movie/bin/python scripts/build_progress_site.py --movies

# Optional later measurements; run each separately without other compute jobs:
julia --project=GLMMHDTurb -t 1 scripts/benchmark_cpu.jl
julia --project=GLMMHDTurb -t 4 scripts/benchmark_cpu.jl
```

`render_validation_movies.py` checks declining entropy error and observed orders,
records the new acoustic errors, renders five videos and updates the two page
records in `progress/tests.json`. The original acoustic check results remain
labeled historical; extra output endpoints change the movie run's timesteps.
