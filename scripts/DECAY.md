# A small decaying hydrodynamic turbulence movie

This uses the same Julia CPU SIMD PLM/LLF integrator as `render_ot_frames.jl`.
The initial condition is a random 3D velocity field, rather than the Orszag–Tang
vortex. All magnetic components and the GLM variable are exactly zero throughout,
so the MHD equations reduce to pure hydrodynamics. The artificial GLM speed is
kept negligible but positive to leave the existing solver's algebra defined.
No external forcing, gravity, explicit viscosity, or cooling is applied.

## Setup

- Periodic unit cube, 64³ cells, four CPU threads, Float32 state.
- Initially uniform density 1 and sound speed 1; initial 3D rms Mach number 10.
- Solenoidal random Fourier modes with 1 ≤ |k| ≤ 3 (in units of 2π/L).
  Mode amplitudes have a Gaussian envelope centered at |k| = 2, width 0.65.
  Seed 42, mean velocity removed, then rms-normalized.
- Ideal-gas total-energy equation with gamma = 1.01; near-isothermal, not strictly
  isothermal. Dissipated kinetic energy heats the gas.
- Five initial **box** crossing times: t_cross = L / v_rms,0 = 0.1; final t = 0.5.
- 121 evenly spaced density snapshots, plus measured diagnostics at each snapshot.

The displayed Mach number is the mass-weighted turbulent rms speed divided by
the mass-weighted rms instantaneous sound speed. The CSV also records the
mass-weighted rms of the local Mach number, and velocity divided by the initial
sound speed. These definitions differ as density and temperature evolve.
The middle-plane image and the density averaged along z use fixed logarithmic
color scales throughout the movie. Their scales differ and are labeled.
Binary data are little-endian Float32, x varying fastest; the Python image rows
are y and columns are x.

## Reproduce

From the repository root, with the existing Julia project dependencies installed:

```sh
julia --project=GLMMHDTurb -t 4 scripts/render_decay_frames.jl 64 scripts/out/decay_new
python3 -m venv .venv-movie
.venv-movie/bin/python -m pip install -r scripts/requirements-movie.txt
.venv-movie/bin/python scripts/make_decay_movie.py scripts/out/decay_new
```

The simulation refuses to overwrite a nonempty output directory. Choose a fresh
directory when rerunning. Pass 32 instead of 64 for a smaller demonstration.
The video encoder comes with `imageio-ffmpeg`; a separate ffmpeg installation is
not needed. The movie has the look of the notebook's figures (`scripts/plot_style.py`:
its fonts, inks and colour maps), which needs `seaborn` and `cmasher`; both are in
the requirements file. A Python that has those two but not `imageio-ffmpeg` works
as well: the encoder is then taken from `.venv-movie`. Outputs include MP4, animated GIF, a PNG preview, binary density
planes, `diagnostics.csv`, and `manifest.txt`.

## Validation and measured result

The completed 64³ run took 33.4 seconds for evolution and snapshot output on this
MacBook using four threads, excluding startup/compilation and movie rendering.
It used 1,665 accepted steps. Mach number fell from 10 to 0.979; kinetic energy
fell to 1.436% of its initial value. The final rms sound speed was 1.224.

Every candidate step is checked using raw conserved variables, before accepting
it. Steps with nonfinite values, density at/below its floor, or pressure at/below
its floor are rejected and retried at half the time step (up to 12 halvings).
The early strong shocks caused 217 rejected trial steps; these states do not
contribute to the trajectory or movie. Maximum relative mass and total-energy
errors among accepted states were 2.53e-9 and 4.92e-9 respectively. Magnetic field
and GLM variable stayed exactly zero. The run aborts if conservation errors
exceed 1e-4.

This is a low-resolution visual demonstration. The initial field has not been
pre-driven to developed turbulence, and the solver's numerical dissipation
affects the measured decay. These results are not a converged physical decay
rate; that would require resolution and solver comparisons.
