# Athena++ CPU reference installation and error comparison

The official [Athena++ source](https://github.com/PrincetonUniversity/athena) is
installed at `external/athena/`. The built executable is
`external/athena/bin/athena`. It runs on the MacBook CPU; no GPU or cluster is
required. The source directory is excluded from the parent repository's Git
tracking and keeps its own upstream Git history and BSD license.

## Installed configuration

- Commit: `823614c90b594472747a0ac2a699e4a454f300d2` (2026-08-13).
- Official `orszag_tang` problem generator, unchanged upstream source.
- Cartesian ideal MHD with constrained transport; HLLD Riemann solver.
- Double-precision evolution; three ghost cells for PPM.
- Apple Clang, OpenMP using the already-installed Homebrew `libomp`, four threads.
- Input selects PPM (`xorder=3`), RK3, CFL 0.3, gamma = 5/3, periodic boundaries.
- No HDF5, MPI, or new system-wide packages required. The stock VTK writer saves
  Float32 output even though the evolution and restart files use double precision.
  This output rounding is much smaller than the measured discretization errors.

The exact configuration, compiler version, source commit and executable SHA-256
are recorded in `scripts/athena/installation.json`. Build output is in
`external/athena/build.log`; generated configuration is in `configure.log`.

## Completed runs

The 64² smoke test and 256² and 512² reference runs finished successfully, with
states at exactly t = 0, 0.1, 0.25, 0.5. The 512² run took approximately 233 seconds;
the 256² run took approximately 20 seconds. Timings include each run segment's
startup and output, and are single measurements, not throughput benchmarks.

The runner advances to each requested endpoint and restarts for the next segment.
This matters because Athena++'s ordinary scheduled outputs may occur just after
the nominal output time; endpoint stepping lands exactly on the comparison time.
Double-precision restart states carry the trajectory between segments. We retain
all restart files and the actual commands and timings in each run manifest.

The **1024² refinement run is complete**, with exact endpoints t=0, 0.1, 0.25,
and 0.5. It took about 1,217 seconds including startup and output; other work
overlapped parts of the run, so this is not a controlled performance sample.
It is now the primary reference, with 512² as the sensitivity check. Every local
grid, including 512², therefore has a finer independent reference. The old runs
and the old analysis remain available.

The updated four-resolution table is in [MHD_SPECTRA.md](MHD_SPECTRA.md). At 64²
and t=0.5, density L1 differences are **6.644% GLM / 5.907% CT**, and magnetic
vector L1 differences are **16.083% / 12.929%**. The restricted 512²–1024²
reference difference is about **0.170% density / 0.300% magnetic**. This supports
the earlier 64² comparison without making Athena an exact solution.

## Matching the initial conditions

Athena++'s generator uses [-0.5,0.5]² with corresponding sign conventions.
Map its coordinates to ours via x_local = x_Athena + 0.5 and
y_local = y_Athena + 0.5. Under this mapping its analytic velocity and magnetic
fields agree with our [local CPU Orszag–Tang setup](GLM_CT_CPU.md).

Athena initializes face B from a discrete curl of a corner vector potential.
For this sinusoidal problem it yields the analytic field multiplied by sinc(1/N)
for Bx and sinc(2/N) for By. Our local driver uses center-sampled analytic values.
This is a finite-resolution initialization difference, not a sign or coordinate
error. The comparison verifies Athena's initial field against these discrete
formulas, and includes t = 0 in the error plots rather than subtracting away the
initial discrepancy. The local gamma is stored in Float32; Athena uses 5/3 in
double precision.

## Error definition and archived 512² results

All Athena MeshBlocks are assembled by their coordinates, with checks that every
cell is covered exactly once. The fine arrays are then volume-averaged to the
local 64² cells. Density, total-energy density, momentum vector, and cell-centered
magnetic vector are compared at identical times. The magnetic comparison is a
cell-field accuracy diagnostic, not a CT face-divergence diagnostic.

For scalar q, relative L1 = sum |q_local - q_ref| / sum |q_ref| and relative
L2 = sqrt(sum |q_local - q_ref|² / sum |q_ref|²). For vectors, the absolute value
is the Euclidean vector norm in each cell. Consequently magnetic error measures
the vector difference, not only the difference of field magnitudes. All norms
are taken after averaging the reference to the comparison grid.

At t = 0.5, against the 512² reference:

| Relative L1 error | Local GLM 64² | Local CT 64² |
|---|---:|---:|
| Density | 6.618% | 5.875% |
| Magnetic vector | 16.082% | 12.933% |

Against the 256² reference, density errors are 6.525% and 5.777%, and magnetic
errors are 15.976% and 12.837%. The choice of reference changes these reported
errors by about 0.1 percentage point. The restricted 512² and 256² reference
fields themselves differ by 0.391% in density and 0.743% in magnetic vector
(relative L1, normalized by the 256² values).

Our CT implementation has smaller errors for these quantities at this time and
resolution. This does not establish a universal GLM-versus-CT ranking. In
particular, the reference has residual numerical error; both local methods use
the simpler PLM/HLL/RK2 algorithm, and initial sampling differs. The completed 1024² refinement and four local resolutions are reported above.

## Use and reproduce

From the project root, inspect or rebuild the executable:

```sh
external/athena/bin/athena -c
sh scripts/build_athena_cpu.sh
```

Run another reference in a new directory:

```sh
.venv-movie/bin/python scripts/run_athena_reference.py --n 512 --out scripts/out/athena_512_new
```

To reproduce the completed finer reference in a fresh directory:

```sh
.venv-movie/bin/python scripts/run_athena_reference.py --n 1024 --out scripts/out/athena_1024_new
```

Generate plots using the completed runs:

```sh
.venv-movie/bin/python scripts/compare_athena_reference.py \
  --reference scripts/out/athena_1024 --check-reference scripts/out/athena_512
```

The completed 1024² run is `--reference`, with the 512² directory as
`--check-reference`. The `.venv-movie` environment already contains the required
NumPy and Matplotlib packages; the reader is Athena++'s own `athena_read.py`.

On a fresh checkout, reproduce the exact source before building:

```sh
git clone https://github.com/PrincetonUniversity/athena.git external/athena
git -C external/athena checkout 823614c90b594472747a0ac2a699e4a454f300d2
sh scripts/build_athena_cpu.sh
```

The build script assumes Homebrew libomp under `/opt/homebrew/opt/libomp`; set
`ATHENA_LIBOMP_PREFIX` to use another location. It is already installed here.

## Artifacts

- `scripts/out/athena_64/`, `athena_256/`, `athena_512/`: VTK fields, double-precision
  restarts, history diagnostics, build configurations and run manifests.
- `scripts/out/errors_athena_512/error_vs_time.png`: errors in four fields, both
  reference resolutions, plus the reference-to-reference difference.
- `scripts/out/errors_athena_512/density_errors.png`: reference and local fields,
  signed error maps, and a common density slice at t = 0.5.
- `errors.csv`: relative L1 and L2 errors; `reference_sensitivity.csv`: reference
  differences; `reference_on_64_t*.npz`: conservatively averaged reference arrays.
- `validation.json`: coordinate and discrete initial-field checks.

The snapshots are checked for finite values and positive density and pressure.
Initial velocity and discrete-curl field checks pass to the expected VTK
Float32 rounding level. Plots were rendered and visually inspected.

New artifacts: `scripts/out/athena_1024/` and `scripts/out/errors_athena_1024/`.
The latter uses the 21-output local 64² study, matching the active notebook.
