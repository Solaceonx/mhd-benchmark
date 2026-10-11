# CPU GLM–CT spectral comparison

The completed study runs matched GLM and CT methods at **64², 128², 256² and
512²**, on four CPU threads, to t = 0.5. Each run saves 21 states and energy
diagnostics. Spectra and reference errors are evaluated at t = 0, 0.1, 0.25 and
0.5. Native CT face fields are now saved at every output time. The equations and
numerical update are unchanged; see [the solver description](GLM_CT_CPU.md).

Athena++ **1024²** is now the primary numerical reference, conservatively averaged
to each local grid. Athena++ 512² provides a sensitivity comparison at every
local resolution, including 512². All local runs now have a finer reference.
These are still finite-reference differences, not exact errors or certified
convergence rates. Initial discrete sampling differs between codes as described
in [the Athena++ setup](ATHENA_REFERENCE.md); t = 0 differences are retained.
The earlier 512²-reference analysis is preserved in `analysis_512_archive/`.
The dense movies still use Athena++ 512², labeled separately from the tables.

## Diagnostics and artifacts

- Absolute and relative L1, L2 and Linf differences for density, magnetic
  vector, momentum vector and total-energy density.
- Density-fluctuation, kinetic-energy and magnetic-energy power spectra.
- Density and magnetic-vector error spectra against each available reference.
- The spectrum of the 1024²-minus-512² reference difference on each common grid.
- Common Fourier and centered-difference divergence spectra of cell B, plus
  local CT's native face-divergence spectrum and normalized divergence statistics.
- Kinetic, magnetic, thermal and total energy histories; wall and stepping times.

Files live in `scripts/out/spectral_study/analysis/`:

| File | Contents |
|---|---|
| `resolution_comparison.png` | L1/L2/Linf differences and CPU stepping costs |
| `N64/`, `N128/`, `N256/`, `N512/` | Power, error, divergence and energy-history plots |
| `spectra.csv` | All Fourier shells, including DC and square-grid corner modes |
| `norms.csv` | Four fields, four times, both methods, available references |
| `divergence.csv` | RMS, maximum and h RMS(div B)/B_rms for each operator |
| `energies.csv` | Local, restricted-reference and native-reference energies |
| `timings.csv` | Per-method stepping time and shared wall time, excluding compilation |
| `summary.json` | Final norms and error-power fractions in specified bands |
| `validation.json` | Parseval checks, provenance and limitations |

## Definitions

Python arrays have axes (y, x, component). The periodic box has L = 1. FFT
coefficients use `norm="forward"`, dividing the forward transform by N². Mode
indices are in **cycles per box**, so derivative wave numbers are 2πm/L. Shell j
contains j ≤ sqrt(mx² + my²) < j+1.

Spectra are **sums** over modes in a shell, not means per mode. A full complex
FFT retains both signs, so no factor of two is applied. Scalar power is |qhat|²;
vector power sums component powers. Summing every shell gives mean(|q|²). No
window is used because the domain is periodic. Plots show nonzero, complete
annuli below axial Nyquist; tables retain every mode for Parseval accounting.
The first shell, plotted at midpoint 1.5, contains the fundamental kL = 1 modes.

Density power uses rho minus its spatial mean. Kinetic power is one half the
power of momentum/sqrt(rho), summing to mean(rho |v|²/2). It is a density-weighted
energy diagnostic, not the unweighted velocity spectrum. Magnetic energy is
one half the vector B power, in the solver's units with energy density B²/2.

Error power Fourier transforms **q_local − q_reference**, with no mean
subtraction or one-half factor. It is not the difference of two power spectra.
Its complete sum is absolute L2 error squared and includes phase/position errors
that ordinary power spectra discard. Magnetic error uses the vector difference,
not the difference of field magnitudes.

Relative norms divide by the corresponding reference norm: mean(|qref|) for L1,
sqrt(mean(|qref|²)) for L2, and max(|qref|) for Linf. Vector absolute values denote
Euclidean length in each cell. Linf is a global maximum normalized by the
reference maximum, not the maximum of pointwise fractional errors near zeros.

Reference conserved fields and cell B are averaged **before** constructing the
kinetic-energy variable or taking FFTs. This matches the local cells' spatial
support and includes the averaging filter. It does not deconvolve cell
averaging. The thermal remainder from restricted total energy includes
unresolved kinetic/magnetic energy. `energies.csv` labels these rows
`comparison_grid`; native Athena energies are separate `native_grid` rows.
Athena curves have four saved time points; connecting dashes are visual guides.

## Divergence operators

The common Fourier derivative is i 2π(mx Bxhat + my Byhat). On even grids,
derivatives along an axis's Nyquist mode are zero, the standard real
trigonometric interpolant convention. The common central difference has symbol
i N sin(2πm/N), also zero at Nyquist. These diagnose the cell-centered
representation of every method.

CT preserves the face finite-volume operator:

    div_face B = N [Bx(i+1,j) − Bx(i,j) + By(i,j+1) − By(i,j)].

Its spectrum uses saved face arrays and measures the actual local CT constraint.
A centered average of a discretely divergence-free face field does not generally
have zero divergence under the common operators. Averaging fine cell B also
does not preserve a face-divergence identity. Thus restricted Athena++ cell-B
divergence can be substantial without a violation of Athena's CT constraint.
Athena face divergence is not reconstructed from the VTK cell-field output.

The local solver evolves Float32. Native CT divergence accumulates roundoff
rather than staying exactly at machine epsilon: its final maximum is 1.07e-4
at 64² and 4.61e-3 at 512². Dimensionless h RMS(div B)/B_rms is 1.07e-6 and
2.93e-6, respectively. These face values and GLM's cell divergence measure
different operators.

## Results

Final relative L1 differences against Athena++ **1024²**, in percent:

| Local N | Density GLM | Density CT | Magnetic GLM | Magnetic CT |
|---:|---:|---:|---:|---:|
| 64 | 6.644 | 5.907 | 16.083 | 12.929 |
| 128 | 3.539 | 3.108 | 8.101 | 6.565 |
| 256 | 1.764 | 1.569 | 4.083 | 3.335 |
| 512 | 0.899 | 0.800 | 1.966 | 1.614 |

The 512²–1024² reference difference at t=0.5, normalized by the 1024²
reference after restriction to each comparison grid:

| Comparison N | Density reference difference | Magnetic reference difference |
|---:|---:|---:|
| 64 | 0.170% | 0.300% |
| 128 | 0.249% | 0.458% |
| 256 | 0.363% | 0.661% |
| 512 | 0.458% | 0.813% |

These measure reference sensitivity, not rigorous error bounds.

In the archived 512²-reference analysis, at 64² and t = 0.5, both methods lose substantial magnetic power toward the grid
limit relative to the averaged reference. CT retains more magnetic power over
many intermediate modes, but that alone does not establish phase accuracy.
The error spectrum supplies the complementary result: CT has **38.8% less
magnetic error power** in shells 4 ≤ kL < 16, but only **3.7% less** in shells
24 ≤ kL < 32. These are reductions in squared error, not L1 or L2. The advantage
is not confined to the very smallest scales.

The current 512²–1024² reference sensitivity is measured on every local grid.
At 64² it is small relative to local errors; it grows in importance toward
512². We do not infer a universal GLM/CT ranking, sharp effective bandwidth,
effective Reynolds numbers, or certified convergence order from this study.

Combined wall times excluding compilation were 0.92 s, 5.49 s, 33.4 s and 265 s
for N = 64, 128, 256 and 512. Per-method step times exclude CFL evaluation,
diagnostics and I/O; combined wall times include those. These are single
exploratory timings with a shared timestep. Some analysis overlapped the 512²
simulation; these are not controlled performance benchmarks or a basis for a
percentage speedup claim.

Both methods conserve total energy closely while their energy partitions differ.
**Magnetic/kinetic changes include ideal transfers and compressional work**;
the curves alone do not isolate numerical resistivity or reconnection heating.
Total-energy plots report drift from each run's own initial energy because the
initial discretizations differ slightly.

The 21-output sweep is separate from the earlier 101-output 64² run. Output
endpoints clip timesteps, so the cadence changes the trajectory slightly.

## Validation and reproduction

Eight tests check a known sinusoid; vector/DC/corner-mode Parseval normalization;
phase shifts with identical power; vector norms; analytic and centered
derivatives; the CT curl/divergence identity; Nyquist handling; and energy
accounting. The updated study checked 496 spectra against real-space means with maximum
relative Parseval residual 5.79e-16. It also checked the error-spectrum/L2
identity, positive states, saved face divergence against Julia diagnostics,
and face-to-cell field reconstruction.

From the project root:

```sh
.venv-movie/bin/python scripts/check_mhd_spectra.py

# Existing simulation directories are never overwritten.
julia --project=GLMMHDTurb -t 4 -e 'include("scripts/run_glm_ct_2d.jl"); for n in (64,128,256,512); run_compare(;n,nframes=21,root=joinpath("scripts","out","spectral_study_new","glm_ct_$n")); end'

# Regenerate analysis tables and plots from the completed study.
.venv-movie/bin/python scripts/analyze_mhd_spectra.py \
  --locals scripts/out/spectral_study/glm_ct_64 \
           scripts/out/spectral_study/glm_ct_128 \
           scripts/out/spectral_study/glm_ct_256 \
           scripts/out/spectral_study/glm_ct_512
```

The runner also accepts N, output directory and frame count as three positional
arguments to `scripts/run_glm_ct_2d.jl`; default frame count remains 101. Use a
count including all analysis times, e.g. 21 or 101 for t_end = 0.5. The analysis
accepts `--reference`, `--check-reference` and `--out`. The defaults now use 1024² as the primary reference and 512² as the check.

Method-comparison spectra are motivated by
[Kritsuk et al. (2011)](https://arxiv.org/abs/1103.5525). For related Orszag–Tang
energy/dissipation analysis see
[Kayanikhoo et al. (2023)](https://arxiv.org/abs/2312.06675). This 2D adiabatic
deterministic test is not a replication of the former paper's 3D isothermal
turbulence experiment. Current/vorticity and cross-spectral coherence are
possible extensions, not included in this study.
