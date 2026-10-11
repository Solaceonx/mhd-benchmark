# GPU GLM–CT comparison protocol

Research question: under which problems, error tolerances and hardware conditions
does an existing GLM implementation outperform an existing CT implementation?
GLM superiority is a hypothesis, not an acceptance criterion. Current local CPU
Orszag–Tang results favor CT in density and magnetic-vector error; those runs do
not establish the ranking of the different GPU implementations.

## Targets and implementation scope

Run the same test harness on Stanford Sherlock and SLAC S3DF. Reuse the original
GPU implementations in `GLMMHDTurb/`, retaining the MacBook CPU and Athena data
as validation references. New work should primarily implement test initialization,
solver adapters, output, instrumentation and analysis, rather than porting the
new CPU RK2 solver as a prerequisite.

The repository includes Julia GPU implementations and optimized CUDA variants.
Identify each by source hash and explicit numerical/build configuration. Begin
with the understandable Julia GPU paths, then evaluate optimized variants only
after their accuracy and robustness have been measured. Existing benchmark
documents are historical records; repeat measurements on the allocated hardware.

The existing GPU kernels use cubic 3D storage. A repeated 2D initial condition
can be compared to a 2D reference after verifying invariance along the third
axis. Report the actual work and allocation as N³. Its elapsed time is not a
hardware speedup measurement against the true N² CPU driver.

## Two complementary comparisons

1. Controlled numerical settings: match physics, initial data, domain, output
   times, precision and flux/reconstruction choices wherever supported. Record
   remaining predictor, cleaning and CT differences. These compare specified
   implementations; they do not completely isolate GLM versus CT mathematically.
2. Practical performance: compare the best validated configuration of each method
   at the same target error and physical endpoint, allowing different grid sizes.
   Give CT comparable optimization effort. Do not select only the fastest GLM
   variant and an unoptimized CT baseline.

## Measurements and decision rules

- Primary plots: density and magnetic-vector error versus elapsed compute time,
  and versus end-to-end elapsed time. Use resolution sweeps to locate measured
  configurations meeting each predeclared tolerance. Mark unmet tolerances rather
  than extrapolating success. Qualify results when reference uncertainty matters.
- Secondary: throughput, peak device memory, mass/energy drift, common cell-field
  divergence, native CT face divergence, positivity violations, floor activation,
  rejected timesteps and failed runs where instrumentation supports them.
- Measure after compilation/warm-up, with device synchronization at timing
  boundaries. Separate computation, host/device transfers and output. Record cold
  startup separately when useful. Queue wait is operational information, not
  solver execution time.
- Repeat at least five times, alternating method order on the same allocated GPU.
  Report median and spread, device model, memory/MIG profile, driver/toolchain,
  precision, fast-math options, CPU allocation, steps and physical endpoint.
- Within each problem, plot the accuracy/time tradeoff rather than declaring a
  winner from a fixed-grid throughput ratio. Different tests may favor different
  methods. Keep cluster/device results separate before comparing trends.

## Initial test sequence

1. Uniform state and exact entropy advection: preservation and smooth convergence.
2. Orszag–Tang: compare to the saved Athena 1024² reference, checking time,
   coordinates, initial sampling, cell averaging and out-of-plane invariance.
3. Circularly polarized Alfvén wave: exact smooth evolving magnetic field; validate
   initialization and reference cell/face representation before measuring errors.
4. Magnetic-loop advection: magnetic transport and dissipation.
5. Shock tubes and rotor: shocks, strong gradients and robustness; add appropriate
   boundary conditions where needed.
6. Decaying turbulence: matched initial realizations and multiple seeds after the
   deterministic tests pass. Use statistical diagnostics appropriate to turbulent
   evolution and distinguish numerical-reference uncertainty.

For chaotic late-time flows, pointwise differences alone do not rank physical
fidelity. Compare spectra, energy evolution and statistical variation as well.

## Sherlock execution policy

Sherlock documents `sh_dev -g 1` for lightweight MIG development instances, and
the `gpu` partition with an explicit `--gpus` / `-G` request for GPU jobs. Use a
development instance for compilation and small correctness checks; request a
full GPU for the principal performance comparison. Record the actual allocation.

Discover eligible GPU features with `node_feat -p gpu`; constrain benchmark jobs
to a supported GPU model rather than assuming the examples in documentation are
the current inventory. Run GLM and CT sequentially inside the same allocation.
Do not mix Sherlock MIG results with full-device S3DF results as method speedups.

Source: [Sherlock GPU documentation](https://www.sherlock.stanford.edu/docs/user-guide/gpu/),
checked 2026-09-24. Account/partition availability must be verified on login.

## Status

This document records the agreed study direction. A transferable first GPU harness is now implemented in
[`sherlock_mhd/`](../sherlock_mhd/README.md), with eight problem configurations,
Sherlock jobs, binary outputs and local analysis. CPU initial-condition and
round-trip checks pass; actual GPU execution remains unverified pending the
Sherlock smoke job. No Sherlock or S3DF jobs have been submitted in this session.
Optimized CUDA variants are not yet wired into this initial Julia-GPU harness.
