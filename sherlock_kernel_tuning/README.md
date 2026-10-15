# Same-scheme GPU launch tuning pilot

Original solver source is loaded unchanged and verified against the benchmark
checksum. This tests `integrator_tiled!` through `step_tiled!` and
`integrator_plm!` through `step_plm!`. Nothing is submitted by local checks.

- Threads per block: 96, 128, 192, 256.
- Original tiled Float32 LLF and PLM Float32 HLLD: tile width 4.
- Production PLM Float16 HLL: tile widths 4 and 6 where both divide the grid.
- Default grids: 96³, 192³, 384³; five timing samples of ten steps each.

Each candidate is compared with its own solver/precision baseline, using the
existing 192-thread default and automatic tile choice. All nine output fields
must be finite, density/internal energy positive, and the maximum absolute
difference from that baseline zero after ten steps. A mismatch is rejected,
rather than relabeled a speedup. Different solvers or precision levels are never
counted as optimization gains against one another.

The initial state is a smooth, nonuniform periodic sine field with a z-velocity
perturbation. This is an early-time throughput pilot, not the full benchmark
suite or a late-time turbulence workload. Timing uses CUDA events plus batch
wall time, with compilation, resets and transfers excluded. A fresh baseline is
timed before every candidate to expose clock drift. No timestep scan, forcing,
diagnostics, or file output enters the timed region. Median and range are saved
in `timings.csv` beside `environment.toml`. Raw timings do not establish a
speedup until differences exceed run-to-run variation and repeat on the target GPU.

## Local check

```sh
julia --project=sherlock_sweep/env sherlock_kernel_tuning/tune.jl --check
```

This loads the source and validates the configuration grid. It does not compile
or execute kernels on a GPU. The pilot still needs an H100 trial; subsequent
accuracy tests should cover exact waves, shocks, blasts, field loops and turbulence.

## Run on Sherlock

Send this directory to a fresh side directory, then run `bash -l submit.sh`
there. The submitter guards against an already queued/running job of the same
name, requests one H100 and uses the existing sweep bundle's environment and
vendored source. It does not change that bundle. `SWEEP_BUNDLE` can override its
default location. Logs are in this directory's `logs/`, and results are under
`$SCRATCH/mhd-results-kernel-tuning/job-<id>/`. The job requires a 48³ GPU smoke
test to pass before starting the larger grids; smoke and full results are saved
in separate subdirectories. A failure to query the queue stops submission.

`TUNE_GRIDS=48 TUNE_SAMPLES=3 TUNE_STEPS=2` gives a short GPU smoke test.
Defaults can also be overridden with those variables. A fresh `TUNE_OUTPUT`
directory is required; existing results are not overwritten.

Next experiments should follow actual profiler evidence: register pressure,
shared-memory use, occupancy and stalls. Earlier upstream A6000 experiments
already found that forcing register caps, extra GPU calls and the streamed
layout can make these kernels slower; those results do not automatically carry
over to the H100. Changes to precision, fluxes, time stepping or reconstruction
must be evaluated as numerical-method changes, separately from launch tuning.
