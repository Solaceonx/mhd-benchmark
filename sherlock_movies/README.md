# Transferable Sherlock MHD test suite

> **Movie-run bundle (v2).** This copy of `sherlock_rerun` holds only `configs/movie_*.toml`: 7 2D tests at 384²
> and 5 3D tests at 256³ (midplane), 241 outputs each, for videos (not timing or accuracy). Follow [MOVIES.md](MOVIES.md); the rest of this file
> describes the original campaign and is kept for reference.

For lyxia's actual storage paths and the staged first run, start with
[SHERLOCK_LAUNCH.md](SHERLOCK_LAUNCH.md). Current project context is in
[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

This folder is self-contained. Transfer it to Sherlock, install its Julia
environment, run the small GPU smoke job, then submit individual physics tests.
Download the returned archives to the Mac for analysis. No cluster access or
job submission has been performed while preparing this folder.

**Status:** CPU initial-condition, serialization and analysis checks are tested.
The driver loads on Julia 1.12.5 on the Mac. GPU compilation/evolution cannot be
verified on this Mac; the Sherlock smoke job is a required first gate. A finished
simulation is not, by itself, a passed accuracy/convergence test.

## What is included

| Path | Purpose |
|---|---|
| `vendor/` | Unmodified snapshots of the original GLM and CT Julia GPU solvers; SHA-256 provenance |
| `env/` | Original Project.toml and Manifest.toml, created with Julia 1.12.5 |
| `configs/` | One file per test: problem, parameters, grids, seeds, output cadence, five repeats |
| `src/run_suite.jl` | GPU adapters, timestep selection, measurements and field output |
| `jobs/` | Setup, smoke and one generic Slurm array job (`test.sbatch`) for every physics test |
| `tools/submit.sh` | Submit one named test; array range and time limit come from its config |
| `tools/pack_results.sh` | Archive one result folder with a SHA-256 checksum |
| `tools/analyze_returns.py` | Local norms, spectra, timing summaries and comparison figures |
| `references/` | Compact Athena 1024² fields averaged to 32², 64², 128² at t=0, .1, .25, .5 |

The numerical kernels are reused. This first package compares the original
**FP32 tiled GLM / full Hancock / HLL** path and the original **FP32 seven-kernel
CT / transverse-free Hancock / HLL** path. Both use gamma=5/3, CFL=0.25, periodic
boundaries and identical initial cell fields. The adiabatic index is set per problem
(5/3, 7/5 or 2). CT hard-codes `const GAMMA = 5f0/3f0`; for 7/5 and 2 the driver parses
the unmodified `vendor/ct_mhd.jl` again at load time with only that constant replaced,
under a new module name (`CTMHD_gamma7_5`, `CTMHD_gamma2`). The vendor file is never
edited and gamma=5/3 always uses the original module; `run.toml` records which was used. GLM uses runtime cleaning parameters
`glm_ch_scale=1`, `glm_cp_coef=0.18`; its library default scale was 0.25. GLM density
and pressure floor parameters are configured to approximately 1e-6 to match CT's
hard-coded 1e-6 floors. Internal floor implementations and predictors still differ.
Existing GLM `@fastmath` annotations are retained; this is not a strict-IEEE build.

**This is not yet a comparison of the fastest optimized GLM and CT variants.**
Use these results to establish behavior, then extend to the optimized CUDA
variants with comparable effort for both methods. No solver should be declared
better solely from this package's fixed-grid throughput.

## 1. Transfer from the Mac

From the repository root, substitute your actual SUNet ID:

```bash
SHERLOCK_USER=your_sunetid
rsync -av sherlock_mhd/ "$SHERLOCK_USER@login.sherlock.stanford.edu:sherlock_mhd/"
ssh "$SHERLOCK_USER@login.sherlock.stanford.edu"
```

On Sherlock, use scratch for the job input/output. For the first upload:

```bash
mkdir -p "$SCRATCH/mhd-study"
cp -R "$HOME/sherlock_mhd" "$SCRATCH/mhd-study/"
cd "$SCRATCH/mhd-study/sherlock_mhd"
```

Do not copy an update over active runs. Scratch storage is not
permanent; download results after the jobs finish.

## 2. Select Julia and install packages

```bash
module spider julia
```

Choose a Julia **1.12.x** module; 1.12.5 matches the bundled lockfile. The default
job command loads `julia`. If that default is a different version, set the actual
module name shown by `module spider`, for example `export JULIA_MODULE=...`.
The setup job deliberately stops on a different minor version rather than
silently replacing the locked package environment.

If Sherlock has no suitable module, use an existing personal Linux Julia 1.12.x
installation with `export JULIA_BIN=/absolute/path/to/julia`. The Mac's Julia
executable cannot be copied to Linux. Obtain Julia from its official distribution
if needed; this package does not download an executable automatically.

```bash
bash tools/submit.sh setup
```

Wait for that job to finish successfully. Inspect `logs/mhd-setup-JOBID.out` and
`.err`. Package downloads/precompilation occur in this setup job, not in timed
measurements. The default persistent depot is `$HOME/.julia-mhd`; override with
`MHD_JULIA_DEPOT` if needed. Use the same environment variables for later jobs.

## 3. Run the small GPU smoke job

```bash
bash tools/submit.sh smoke
```

It runs every config in `configs/` on **32³** grids, each method once, with three outputs
to at most t=.01. It checks allocation, GPU compilation, evolution, basic state
validity and output for both methods. It does not certify accuracy or convergence.
Read `logs/mhd-smoke-JOBID.out` and the returned `driver.log` before proceeding.

Sherlock also documents `sh_dev -g 1` for lightweight MIG development instances.
Inside such an allocation, an interactive smoke run is:

```bash
source jobs/common.sh
prepare_environment
"$MHD_JULIA" --project=env src/smoke.jl "$MHD_RESULTS/manual-smoke-$(date +%Y%m%d-%H%M%S)"
```

Use a full GPU allocation for the principal performance study. Do not mix MIG
and full-GPU timings. `node_feat -p gpu` lists features visible to your account;
select a supported GPU model for production jobs rather than assuming the old
example GPU models in documentation are the current inventory.

## 4. Submit physics tests individually

Each array task runs **both GLM and CT on the same GPU**, sequentially, with five
repetitions and alternating method order. Array concurrency is capped at one per
test by default. No script automatically launches the entire suite. List the tests with
`bash tools/submit.sh list`.

The suite implements the FLASH user guide MHD test problems (section 35.2) that can run
in these solvers, plus exact-solution convergence tests. Both solvers support only a
periodic N³ Cartesian unit cube, so FLASH problems are rescaled to it; the config header
comment of each test states its mapping. "div B" marks tests where the two div(B)
treatments can actually differ: in grid-aligned 1D problems div(B)=0 trivially.

| Submit name | γ | End time | Outputs | Reference | div B | Origin / adaptation |
|---|---|---:|---:|---|:-:|---|
| `uniform` | 5/3 | .1 | 21 | exact | | preservation gate |
| `entropy` | 5/3 | 1 | 21 | exact | | diagonal entropy wave, constant B |
| `alfven` | 5/3 | 1 | 21 | exact | | circularly polarized wave along x |
| `alfven_oblique` | 5/3 | 1/√5 | 21 | exact | ✓ | same wave with k ∥ (1,2) (Tóth 2000) |
| `brio_wu` | 2 | .05 | 11 | none yet | | FLASH 35.2.1, periodic L\|R\|L; see below |
| `brio_wu_rotated` | 2 | .05/√2 | 11 | none yet | ✓ | Brio–Wu with shock normal (1,1)/√2 |
| `orszag_tang` | 5/3 | .5 | 21 | Athena 1024² | ✓ | FLASH 35.2.2 (Athena normalization, same dynamics) |
| `orszag_tang_3d` | 5/3 | 1 | 11 | none | ✓ | FLASH 35.2.2 3D; Helzel et al. (2011) z perturbation ε=.2 |
| `rotor` | 7/5 | .15 | 16 | none | ✓ | FLASH 35.2.5 / Balsara–Spicer first rotor, u0=2 |
| `current_sheet` | 5/3 | 2.5 | 21 | none | ✓ | FLASH 35.2.6 / Gardiner–Stone; [0,2]² box halved, so t=2.5 ≡ t=5 |
| `field_loop` | 5/3 | 2 | 21 | advected IC at t=1,2 | ✓ | FLASH 35.2.7 2D, v=(2,1,√5): B_z growth ∝ w·div(B) |
| `field_loop_novz` | 5/3 | 2 | 21 | advected IC at t=1,2 | ✓ | Gardiner–Stone (2005) in-plane loop, v_z=0 |
| `field_loop_3d` | 5/3 | 1 | 11 | advected IC at t=1 | ✓ | FLASH 35.2.7 3D / Gardiner–Stone (2008) in a cube, see below |
| `blast_2d_b{0,50,100}` | 7/5 | .01 | 11 | none | ✓ | FLASH 35.2.8, 2D cylinder, B_x = {0,50,100}/√4π |
| `blast_3d_b{0,50,100}` | 7/5 | .01 | 11 | none | ✓ | FLASH 35.2.8, 3D sphere, B_x = {0,50,100}/√4π |
| `decay` | 5/3 | .5 | 5 | none | ✓ | 3D unforced Mach 2 decay (not a FLASH problem) |
| `decay_supersonic` | 1.01 | .2 | 11 | none | ✓ | after Kritsuk et al. (2011): Mach 9, B₀ = 0.3 ẑ, 64³–256³, two seeds |
| `kh_uniform` | 5/3 | 3 | 31 | AthenaK 2048² | ✓ | Tomida et al. (2026) §III.1: double shear layer, β = 2×10⁵ |
| `kh_localized` | 5/3 | 3 | 31 | AthenaK 2048² | ✓ | Tomida §III.2: field only on the lower layer; upper-layer field = leakage |
| `kh_localized_dtjump` | 5/3 | 3 | 31 | none | ✓ | Tomida §III.3: dt × 1/40 for 100 steps at t = 1.5 (GPU driver only) |
| `throughput_ot3d` | 5/3 | .05 | 2 | none | | 96³–384³ cell-update rate and memory; N divisible by 6 for GLM TB=6 |

Default grids are 32, 64, 128 (decay: 32, 64 × seeds 42, 43, 44). The 3D tests save
snapshots only at `snapshot_frames` to keep returns small; diagnostics are still written
at every output.

Adaptation notes:

- **Brio–Wu.** Periodicity forces a second interface at x=0. The region x∈[.25,.75] at
  t=.05 is exactly the standard tube on [0,1] at t=.1 with N/2 cells
  (x_std = 2x−.5, t_std = 2t); waves from the other interface stay outside it until then.
  The rotated version has the same scaled time along its normal. A 1D high-resolution
  reference still has to be generated (Athena++ `shock_tube` is available locally).
- **3D field loop.** Gardiner & Stone use a 1×1×2 box with loop axis (−1,0,2). A unit cube
  needs a lattice-vector axis whose periodic images stay farther apart than the loop
  diameter 0.6; axis (−1,0,1)/√2 gives 0.707. Velocity (1,1,2) returns the loop at t=1.
- **Blasts** use `pressure_policy = "floor"`: with β≈2.5e-4 unfloored pressure goes
  negative within a few steps (seen for GLM on CPU). The run continues with each solver's
  own 1e-6 floor, the CFL scan uses the floored pressure, and `diagnostics.csv` records
  `nonpositive_pressure_cells` per output and `run.toml` the first time it happened. All
  other tests stop at the first nonpositive pressure. Images of the periodic box begin to
  interact only near t=.01.
- **Not implemented.** The magnetized accretion torus (35.2.3) needs cylindrical geometry,
  a pseudo-Newtonian gravity source and outflow/diode boundaries. The magnetized Noh
  Z-pinch (35.2.4) needs inflow/outflow boundaries: in a periodic box the flow diverges
  at the box edges, and a rarefaction moving inward at about 4.5 v0 contaminates the
  centre well before the FLASH end time. Neither solver has these capabilities, and
  adding them means changing the vendored kernels.

All initial magnetic fields come from an edge-centred vector potential, so the initial
face divergence cancels to round-off for every problem (Orszag–Tang now matches Athena's
discrete-curl initialization). Discontinuous hydro states are cell averages over 4³ or
8² subsamples.

Start with Orszag–Tang at 64³ only (array index 1):

```bash
bash tools/submit.sh orszag_tang --array=1
```

Then submit whichever complete test sweep you want:

```bash
bash tools/submit.sh alfven_oblique
bash tools/submit.sh field_loop
bash tools/submit.sh blast_3d_b100
```

Additional arguments go directly to `sbatch`, e.g. `--constraint=FEATURE`,
`--time=02:00:00` or `--dependency=afterok:JOBID`. Use an actual feature discovered
with `node_feat`. Command-line options override the job file's directives.

Array index i maps to grid `grids[i % len(grids)]` and seed `seeds[i ÷ len(grids)]`, so
ordinary arrays map 0,1,2 to N=32,64,128 and decay maps 0,1 to seed42/N32,64 and so on.
`submit.sh` derives the array range from the config, and uses `slurm_time` from the
config when present (the 3D tests request two hours). An explicit `--array` or `--time`
on the command line replaces them. Grids must be multiples of 4. The existing kernels
use N³ storage/work even for planar flows. A new test is only a new config file.

Jobs have a one-hour default Slurm limit and per-run step/wall limits. Failed cases are
recorded; there are no silent timestep retries or automatic resubmissions.
Use `squeue -u "$USER"` to inspect jobs and `scancel JOBID` to stop one.

## 4b. Multi-code comparison (GLM variants, CT, AthenaK, RAMSES, Athena++)

The GPU driver exposes six methods from the unmodified vendor kernels:

| Method | What it is |
|---|---|
| `glm` | original suite path: `step_tiled!`, Float32 tile, LLF (`use_hlld=false`) |
| `glm_hll16` | advisor's production `step_plm!`: f16 shared tile, HLL; TB=6 when N%6==0 |
| `glm_hlld` | `step_plm!` Float32 tile, HLLD |
| `glm_ppm` | `step_plm!` lean parabolic PPM, f16, HLL |
| `glm_constch` | `glm_hll16` with constant c_h (Tomida et al. 2026 "D1c"): c_h fixed at its t=0 value, dt capped so c_h·dt/dx never exceeds the variable scheme's value |
| `ct` | CT prototype |

The core campaign defaults to the original `glm` path and `ct`. A config's
explicit `methods` list can enable the other GLM kernels for targeted follow-up
tests. External codes run the same initial data: Julia
`problems.jl` writes a spec (`tools/write_spec.jl`); `codes/common/abel_problems.hpp` rebuilds
it inside AthenaK and Athena++; RAMSES reads the exact Float32 state. `tools/check_external_ic.jl`
compares each code's t = 0 output with Julia: all problems agree to ≤ 1.2×10⁻⁷ (Float32 rounding).

| Code | Where | Variants | Notes |
|---|---|---|---|
| AthenaK (pinned commit) | GPU, Kokkos/CUDA | `matched` PLM/HLLE/RK2, `best` PPM4/HLLD/RK3, `reference` | **Double precision only**: its single-precision build fails to compile at this commit (27 type errors), so GPU cost comparisons with the Float32 GLM/CT kernels carry a precision difference. A small build-time patch (`codes/athenak/patch_athenak.py`) makes timesteps land on output times; physics untouched |
| RAMSES (pinned) | CPU, MPI | `matched` HLL, `best` HLLD | Uniform octree (levelmin = levelmax), so N must be a power of two; `exact_output_time`. Cost is CPU seconds and is never mixed with GPU cost |
| Athena++ (pinned) | CPU, MPI | `reference` | Cross-check of the AthenaK references at half their resolution |

References: `--reference` runs AthenaK `best` in double precision at `reference_n` (2048² for
planar tests, as true 2D; 384³ in 3D) and saves every output block-averaged to each
comparison grid. Exact-solution tests use their exact solutions; current sheet and turbulence
have no pointwise reference by design (numerical reconnection and chaos).

Workflow on Sherlock:

```bash
bash tools/fetch_codes.sh                 # login node: clone pinned sources into $SCRATCH/mhd-codes
bash tools/submit.sh build-gpu-codes      # AthenaK for the GPU model the job lands on
bash tools/submit.sh build-cpu-codes      # RAMSES and Athena++ (MPI)
bash tools/submit.sh orszag_tang                        # GLM variants + CT
bash tools/submit.sh orszag_tang --code athenak:best    # one external code/variant
bash tools/submit.sh orszag_tang --reference            # AthenaK reference
bash tools/submit.sh matrix                             # prints every command of the full study
```

Use the same `--constraint` (GPU model) for the AthenaK build and every GPU run, so timings
are comparable and the AthenaK build matches the architecture. Module names are overridable:
`MHD_PYTHON_MODULES` (needs NumPy), `MHD_GPU_BUILD_MODULES`, `MHD_GPU_RUN_MODULES`,
`MHD_CPU_BUILD_MODULES`; check them with `module spider`. External runs write the same
archive layout as the GPU driver, and `tools/postprocess_external.jl` computes their
diagnostics with the same Julia functions.

Resource estimate: `julia tools/estimate_cost.jl [--csv costs.csv]` prints GPU-hours, CPU
core-hours and returned data for the whole matrix, the largest consumers, and any run expected
to exceed its time limits. Its throughputs are assumptions; after the smoke job and the first
real runs, replace `RATES` with the measured `million_cell_steps_per_second` values.
External runs delete each code's native output only after conversion and Julia
postprocessing succeed (`--keep-native` keeps it). Failed runs keep native output
for diagnosis and return a nonzero job status.

For one-command submission of the full comparison, use
`bash tools/submit_campaign.sh --gpu-feature=GPU_SKU:H100_SXM5` on Sherlock after
uploading the current bundle. This fetches pinned sources and submits the smoke,
build and small pilot jobs first. Every full-matrix job has `afterok` dependencies
on those gates; failed gates prevent expensive dependent jobs from starting.
The command records every submitted Slurm ID in `campaigns/TIMESTAMP.tsv`. If
Sherlock rejects a submission partway through (for example, a queue limit),
continue with `bash tools/submit_campaign.sh --gpu-feature=GPU_SKU:H100_SXM5
--resume campaigns/TIMESTAMP.tsv` once there is room. The resume option reuses
existing job IDs. `--dry-run` prints the matrix without submitting jobs.

The cost model uses assumed rates, especially for CT and AthenaK, and has not
been calibrated on the selected Sherlock H100 yet. The central full-matrix
estimate is about 23 GPU-hours, 773 CPU core-hours and 59 GiB of uncompressed
returns, plus setup, builds and pilot runs. The pessimistic rates give about
43 GPU-hours and 1,343 CPU core-hours. Three long CPU tests now request three
hours to cover their pessimistic modeled wall times.

Not implemented: a driven-dynamo comparison (Pakmor et al. 2026) needs one forcing
implementation shared by every code; the CT prototype has none.

## 5. Download the results

Raw results default to `$SCRATCH/mhd-results/TEST-JOBID-ARRAYINDEX/`.
`MHD_RESULTS` overrides that directory. A normally exiting job, including an
ordinary simulation failure, writes `returns/TEST-JOBID-ARRAYINDEX.tar.gz` and a
`.sha256` file **inside this bundle**. Archives include driver logs, NVIDIA and
Slurm details, the actual Julia environment, original input, and per-run results.

If Slurm kills the job before packing finishes, raw files remain in scratch.
An interrupted `run.toml` may still say `running`; the analyzer treats that as
incomplete. After the job has stopped, pack it manually:

```bash
bash tools/pack_results.sh "$SCRATCH/mhd-results/TEST-JOBID-ARRAYINDEX"
```

Packing refuses to overwrite an existing archive. Neither packing nor analysis
removes raw results. Snapshots support analysis; automatic simulation restart
from these files is not implemented.

On the Mac, substitute the actual absolute scratch path from Sherlock:

```bash
mkdir -p scripts/out/sherlock_downloads
rsync -av "$SHERLOCK_USER@login.sherlock.stanford.edu:/ACTUAL/SCRATCH/PATH/mhd-study/sherlock_mhd/returns/" scripts/out/sherlock_downloads/
```

Verify and extract a returned archive in a new directory:

```bash
cd scripts/out/sherlock_downloads
shasum -a 256 -c TEST-JOBID-ARRAYINDEX.tar.gz.sha256
mkdir -p extracted
tar -xzf TEST-JOBID-ARRAYINDEX.tar.gz -C extracted
```

From the repository root, using the existing movie/analysis environment:

```bash
.venv-movie/bin/python sherlock_mhd/tools/analyze_returns.py \
  scripts/out/sherlock_downloads/extracted \
  --out scripts/out/sherlock_analysis
```

To show the results in the progress notebook (`progress/gpu-suite.html`), rebuild just
that page from the same extraction directory:

```bash
.venv-movie/bin/python scripts/build_progress_site.py --gpu-only
```

The analyzer needs Python >=3.11, NumPy and Matplotlib. It writes `norms.csv`,
`spectra.csv`, `diagnostics.csv`, `timings.csv`, `timing_summary.csv`,
`analysis_summary.json`, and per-case density/error plots. Failed/incomplete
cases remain visible and are excluded from timing medians. Existing CPU results
and the progress notebook are not overwritten. The returned fields can also
be used to make movies and add GPU results to the notebook later.

## Output format and timing scopes

Each `glm_r1/`, `ct_r1/`, etc. contains `run.toml` and `diagnostics.csv`.
Repeat 1 additionally contains snapshots. Later repeats keep diagnostics and
exact-reference norm summaries but omit binary fields to limit return size.

- `state_###.bin`: little-endian Float32, Julia/Fortran order, x index fastest.
- `ref_###.bin`: same layout, the reference state at that output when one exists
  (exact translated solution, or advected initial loop). The analyzer reads these files;
  it no longer re-implements the analytic solutions.
  Fields: rho, mx, my, mz, E, Bx, By, Bz, psi. CT's psi slot is zero.
- Planar tests: shape `(N,N,9)`, at z index 1. The full 3D volume is checked for
  z variation at every output. Alfvén and the 2D field loop have nonzero vz/Bz but stay
  z invariant.
- 3D tests (`orszag_tang_3d`, `field_loop_3d`, `blast_3d_*`, `decay`): full `(N,N,N,9)`.
- CT: `face1_###.bin`, `face2_###.bin`, `face3_###.bin`, storing the lower periodic
  face in each direction, same spatial shape as the corresponding state file.
- `run.toml` maps each frame to its actual physical time and records kernel
  choice, parameters, status, timing scopes and any exception. The environment
  record includes source hashes and the real GPU model.

GLM and CT use the same physical CFL rule but independently chosen timesteps.
The original GLM wrapper infers cleaning speed from the actual timestep, so
clipping to output times also changes its cleaning speed. This behavior is
preserved and recorded. Compare identical configured output cadences.

`stepping_seconds` measures synchronized elapsed time in the original evolution
wrapper, including launches and the completion wait. `cfl_seconds` measures the
common signal scan/reduction and CT cell-field preparation. `solve_seconds` is
their sum. Compilation/warm-up, host transfers, CPU diagnostics and output are
separate. The first repeat's `evolution_wall_seconds` includes snapshot output;
later repeats omit binary output. The analyzer never combines those two scopes
into one end-to-end median.

Diagnostics use unfloored pressure computed from conserved fields and stop on
nonpositive/nonfinite states. **Observed cells at floors are not counts of
internal floor activations.** Those internal branches are not instrumented.
`allocated_array_bytes` reports explicit live arrays, including adapter scratch;
it is **not peak GPU memory** and omits CUDA context/pool overhead. CT's original
scratch footprint is much larger than GLM's tiled implementation, so record
that implementation distinction when interpreting memory or speed.

At N=128 a planar pair's 21 snapshots are about 28 MiB uncompressed. At N=64,
the decay pair's five full-volume snapshots are about 105 MiB per seed. A 3D test at N=128
writes about 75 MiB per GLM snapshot and 100 MiB per CT snapshot (state plus faces),
so `orszag_tang_3d` (5 snapshots) returns roughly 0.9 GiB for its N=128 task; trim
`snapshot_frames` if that is too large. Repeated
timings add small tables rather than duplicate full field outputs.

## Verification and sources

Local checks can be repeated without an NVIDIA GPU:

```bash
julia --startup-file=no tools/check_initial_conditions.jl
julia --project=env tools/check_roundtrip.jl /tmp/NEW-mhd-format-fixtures
python3 tools/check_analysis.py
python3 tools/analyze_returns.py /tmp/NEW-mhd-format-fixtures --out /tmp/NEW-mhd-format-analysis
```

The format fixtures are synthetic CPU data, prominently labeled as such; they
must not be reported as GPU results. The original solver files are byte-identical
to the local source snapshot. The actual GPU smoke job remains pending.

Every config was also evolved to its end time on the Mac with the original GLM physics
through its KernelAbstractions CPU path (`step_fused!`, N=32 planar / N=16 3D). That was
only a check that the initial conditions evolve sensibly. It is not a GPU result, and CT
cannot run on the CPU.

Official documentation checked 2026-09-24:

- [Sherlock GPU requests and development instances](https://www.sherlock.stanford.edu/docs/user-guide/gpu/)
- [Julia modules on Sherlock](https://www.sherlock.stanford.edu/docs/software/using/julia/)
- [Sherlock job and scratch-storage guidance](https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/)
- [CUDA.jl synchronized timing guidance](https://cuda.juliagpu.org/stable/development/profiling/)
