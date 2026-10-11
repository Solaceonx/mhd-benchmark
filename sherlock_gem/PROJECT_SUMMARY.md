# GLM vs CT on GPUs: project summary

> **GEM reconnection bundle.** Holds `configs/gem.toml` (accuracy test) and the two GEM movie configs.
> Follow [GEM.md](GEM.md); the rest of this file describes the original campaign and is kept for reference.

*Status as of 2026-09-25. Everything below has been built and checked on the Mac. **No run
has executed on a GPU yet**; the Sherlock smoke job is the first real test.*

## Goal

Show, with a fair and reproducible comparison, where the advisor's GPU GLM (Dedner) MHD code
does better than constrained transport (CT), and where it does not. The main claim to test is
**accuracy per unit of cost**. CT tends to win at a fixed grid (on Orszag–Tang by about the
same amount as 1.1–1.25× more resolution). GLM's case to test is speed and memory:
a fused update may allow finer grids for the same budget. Nine floats describe one
state; the current driver holds two states plus a signal array (about 19 Float32
values per cell, before shared memory and CUDA overhead). Use measured cost and
recorded allocations rather than the single-state size for comparisons.

Three recent papers frame the comparison:

- **Kritsuk et al. (2011):** divergence cleaning loses magnetic spectral bandwidth compared
  with CT.
- **Tomida et al. (2026):** conventional Dedner, with c_h ∝ 1/Δt, leaks field and reacts badly
  to sudden timestep changes. They propose a constant c_h.
- **Pakmor et al. (2026):** no systematic CT advantage; the codes differ mainly in their
  dissipation.

## What was done, in order

### 1. Review of the original bundle
`sherlock_mhd/` is a self-contained Sherlock package. It holds the advisor's GPU GLM solver
and a lean CT prototype (both Julia/CUDA, vendored unmodified), a driver, 8 tests and an
analyzer. Findings:

- **The GLM path runs a Rusanov (LLF) flux, not HLL** (`use_hlld=false` selects `llf_x`), while
  CT uses HLL. This is the advisor's code, and was raised with him rather than changed here.
- **Half the tests could not tell GLM from CT.** The uniform state, entropy wave, grid-aligned
  Alfvén wave and grid-aligned shock tube all have ∇·B = 0 trivially.
- **Other differences were mixed in with the ∇·B method:** full vs transverse-free Hancock
  predictor, the Riemann solver, CT's arithmetic edge-EMF averaging, and different floors.
- **"Divergence" meant different things for the two methods:** a cell-centred operator for
  GLM, the face constraint for CT.
- **A job-script bug:** `julia … | tee` recorded tee's exit status, so failed runs looked
  successful. Now fixed.

### 2. FLASH MHD test problems adapted to the periodic GPU solvers
Both solvers only support a periodic N³ unit cube, and CT hard-codes γ = 5/3.

- **Added:**
  - Brio–Wu, grid-aligned and rotated 45°
  - Orszag–Tang 3D
  - the Balsara–Spicer rotor (the old one had a taper bug and γ = 5/3)
  - the Gardiner–Stone current sheet
  - three field loops (FLASH v_z = √5, in-plane, and 3D)
  - 2D and 3D blasts at B = 0, 50/√4π and 100/√4π
  - the oblique Alfvén wave
- **Initial fields** all come from an edge vector potential, so ∇·B = 0 to round-off at t = 0.
- **Discontinuous initial states** are cell averages.
- **Per-problem γ** (5/3, 7/5, 2, 1.01): the unmodified CT vendor file is re-parsed at load
  time with only its `GAMMA` constant replaced. The file on disk is never edited.
- **Pressure floors:** the magnetized blasts drive pressure negative within a few steps. Those
  tests continue on the solvers' own floors and count the affected cells; every other test
  stops at the first nonpositive pressure.
- **Not possible without changing the solvers:** the accretion torus (cylindrical geometry,
  gravity, outflow boundaries) and the Noh Z-pinch (inflow/outflow boundaries).

### 3. Multi-code comparison
Codes added, all pinned to exact commits:

| Code | Runs on | Role |
|---|---|---|
| GLM (advisor): original tiled path (+4 kernels on request) | GPU | method under test |
| CT prototype | GPU | same-code CT baseline |
| **AthenaK** | GPU (Kokkos/CUDA) | production GPU CT: the fair cost competitor |
| **RAMSES** | CPU (MPI) | CT from the Kritsuk and Pakmor comparisons; accuracy only |
| **Athena++** | CPU (MPI) | independent cross-check of the references |

GLM variants, all the advisor's unmodified kernels:

| Variant | Kernel |
|---|---|
| `glm` | original tiled LLF path |
| `glm_hll16` | **his production f16/HLL kernel** |
| `glm_hlld` | HLLD |
| `glm_ppm` | PPM |
| `glm_constch` | production kernel with constant c_h (Tomida's fix, applied by the driver) |

The active campaign compares the original tiled `glm` path with `ct`. The
other GLM variants remain available for targeted follow-up runs and do not
gate the core comparison.

Note (2026-09-28): this campaign ran the Kelvin–Helmholtz and current-sheet tests
with the setups in these configs. A later literature check (see the Marlowe
bundle) found they differ from Tomida et al. (2026) and from Gardiner & Stone
(2005); the Marlowe bundle carries the corrected versions.

- **Identical initial data in every code.** Julia `problems.jl` is the source of truth. A
  shared C++ header rebuilds the same state inside AthenaK and Athena++, and RAMSES reads
  Julia's exact Float32 state. For every problem, each code's t = 0 state matches Julia to
  ≤ 1.2×10⁻⁷.
- **Identical output times.** Every code lands exactly on the output times. AthenaK needed a
  small build-time patch for this; its physics is untouched.
- **New tests:**
  - Tomida-style Kelvin–Helmholtz with a uniform field, and with a field localized on one layer
    (to measure leakage)
  - the Tomida timestep-drop protocol (GPU only)
  - a Kritsuk-style Mach 9, γ = 1.01 decay
  - a throughput benchmark at 96³–384³ (sizes that let the production kernel use its fastest
    6-cell tiles)
- **References.** AthenaK in double precision at 2048² (2D) or 384³ (3D), saved block-averaged
  onto each comparison grid. Athena++ at half that resolution measures how sensitive the
  results are to the reference.
- **Checked on the Mac CPU:** Orszag–Tang at 64² gives density L1 errors of 7.3% (AthenaK
  matched), 3.9% (AthenaK best), 6.6% (RAMSES matched) and 4.7% (RAMSES best). That is
  consistent with the earlier CPU GLM/CT study (6.6% / 5.9%).

### 4. Progress notebook (`progress/gpu-suite.html`)
- a status matrix with one dot per code
- any two methods side by side against the reference, per test, grid, field and output time
- every method in one table
- time histories
- three panels: **accuracy vs grid, accuracy vs GPU-seconds, and throughput**

Hue encodes the code (the palette is checked for colour-vision deficiency) and line style the
variant. CPU-measured runs never enter the GPU-cost panel. It rebuilds in seconds with
`scripts/build_progress_site.py --gpu-only`.

## What will run on Sherlock

**One-time setup:**
```
bash tools/fetch_codes.sh                  # pinned sources into $SCRATCH/mhd-codes
bash tools/submit.sh setup                 # Julia environment
bash tools/submit.sh build-gpu-codes       # AthenaK for the chosen GPU model
bash tools/submit.sh build-cpu-codes       # RAMSES + Athena++ (MPI)
bash tools/submit.sh smoke                 # all 25 tests at 32³ on the GPU driver
```

**The study:** 25 tests, 145 commands in total, all printed by `bash tools/submit.sh matrix`:

`bash tools/submit_campaign.sh --gpu-feature=GPU_SKU:H100_SXM5` submits the
full matrix with smoke, builds, and five small pilot jobs as Slurm success
dependencies. This is the one-command path for a complete campaign. The first
H100 performance measurements are still pending; the resource model in
`tools/estimate_cost.jl` is an estimate based partly on assumed throughput.

| Jobs | Commands | Array tasks | Each task |
|---|---:|---:|---|
| GPU driver (GLM + CT) | 25 | 80 | `glm` and `ct` × 5 repeats, same GPU, rotating order |
| AthenaK `matched` and `best` | 48 | 156 | 5 repeats; repeat 1 writes snapshots |
| RAMSES `matched` and `best` (CPU) | 46 | 150 | 5 repeats, 32 MPI ranks |
| AthenaK references | 13 | 13 | one high-resolution run, saved restricted |
| Athena++ reference checks (CPU) | 13 | 13 | one run at half the reference resolution |

**Rules for fair timing:**
- Use the same `--constraint` (GPU model) for the AthenaK build and every GPU run.
- Timing medians use repeats 2–5 (repeat 1 includes snapshot output).
- GPU and CPU seconds are never mixed.

## Deliverables once the runs finish

1. **Raw, reproducible archives** (`returns/*.tar.gz` with SHA-256). Each holds the input
   config, the environment (GPU model, commits, source hashes, modules), per-repeat `run.toml`,
   `diagnostics.csv`, snapshots and reference files. Failures are kept, never retried silently.
2. **The notebook page** filled in for all 25 tests.
3. **Accuracy per cost**, the headline figure: error vs GPU-seconds for GLM variants, CT and
   AthenaK, per test and grid. This is where GLM's speed and memory advantage either shows up
   or doesn't.
4. **Throughput and memory**: cell-updates per second vs N up to 384³, and array memory.
   Caveat: AthenaK runs in fp64 only (its fp32 build does not compile at the pinned commit),
   so this is fp32 GLM/CT vs fp64 AthenaK, and the page labels it.
5. **Convergence**: error vs N on the exact tests (entropy, both Alfvén waves, field loops)
   and against the references elsewhere, with reference sensitivity from Athena++.
6. **∇·B-specific evidence**: field-loop energy decay and B_z growth; the Tomida
   timestep-drop response for `glm` vs `glm_constch` vs `ct`; the rotated shock tube.
7. **Robustness**: counts of nonpositive-pressure cells and failures in the strongly
   magnetized blasts, for every method.
8. **Turbulence**: energy histories and spectra for the Mach 2 and Mach 9 decays.

**Analysis still to write once data exists:**
- the KH leakage metric (field at the upper layer against its 3.7×10⁻⁶ initial tail)
- Kritsuk-style effective Reynolds numbers and spectral bandwidth
- a combined cross-test summary table

The raw quantities they need are already saved.

## Known limitations and open items

- **Not yet run on a GPU:** the GLM variant launches, constant-c_h GLM, the CT γ variants and
  AthenaK's CUDA build are untested. Module names in the job scripts are guesses you can
  override (`MHD_*_MODULES`).
- **The GLM LLF-vs-HLL question** is still with the advisor. The suite now runs his production
  HLL and HLLD kernels, so the headline comparison no longer depends on it.
- **Only the CT prototype and AthenaK share hardware with GLM.** RAMSES and Athena++ answer
  accuracy questions only.
- **Not implemented:** accretion torus, Noh Z-pinch (both need boundary/geometry support in
  the solvers), and a driven dynamo (every code would need one shared forcing implementation).
- **Storage and launch:** see [SHERLOCK_LAUNCH.md](SHERLOCK_LAUNCH.md) for lyxia's
  exact paths, module checks, staged first jobs and download commands. The synthetic
  return fixture is retained locally under `scripts/out/sherlock-format-fixtures/`,
  outside this transfer bundle.
