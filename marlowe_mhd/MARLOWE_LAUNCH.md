# Marlowe launch guide

This is the Marlowe copy of the GLM/CT/AthenaK GPU study. Its scientific inputs
match the Sherlock bundle except for four tests corrected after a literature
check on 2026-09-28 (the Sherlock campaign had already run them):

- `current_sheet`: vx = 0.1 sin(πy) on [0,2]², the original Gardiner & Stone (2005)
  and RAMSES form; the Sherlock run used FLASH's sin(2πy) (twice the wavenumber).
- `kh_uniform`, `kh_localized`, `kh_localized_dtjump`: Tomida et al. (2026)
  parameters (p = 10, v = ±1, 0.01 sin 2πx perturbation, σ = 0.2, Bx = 0.01,
  timestep drop at t = 3.0 run to 3.2); the Sherlock runs used p = 1, v = ±0.5.
  The timestep-drop test also runs `glm_hll16` and `glm_constch` so the c_h
  policy is compared on one kernel.

Do not pool these four tests across the two sites. RAMSES and Athena++ CPU
baselines remain on Sherlock; this package submits only the GPU driver and
AthenaK unless explicitly changed.

## What will run

- Original Float32 CUDA GLM and CT implementations on identical initial states.
- AthenaK Float64 `matched` (PLM/HLLE/RK2) and `best` (PPM4/HLLD/RK3).
- AthenaK high-resolution references where a config defines `reference_n`.
- The same 25 test configurations, grids, seeds, CFL numbers and output
  cadences as the Sherlock campaign, apart from the four tests listed above.

`SCIENTIFIC_INPUTS.sha256` protects the configs, initial-condition generators,
vendored kernels, pinned source versions and compact references from accidental
changes during transfer.

## 1. Package and transfer from the Mac

From `/Users/lyxia/Projects/research/abel_mhd`:

```bash
bash marlowe_mhd/tools/package_marlowe.sh
rsync -av --progress \
  marlowe_mhd.tar.gz marlowe_mhd.tar.gz.sha256 \
  lyxia@login.marlowe.stanford.edu:~/
```

## 2. Unpack in project scratch

Replace `PROJECT_ID` with the directory name assigned to the Marlowe project.

```bash
ssh lyxia@login.marlowe.stanford.edu
export MARLOWE_PROJECT=PROJECT_ID
(cd "$HOME" && sha256sum -c marlowe_mhd.tar.gz.sha256)
mkdir -p "/scratch/$MARLOWE_PROJECT/mhd-study"
cd "/scratch/$MARLOWE_PROJECT/mhd-study"
tar -xzf "$HOME/marlowe_mhd.tar.gz"
cd marlowe_mhd

cp site/marlowe.env.example site/marlowe.env
sed -i "s/REPLACE_WITH_PROJECT_ID/$MARLOWE_PROJECT/" site/marlowe.env
source site/marlowe.env
```

The checksum command assumes both uploaded files are in `$HOME`.

## 3. Discover modules and run preflight

Check Julia and Python before editing `site/marlowe.env`:

```bash
module spider julia
module spider python
module spider nvhpc
module spider cmake
```

Use a Julia 1.12.x module or set `JULIA_BIN` to a Linux Julia 1.12 executable.
If Python/NumPy need modules, put their exact module names in
`MHD_PYTHON_MODULES`. If `cmake` is not provided by the base environment, append
its exact module name to `MHD_GPU_BUILD_MODULES`.

```bash
bash tools/marlowe_preflight.sh
```

The preflight verifies the account association, current preempt limits,
writable scratch paths, Python, NumPy, Julia, CUDA, and all scientific-input
checksums. It does not request a GPU.

## 4. Install and validate the Julia environment

```bash
setup_id=$(bash tools/submit.sh setup --parsable)
echo "$setup_id"
sacct -j "$setup_id" -X --format=JobID,State,ExitCode,Elapsed
tail -n 60 "logs/mhd-setup-$setup_id.out"
```

Wait for `COMPLETED 0:0`. The setup job runs the CPU initial-condition tests and
installs the locked Julia packages. It does not certify GPU evolution.

Marlowe's documentation does not state whether compute nodes can reach the
internet, and package installation downloads from the Julia registry. If the
setup job fails while downloading, run the same step on the login node (it uses
no GPU and a few CPU minutes):

```bash
source site/marlowe.env && source jobs/common.sh && load_site_profile && prepare_environment
load_modules "$MHD_GPU_RUN_MODULES"   # precompilation records the CUDA runtime version
"$MHD_JULIA" --project=env tools/setup.jl
```

CUDA.jl is configured to use the system CUDA from `nvhpc`
(`env/LocalPreferences.toml`), so no CUDA runtime is downloaded on GPU nodes.

## 5. Submit only the GPU gates

```bash
bash tools/submit_campaign.sh --gates-only
```

This submits, in dependency order:

1. all-test 32³ GLM/CT smoke;
2. Hopper/SM90 AthenaK build;
3. 64³ Orszag–Tang GLM/CT pilot;
4. 96³ tiled-kernel throughput pilot;
5. 64³ AthenaK matched pilot.

The command prints a ledger path such as `campaigns/DATE.tsv`. Check every gate:

```bash
ledger=campaigns/REPLACE_WITH_PRINTED_LEDGER.tsv
ids=$(awk -F '\t' 'NR>1 {print $2}' "$ledger" | paste -sd, -)
sacct -j "$ids" -X --format=JobID,JobName%36,State,ExitCode,Elapsed
```

## 6. Submit production after the gates pass

```bash
bash tools/submit_campaign.sh --resume "$ledger"
```

Every production job remains behind any unfinished gate. Arrays default to one
concurrent task, so the account's live GPU cap controls scheduling without
changing the scientific matrix. Do not add a Sherlock GPU feature constraint on
Marlowe.

Marlowe Basic jobs can be preempted and restarted from the beginning. The short
and medium tests are suitable for `preempt`. Before launching the longest 256³
decay, 384³ throughput, or high-resolution reference tasks, either obtain access
to the non-preemptible `batch` partition or add restart/checkpoint support.

## Results

Raw runs go to `/scratch/$MARLOWE_PROJECT/mhd-results`. Each completed task also
writes a compact archive and checksum to `returns/` inside the uploaded bundle.
Download those archives to the Mac for the existing analysis and progress site.

What each run records, for benchmarking:

| Quantity | Where |
|---|---|
| Stepping, CFL-scan and solve time per repeat (5 repeats; repeat 1 also writes snapshots) | `run.toml` (`stepping_seconds`, `solve_seconds`), AthenaK `code_seconds` |
| Throughput | `million_cell_steps_per_second` (N³ cells × steps / solve time) |
| GPU memory | GPU driver: `allocated_array_bytes`, `device_memory_after_warmup_bytes`, `device_memory_peak_bytes`, CUDA pool bytes. AthenaK: `device_memory_peak_bytes` and `device_memory_above_baseline_bytes` from `nvidia-smi` sampling |
| Accuracy | errors against exact/driver references in `run.toml`; AthenaK references block-averaged to each grid |
| Conservation, ∇·B, floors, ψ | `diagnostics.csv` at every output (`mass_drift`, `energy_drift`, divergence measures, `psi_rms`, `psi_max`, `nonpositive_pressure_cells`) |
| KH leakage | `leak_band_bmax`, `leak_band_brms`: field at the field-free upper layer, every output |
| Timestep-drop response | `dense_diagnostics.csv`: every step from the drop until 300 steps after it (repeat 1) |
| Provenance | GPU model and clocks (`nvidia-smi -q`), source hashes, build commits, modules, Slurm record |

Analyze Marlowe returns against this bundle's configs (four tests differ from Sherlock):

```bash
MHD_ANALYSIS_BUNDLE=marlowe_mhd .venv-movie/bin/python scripts/build_progress_site.py \
  --gpu-only --gpu-returns PATH/TO/EXTRACTED/MARLOWE/RETURNS
```
