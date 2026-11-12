# Tom Abel's nested-grid turbulence (NestedMHD) on Sherlock's H100s

Tom's code: github.com/yipihey/NestedTurbulence (MIT), Julia package `NestedMHD` in `core/`. Report:
yipihey.github.io/NestedTurbulence. Static ratio-2 nested patches centred in a periodic box, each level with its own
CFL step (Berger–Oliger order, time-interpolated ghosts, refluxing, CT EMF correction); GLMMHDTurb's PLM/MUSCL-Hancock
physics with GLM or CT (the GS05 CT fix of Solaceonx/GLMMHDTurb@sweep-fixes). Tom had run it on CPUs only.

Not vendored here: the clone lives in `../external/NestedTurbulence` (git-ignored) and is sent to Sherlock as is.

- `env/`: CUDA.jl 6.5.1 + KernelAbstractions 0.10.4 (Tom's kernels use `return` inside `@kernel`, which
  KernelAbstractions 0.9, and so the sweep's CUDA.jl 5 environment, refuses). Depot `$HOME/.julia-mhd`.
  CUDA.jl's own toolkit artifacts need glibc 2.27 and Sherlock has 2.17 (job 51642135 failed on libcublasLt), so
  `env/LocalPreferences.toml` selects the local toolkit and the jobs load `devel cuda/12.4.0`. The preferences only
  apply because the two JLLs are listed under `[extras]` in `env/Project.toml`.
- `precompile.sbatch`: installs and precompiles `env/` on a CPU node (no GPU hours); it must print
  `local toolkit true` before the GPU job is worth running.
- `setups.jl`: Tom's 3-D Orszag–Tang, and his Mach-10 driven set-up (`core/bench/run_mach10.jl`) in 3-D.
- `test_gpu_compile.jl` (Mac, no GPU): records every kernel launch of a short CPU run of each set-up and compiles it
  for sm_90 with the GPU argument types. 2026-11-09: 73 launches, all compile; a known-bad kernel is caught.
  `julia --project=env test_gpu_compile.jl` (official Julia: side_projects/.julia-dist; Homebrew's Pkg is broken).
- `gpu_check.jl` / `gpu_check.sbatch`: CPU/GPU agreement on a nested Orszag–Tang, speed (uniform 128³–512³ and a
  3-level nest, CT/GLM, Float32/Float64), and one crossing time of driven Mach-10 turbulence nested
  (3 × 64³, finest 256³ in the central 1/4) and uniform 256³. Output `$SCRATCH/mhd-results-nested/gpu_check/`.

gpu_check, job 51686178 (2026-11-09): CPU and GPU agree to round-off on every level (Float64 1e-8, Float32 1e-6;
same step counts), CT keeps div B at 1e-15; uniform 1.6–2.1e9 cell updates/s (Float32), 1.15e9 (Float64); the 3-level
nest 1.1e9 with GLM but 0.37e9 with CT. The driven Mach-10 nest (Courant 0.4) failed after ~55 min: level 2's CFL step
collapsed (ceil(Int, remaining/dt) overflowed at 8e19). In 3-D these kernels are unsafe at Courant 0.4 (sum rule;
sherlock_checks, 2026-10-12); Tom's run_mach10.jl is 2-D.

**Fixed 2026-11-10.** Two causes, both 3-D only. (1) Courant 0.4 (sum rule) is unstable in 3-D even uniform: 0.25.
(2) `prolong_cells_kernel!` (interlevel.jl) adds three independently MonCen-limited slopes, so a ghost cell at a corner
of its coarse cell can reach q0 − 3/2·q0 < 0 next to a near-vacuum (2-D: at most 0): negative ghost density beside the
face where |B| ran away (every 3-D nest, CT or GLM, 2 or 3 levels, t ≈ 0.034). Fix: one α scales every slope of the
cell so the fine density stays ≥ the lowest of the six coarse neighbours; non-positive thermal energy → coarse state.
Branch `gpu-3d-fixes` (4fe5682) of `../external/NestedTurbulence` (local; not in Tom's repository), on Sherlock as
`$SCRATCH/mhd-study/NestedTurbulence-gpu3d` (`NESTED_DIR`). Job 51745822: 3 × 64³ and 3 × 32³ CT, 2 × 32³, GLM all reach
t = 0.1 at Mach 10.1–10.3, CT div B 1e-15; Tom's 2-D nest unchanged. Driving's per-root-step `totals(H)` copies the root
to the host: it dominates the wall time of these small runs (75 Mcell/s for 3 × 64³).

- `driven_debug.jl` / `driven_debug.sbatch`: that failure, small: nest 3 × 32³ and uniform 128³ at Courant 0.4 and
  0.25, then 3 × 64³ at 0.25; per root step per level the CFL step, density range, max speed and where the largest
  signal speed is (`<case>.csv`); a watchdog saves every level's state when a step collapses. Output
  `$SCRATCH/mhd-results-nested/driven_debug/`. Checked on the Mac CPU (`NESTED_BACKEND=cpu ... tiny_c040`).

- `compare.jl` / `compare.sbatch`: nest (3 × 64³, 3 × 128³) against uniform at its finest resolution (256³, 512³),
  histograms and 3-D snapshots every 0.05. Figures and the site page: `scripts/nested_figures.py`, `scripts/build_nested.py` (Mac).
- `movie.jl` / `movie.sbatch`: the same nest with a mid-plane slice of every level (ln ρ, |v|, log10 |B|) every
  0.002 to t = 1 (501 frames; nest64 ~37 MB, nest128 ~300 MB) in `$SCRATCH/mhd-results-nested/movie/`.
  `plot_movie.py` (Mac): the movie. Checked on the Mac CPU (`NESTED_BACKEND=cpu ... movie.jl ... tiny`).

- `bigrun.jl` / `bigrun.sbatch`: a long nest that stops and resumes (checkpoint every 30 min, bit-identical on resume,
  checked on the Mac CPU), with an Alfvén-speed cap (`BIG_VA_MAX`, default 30 c_s) from our commit 1164c73 of
  `../external/NestedTurbulence` (on Sherlock as `NestedTurbulence-cap`). Case `deep192`: root 128³ + 3 levels of 192³
  (finest 1024³-equivalent over the central 3/16; level 3 512³-equivalent over 3/8). Histograms per level every 0.01,
  slices every 0.002, 3-D snapshots every 0.05 in `$SCRATCH/mhd-results-nested/big/`. Job 52101427 (2026-11-12).

Send (Mac), then submit (Sherlock):

```bash
cd /Users/lyxia/Projects/research/abel_mhd && COPYFILE_DISABLE=1 tar --no-xattrs -C external -cz NestedTurbulence | ssh sherlock 'tar -xz -C $SCRATCH/mhd-study' && COPYFILE_DISABLE=1 tar --no-xattrs -cz sherlock_nested | ssh sherlock 'tar -xz -C $SCRATCH/mhd-study'
```

Sherlock's git (1.8) has no `git -C`: write the commit to `NestedTurbulence/COMMIT` after sending.

```bash
cd $SCRATCH/mhd-study/sherlock_nested && mkdir -p logs && p=$(sbatch --parsable precompile.sbatch) && sbatch --dependency=afterok:$p gpu_check.sbatch
```
