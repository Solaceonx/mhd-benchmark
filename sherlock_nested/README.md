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

- `driven_debug.jl` / `driven_debug.sbatch`: that failure, small: nest 3 × 32³ and uniform 128³ at Courant 0.4 and
  0.25, then 3 × 64³ at 0.25; per root step per level the CFL step, density range, max speed and where the largest
  signal speed is (`<case>.csv`); a watchdog saves every level's state when a step collapses. Output
  `$SCRATCH/mhd-results-nested/driven_debug/`. Checked on the Mac CPU (`NESTED_BACKEND=cpu ... tiny_c040`).

Send (Mac), then submit (Sherlock):

```bash
cd /Users/lyxia/Projects/research/abel_mhd && COPYFILE_DISABLE=1 tar --no-xattrs -C external -cz NestedTurbulence | ssh sherlock 'tar -xz -C $SCRATCH/mhd-study' && COPYFILE_DISABLE=1 tar --no-xattrs -cz sherlock_nested | ssh sherlock 'tar -xz -C $SCRATCH/mhd-study'
```

Sherlock's git (1.8) has no `git -C`: write the commit to `NestedTurbulence/COMMIT` after sending.

```bash
cd $SCRATCH/mhd-study/sherlock_nested && mkdir -p logs && p=$(sbatch --parsable precompile.sbatch) && sbatch --dependency=afterok:$p gpu_check.sbatch
```
