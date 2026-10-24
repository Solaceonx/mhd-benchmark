# mini-ramses-ism in the comparison suite

Eric Moseley's version of RAMSES II (bitbucket.org/ermoseley/mini-ramses-ism, branch `develop`, pinned at `579d842`,
2026-10-22): the GPU rewrite of RAMSES in CUDA Fortran, still called mini-ramses. Constrained transport, MUSCL-Hancock
with the MonCen slope, on a uniform grid here (levelmin = levelmax = log2 N), one H100 per run, no MPI.

## Set-ups

As classic RAMSES in the sweep (`sherlock_sweep/codes/external.py`):

| Label | Riemann solver | Courant |
|---|---|---|
| `miniramses_matched` | HLL | 0.5 |
| `miniramses_best` | HLLD | 0.8 |

Double precision (`NPRE=8`) by default, as the other external codes; `MINIRAMSES_NPRE=4` builds and runs Eric's
single-precision production setting instead (the records say which). Eric's switches to LLF below a density or
pressure threshold (`switch_llf_dmin`, `switch_llf_pmin`, his newest commit) are off, as upstream.

## What runs

`runs.json` (`python3 make_tasks.py` on the Mac): every run classic RAMSES has in the sweep, the same tests, grids and
seeds, 130 per set-up; plus the speed test `throughput_ot3d` at 64³–512³ with its repeats, which classic RAMSES did not
run. All in 3D: the GPU path of mini-ramses-ism is 3D only (`gpu/gpu_utils.cuf`: "CUDA GPU path requires NDIM=3", job
49368458), so a planar test runs on N³ cells that are the same at every z, as AthenaK's planar runs in the sweep; the
patched `condinit` reads the sweep's plane and repeats it through z, and the records keep the z = 0 plane. 268 runs,
about 43 GPU hours by the estimate in `make_tasks.py`, in seven tasks. Results: `$SCRATCH/mhd-results-sweep/<test>-miniramses-<set-up>-run-N<n>-s<seed>/`,
the sweep's layout and records (`run.toml`, `diagnostics.csv`, `state_###.bin`), through the sweep's own driver:
`miniramses_run.py` adds only the namelist, the run and the reading of mini-ramses's outputs.

## The patch (`prepare_patch.py`)

Three files replace upstream ones through the Makefile's `PATCH=`; nothing in the solver changes.

- `condinit.f90` and `abel_ic.f90`: the initial state is the sweep's own Float32 data (`tools/write_spec.jl --state`),
  as for classic RAMSES, and the face-centred field is set from the sweep's face files instead of the curl of
  `vecpotentialinit`'s vector potential.
- `newdt_fine.f90`: the step before an output is shortened so that the output lands on its time (classic RAMSES's
  `exact_output_time`); upstream writes it at the first step past it.
- mini-ramses ends a run as soon as the time reaches the last output time, before the step that would write that
  output; the namelist therefore lists one more output time, 10⁻⁹ t_end after the end.

## Tested on the Mac (CPU build, gfortran 15, double precision)

The same driver as on Sherlock, with the CPU builds (`make COMPILER=GNU NDIM=2|3 MPI=0 HYDRO=1 MHD=1 NPRE=8
PATCH=... ramses`, 13 s):

| Run | mini-ramses-ism | classic RAMSES in the sweep |
|---|---|---|
| field loop, 64², HLL: error of B at t = 2 | 0.231, 1440 steps | 0.236, 1440 steps |
| field loop 3D, 32³, HLL: error of B at t = 1 | 0.271, 510 steps | 0.275, 510 steps |

Every output lands on its time; the magnetic energy at t = 0 is the sweep's to every digit. **Not tested:** the GPU
build, which needs the NVIDIA compiler; the pilot does that first.

Since the switch to 3D (2026-10-24): `abel_ic.f90` compiled alone (gfortran, NDIM=3) reads a plane as the same state
at every z, cell values, faces and coarse cells alike, and a full N³ file as before; the reader keeps exactly the z = 0
plane of a 3D output (synthetic output in mini-ramses's layout). The Mac copy of the source is gone and the repository
is private since 2026-10-23, so the CPU build was not repeated.

**On Sherlock:** nvhpc 25.1 (and 24.7) build with `CUDA_ARCH=sm_90`; nvhpc 25.1 stops with an internal compiler error on
`cooling/cooling_module.f90` at -O4, which `build.sh` builds at a lower level and keeps (`make -o`); the newer nvhpc
modules are visible only after `module load devel`; the job asks for H100s (`GPU_SKU:H100_SXM5`).

## Commands

On the Mac, in the project directory (sends the bundle; `submit.sh` gets the source and submits the pilot and, depending
on it, the seven tasks; it refuses while a `miniramses` job is still queued):

```bash
cd /Users/lyxia/Projects/research/abel_mhd && COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - sherlock_miniramses | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-study && tar -xzf - && bash -l sherlock_miniramses/submit.sh'
```

On Sherlock, any time:

```bash
bash -l /scratch/users/lyxia/mhd-study/sherlock_miniramses/status.sh
```

If the pilot fails, the seven tasks wait on it forever (`squeue` shows `DependencyNeverSatisfied`): cancel them with
`scancel -n miniramses` and send the status output. The compiler: the newest `nvhpc` module that builds the code
(`compiler.sh`, as for RAMSES II); `MINIRAMSES_NVHPC_MODULE` chooses one.

Fetch to the Mac when the tasks are done:

```bash
cd /Users/lyxia/Projects/research/abel_mhd && ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash /scratch/users/lyxia/mhd-study/sherlock_miniramses/tools/pack_for_mac.sh' | tar -xzf - -C sweep_results
```
