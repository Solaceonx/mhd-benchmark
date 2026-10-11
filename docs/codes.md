# Codes and set-ups

Every external code runs twice where it can: **matched** (the scheme closest to Tom's kernels: piecewise linear,
a two-wave Riemann solver, second order in time) and **best** (what its authors would use). Versions are pinned in
`sherlock_sweep/codes/versions.env`.

## Tom Abel's kernels

Julia with KernelAbstractions and CUDA, single precision, one H100. Source: `GLMMHDTurb/` after
`bash scripts/fetch_solver.sh`; the bundles use the two files in their `vendor/` unedited.

| Label | What it is |
|---|---|
| `glm` | The original tiled kernel: LLF fluxes, Float32 tiles |
| `glm_hll16` | The production kernel: HLL fluxes, tiles stored in half precision |
| `glm_hlld` | HLLD fluxes, Float32 tiles |
| `glm_ppm` | PPM reconstruction, half-precision tiles |
| `glm_constch` | As production, with a constant cleaning speed c_h |
| `ct` | The constrained-transport prototype |
| `hydro` | The Euler kernel (`step_hydro!`), used for driven turbulence only |

The timestep comes from the sum over the three directions of |v| + c_fast. The sweep ran it at Courant number 0.25;
the code's default is 0.7. The other codes use 0.3 against the fastest single direction, so the kernels took about
3.3 times more steps than AthenaK on the same problem ([sweep results](results-sweep.md)).

## External codes

| Code | Matched | Best | Notes |
|---|---|---|---|
| AthenaK | PLM, HLLE, RK2 | PPM4, HLLD, RK3 | H100; also the 4096² and 512³ references |
| RAMSES | HLL | HLLD | CPU, commit `5f0ed46`; needs an NDIM=2 build for planar tests |
| Athena++ | PLM | PPM, HLLD | CPU; also the 1024² Orszag–Tang reference of the first study |
| AREPO | | Powell cleaning, HLLD, global timestep, Courant 0.3 | CPU; a static Cartesian mesh (`arepo_static`) and the moving mesh (`arepo_moving`) |
| RAMSES II | | CT, double precision, CUDA Fortran | H100; timed on 3D Orszag–Tang only |
| MHD-ALE | | moving-mesh finite elements on MFEM, RK3, CFL 0.5 | CPU; fork [yipihey/MHD-ALE](https://github.com/yipihey/MHD-ALE); 2D has no v_z or B_z |

For hydrodynamic turbulence RAMSES runs its hydro solver: matched is HLL with slope limiter 1 at Courant 0.5, best
is HLLC with slope limiter 2 at 0.8.

## How a code gets the tests

All initial conditions live in one place, `sherlock_sweep/src/problems.jl`. AthenaK and Athena++ get them through a
generated header (`codes/common/abel_problems.hpp`), RAMSES and AREPO through a state file written by the driver,
MHD-ALE through a test case that includes the same header. Every code's output is converted to the same cell-centred
arrays before anything is measured, and every measurement is one implementation (`sherlock_sweep/src/`,
`sherlock_turb/measure.jl`).

## Not included

SWIFT and ART, the other codes of Ugalino et al. 2026, were considered and dropped. ART has no public repository.
