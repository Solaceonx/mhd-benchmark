# abel_mhd

Rotation project with Tom Abel (tabel@stanford.edu) at KIPAC, Stanford.

## Background

Tom has a GPU-accelerated MHD code written in Julia (using
KernelAbstractions.jl) that supports both Dedner (GLM) and constrained
transport (CT) divergence-cleaning formulations, with a choice of Riemann
solvers and PLM/PPM spatial reconstruction. Early results suggest Dedner is as
accurate as CT but substantially faster on GPUs — a point not yet well
appreciated in the literature (see the recent Teyssier preprint). The code is
already faster than Ramses/AthenaK and reaches throughput on the order of
GigaCells/sec; f16 precision appears sufficient for these problems.

Existing code/benchmarks live at
[yipihey/GLMMHDTurb](https://github.com/yipihey/GLMMHDTurb):
- `REPORT.md` — write-up of results so far
- `BENCHMARKS.md` — GPU throughput benchmarks
- `GPU_SOLVER_OPTIMIZATION_PLAYBOOK.md` — porting/optimization notes

Proposed first project: extend/benchmark this code (possibly on newer SLAC/
Stanford GPUs and Apple Metal), write it up as a paper, likely with Tom, Eric
Moseley (KIPAC), and Romain Teyssier (Princeton) as co-authors. Related prior
work: Wang & Abel (2010), Wang, Kaehler & Abel (2012).

## Status

Not started — picking this up in September 2026. To do first: read
GLMMHDTurb's REPORT.md/BENCHMARKS.md, decide whether to fork/vendor that repo
here, and get up to speed on MHD + GPU computing (Julia, KernelAbstractions.jl).

## Background reading

- Wang & Abel 2010; Wang, Kaehler & Abel 2012 (Dedner MHD on GPUs)
- Recent Pakmor & Springel papers on related MHD schemes
- Recent Teyssier preprint (Dedner vs. CT comparison)
