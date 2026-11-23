# MHD benchmark

One set of magnetohydrodynamics tests, run with every code we could get onto the same grids and measured with the
same diagnostics. The question behind it: can divergence cleaning (Dedner/GLM) on a GPU stand in for constrained
transport (CT), and at what cost in accuracy? A rotation project of Luke Xia with Tom Abel (KIPAC, Stanford), with
Eric Moseley; started July 2026, most runs from September to November 2026 on Stanford's Sherlock cluster.

To continue the work, on another computer or with another assistant, start with [HANDOFF.md](HANDOFF.md).

## What was run

Fifteen codes, 51 set-ups, about 5,900 runs, all from the same initial states and measured by one analysis pipeline.

| Code | Method | Divergence control | Where |
|---|---|---|---|
| Tom Abel's GLM kernels (Julia) | finite volume | Dedner cleaning | H100, single and half precision |
| Tom Abel's CT prototype | finite volume | constrained transport | H100 |
| AthenaK | finite volume | constrained transport | H100 |
| AthenaPK | finite volume | Dedner cleaning | H100 |
| Cholla | finite volume | constrained transport | H100 |
| gPLUTO | finite volume | constrained transport or cleaning | H100 |
| RAMSES II | finite volume | constrained transport, upwind CT | H100, double and single precision |
| Castro | finite volume | constrained transport | H100 |
| Nyx | finite volume (hydrodynamics only) | none | H100 |
| Athena++ | finite volume | constrained transport | CPU |
| RAMSES | finite volume | constrained transport | CPU |
| Enzo | finite volume | Dedner cleaning or constrained transport | CPU |
| AREPO | moving mesh | Powell source terms | CPU |
| GIZMO | meshless finite mass | Dedner cleaning and constrained gradients | CPU |
| Phantom | smoothed particles | cleaning | CPU |
| MHD-ALE (Tom Abel's fork) | moving-mesh finite elements | exact by construction | CPU |

Most grid codes ran two set-ups: a textbook second-order one and the one their authors recommend for accuracy. The
suite has 24 tests (17 planar, 7 in 3D) on grids of 32 to 512 cells a side, plus the Mach 9 decaying turbulence of
Kritsuk et al. (2011) for every code, driven turbulence at Mach 3 to 50, and Tom Abel's nested grids against uniform
ones. Details: [codes](docs/codes.md), [tests](docs/tests.md).

## What came out

- **The Riemann solver matters more than the reconstruction** wherever the flow is not smooth. Within one code, HLLD
  instead of a two-wave solver reaches the same error on reconnection 13 to 65 times sooner, more than any change of
  reconstruction.
- **GPUs win on time to accuracy, and single precision is often free.** The best GPU set-ups reach a standard accuracy
  tens to thousands of times sooner than a CPU node. The fastest, RAMSES II's upwind-CT HLLD scheme, loses nothing in
  single precision; high-order schemes hit a single-precision round-off floor.
- **Mach 9 decaying turbulence.** The extra compressive power that two-wave solvers show near the grid scale is missing
  vortical power. The recommended high-order set-ups of five codes fail this test; with its first-order flux correction
  switched on, AthenaPK's finishes and is the least diffusive of all.
- **Tom Abel's GLM kernels** are the fastest per cell update (about 2 × 10⁹ per second on one H100) and finish nearly
  every run, but divergence cleaning leaks field into a field-free layer where CT codes do not.
- **Faults found on the way**: RAMSES's turbulence driving (fixed in RAMSES II, open upstream as
  ramses-organisation/ramses#361), AthenaPK's density floor, MHD-ALE's projection tolerance, Castro's start-up check of
  the initial field, and several in the GLM kernels and the CT prototype:
  [issues found](docs/issues-found.md). Castro converges at first order on Alfven waves; the cause is not traced.
- **Being rerun:** Enzo's CT set-up, which ran with a non-default electric field (`MHD_CT_Method` 2) and was first
  order on Alfven waves; it is being rerun with the default.

## Where things are

| Path | What |
|---|---|
| `docs/` | The write-up: [codes](docs/codes.md), [tests](docs/tests.md), [sweep results](docs/results-sweep.md), [turbulence](docs/results-turbulence.md), [issues found](docs/issues-found.md), [running on Sherlock](docs/sherlock.md), [notebook and figures](docs/notebook.md), [data](docs/data.md), [open items](docs/status.md) |
| `SWEEP.md` | Plan of the comprehensive sweep, with the later additions |
| `sherlock_sweep/` | The sweep as it ran on Sherlock: tests, adapters for each code, the job coordinator |
| `sherlock_turb/` | Driven turbulence: one driver for Tom's kernels and RAMSES |
| `sherlock_checks/` | Tom's kernels at other Courant numbers and with single-precision tiles |
| `sherlock_newcodes/` | AthenaPK, Cholla, gPLUTO, Enzo, Phantom, Castro and Nyx: patches, runner, build and production jobs |
| `sherlock_arepo/`, `sherlock_gizmo/`, `sherlock_ramses2/`, `sherlock_miniramses/`, `sherlock_mhdale/` | The other add-on codes |
| `sherlock_order/`, `sherlock_fixes/`, `sherlock_nested/` | Reconstruction x Riemann solver study; Tom Abel's kernels with the fixes; nested grids |
| `sherlock_mhd/`, `sherlock_rerun/`, `sherlock_movies/`, `sherlock_gem/`, `marlowe_mhd/` | The first campaign (21 tests) and its follow-ups |
| `scripts/` | Analysis, figures, the notebook builders, the first CPU studies on a laptop |
| `scripts/solver_checks/` | Small CPU runs that trace the sweep's findings to places in the solver source |
| `progress/` | The notebook: nine pages of figures and movies (open `progress/index.html`) |
| `paper_figures/` | Draft figures for our paper, one PDF each |
| `turb_results/` | The measured series, spectra and slices of the Mach 3 and 10 turbulence runs |

## What is not here

Tom Abel's solver source is public at [yipihey/GLMMHDTurb](https://github.com/yipihey/GLMMHDTurb) but has no licence,
so it is not copied into this repository (nor are gPLUTO's and MHD-ALE's, for the same reason). `bash scripts/fetch_solver.sh` puts it where the bundles expect it, at the
commit every run used, and checks it against the recorded checksums. The other codes are fetched by the build
scripts at pinned commits. The raw results (about 2.5 TB on Sherlock's scratch, 55 GB of archives on a laptop) are not
here either: [data](docs/data.md).
