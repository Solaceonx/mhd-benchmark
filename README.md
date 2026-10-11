# MHD benchmark

One set of magnetohydrodynamics tests, run with every code we could get onto the same grids and measured with the
same diagnostics. The question behind it: can divergence cleaning (Dedner/GLM) on a GPU stand in for constrained
transport (CT), and at what cost in accuracy? A rotation project of Luke Xia with Tom Abel (KIPAC, Stanford), with
Eric Moseley; started July 2026, most runs from September and October 2026 on Stanford's Sherlock cluster.

## What was run

| Code | Divergence control | Where | Runs |
|---|---|---|---|
| Tom Abel's GLM kernels (Julia, five variants) | Dedner cleaning | one H100 | every test |
| Tom Abel's CT prototype | constrained transport | one H100 | every test |
| AthenaK | constrained transport | one H100 | every test, two set-ups |
| RAMSES | constrained transport | CPU | every test, two set-ups |
| Athena++ | constrained transport | CPU | every test, two set-ups |
| AREPO | Powell source terms | CPU | 17 planar tests, static and moving mesh |
| RAMSES II | constrained transport | H100 | speed test only |
| MHD-ALE | exact by construction | CPU | 13 planar tests; 45 of 62 runs complete |

26 tests, grids from 32 to 512 cells a side (references to 4096²), 1,261 runs in the main sweep plus 164 with AREPO,
and a pilot of driven turbulence at Mach 3 to 50 (48 runs so far, plus 4 of a cross-check). Details: [codes](docs/codes.md), [tests](docs/tests.md).

## What came out

- **Speed.** The GLM kernels reach 2.0 × 10⁹ cell updates per second on one H100 (3.1 to 3.4 × 10⁹ on grids divisible
  by 6), against 0.2 to 0.4 × 10⁹ for AthenaK and 0.9 × 10⁹ for RAMSES II's constrained transport on the same card.
  [Sweep results](docs/results-sweep.md).
- **Accuracy.** The single-precision GLM kernels are second order and within about 25 % of same-order CT codes on
  shock tubes, Kelvin–Helmholtz, blasts and reconnection. The half-precision kernels stop converging past 64 to 128
  cells.
- **Leakage.** GLM lets magnetic field leak into a field-free vortex, 8 to 13 times the reference at 512², and it
  levels off instead of converging away. CT codes do not. This reproduces Tomida et al. 2026.
- **Robustness.** The simplest GLM kernel finishes 139 of 140 runs, including Mach 9 turbulence where AthenaK's and
  RAMSES's sharper set-ups stall.
- **Turbulence.** With one definition of the driving, Tom's Euler kernel and RAMSES agree to 1 % in Mach number.
  [Turbulence results](docs/results-turbulence.md).
- **Faults found on the way**, in RAMSES's turbulence driving, in the CT prototype, in the half-precision kernels and
  elsewhere: [issues found](docs/issues-found.md).

## Where things are

| Path | What |
|---|---|
| `docs/` | The write-up: [codes](docs/codes.md), [tests](docs/tests.md), [sweep results](docs/results-sweep.md), [turbulence](docs/results-turbulence.md), [issues found](docs/issues-found.md), [running on Sherlock](docs/sherlock.md), [notebook and figures](docs/notebook.md), [data](docs/data.md), [open items](docs/status.md) |
| `SWEEP.md` | Plan of the comprehensive sweep, with the later additions |
| `sherlock_sweep/` | The sweep as it ran on Sherlock: tests, adapters for each code, the job coordinator |
| `sherlock_turb/` | Driven turbulence: one driver for Tom's kernels and RAMSES |
| `sherlock_checks/` | Tom's kernels at other Courant numbers and with single-precision tiles |
| `sherlock_arepo/`, `sherlock_ramses2/`, `sherlock_mhdale/` | Add-on codes |
| `sherlock_mhd/`, `sherlock_rerun/`, `sherlock_movies/`, `sherlock_gem/`, `marlowe_mhd/` | The first campaign (21 tests) and its follow-ups |
| `scripts/` | Analysis, figures, the notebook builders, the first CPU studies on a laptop |
| `scripts/solver_checks/` | Small CPU runs that trace the sweep's findings to places in the solver source |
| `progress/` | The notebook: nine pages of figures and movies (open `progress/index.html`) |
| `paper_figures/` | Draft figures for our paper, one PDF each |
| `turb_results/` | The measured series, spectra and slices of the Mach 3 and 10 turbulence runs |

## What is not here

Tom Abel's solver source is public at [yipihey/GLMMHDTurb](https://github.com/yipihey/GLMMHDTurb) but has no licence,
so it is not copied into this repository. `bash scripts/fetch_solver.sh` puts it where the bundles expect it, at the
commit every run used, and checks it against the recorded checksums. The other codes are fetched by the build
scripts at pinned commits. The raw results (about 2.5 TB on Sherlock's scratch, 55 GB of archives on a laptop) are not
here either: [data](docs/data.md).
