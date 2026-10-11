# Open items

As of 2026-10-11.

## Waiting on results

- **Courant-number and single-precision checks** (`sherlock_checks/`): built and checked without a GPU; to be
  submitted. They answer the first two items of "Not run" below.
- **Turbulence data to fetch.** The Mach 30 and 50 runs and the projection cross-check are finished on Sherlock
  ([results](results-turbulence.md)); the Turbulence page and its figures still show Mach 3 and 10 only.
- **MHD-ALE.** 45 of 62 runs complete; whether the stops on fine grids come from the grid or the rank count needs
  the two short runs of `rank_check.sh` ([issues found](issues-found.md#mhd-ale)).

## To report

- The RAMSES projection fault and the negative densities, to the RAMSES and RAMSES II maintainers.
- The seven proposals for Tom's kernels ([issues found](issues-found.md#toms-kernels)).
- Questions for the MHD-ALE author: the wave-speed line, recommended settings, permission to publish results (the
  code has no licence).

## Not run

- A Float32 run of the production kernel, to confirm that half precision is what stops its convergence.
- The kernels at their default Courant number, or at 0.4.
- A second pass with a pressure floor for the 70 runs stopped at the first non-positive pressure.
- Stages 2 to 4 of the sweep: explicit resistivity for GEM, a driven dynamo, 1024³ references on several GPUs.
- The full turbulence matrix: subsonic, compressive driving, 512³, a mean field, AthenaK and Athena++.

## To decide

- What to copy off Sherlock's scratch before the purge starts on 2026-12-29 ([data](data.md)).
- Which paper figures to finish: the drafts in `paper_figures/` were paused before polishing.
- The test lists of the AthenaK and GIZMO methods papers have not been compared with ours yet.
