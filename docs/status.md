# Open items

As of 2026-10-11.

## Waiting on results

- **Turbulence at Mach 30 and 50.** RAMSES ran; the measurement was fixed for negative densities and resubmitted;
  the GPU runs were queued for an H100. Results are not fetched, and the Turbulence page's layouts assume two Mach
  numbers.
- **Cross-check of the RAMSES projection.** Tom's Euler kernel driven with RAMSES's released projection
  (`gpu-hydro_ramsesproj`, 128³) was queued; not fetched.
- **RAMSES II speed test.** The build passed after the compiler workaround and the job was submitted; no result in
  hand.
- **MHD-ALE.** Tom's fork builds on Sherlock, its regression suite passes, and our 13 tests start from the right
  states. Whether the production runs finished is not recorded here.

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
