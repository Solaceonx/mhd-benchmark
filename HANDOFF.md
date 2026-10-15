# Handoff

Written 2026-10-11 (late evening, Pacific) so the work can go on from another computer or with another assistant.
Read this first, then `README.md` and `docs/` (codes, tests, results, issues found, Sherlock, notebook, data, status).

## 1. Who and how

- **lyxia** (Luke Xia) runs the project with Tom Abel (KIPAC); Eric Moseley is a collaborator.
- **Only lyxia can reach Sherlock** (password + Duo). The assistant writes and tests everything locally, then hands
  over exact commands, one per code block:
  - *sending or fetching files:* one Mac command, `tar … | ssh … 'cd /scratch/users/lyxia/mhd-study && …'`
    (patterns in `docs/sherlock.md` and in each bundle's README);
  - *checks:* plain "on Sherlock" commands, usually `bash -l …/status.sh`.
- lyxia pastes the output back. Diagnose from that; do not ask them to investigate.
- lyxia is short on assistant usage: short replies, exact commands, no long explanations unless asked.

## 2. Rules that must hold

- **Tom Abel's solver source is never edited and is not in this repository** (public, no licence).
  `bash scripts/fetch_solver.sh` restores it. To run his kernels differently, call them from our driver
  (`sherlock_sweep/src/run_suite.jl`), as `plm_unlisted!` does.
- **The website pages are unlisted.** They live in the `mhd/` folder of the website repository
  (`Solaceonx/solaceonx.github.io`). Nothing on the home page or research page may link to them; every page carries
  `noindex`. Do not write their address into this repository's documents.
- **Website commits:** author `Luke Xia <lyxia@mac.lan>` (`git -c user.name=… -c user.email=…`); a bot commits there
  every two hours, so `git pull --rebase` before pushing.
- **The notes page** (`progress/notes.html` → `mhd/notes.html`) is lyxia's private notebook: encrypted in the
  browser, synced to the `notes` branch of the website repository with a token only lyxia holds. Never read the
  notes, never record the password or a token. Test the page with requests to `api.github.com` stubbed, then clear
  the test browser's storage.
- **On Sherlock:** never overwrite a script that may be running (extract to a side directory, then `mv`); never
  resubmit a job that is still in `squeue`; try builds on a login or dev node before queueing for an H100.
  The login banner mentions `/etc/agents/AGENTS.md`; it has not been read or acted on, and that is lyxia's call.
- **Other papers' figures:** only CC BY ones may be copied (Tomida et al. 2026). `paper_figures/` holds figure PDFs
  and nothing else.
- **lyxia's Mac:** keep compute light (rendering all test movies takes about 35 minutes: run it in the background and
  say so); the project may use about 100 GB; raw movie frames are deleted after rendering.
- **Wording:** Tom's kernels are labelled "Abel et al." on pages and movies, not "advisor".
- **Look:** STIX text, one colour per code family, cmasher or `mako` colour maps (`scripts/plot_style.py`). Fields
  to type in must look like boxes; lists should be compact. Shared analysis plots have no gridlines.

## 3. What is where

| | In this repository | Only on lyxia's Mac | Only on Sherlock |
|---|---|---|---|
| Code | every bundle, script and page | Tom's source (`GLMMHDTurb/`, `*/vendor/*.jl`), `external/` | built codes in `$MHD_CODES` |
| Data | `turb_results/`, `progress/sweep_status/`, `progress/setups.json` | `sweep_results/` (3.3 GB), `*/returns/` (50 GB of archives), `scripts/out/` (4 GB) | `$SCRATCH/mhd-results-*` (2.5 TB; purged 90 days after last change, from 2026-12-29) |
| Media | `progress/media/` except the three folders at right | `progress/media/tests/` (test movies; also on the website), `media/movies/`, `media/gpu/` | |

**Publishing the site works only on the Mac.** `scripts/publish_site.py CLONE` deletes `CLONE/mhd` and rebuilds it
from the local `progress/` and `scripts/out/`. On a machine without those media it would drop the movies. Elsewhere,
either edit files inside the website clone's `mhd/` directly, or first copy `CLONE/mhd/media/*` into
`progress/media/` and `CLONE/mhd/files/scripts/out` into `scripts/out`.

To publish (on the Mac), never piping the publisher's output, and stopping if it exits non-zero:

```bash
python3 scripts/publish_site.py /path/to/site_clone
```

then, in the clone: `git add mhd`, commit as above, `git pull --rebase`, `git push origin HEAD:main`.

Tools: figure and page scripts run with the conda `python3` (numpy, matplotlib, seaborn, cmasher); movies with
`.venv-movie/bin/python` (`scripts/requirements-movie.txt`); Julia 1.12 with `--project=sherlock_sweep/env`.

## 4. In flight (as of this note)

**Sweep page campaign coverage added 2026-10-14:** `build_sweep_plan.py` now derives a separate driven-hydro
summary from `turb_results/*/run.toml`: 48 primary records (12 each at Mach 3, 10, 30, 50), plus 12 forcing checks.
It links high-Mach results, spectra and existing dense Euler movies without folding them into the standard MHD
matrix or planned magnetized-driving stage. Counts describe reported completion, not physical validity.
The first accounting fetch ended after printing a fresh remote export, without transferring it to the Mac.
The fetch script now reports query stages and returns a successful sweep export even if a later query fails;
incomplete accounting never replaces an older report. Local fixture checks cover complete and partial transfers.
lyxia has been asked to rerun it and paste the final output; fresh accounting has still not arrived locally.
The retry failed at `sacct` and the EXIT fallback wrote into the redirected accounting file instead of SSH.
The archive now uses a dedicated stdout descriptor; query wrappers run in an isolated subshell, with both stdout
and stderr diagnostics printed on failure. Seven local checks pass, including execution of the remote shell body
against a wrapper that calls `exit`. The actual reason for Sherlock's sacct refusal remains unknown pending a retry.

**2026-10-14 accounting / movie-comparison follow-up:** `scripts/fetch_sweep_accounting.sh` is ready for lyxia to
run on the Mac (read-only Slurm queries, no submission). It fetches fresh sweep status, allocation accounting since
September 1, the sweep ledger, and fairshare configuration into `progress/sweep_status/`.
As of preparing this update the new report has not arrived; the last saved CPU-lane total is 93,857 core-hours
from 2026-10-06, not current account-wide usage. Never convert `sshare RawUsage` directly into lifetime core-hours.
Once the fetch finishes, run `.venv-movie/bin/python scripts/sweep_accounting.py`, then
`.venv-movie/bin/python scripts/build_sweep_plan.py` and publish from the Mac. The new page distinguishes actual
allocation accounting, GPU host CPUs, and plan estimates; CSV/JSON downloads appear when accounting exists.

`scripts/submit_turb_compare_movies.sh` is ready for the user to run on the Mac: it uploads an isolated bundle and
submits two GLM/LLF GPU runs plus four corrected RAMSES hydro runs (HLL/HLLC, Mach 30/50, 256³, seed 1, 201 frames).
The CPU array uses 64 cores, concurrency one; no such jobs have been submitted by the assistant.
Results: `$SCRATCH/mhd-results-turb-compare-movie`. RAMSES needs ~250 GB temporary scratch for the active run;
the old four RAMSES evolutions used ~320 core-hours total, before denser output/conversion overhead.
Fetch with `scripts/fetch_turb_compare_movies.sh`; render with
`.venv-movie/bin/python scripts/make_turb_movie.py --compare --mach 30 50`; rebuild the high-Mach pages.
The renderer reuses existing Euler movie inputs and creates a synchronized 2 × 2 comparison with one common colour
scale. Nonpositive RAMSES densities are kept raw, floored only for display and counted in provenance.
See `sherlock_turb/README.md`, "More movies: matched comparisons". Local accounting and native-format reader checks
pass; the new RAMSES density-only movie runner has not yet run on Sherlock.

1. **Courant-number and single-precision checks** (`sherlock_checks/`): job `mhd-checks` 47440155 was still waiting
   for an H100. On Sherlock: `bash -l /scratch/users/lyxia/mhd-study/sherlock_checks/status.sh` prints the trial's
   verdict ("control passes" must appear for both grids) and a table beside the sweep's Courant 0.25 runs.
   Then: read the table, write the result into `docs/results-sweep.md` and `docs/issues-found.md` (does
   `glm_hll32` converge and survive the blasts and Mach 9? how do errors and stops change at 0.4 and 0.7?).
   The kernels have never run on a GPU in these variants; only compiled.
2. **"advisor" → "Abel et al.": done on the site.** The test movies were re-rendered with the new label and
   published on 2026-10-11. Still saying "advisor": the local-only GPU-suite page's own movies
   (`progress/media/movies/`; redo them with `MOVIE_RENDER_ARGS="--fields density,magnetic,vorticity"` and
   `scripts/render_movies_from_archives.sh`, then `scripts/render_preview_movies.sh`, about 35 minutes), and the old
   bundles' own documents and code comments (`sherlock_mhd/` and its siblings), left as they ran.
3. **Turbulence movies completed 2026-10-14.** The four dense H100 runs (Mach 3, 10, 30, 50; Euler/HLL;
   256³, seed 1) completed on 2026-10-12; lyxia fetched their 804 density slices. `scripts/make_turb_movie.py`
   checks the inputs and renders individual 1440 × 1600 clips plus a 3072 × 896 synchronized compilation,
   all at 20 fps with 201 saved snapshots over 10T and endpoint holds. PNG posters and JSON provenance accompany
   the MP4s in `progress/media/tests/`; these assets are local-only here and tracked in the website repository.
   `build_highmach_results.py --page-only` embeds individual clips; `build_turbulence.py --page-only` embeds the
   compilation. Commands to fetch and regenerate are in `sherlock_turb/README.md`, section "Movie".
4. **GEM movie with four codes.** Fetch (Mac):
   ```bash
   ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-results-sweep && find gem-gpu-N128-s42-all/data/glm_r1 gem-athenak-best-run-N128-s42/data gem-ramses-best-run-N128-s42/data -name "state_*.bin" | tar -cf - -T -' | tar -xf - -C ~/Projects/research/abel_mhd/sweep_results
   ```
   then `.venv-movie/bin/python scripts/render_sweep_movie.py gem --n 128 --labels glm athenak_best ramses_best arepo_moving`,
   `python3 scripts/build_test_pages.py`, publish. Until then the GEM page shows AREPO only.
5. **Two 3D blasts have no movie** (`blast_3d_b0`, `blast_3d_b100`): only six slices per run exist. A movie needs
   new runs with dense output; offered to lyxia, not built. The sweep driver can do it through a config with more
   `frames` and `slice_frames = "all"`, run like the checks.
6. **MHD-ALE:** 45 of 62 runs done; tests with a mean field stop at 256² and 512² (16 ranks). Whether the grid or
   the rank count triggers it: `bash -l /scratch/users/lyxia/mhd-study/sherlock_mhdale/rank_check_status.sh`
   (run `rank_check.sh` first if it reports nothing).

## 5. Next, in the order I would take it

1. Read the checks' table (item 1): it can change the headline numbers of the paper.
2. Finish items 2 to 4 (labels, turbulence movie, GEM).
3. Tell the RAMSES and RAMSES II maintainers about the driving fault and the negative densities; give Tom the seven
   proposals for his kernels (`docs/issues-found.md`). A draft message was given to lyxia; whether it was sent is
   not known.
4. Paper figures: nine drafts in `paper_figures/` were paused before polishing (`scripts/manuscript_figures.py`);
   there is no manuscript text.
5. Decide what to copy off Sherlock's scratch before 2026-12-29 (`docs/data.md`).
6. lyxia is going through the codes and tests one at a time for their notes: they ask "tell me about X" and get a
   short structured rundown (what it is, set-up, what it tests, how we ran it, what we found). Done so far:
   RAMSES, Orszag–Tang (with `docs/figures/orszag_tang_setup.png`), the planar blast with Bx = 50/√4π.

## 6. How the pieces are rebuilt

**GPU launch-tuning pilot added 2026-10-14:** `sherlock_kernel_tuning/` compares thread counts and supported tile
sizes in the unmodified `integrator_tiled!` and `integrator_plm!`. Each solver/precision setup has its own control;
changed outputs are rejected before timing. Local configuration/source checks pass; no GPU run has been submitted
and no speedup is established. The submitter requests an H100 and the job gates larger grids on a 48³ smoke trial.
See its README. This pilot times integrators only; it is not a replacement for the accuracy/robustness suite.
The user deferred this optimization work; do not pursue or submit it unless they return to it.

**Focused Mach 30/50 pages and Mach 9 audit added 2026-10-14:**
`.venv-movie/bin/python scripts/build_highmach_results.py` builds `test-driven-mach-30.html` and
`test-driven-mach-50.html`, with four focused plot pairs, numeric summaries, and their own shared diagnostics and
downloads. Run it after `scripts/driven_test_diagnostics.py`; `--page-only` reuses focused plots. Both are linked
from Tests and Turbulence and published as test pages. `scripts/build_test_pages.py` preserves these extra pages.
The Mach 9 page now includes `scripts/mach9_audit.py`'s saved-output audit: 43 finite complete, 37 stopped,
8 complete-but-nonfinite, 4 skipped (first repeat per method/grid/seed). The result tables flag nonfinite saved
histories even if the code reports completion. See `docs/results-sweep.md`; 3D root causes are not all isolated.

**Shared test analysis added 2026-10-13:** all 25 test pages plus the driven-turbulence page have eight quantitative
diagnostic sections, using real fetched records (257 PNG/SVG figure pairs after the comparison-layout update). Builders:
`scripts/test_diagnostics.py` and `scripts/driven_test_diagnostics.py`, followed by the page builders below.
See `docs/test-analysis.md`. Missing reference snapshots are marked explicitly, not filled with synthetic values;
The missing inputs were fetched on the Mac on 2026-10-13: planar 128² histories, other-grid endpoints,
64³ full-volume histories, restricted AthenaK references, environment records, and 61 MHD-ALE run records
(45 complete and finite, 16 stopped). The current-sheet N512 MHD-ALE record remains absent; missing final
states of stopped runs are not download failures. Larger 3D error histories still require additional files.
`bash scripts/fetch_test_analysis.sh` downloads existing missing inputs without submitting jobs.
The common accuracy figure is L1/L2/Linf **versus physical time**, not snapshot bars. Error convergence and cost
use the same endpoint errors. Do not treat an independently forced turbulence realization as a pointwise truth.
All science curves now share axes across Abel et al. and external codes, with two legend blocks and page
selectors that emphasize an Abel kernel and/or external family. Runtime bars combine and sort all codes;
accuracy/cost keeps GPU/CPU resource units separate with common accuracy scales. Downloads show every curve.
The selector script is `scripts/test_analysis_view.js`, embedded by the page builder; it loads vector plots
only when emphasis is requested, retaining PNG fallback. No gridlines. Rebuild both diagnostic builders after
changing the shared figure layout.

| Thing | Command |
|---|---|
| Turbulence page and figures | `python3 scripts/build_turbulence.py` (`--page-only` skips the figures) |
| Test pages | `python3 scripts/build_test_pages.py` |
| Annotated solver page, equations and side glossary | `.venv-movie/bin/python scripts/build_solver_walkthrough.py` (pinned source required; checksum checked) |
| Set-up equations (notes page and test pages) | `python3 scripts/build_notes_setups.py` (edit `SETUPS` in it; mathtext only: words stay outside the dollar signs) |
| Sweep and Papers pages | `python3 scripts/build_sweep_plan.py`, `python3 scripts/build_paper_figures.py` |
| Check GPU code without a GPU | `julia --project=sherlock_sweep/env sherlock_checks/test_gpu_compile.jl` (and the one in `sherlock_turb/`) |
| Local preview | `python3 scripts/serve_progress.py` (port 8765) |

Notes page internals: one note per id (`code/…`, `test/…`, `turb/…`, `other/…`, `term/<slug>` for glossary terms);
renaming a glossary term changes its id and orphans its text. The page holds only a salt and a check value for
the password; changing the password means new values in the page and re-encrypting.

## 7. Mistakes worth not repeating

- Piping the publisher through `cut` hid a failed link check and the commit went ahead. Check its exit code.
- A string replacement that ended in a `#` comment swallowed the rest of a one-line statement.
- A first colour map for |B| (`cmr.voltage`) washed out on a log scale: look at a contact sheet of last frames
  before publishing new maps.
- Unlocking the notes page locally with the real password pulls lyxia's real notes: stub the network first.
- On Sherlock, `set -e` plus `diff` (exit 1 on a difference) ended a script silently. Rehearse job scripts against
  stand-ins for Slurm before sending them.
- Numbers quoted from memory were wrong more than once. Compute them from the records, or read the config.
