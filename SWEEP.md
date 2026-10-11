# Comprehensive sweep: plan (2026-09-30)

**Status (2026-10-01, evening):** stage 1 has been running on Sherlock since 2026-09-30 (`sherlock_sweep/`, instructions
`sherlock_sweep/SWEEP_RUN.md`; checks and the bugs found on the way in `sherlock_sweep/VALIDATION.md`). Patches 1–9 are
applied there. At the status export of 22:20 UTC, of 1,670 method runs 1,497 were finished, 140 stopped by their solver
(results), 29 not finished (25 of them 3D CPU runs), 4 taken out. Which runs exist, the grid, costs and schedule: the
progress site's Sweep page (`progress/sweep-plan.html`, `scripts/build_sweep_plan.py`). The paper figures (item 8
below) are built: see "Paper figures". Stages 2–4 still need code.

Goal: every test at every resolution with every code, including faithful reproductions of the
comparison papers (Tomida et al. 2026, Kritsuk et al. 2011, Pakmor et al. 2026, GEM / Birn et al. 2001,
Gardiner & Stone 2005/2008), so that each paper's main result is reproduced *and* extended to more
methods, resolutions and tests. Decisions (lyxia, 2026-09-30): Sherlock owner nodes; build turbulence
driving, explicit resistivity and multi-GPU 1024³ references; all five GLM kernels; CPU codes at full scale.

## Codes (every test)

| Code | Variants | Runs on |
|---|---|---|
| GLM (T. Abel) | original tiled LLF, production f16/HLL, HLLD, PPM, constant c_h | H100 |
| CT prototype (T. Abel) | as vendored | H100 |
| AthenaK | matched (PLM/HLLE/RK2), best (PPM/HLLD/RK3) | H100; multi-GPU for 1024³ |
| RAMSES | matched (HLL), best (HLLD) | CPU (needs an NDIM=2 build for 2D tests) |
| Athena++ | matched, best | CPU |

## Resolutions

| Tests | Ladder | Truth |
|---|---|---|
| 2D (waves, shock tubes, Orszag–Tang, rotor, field loops, blasts, KH ×3, current sheet, GEM) | 32² … 512² (KH from 64²; Tomida used h = 1/128, 1/256) | exact, or AthenaK 4096² (true 2D; Athena++ 2048² check) |
| 3D (Orszag–Tang 3D, 3D blasts, 3D field loop, Mach 2 decay) | 32³ … 512³ | AthenaK 512³ in stage 1 (for ≤ 256³); 1024³ multi-GPU in stage 4 |
| Kritsuk decay (Mach 9, B0 = 0.3, driven start) | 128³, 256³, 512³ | AthenaK 1024³ |
| Pakmor dynamo (driven, Mach 0.2, to t = 100) | 32³ … 256³ (GPU codes also 512³) | convergence across codes |
| GEM ideal and resistive (η = 0.005 d_i v_A) | 64² … 512² | AthenaK 4096²; resistive converges |
| Throughput | 96³ … 512³ | timing only |

Timing repeats: 5 up to 128, 3 at 256, 1 at 512 and above (timing noise is small at large N;
accuracy-per-cost uses the repeated grids).

## What has to be built

1. **GPU driver, checkpoint/restart.** Owner nodes preempt; runs over ~1 h need to resume, not
   restart. Checkpoint the device state every ~15 min; a requeued job resumes in the same directory;
   a resumed repeat is marked `timing_valid = false`.
2. **GPU driver, reduced outputs.** At 512 keep: diagnostics at every output, power spectra
   (kinetic, solenoidal/compressive, magnetic, density) computed on the GPU, midplane slices, and
   states block-averaged to ≤256 for errors. Full states only at small N. Same for AthenaK
   (restricted outputs exist; add spectra). Methods split into separate array tasks at large N.
3. **Explicit resistivity.** GLM: operator-split η∇²B + ohmic heating. CT: resistive EMF ηJ on edges
   (keeps ∇·B = 0). AthenaK `ohmic_resistivity`, RAMSES `eta_mag`, Athena++ `eta_ohm`. Diffusive
   timestep limit in the driver.
4. **Turbulence driving, identical in every code.** Pakmor et al. (2026) spec: solenoidal
   acceleration from integer modes 1 ≤ |k|L/2π ≤ 3 with parabolic P(k), peak P0 = 2e-4, six OU
   processes per mode (σ² = 1, τ = 1) updated every Δt = 0.005, one seeded realization shared by all
   codes via a mode table; isothermal by pressure reset (cs = 1, γ = 1.01); ρ = p = 1, Bz = 1e-4.
   GLM/CT: force applied in the driver after each step. AthenaK/Athena++: user source term reading the
   table. RAMSES: the same table through its analytic-acceleration hook.
5. **Kritsuk start state.** Drive (k/k_min ≤ 2) from uniform ρ, p and B0 = 0.3 ẑ with random
   large-scale velocity to Ms ≈ 9–10 for 3 t_d = 0.15 with AthenaK at 1024³; restrict the final state
   (face fields conservatively) to 128–512; every code starts its decay from that file (as Kritsuk et
   al. did from their 1000³ STAGGER snapshot). Needs "state file" initial conditions in AthenaK,
   Athena++ and the GPU driver (RAMSES already reads Julia's state).
6. **Multi-GPU AthenaK** (MPI + CUDA, one 8-GPU node) for 1024³ references and the Kritsuk start.
7. **RAMSES NDIM=2 build** so 2D tests are not run as cubes on CPUs.
8. **Analysis:** reproduce each paper's figures (Tomida leakage and dt-jump; Kritsuk spectra,
   bandwidth, effective Re; Pakmor growth rate and saturation; GEM flux; G&S loops and sheets) with
   every code added.

## Cost (calibrated estimator, `sherlock_sweep/codes/estimate.py`)

Step counts and H100 throughputs are medians of the completed Sherlock runs; RAMSES 3D per-core speed is measured on
Sherlock; RAMSES 2D and Athena++ are Mac timings scaled by RAMSES's Sherlock/Mac ratio.

| Stage | GPU-h | core-h |
|---|---|---|
| 1 · standard sweep (in the bundle; 1,265 runs in 227 Slurm jobs) | ~220 [188–307] | ~68k [47k–129k]; 55k of it 3D RAMSES/Athena++ at 512³ (`--cpu-max-n3d 256` drops it) |
| 2 · resistivity (GEM, current sheet) | ~24 | ~0.3k |
| 3 · driving (Pakmor dynamo; Kritsuk start at 1024³ + decay) | ~320 | ~45k |
| 4 · 1024³ AthenaK references (8 GPUs) | ~134 | — |

Stage 1 storage: ~2.6 TB kept on scratch; ~55 GB returned to the Mac (files ≤ 50 MB; ~2 GB of summaries alone).
Wall clock for stage 1: gates ~5 h, GPU work under a day, CPU work 2–6 days (the 20–33 h 3D RAMSES runs at 512³ set
the end).

## Order

1. Infrastructure (items 1, 2, 7) → submit the standard sweep (existing tests, new ladders, all codes).
2. Resistivity (3) → resistive GEM and current sheet.
3. Driving (4) → Pakmor dynamo; then the Kritsuk start (5, 6) and decay.
4. References at 1024³ / 4096² and the paper-reproduction analysis (8).

## Paper figures

`progress/paper-figures.html` ("Papers" on the progress site) redraws the figures of the comparison papers from the
sweep's own output, one figure of ours per figure of a paper, with every code in place of the paper's.

1. Copy the files the figures read from Sherlock (about 1.3 GB: every part's `run.toml` and CSVs, and the binary frames
   the images show; `scripts/sweep_data.py`, table `WANTED`). One command on the Mac, one password and Duo prompt; it
   asks only for what is not here yet, so running it again later picks up runs that finished since:

       cd /Users/lyxia/Projects/research/abel_mhd && python3 scripts/sweep_fetch.py list | ssh lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-results-sweep && while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done | tar -cf - -T -' | python3 scripts/sweep_fetch.py receive

   The files land in `sweep_results/` with the paths they have on Sherlock. `python3 scripts/sweep_fetch.py check`
   says what is here. The same stream brings the status export (`sweep_state/runs.json` on Sherlock) when it is newer
   than `progress/sweep_status/export_latest.json`; to refresh it first, on Sherlock:
   `cd /scratch/users/lyxia/mhd-study/sherlock_sweep && module load python/3.12.1 && python3 tools/sweep.py runs --json sweep_state/runs.json`.
   Then `python3 scripts/build_sweep_plan.py` updates the Sweep page.
2. `python3 scripts/build_paper_figures.py` draws every figure (`scripts/paper_figures.py`; one module per paper:
   `paper_tomida.py`, `paper_kritsuk.py`, `paper_gem.py`, `paper_stone.py`) into `progress/media/papers/` and writes the
   page. `--page-only` rewrites the page from the figures already drawn.

What is covered (the page lists every figure of every paper with its state): Tomida et al. 2026 Figs. 1, 2, 4–8 (same
tests on our periodic double layer; leakage is read inside the second, field-free vortex, because by t = 3 the flow has
carried field everywhere else, and summarised as one number per run against resolution); Kritsuk et al. 2011 Figs. 1–5
and Table 1 (same measurements, on our smooth-start Mach 9 decay rather
than their driven state); Birn et al. 2001 Fig. 1 (ideal MHD, so numerical reconnection); Gardiner & Stone 2005 Figs.
2/3/6–13 and 2008 Figs. 1, 2, 7, 8. Not possible yet: Pakmor et al. 2026 (needs the driven runs of stage 3), Tomida's
c_h scan and applications, the 3D wave tests of Gardiner & Stone 2008. Ugalino et al. 2026 (galactic dynamo with
RAMSES, ART, AREPO, SWIFT) has no idealized tests; its Fig. 5 is drawn for our Mach 9 decay as an analogue.

The papers' own figures: Tomida et al. 2026 is CC BY 4.0, so its figures are stored under
`progress/media/papers/original/` and shown beside ours. The other papers carry no reuse licence; the page links to
each figure in the paper, draws the numbers a paper prints (Gardiner & Stone's fitted decay curves, Kritsuk's Table 1
range) on our axes, and shows any copy saved by hand as `progress/media/papers/original/<figure id>.png` beside ours.

Figures carry a computed "In these runs" line (the `finding` of each entry). A run whose state stops being finite while
its code runs on (Athena++ PPM/HLLD/RK3 at Mach 9: recorded as complete on Sherlock) is cut at its last finite output
and counted as stopped on both pages (`sweep_data.Sweep.finite_rows`, `sweep_status.mark_nonfinite`).

## Colour maps of images

`scripts/plot_style.py` holds one table, `FIELD_MAPS`, with the colour map of each kind of quantity: gas (density,
pressure) `mako`, field strength `cmr.freeze`, signed quantities `cmr.iceburn` (black at zero), a second signed quantity
`cmr.wildfire`. All are dark where the value is low. The Papers page, the manuscript figures and the Turbulence page
take their maps from it (`field_map("density")` and so on); changing a name there changes every figure at its next
build. Needs seaborn and cmasher (in the conda Python the figure scripts run with; `scripts/requirements-movie.txt`
lists them for `.venv-movie`).

Movies use the same look as the figures: `apply_movie_style()` in `scripts/plot_style.py` (white ground, the notebook's
inks and fonts, the colour of the code for a run, grey for the reference it is compared with), and the code colours
themselves are one table there (`FAMILY`; `progress/style.css` uses the same green, orange and grey for the three
columns of the first pages). Every page is rendered this way:

- 01 Orszag-Tang: `.venv-movie/bin/python scripts/build_progress_site.py --ot-only --movies` (stills, plots, 24 movies;
  `--ot-only` keeps the GPU-suite data, so no archives are needed);
- 02, 03 the decay pages: `scripts/make_decay_movie.py`, `scripts/make_spectral_gap_movie.py`;
- 04, 05 the wave checks: `scripts/render_validation_movies.py`;
- 06 GPU suite: `bash scripts/render_movies_from_archives.sh` (the eleven dense movies) and
  `bash scripts/render_preview_movies.sh` (the ten short ones from the ordinary 128² returns), both with
  `MOVIE_RENDER_ARGS="--fields density,magnetic,vorticity"` to render the three fields the page shows; its stills
  come from `scripts/rebuild_gpu_page.sh`.

`.venv-movie` has everything these need (`scripts/requirements-movie.txt`).

## Figures for our paper

`paper_figures/` holds the figures meant for our own paper, drafts and final versions, one PDF each and nothing else
(no scripts, data or previews). `python3 scripts/manuscript_figures.py` redraws all of them from `sweep_results/`
(a few seconds); `--preview DIR` also writes PNG copies somewhere else for looking at them, and a figure's name as
argument draws only that one. Typography is the project's (`scripts/plot_style.py`: STIX text, Computer Modern
maths); widths are 7.1 in (both columns) with 7–8.5 pt text. Colours and line styles are those of the Papers page
(`scripts/paper_common.py`).

- `cost_vs_error.pdf`: error against GPU time on one H100 for the entropy wave, the oblique Alfvén wave and the field
  loop (errors and timings from `run.toml`; a pale symbol is a run that was interrupted and resumed, its parts added up).
- `speed.pdf`: cell-updates per second against grid size (`throughput_ot3d`; the f16 kernels are about 1.7 times
  faster on grids divisible by 6, drawn with filled symbols), and the time to finish 3D Orszag–Tang at 512³.
- `robustness.pdf`: every test, grid and code: finished, or how far the solver got (the timing run is left out, so the
  totals are of 133 runs per GPU kernel, not 140). AREPO has two columns, for the planar tests it ran.
- `convergence.pdf`: errors of the three wave tests against the grid; AREPO is in the row of the other codes.
- `arepo.pdf`: the field loop on AREPO's static mesh (the magnetic energy grows by orders of magnitude at every grid)
  against its moving mesh (constant); the timestep of the seven static-mesh runs AREPO ended itself; AREPO's own
  measure of the divergence of B against the grid, moving and static.

AREPO is read by `scripts/sweep_data.py` from the fetched `run.toml` files (labels `arepo_moving`, `arepo_static`; it
is not in the sweep's plan or status export, and not on the Sweep and Papers pages). A static-mesh run that finished
is not necessarily right: the field loop at 32² to 128² ran to the end with relative errors of B from 2.5 to 330.

The sweep ran the GLM kernels and the CT prototype at Courant number 0.25 in their own timestep rule, which divides by
the *sum* of the signal speeds of the three directions (`compute_dt` in `vendor/glmmhd_turb.jl`; the default there is
0.7). The other codes use 0.3 (RAMSES 0.5 and 0.8) against the fastest single direction. That is why the GPU kernels
take about three times as many steps; time-to-solution numbers are for this setting.

## Keeping track of the data

Sherlock purges scratch files 90 days after their last change (the first sweep outputs from about 2026-12-29), and
only lyxia can reach it. Two tools keep the copy here honest:

- `scripts/sweep_fetch.py` copies what the pages and figures read (about 1.3 GB), and any further files named in
  `progress/sweep_status/extra_wanted.txt`.
- `scripts/sweep_inventory.py` lists every file under the result directories on Sherlock (one ssh command, see its
  header), stores the listing under `progress/sweep_status/inventory_*.tsv.gz`, and reports by kind of file (records,
  diagnostics, states, slices, reference averages, restricted 3D states, logs, the codes' own outputs) how much is
  there, how much is here, whether the fetch list covers it and when the purge would first take it.
  `python3 scripts/sweep_inventory.py extra KIND ... [--max-mb N]` adds the rest of a kind to the fetch.

## RAMSES II (GPU)

`sherlock_ramses2/` (its README has the commands) times RAMSES II's CUDA Fortran backend, whose MHD is constrained
transport in double precision, on 3D Orszag–Tang to t = 0.05 at 64³–512³ on one H100, for `nsubgrid` = 1, 2 and both
precisions, and returns the energies at t = 0.05 for checking against the sweep's `throughput_ot3d` records. Speed
only so far; the full test matrix would need our initial conditions, a reader for its output format and a run adapter.

## AREPO

`sherlock_arepo/` (its README has the commands and the state of testing) runs the public AREPO release, with Powell's
scheme for div B, on the 17 planar tests at every grid, on a static Cartesian mesh (`arepo_static`) and on its moving
mesh (`arepo_moving`): 164 CPU runs. `arepo_run.py` plugs it into `sherlock_sweep/codes/external.py` without changing
the sweep bundle, so its run directories and records sit beside the other codes' under `mhd-results-sweep/`. The
plan in `sherlock_sweep/codes/plan.py` does not list these runs yet, so the Sweep page and the figures do not show
them until the local readers are extended. SWIFT and ART (the other codes of Ugalino et al. 2026) are not planned.

## MHD-ALE (Tom's MFEM-based code)

`sherlock_mhdale/` (its README has the details): github.com/ruijie-xi/MHD-ALE at commit `0e0564ed`, on the author's
MFEM fork; a moving-mesh finite-element method with div B = 0 by construction, MPI on CPUs. Read from its source: in
2D it has no v_z or B_z (4 of the 17 planar tests cannot run), and its remap supports boxes periodic in y only, so
`rotor`, the `blast_2d_*` and `brio_wu` can run like for like in boxes with walls, and the other 8 only without
remap unless the remap is extended (which would be our change). Our additions (test case from `abel_problems.hpp`,
sampling onto the suite's grid, output on the output times) do not touch the solver. First step: `check.sh` submits
a trial job (build, the author's Taylor-Green test, starting states against the suite's, a few steps per box).

## Driven turbulence (pilot)

`sherlock_turb/` (README there): hydrodynamic turbulence driven to a steady state, Mach 3 and 10 on 128³ and 256³, in
Tom Abel's Euler and MHD kernels and in RAMSES, with one definition of the driving. Results on Sherlock in
`$SCRATCH/mhd-results-turb`; the small files of every run are copied to `turb_results/` (command on the page).
`python3 scripts/build_turbulence.py` draws the figures (`scripts/turb_figures.py`) and writes
`progress/turbulence.html` ("Turbulence" on the progress site; local only, not published).

The pilot found a fault in RAMSES's driving module (also in RAMSES II): the Helmholtz projection takes the direction of
a wave vector from the absolute values of its components, so a "solenoidal" force is one fifth compressive. RAMSES
therefore ran twice, as released (`ramses-*`) and corrected (`ramsesfix-*`); only the corrected runs share their driving
with Tom's kernels. Evidence and the three-line correction: `sherlock_turb/README.md`.

## Checks of the solver kernels

`scripts/solver_checks/` runs the solver's own functions (loaded unedited from `sherlock_sweep/vendor/`) on small planar
problems on the CPU, to trace what the sweep found to places in the source: the CT prototype's halved slope, the
half-precision tile (lost convergence, overflow), HLLD's degenerate-case tolerance, c_h following 1/Δt, the Courant
number, and cell-centred CT against the leakage. Findings, numbers and commands: `scripts/solver_checks/README.md`.

## Ask the advisor

- Can his kernels run thin slabs (N×N×n_z)? 2D runs would become ~N/n_z times cheaper.
- c_h policy: from the CFL dt, or fixed; the dt-proportional c_h misbehaves on shortened steps.
