# Notebook and figures

`progress/` is a static site. Open `progress/index.html`, or serve the repository root with
`python3 scripts/serve_progress.py` (it answers byte ranges, which the synchronised movies need). Run
`bash scripts/fetch_solver.sh` first if a builder needs Tom's kernels. `progress/README.md` says how to refresh a
page or add a test.

## Pages

| | Page | Shows | Built by |
|---|---|---|---|
| 01 | Orszag–Tang | GLM, CT and an Athena++ reference side by side, 64² to 512², on a laptop CPU | `build_progress_site.py` |
| 02 | Hydro decay | Decaying hydrodynamic turbulence at 64³ | `make_decay_movie.py` (`DECAY.md`) |
| 03 | Prepared decay | A spectrum with a gap cut out, and how it refills | `run_spectral_gap.jl`, `make_spectral_gap_movie.py` (`SPECTRAL_GAP.md`) |
| 04 | Acoustic check | A sound wave against the exact answer | `run_acoustic_movie.jl`, `render_validation_movies.py` |
| 05 | Entropy wave | Convergence against the exact answer | `run_entropy_wave.jl`, `render_validation_movies.py` |
| 06 | GPU suite | The first Sherlock campaign: 21 tests, stills and movies | `build_gpu_suite.py`, `render_suite_movies.py`, `rebuild_gpu_page.sh` |
| 07 | Sweep | Every run of the sweep: finished, stopped by its solver, or missing | `build_sweep_plan.py` |
| 08 | Papers | 43 figures of the comparison papers redrawn from the sweep | `build_paper_figures.py`, `paper_*.py` |
| 09 | Turbulence | The driven-turbulence pilot | `build_turbulence.py`, `turb_figures.py` |
| 10 | Tests | One page a test problem: set-up, movie, completion and eight quantitative diagnostic sections | `test_diagnostics.py`, `build_test_pages.py` |

All builders are in `scripts/`. Page 06 needs the archives in `sherlock_mhd/returns/`, which are not in this
repository, and its stills and movies are not either ([data](data.md)); the other pages work from a clone.

The shared test diagnostics use fetched sweep records and available state/reference files. They have no
gridlines, separate code families/variants clearly, and explain unavailable comparisons. The driven pilot has
the same sections with its reference-dependent metrics explicitly unavailable. See [test analysis](test-analysis.md)
for rebuilding, metric definitions, data coverage, validation, and the missing-input fetch command.

`progress/notes.html` is a private notes page, one note per code and per test, encrypted in the browser with a
password; no page links to it.

## Movies of the test pages

`progress/media/tests/` (not in this repository; on the website). Three sources:

- the movie runs of the first campaign (241 outputs; `MOVIE_RENDER_ARGS="--fields density,magnetic,vorticity --family-maps
  --out progress/media/tests --manifest progress/test_movies.json" bash scripts/render_movies_from_archives.sh`) and short
  previews from its ordinary runs (`scripts/render_preview_movies.sh` with the same arguments);
- the sweep's own frames for a test those did not cover (`scripts/render_sweep_movie.py`, used for GEM);
- the driven-turbulence movie runs (`sherlock_turb/movie.sh`, then `scripts/make_turb_movie.py`).

On these pages the colour maps change with the family of tests (`plot_style.FAMILY_MAPS`, all from cmasher except
`mako`); within a page a quantity keeps its map.

## One look

`scripts/plot_style.py` holds the choices every figure and movie uses: STIX text with Computer Modern mathematics,
one colour per code family (CT prototype orange, GLM green, AthenaK blue, RAMSES purple, Athena++ magenta, AREPO
ochre), and one colour map per kind of field (density `mako`, magnetic `cmr.freeze`, signed quantities
`cmr.iceburn`, current `cmr.wildfire`). `SWEEP.md` has the reasoning under "Colour maps of images".

## Figures for the paper

`python3 scripts/manuscript_figures.py [NAME]` writes one PDF per figure to `paper_figures/`, which holds nothing
else: `cost_vs_error`, `speed`, `robustness`, `convergence`, `leakage`, `divergence`, `field_loop`, `turbulence`,
`arepo`. They are drafts.

## Other papers' figures

Only Tomida et al. 2026 is published under CC BY 4.0, so only its figures are stored
(`progress/media/papers/original/`) and shown next to ours, with credit. For the other papers the page links to the
figure in the paper and draws the numbers the paper prints on our axes.

## The first CPU studies

Before there was GPU time: `scripts/GLM_CT_CPU.md` (a 2D CPU version of the GLM and CT updates),
`scripts/ATHENA_REFERENCE.md` (Athena++ references and error plots), `scripts/MHD_SPECTRA.md` (spectra, divergence,
energy and cost from 64² to 512²), `scripts/CPU_BENCHMARKS.md` (what those timings can and cannot say),
`scripts/GPU_COMPARISON_PLAN.md`.
