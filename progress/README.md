# MHD Field Notes

Open `index.html` for the Orszag–Tang comparison. Every physics test has its own
HTML page with the same **GLM / CT / The Truth** columns:

- `index.html`: active MHD comparison, four resolutions, checkpoint controls,
  density / magnetic fields, matched GLM / CT / Athena movies, errors, spectra and energies.
- `hydro-decay.html`: initial near-isothermal 64³ decay and its movie.
- `isothermal-decay.html`: prepared hydro decay and the parked spectral-gap branch.
- `acoustic-wave.html`: small-amplitude sound-wave validation with numerical and linear-reference movies.
- `entropy-wave.html`: exact finite-amplitude magnetized entropy advection, convergence and three linked movies.
- `gpu-suite.html`: the multi-code Sherlock suite (`sherlock_mhd/`): GLM variants, CT, AthenaK,
  RAMSES, with AthenaK references and an Athena++ cross-check. A status matrix (one dot per code)
  for every test × grid; any two methods side by side against the reference; every method in
  one table; time histories; accuracy vs grid, accuracy vs GPU cost, and throughput.
  Hue encodes the code (validated for colour-vision deficiency); line style and marker the
  variant. CPU-measured runs never enter the GPU cost panel.

- `sweep-plan.html`: the comprehensive sweep (`sherlock_sweep/`): which runs exist (one mark per method, test and
  grid), the plan and its cost. Built by `scripts/build_sweep_plan.py` from the bundle's plan and the newest status
  under `sweep_status/`.
- `paper-figures.html`: the figures of the comparison papers redrawn from the sweep with every code, each marked as
  the same test, the same measurement on a different set-up, or added here; figures of Tomida et al. (2026, CC BY 4.0)
  are shown beside ours, the other papers are linked. Built by `scripts/build_paper_figures.py` from `sweep_results/`
  (copied from Sherlock by `scripts/sweep_fetch.py`; see `SWEEP.md`, "Paper figures").

The hydro pages explicitly distinguish hydro-only implementations from GLM/CT
magnetic evolution. The reference column distinguishes numerical references,
analytical approximations, controls and missing data.

`solver.html` is an annotated reading of the Abel et al. GLM-MHD source, with
clickable line explanations, a searchable side glossary, and the five benchmark
configurations. Rebuild with `python3 scripts/build_solver_walkthrough.py` after
restoring the pinned source using `scripts/fetch_solver.sh`. The builder verifies
the source checksum and extracts the excerpts without editing the solver.

## Open locally

The site uses plain HTML, CSS and JavaScript, with no dependencies or external
requests. It works as a local file. For consistent video playback and notes
storage, serve the **repository root** (media links also reference `scripts/out`):

```sh
.venv-movie/bin/python scripts/serve_progress.py
```

Then visit <http://127.0.0.1:8765/progress/>. The preview is local to this computer.
This server supports byte-range requests so native video seeking works reliably.
On narrow screens the three-column comparison scrolls horizontally rather than
changing the order of the methods. Click an image to enlarge it; Escape closes
the viewer. All three Orszag–Tang movies share one transport: native play, pause,
seeking and playback speed in any player control the group. The shared
grid and field controls change all movies together and reset playback. The saved
checkpoint slider controls still images only. Multi-video validation pages also
link native playback controls.

Notes save to this browser's local storage per test and column. They are not
written into run data or synchronized between browsers, file URLs or origins.

## Refresh results

After updating the analysis outputs, regenerate the static data and images:

```sh
.venv-movie/bin/python scripts/build_progress_site.py
```

To regenerate all 24 Orszag–Tang movies (four grids × two fields × three methods):

```sh
.venv-movie/bin/python scripts/build_progress_site.py --movies
```

This reads saved runs and never launches simulations. Orszag–Tang metrics come
from `scripts/out/spectral_study/analysis/`, state images from the corresponding
local runs, and reference fields from the Athena directory recorded by the
analysis. The current resolutions are 64, 128, 256 and 512, with four common
checkpoints. The older movie and SIMD animation remain linked as historical
artifacts. Generic-page summaries are curated from the experiment documents;
update those records when the corresponding experiments change.

The Athena movie prefers `scripts/out/athena_512_movie41/`: a separate 512² run with
41 saved states at t = 0, 0.0125, …, 0.5, averaged to each selected grid for matched display. Matching
GLM and CT states live in `scripts/out/spectral_study_movie41/glm_ct_N/`. The builder falls back to
the original 21-state `athena_512_movie` and spectral-study runs when the denser runs are absent. Movie reference resolution
is separate from the quantitative reference and is labeled on the page.
All three movies have the same timestamps, color limits and repeated-frame
timing; no intermediate fluid states are synthesized. Extra endpoint clipping
changes the Athena timestep sequence slightly. The quantitative tables and four-checkpoint snapshot viewer now use the
completed 1024² run; movies retain the dense 512² run.
`movie_validation.json` records conservation and differences from that original
reference. To reproduce the movie run in a new directory:

```sh
.venv-movie/bin/python - <<'PY'
import runpy, sys
times = [f"{i/80:.10g}" for i in range(1, 41)]
sys.argv = ["run_athena_reference.py", "--n", "512", "--out",
            "scripts/out/athena_512_movie_new", "--times", *times]
runpy.run_path("scripts/run_athena_reference.py", run_name="__main__")
PY
```

The builder reads the completed `athena_512_movie41` directory by convention.
Playback event tests can be rerun with `node --test scripts/check_video_sync.cjs`.

`data.js`, the HTML pages and `media/` are generated. Keep them alongside
`app.js`, `sync-videos.js`, `style.css`, and the original `scripts/out` artifacts for offline use.
The built date records the site refresh, not the simulation execution time.

## Refresh GPU suite results

Download and extract Sherlock archives as described in `sherlock_mhd/README.md`, into
`scripts/out/sherlock_downloads/extracted/`, then rebuild only the GPU page:

```sh
.venv-movie/bin/python scripts/build_progress_site.py --gpu-only
```

This reuses the existing Orszag–Tang data and media, and regenerates `media/gpu/` from
scratch. `--gpu-returns DIR` reads another extraction directory. For each test, grid and seed
the newest result group wins (by `environment.toml` creation time). Smoke runs appear as
their own 8³ column. Snapshots use repeat 1; timings use every completed repeat. Every saved
physical output is rendered; no intermediate fluid states are synthesized.
Reference fields come from the driver's `ref_###.bin` files, or from the bundled Athena
Orszag–Tang data. Groups whose device is labelled as a CPU or synthetic fixture are flagged on
the page as not GPU measurements. The test list comes from `sherlock_mhd/configs/`; the
family, origin and "what it measures" text per test comes from `gpu_tests.json`.

## Add a physics test

Add a record to `tests.json` with a unique `id`, `page`, `number`, `title`, `nav`,
`status`, `description`, `chips`, `document`, and `kind: "generic"`. Supply exactly
three `columns`, in GLM / CT / reference order. Each column has `subtitle`,
`status` and `cards`. Cards support `title`, `text`, `stats` (label/value pairs),
`items`, `image`, `video`, `poster`, `caption`, `links`, `formula`, and `empty`.
Use paths relative to this directory and describe missing runs explicitly.
Run the builder to generate the new page and update navigation on every page.

See [CPU benchmark audit](../scripts/CPU_BENCHMARKS.md) for exact-wave results,
benchmark limitations and the prepared (not yet executed) repeated timing runner.
