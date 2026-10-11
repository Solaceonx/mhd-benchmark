# Data

## In this repository

| Path | What | Size |
|---|---|---|
| `turb_results/` | Mach 3 and 10 turbulence: per run `run.toml`, `series.csv`, `spectra.csv`, `pdf.csv`, a density slice at 10 T; plus the calibration and `summary.csv` | 75 MB |
| `progress/data.js`, `progress/*.json` | The numbers behind the notebook pages | 6 MB |
| `progress/sweep_status/` | The sweep's last status export, and an inventory of every file on Sherlock | 2 MB |
| `progress/media/` (without `gpu/` and `movies/`) | Figures and short movies of pages 01 to 05, 08 and 09 | 40 MB |
| `sherlock_*/references/` | Athena++ 1024² Orszag–Tang reference, averaged to 32², 64², 128² | 2 MB each |
| `paper_figures/` | Draft figures for the paper | 1 MB |

## Not in this repository

| What | Where | Size |
|---|---|---|
| All sweep output | Sherlock, `$SCRATCH/mhd-results-sweep` | about 2.5 TB |
| Fetched sweep records and the frames the figures need | laptop, `sweep_results/` | 3.3 GB |
| First campaign, verified archives | laptop, `sherlock_mhd/returns/` | 34 GB |
| Dense-output movie runs, archives | laptop, `sherlock_movies/returns/` | 15 GB |
| Rerun archives | laptop, `sherlock_rerun/returns/` | 0.25 GB |
| First CPU studies (Orszag–Tang 64² to 512², Athena++ references) | laptop, `scripts/out/` | 4 GB |
| Stills and movies of the GPU-suite page | laptop, `progress/media/gpu/`, `progress/media/movies/` | 0.35 GB |
| Turbulence output | Sherlock, `$SCRATCH/mhd-results-turb` | |

## Sherlock deletes scratch

Files on `$SCRATCH` are removed 90 days after their last change. The first sweep outputs go from about
**2026-12-29**. What to keep beyond the fetched records is still to be decided.

## Keeping track

- `scripts/sweep_inventory.py` lists every file on Sherlock in one ssh command and reports, by kind, what is there,
  what is on the laptop, what is in the fetch list, and when it would be purged. The last inventory
  (2026-10-07) has 87,733 files.
- `scripts/sweep_fetch.py` copies the fetch list (`scripts/sweep_data.py: WANTED` plus
  `progress/sweep_status/extra_wanted.txt`), resumable file by file.
- Every run directory holds a record of what made it: commit, options, compiler, modules, GPU, and its log.
- On the laptop only verified archives are kept, never a second unpacked copy; the GPU-suite page is rebuilt from
  them by `scripts/rebuild_gpu_page.sh`, which unpacks to a temporary directory and deletes it.
