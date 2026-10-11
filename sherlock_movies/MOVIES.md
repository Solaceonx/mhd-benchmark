# Sherlock movie runs (v2, 2026-09-29)

Dense-output runs for videos only: nothing here is timed or enters the accuracy comparison. Each
`configs/movie_<test>__<method>.toml` has the physics of `<test>.toml` in `sherlock_rerun`
(Kelvin–Helmholtz, current sheet) or `sherlock_mhd` (the rest); only the output settings differ.

| | 2D tests (7) | 3D tests (5) |
|---|---|---|
| Tests | kh_uniform, kh_localized, orszag_tang, rotor, current_sheet, field_loop, blast_2d_b50 | orszag_tang_3d, blast_3d_b50, field_loop_3d, decay (Mach 2), decay_supersonic (Mach 9) |
| GLM and CT | 384², 241 outputs; one job per method | 256³, 241 outputs of the z midplane only (`slice_output`) |
| Truth panel | AthenaK PPM/HLLD/RK3 at 1536² (true 2D), block-averaged to 384² | AthenaK on the same 256³ grid (PPM/HLLD/RK3; PLM/HLLE/RK2 for Mach 9, where the high-order setup stalls) |

`slice_output` is an output-only option added to `src/run_suite.jl` and `codes/external.py` for these
runs: diagnostics still use the full volume; only the saved snapshot is the midplane. The two decay
movies use colour limits that follow the run (`movie_scale = "adaptive"`), smoothed so they do not flicker.

**Fixes after the first submission (2026-09-29):** the 2D jobs request 128 GB (a 384² run is a z-invariant
384³ cube; 16 GB was OOM-killed at start-up). GLM configs set `glm_ch_at_outputs = "cfl"`: the original
driver sets c_h = courant·dx/(3dt) from the actual dt, so every step shortened to land on one of 241 outputs
(sometimes by a tiny leftover of the Float64 clock) inflated c_h, and GLM runs stopped on a nonphysical state
(Mach 9 decay at t = 0.034, Orszag–Tang 3D at t = 0.77). With the option those steps keep the CFL-step c_h;
the movie labels say so. The Kelvin–Helmholtz CT runs take about 2 h at 384² (including per-output
diagnostics of the 384³ volume), so they allow 4 h (`max_run_seconds = 14400`). The other 2D jobs spend as long on per-output diagnostics
of the 384³ volume as on the solve; a 1 h limit killed four of them before they could pack, so they allow 3 h. Retry only failed parts: `bash tools/submit_movies.sh --retry NAME:gpu NAME:truth ...`.

**Jobs:** one 32³ smoke check of all 24 configs, 24 GPU-driver jobs (after smoke), 12 AthenaK jobs: 37,
on `gpu,owners`, H100 SXM5, `--requeue`. Estimated cost: about 8 GPU-h. The Kelvin–Helmholtz CT jobs are
the longest (about 1 h at 384²) and may be preempted and restarted on owner nodes.
**Returned data:** about 35 GB (2D 384² states are 5.3 MB per output).

**Reused, not rebuilt:** the AthenaK executable of the four-test rerun
(`$SCRATCH/mhd-codes-rerun/athenak/build-HOPPER90-fp64`); the submit script stops if it is missing.
Results go to `$SCRATCH/mhd-results-movies`.

## Commands

From the Mac:

```
cd /Users/lyxia/Projects/research/abel_mhd
scp sherlock_movies.tar.gz sherlock_movies.tar.gz.sha256 lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/
```

On Sherlock:

```
cd /scratch/users/lyxia/mhd-study && sha256sum -c sherlock_movies.tar.gz.sha256 && tar -xzf sherlock_movies.tar.gz && cd sherlock_movies && sha256sum -c --quiet BUNDLE_SHA256SUMS && unset MHD_CODES MHD_RESULTS && bash tools/submit_movies.sh
```

Watch: `squeue -u lyxia -o '%.18i %.40j %.8T %.10M %.12P %R' | grep -E 'JOBID|movie'`

Download when done (Mac):

```
rsync -av lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/sherlock_movies/returns/ /Users/lyxia/Projects/research/abel_mhd/sherlock_movies/returns/
```

Then extract into `scripts/out/sherlock_downloads/extracted_movies/`, run
`scripts/render_suite_movies.py scripts/out/sherlock_downloads/extracted_movies`, and rebuild the page.
The dense movies replace the previews on the GPU suite page.
