# Running the comprehensive sweep (stage 1)

Every test at every grid with every code: the five GLM kernels and the CT prototype (T. Abel's kernels, unmodified,
in `vendor/`), AthenaK (GPU), RAMSES and Athena++ (CPU), each in a setup matched to the GPU kernels and in its best
setup, against exact solutions or AthenaK references. The plan, costs and schedule are on the progress site's
**Sweep plan** page (`progress/sweep-plan.html`, built from this bundle by `scripts/build_sweep_plan.py`).

| | Stage 1 (this bundle) |
|---|---|
| Tests | 26 (2D: waves, shock tubes, Orszag–Tang, rotor, blasts, field loops, Kelvin–Helmholtz ×3, current sheet, GEM; 3D: Orszag–Tang, blasts, field loop, Mach 2 and Mach 9 decay; throughput) |
| Grids | 32–512 per side (KH, GEM, Mach 9 decay from 64; KH Δt jump from 128) |
| Codes | GLM ×5, CT, AthenaK matched/best, RAMSES matched/best, Athena++ matched/best; AthenaK references at 4096² (2D, true 2D) and 512³ (3D); Athena++ check at 2048² / 256³ |
| Jobs | 1,265 runs packed into 227 Slurm jobs (109 GPU, 118 CPU): small runs share jobs of ~2 h, the 139 runs longer than an hour have a job each |
| Cost (central, [optimistic–pessimistic]) | ~220 GPU-h [188–307]; ~68k core-h [47k–129k], 80% of it 3D RAMSES/Athena++ at 512³ |
| Storage | ~2.6 TB kept on scratch; ~55 GB returned (files ≤ 50 MB), or ~2 GB of summaries alone |
| Time | gates ~5 h; GPU work under a day; CPU work 2–6 days depending on the cluster |

`python3 tools/sweep.py plan` prints the same numbers from `codes/plan.py` and `codes/estimate.py`.

## What happens

1. `tools/start_sweep.sh` (login node) fetches the pinned sources into `$SCRATCH/mhd-codes-sweep`, then submits
   the **gates** and the **coordinator**:
   - `setup`: Julia environment and every initial condition (CPU);
   - `build-gpu`: AthenaK (CUDA, fp64) on an H100; `build-cpu`: RAMSES (NDIM=3 and NDIM=2) and Athena++ (HLLD and HLLE);
   - `smoke`: every config and every method at 32³ on the H100 (a half-precision GLM kernel stopping on a
     nonphysical state, as they do on the strongest blasts, is recorded as a result; any other failure fails the gate);
   - `resume-check`: each of the six GPU-driver methods is interrupted (after a checkpoint, and after an output that
     follows one) and resumed; outputs, diagnostics and spectra must equal an uninterrupted run exactly (2D and 3D);
   - three pilot jobs, each a real (small) packed job whose results the sweep keeps: the GPU driver (current sheet and
     3D Orszag–Tang at 32, all methods), AthenaK (current sheet 64 as a cube, 3D Orszag–Tang 32), and the CPU codes
     (RAMSES 2D and 3D, Athena++ HLLE and HLLD).
2. The coordinator (`jobs/20_sweep_coordinator.sbatch`, 1 CPU on `normal`) runs `tools/sweep.py tick` every 5 minutes.
   Once every gate has completed it submits the sweep in waves under the per-user submit limit: at most 90 queued
   jobs, of which at most 30 CPU jobs, cheapest first. Each job's run list is written to `sweep_state/tasks/`.
   **One job per unit at a time**: nothing is submitted for a unit that still has a job in the queue. A preempted job
   is requeued by Slurm itself (every job has `--requeue`; same job ID) and the coordinator leaves it alone. A job
   that has left the queue in TIMEOUT (1.5× time) or OUT_OF_MEMORY (2× memory), or in PREEMPTED, NODE_FAIL or
   BOOT_FAIL without having been requeued, is resubmitted, up to 3 attempts. When every job has ended, a job that
   left runs unfinished is submitted once more. After 46 h the coordinator hands over to a new coordinator job. If a
   gate fails it stops and says so in `sweep_state/status.txt`.
3. A GPU-driver job runs its whole run list in one Julia process (start-up and compilation once); an external-code job
   runs its list one run after another. A failed run is recorded and the job goes on to the next. For the external
   codes, repeats 2–5 only time a run that works: when repeat 1 fails they are not run.
4. Every run resumes if its job is requeued or resubmitted: run directories are named by what they hold
   (`$SCRATCH/mhd-results-sweep/<test>-gpu-N<n>-s<seed>-<all|method>`, `<test>-<code>-<variant>-<mode>-N<n>-s<seed>`);
   finished runs are kept. The GPU driver checkpoints every 15 min; AthenaK writes restart files at ~4 output times per
   run; RAMSES restarts from its last output; Athena++ from its last finished segment. All four were killed mid-run and
   resumed on the Mac (GPU driver: the resume gate on Sherlock) and reproduce an uninterrupted run exactly.
   **One job per run directory at a time**: the drivers write `<run>/data.lock` naming their Slurm job, and a job
   that finds a run held by another *running* job skips it untouched (two jobs in one directory resume from,
   overwrite and delete each other's files).
5. Each finished run packs `returns/<run>.summary.tar.gz` (manifests, diagnostics, spectra, logs) and
   `returns/<run>.tar.gz` (everything up to 50 MB per file). Larger files (3D states from 128³ up) stay on scratch and
   are listed in `<run>/kept_on_scratch.txt`. Packing never fails a job; a run it could not pack has no archives,
   and `audit` lists it.

## Commands

Mac (package is already built; `BUNDLE_SHA256SUMS` covers every file):

```
cd /Users/lyxia/Projects/research/abel_mhd
scp sherlock_sweep.tar.gz sherlock_sweep.tar.gz.sha256 lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/
```

Sherlock, login node (first a dry run that submits nothing, then the start):

```
cd /scratch/users/lyxia/mhd-study && sha256sum -c sherlock_sweep.tar.gz.sha256 && tar -xzf sherlock_sweep.tar.gz && cd sherlock_sweep && sha256sum -c --quiet BUNDLE_SHA256SUMS && bash tools/start_sweep.sh --dry-run
```

```
bash tools/start_sweep.sh
```

Options (pass to `start_sweep.sh`): `--cpu-max-n3d 256` drops 3D RAMSES/Athena++ at 512³ (saves ~55k core-h);
`--skip-cpu` runs GPU work only; `--only TEST ...` a subset; `--cap`, `--cpu-cap` the queue ceilings;
`--gpu-partition`, `--cpu-partition`, `--gpu-feature`.

Watch:

```
cat /scratch/users/lyxia/mhd-study/sherlock_sweep/sweep_state/status.txt
```

```
cd /scratch/users/lyxia/mhd-study/sherlock_sweep && module load python/3.12.1 && python3 tools/sweep.py status
```

Runs (1,265): complete / running / failed / not started, with failures grouped by error:

```
python3 tools/sweep.py runs
```

Check the results (read-only; do this when the sweep ends, and after any trouble). It checks every finished run's
records and files (each output once and in order in `run.toml`, `diagnostics.csv` and `spectra.csv`; every file
present with the size its shape implies), sorts failures into solver stops, file errors and others, and lists jobs
that ran twice at the same time, runs left unfinished, and runs not packed:

```
python3 tools/sweep.py audit
```

Then, as it says: `python3 tools/sweep.py redo --audit` deletes the damaged runs and hands their jobs back to the
coordinator (`redo RUN ...` for named runs; more than 40 needs `--yes`), and `python3 tools/sweep.py repack --audit`
packs the unpacked ones.

Status for the site (the Sweep page shows which runs exist; this makes it exact). On Sherlock:

```
cd /scratch/users/lyxia/mhd-study/sherlock_sweep && module load python/3.12.1 && python3 tools/sweep.py runs --json sweep_state/runs.json
```

On the Mac (fetches it and rebuilds the page):

```
scp lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/sherlock_sweep/sweep_state/runs.json /Users/lyxia/Projects/research/abel_mhd/progress/sweep_status/export_latest.json && python3 /Users/lyxia/Projects/research/abel_mhd/scripts/build_sweep_plan.py
```

Download summaries first (small), then the data (Mac):

```
rsync -a --include='*.summary.tar.gz*' --exclude='*' lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/sherlock_sweep/returns/ /Users/lyxia/Projects/research/abel_mhd/sherlock_sweep/returns/
```

```
rsync -a lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/sherlock_sweep/returns/ /Users/lyxia/Projects/research/abel_mhd/sherlock_sweep/returns/
```

## If something goes wrong

- **A gate failed.** Read `logs/sweep-*` for that job and `sweep_state/status.txt`. After a fix (delete the failed
  pilot's run directories if a pilot failed: a run recorded as failed is kept, not redone), resubmit only the failed
  and cancelled gates and restart the coordinator; completed gates (setup, builds) are kept:
  `python3 tools/sweep.py regate && bash tools/restart_coordinator.sh` (after `module load python/3.12.1`; every
  submission carries the sweep's module versions and paths, whichever shell runs it).
- **The coordinator stopped** (e.g. node failure, or after fixing a failed gate by hand): `bash tools/restart_coordinator.sh`
  from the bundle directory. It refuses if one is already queued; ticks are idempotent.
- **A lane is PAUSED** (`status.txt`): most of that lane's finished jobs failed within 15 minutes, which points to a
  systematic problem; nothing new is submitted in that lane. Read a failed job's log in `logs/`, fix, then
  `python3 tools/sweep.py retry --lane cpu --clean-error 'TEXT FROM THE ERROR'` (or `--lane gpu`): it deletes the runs
  that died of that error and hands their jobs back to the coordinator, which resubmits them within the caps.
- **A run FAILED.** Not retried automatically: a solver that stops (e.g. a pressure floor, a runaway step count) is a
  result, recorded in its `run.toml`. To rerun after a code fix: `python3 tools/sweep.py redo RUN_DIRECTORY_NAME ...`
  (deletes those runs and hands their jobs back; the jobs' other runs are kept and skipped), or by error text:
  `python3 tools/sweep.py retry --lane cpu --clean-error 'TEXT'`. Both work whatever state the run's job is in: a
  waiting job will run the deleted runs itself; a handed-back job is resubmitted once its unit has no job left in the
  queue; a run its running job may be working in is left alone (repeat the command later).
  `--clean-error` takes every run with a part (a method or a repeat) that failed with the text: repeat 1 can fail
  while the timing repeats succeed. `python3 tools/sweep.py retry 'JOB_KEY'` hands a whole job back (keys as printed
  by `status`).
- **A run is not worth its cost** (e.g. a code that stalls: every repeat runs to the step cap, and at 512³ that is
  days): `python3 tools/sweep.py skip 'JOB_KEY' ...` takes long runs (those with a job of their own) out of the sweep:
  their jobs are cancelled and never submitted again, and they leave the counts. `skip` lists them, `skip --undo`
  puts them back. What they wrote stays on scratch.
- **A run shows as running but no job is working on it** (`audit` lists these with the last thing the run logged):
  `python3 tools/sweep.py redo --audit` hands their jobs back. A run whose first attempt was stopped while it wrote
  its initial data starts over by itself when its job runs again.
- **A run is "failed" although its code never failed** (the job was preempted, requeued, cancelled or lost a node
  while the code ran; before patch 10 the driver recorded that as a failure and no later job resumed it): `audit`
  lists them ("recorded as failed, but the code had been killed") and `python3 tools/sweep.py redo --audit` hands
  their jobs back without deleting anything; the run continues from its last output. Since patch 10 the driver
  records nothing when a code is killed, and a resumed RAMSES run writes a restart point about every 20 minutes
  (only the newest is kept), so a preemption costs minutes, not the hours since the last output.
- **Two jobs of one unit in the queue** (`audit` says so; the coordinator before patch 7 could create them):
  `python3 tools/sweep.py dedupe` cancels the extra one (it keeps the job that is running).
- **Scratch**: files untouched for 90 days are purged. Copy what you need to keep to `$GROUP_HOME` or the Mac.

## Known limits

- Planar tests run as N³ cubes in the GPU driver and AthenaK (the kernels only evolve cubes; like-for-like GPU timing),
  so the Kelvin–Helmholtz tests at 512 are the largest GPU items. RAMSES and Athena++ run them as true 2D grids.
- AthenaK PPM/HLLD/RK3 stalls (Δt collapse) at Mach 9; it runs only at 64³ and 128³ for `decay_supersonic`, and the
  Mach 9 reference uses AthenaK PLM/HLLE/RK2.
- RAMSES matched (HLL) also stalls at Mach 9 (found in the sweep: at 64³ and 128³ it reaches 10–20% of the run within
  the step cap; 2.7 h per repeat at 128³ on 32 ranks). Its 256³ and 512³ runs would cost an estimated 20–50 thousand
  core-hours to record the same stall, nearly all of it the two 512³ runs (the plan's model said 7.7 thousand); take them out with
  `python3 tools/sweep.py skip 'decay_supersonic|ramses:matched|run|N256|s42' ...` (both seeds, 256 and 512).
- AthenaK at 512³ (fp64) needs an estimated 55–60 GB of the H100's 80 GB; if a 512³ AthenaK run runs out of device
  memory it fails visibly and is not retried.
- The 512³ runs have no reference in stage 1 (their truth is the 1024³ multi-GPU AthenaK of stage 4); their spectra,
  energies and divergence diagnostics are complete.
- Every run carries a step cap of 4× its expected steps (`max_steps_per_n`), so a Δt collapse ends a run instead of
  burning hours.
