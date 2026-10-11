# What was checked before shipping (Mac, 2026-09-30)

The Mac has no NVIDIA GPU, so GPU code paths were checked with JLArrays (GPUArrays' CPU reference backend, scalar
indexing disallowed, the same generic kernels and reductions CUDA runs). The first real GPU runs are the Sherlock
gates (`smoke`, `resume-check`), which run before any sweep job is submitted.

## Checks

| What | Result |
|---|---|
| `vendor/` GLM and CT kernels | SHA-256 identical to T. Abel's `GLMMHDTurb` repository |
| Every config through the driver's validation | all 28 pass (`check_config`) |
| Initial conditions of every config (setup gate's test) | 305/305 pass |
| GPU-side diagnostics (`src/lean.jl`) vs the original host `Problems.diagnostics` | equal to 1e-9 on 6 problems (2D/3D, KH leakage band, GEM flux, CT without ψ, no faces), on CPU arrays and on JLArrays |
| GPU-side error norms vs `solution_errors` | 2e-13; on z = 1 plane views (used for planar tests) vs the full cube 1.5e-15 |
| Block averages | match a host reference; planar input stays planar |
| Power spectra (`src/spectra.jl`) | Parseval exact for a band-limited field; solenoidal/compressive split exact on single modes; JLArrays vs CPU 2e-15 |
| Planar initial states (`plane=true`, and the faster fill) | bit-identical to the cube for all 17 problems; 256² build 6.9 s → 0.34 s |
| RAMSES NDIM=2 build | initial state equals the Julia state exactly; mass and energy conserved |
| RAMSES kill mid-run → resume | steps, every state and diagnostics identical to an uninterrupted run |
| AthenaK (CPU build) kill mid-run → resume from restart file | all 17 outputs, diagnostics and spectra identical |
| Athena++ HLLE build; kill mid-run → resume | all outputs and diagnostics identical (5 segments reused) |
| External post-processing (diagnostics, spectra, slices, block averages, temporary frames deleted) | checked on AthenaK 3D output; a rerun is a no-op |
| Reference runs measure themselves | GEM reference: reconnected flux Φ(0) = 0.2/25.6 as expected |
| Packing (size cap, summary archive, checkpoints excluded) | checked with BSD tar; Sherlock uses GNU tar (same exclusion semantics) |
| Coordinator (`tools/sweep.py`) against a simulated Slurm whose `squeue`/`sacct` follow real Slurm's array-ID conventions | gates wait; waves stay under 90 jobs with 30 reserved for CPU; TIMEOUT resubmitted with 1.5× time; FAILED left alone; a failed gate stops it; `retry`; finishes with every job |
| Coordinator against a simulated Slurm that requeues preempted jobs as Slurm does (same ID, a new accounting record, which can lag behind the queue) | the version deployed before patch 7 submits a second job for a requeued one (the bug, reproduced); patch 7 does not, in or out of the lag; a preempted job that was not requeued is resubmitted after 10 min; a handed-back unit waits for its running job; a failing `squeue` stops the step; a state directory left by the old version is taken over (`dedupe` keeps the running job); a whole sweep driven to its end never has two jobs of one unit |
| Two jobs in one AthenaK run directory (Mac, real build): the requeued job and a resubmission, both resuming | deployed drivers: both run, the log shows the pattern seen on Sherlock (restart header, the first job's full summary, then the second job's end); patched drivers: the second exits untouched (status 75), the first finishes alone |
| The run guard through the real `jobs/external.sbatch` | a run held by a running job is skipped untouched and not packed; a lock whose job is waiting or gone is taken over; a run killed mid-way is refused to another job while its job runs and resumed by the same job (requeue); no lock is left after a run |
| `retry --clean-error`, `redo` and the old Athena++ failures recreated with the pre-patch-5 driver (repeat 1 failed, repeats 2–5 complete) | the patch-7 `--clean-error` finds none (as on Sherlock); patch 8 removes them; `redo` by name removes them whether their job has ended, waits or runs; rerun with the current driver, all 15 parts complete |
| A run stopped after its solver finished but before its output was converted, then resumed (real AthenaK and RAMSES builds) | the patch-8 driver fails on AthenaK with `Duplicate output at frame 20` (as on Sherlock) and leaves one more output file each time; the fixed driver converts the outputs, on the directory as those retries left it and straight after the interruption: all 21 states and the diagnostics identical to an uninterrupted run. RAMSES: identical both ways. Mid-run kill and resume still reproduce an uninterrupted run |
| RAMSES interrupted and resumed with restart points (real 3D build at 32³; `foutput` every 3–7 steps) | killed twice and resumed; a run left by the patch-9 driver resumed by the new one; the solver finished and its conversion lost; the newest output cut short; an output damaged without getting smaller; restart points pruned every 50 ms while RAMSES runs: in every case all 11 states, the step counts and `diagnostics.csv` equal an undisturbed run bit for bit, only the frame outputs are left at the end, and a first attempt writes no restart points. A 2D run equals the patch-9 driver's |
| A code killed from outside (stand-in `srun` that kills RAMSES and reports it as Slurm does) | the patch-9 driver records `exit code 143` as a failure and, run again, keeps it (the bug); the new driver records nothing, exits 76 and the next attempt completes, identical to an undisturbed run; a failure recorded by the old driver (143, and exit 1 with "tasks: Killed") is continued; a crash of the code itself (139) and an out-of-memory kill stay failures. On the sweep's own records: exactly the eight 512³ RAMSES runs are read as killed, the 39 stalls are not |
| `audit` and `redo --audit` with a killed run, against the simulated Slurm | listed, its job counted among those to hand back; handed back with every file kept; resubmitted by the coordinator; a run that failed by itself is left alone |
| `runs --json` and the site's reading of it (`scripts/sweep_status.py`) | every run and part exported; finished, stopped by the solver, failed-to-rerun and unfinished parts are told apart, including the old Athena++ time failures (far below the step cap) against stalls (at the cap) |
| `skip` against the simulated Slurm | the running and the waiting job of skipped runs are cancelled; the coordinator never submits them again through to the end of the sweep; counts and plan drop them, their hours stay in the right lane; `--undo` restores; packed runs and pilots are refused |
| Driver, with the real Athena++ build | a run directory holding half-written initial data and no `input.toml` starts over and completes; with a step cap that fails repeat 1, repeats 2–5 are not run, also when a later job passes over the run |
| `audit` on real local results (88 finished parts of all four external codes and the GPU driver) | none flagged; with damage injected (an output twice in a CSV, a short file, a missing file, a file error, an unmeasured run, missing archives) each is found; solver stops are not flagged; `redo --audit` and `repack --audit` clear a second audit |
| Packed jobs through the real `jobs/external.sbatch` (Mac, with stand-ins for `module`, `srun` and the code paths) | a CPU job (RAMSES 2D and 3D, Athena++ HLLE and HLLD) and an AthenaK job (a run and a GEM reference): each run packed, a failed run does not stop the job; resubmitted, finished runs are skipped and interrupted ones continue |
| GPU-driver run lists (`run_suite.jl --runs`) | parsing, validation and the legacy index path checked on the Mac; on Sherlock the `pilot:gpu` gate passed (two packed runs, all six methods, one process) |
| On Sherlock (H100) | `resume-check` passed (all six methods, 2D and 3D, interrupted and resumed runs identical to uninterrupted ones, including spectra, slices and block averages); `smoke` passed |

## Bugs found and fixed while building this

1. **GPU driver resume could record an output twice**: a preemption after an output but before the next checkpoint made
   the resumed run write that output again (diagnostics row and manifest entry). Rows after the checkpoint are now dropped
   on resume, and the resume gate stops runs at exactly that point (`MHD_STOP_AFTER_CHECKPOINT=output`).
2. Spectra rows wrote the frame number as a float, which the resume truncation could not parse.
3. **RAMSES restarts read conservative outputs as primitive variables** (missing `read_conservative`); a resumed run
   continued from a wrong state. Found by the kill/resume test.
4. Pilot jobs were submitted without their dependency on the builds.
5. The coordinator let GPU work take every queue slot; CPU work now has its own lane.
6. Post-processing a 2D reference (4096²) would have built a 4096³ Julia state (2.4 TB); planar states are now built
   as one plane.
7. **The coordinator counted queued jobs with `squeue -o %A`**, which on real Slurm is each array element's own ID, not
   the array's base ID it recorded; the CPU lane limit would have silently stopped working. Now `%F`. (Found on a
   second pass; the first simulated Slurm printed base IDs, so it could not catch this.)
8. **1,670 tiny jobs**: half the GPU tasks were under 2 minutes of work, each paying ~1.5 min of start-up and a queue
   wait. Runs are now packed into ~2 h jobs (227 jobs); the GPU driver runs a job's list in one Julia process.
9. **The smoke gate failed on Sherlock (2026-09-30)** because it counted any stopped run as a failure: the f16 GLM
   kernels stop on a nonphysical state on the strongest blasts at 32³ (they did in the first campaign too). Such stops
   are now recorded as results; any other failure still fails the gate. `tools/sweep.py regate` resubmits only the
   failed and cancelled gates.
10. **The external-code pilots failed on Sherlock (2026-09-30)**: `regate`, run from a login shell, submitted them
    without the sweep's environment, and the job scripts' fallback `py-numpy` module is Sherlock's Python 2.7 build.
    `tools/sweep.py` now pins the environment on every submission (recorded at start; defaults mirror
    `tools/sweep_env.sh`), and the job scripts' fallbacks name exact module versions.
11. **Every CPU job submitted by the coordinator failed on Sherlock (2026-09-30)**: `srun: fatal: SLURM_MEM_PER_CPU,
    SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive`. The coordinator is itself a job (`--mem=4G`); with
    `--export=ALL` its `SLURM_MEM_PER_NODE` reached the CPU jobs, whose own `--mem-per-cpu` sets `SLURM_MEM_PER_CPU`.
    (The CPU pilot passed because it was submitted from a login shell.) Submissions now drop the submitting job's
    `SLURM_*` variables, the job scripts drop an inherited `SLURM_MEM_PER_NODE` when both are set, and the coordinator
    pauses a lane when most of its finished jobs fail within 15 minutes, instead of submitting the rest to fail too.
    `retry --lane cpu --clean-error TEXT` removes runs that died of TEXT (and their archives) and hands their jobs back
    to the coordinator; jobs waiting to be resubmitted no longer count toward the pause. `tools/sweep.py runs` reports
    progress by run from each run's own records.
12. **Athena++ runs of tests with irrational output times failed conversion** (`brio_wu_rotated`, `alfven_oblique`;
    GEM would have too): Athena++ prints the time it reached with 6 significant digits, outside the frame tolerance. It
    lands on each target exactly, so a printed time equal to the target to that precision is now the target.
13. **RAMSES 3D at 128^3 ran out of octs on some ranks**: `ngridmax` was 1.5x the average per rank, too little for the
    ghost octs of small domains. Now 4x + 50,000 (at most the whole tree). And RAMSES reads the whole initial state on
    every rank (6.4 GB at 512^3), so its 3D jobs now request that much more memory per rank and fewer ranks per node.
14. **An AthenaK run resumed from a restart file lost its later outputs when the output times are not short decimals**
    (`brio_wu_rotated` at 512^3 on Sherlock): AthenaK writes each output block's `last_time` back into its parameter
    text with 6 significant digits, so every output after a restart is offset by that truncation and failed the
    time match. Reproduced on the Mac; the resume now passes the exact time on the command line, and a killed and
    resumed run again equals an uninterrupted one (times, steps, all outputs).
15. **The coordinator ran some preempted jobs twice at once (found 2026-10-01).** Every job is submitted with
    `--requeue`, so Slurm requeues a preempted job itself, under the same ID; for a short while after the preemption
    the accounting still shows PREEMPTED. A coordinator step in that window submitted a second job for the unit, and
    the two then worked in the same run directories: each resumes from, overwrites and deletes the other's files
    (the AthenaK 4096² reference of `kh_uniform` died at its last output with `bin/... could not be opened`, after
    the other job had converted and removed the native output). Fixed in three places: the coordinator submits
    nothing for a unit that has a job in the queue (one function submits, and it checks); the drivers hold a run
    directory for their job (`data.lock`) and skip a run held by another running job; `dedupe` cancels duplicates
    still queued. `audit` checks every run's records and files for what such a collision leaves, and `redo` reruns
    the damaged ones. Also: `retry --clean-error` now works whatever state a run's job is in; the hand-back list is
    append-only (`requeue.jsonl`), so a request cannot be lost to a coordinator step running at the same moment;
    packing never fails a job; hours used count every incarnation of a requeued job.
16. **`retry --clean-error` missed runs in which only repeat 1 had failed (found 2026-10-01).** The nine Athena++
    runs hit by the 6-digit time bug (12) stayed failed through two retries: their timing repeats had succeeded (they
    need only the final time, which matched), and `--clean-error` took only runs whose every part had failed. My
    earlier explanation (their jobs were not in a failed state) was wrong. Reproduced on the Mac with the driver as it
    was before patch 5: the failure pattern (60% and 20% of the run reached) is exactly what 6-digit times give.
    `--clean-error` now takes a run with any part that failed with the text; `redo RUN ...` removes runs by name.
17. **A run whose first attempt was stopped while writing its initial data could never run again**: the next attempt
    found a non-empty directory without `input.toml` and refused it. `input.toml` is written last, before any run
    starts, so such a directory holds no result; the driver now starts it over. (A job that met another job in that
    state failed at once while the other finished the run: one way a job can be FAILED with its runs complete.)
18. **Runs that stall cost far more than planned.** RAMSES matched (HLL) stalls at Mach 9, so every repeat ran to the
    step cap (2.7 h per repeat at 128³, five repeats, against 1.9 h planned for all five). Repeats 2–5 of the external
    codes exist to time a run that works: they are no longer run when repeat 1 fails. `skip` takes long runs out of
    the sweep (jobs cancelled, never resubmitted, left out of the counts); recommended for RAMSES matched Mach 9 at
    256³ and 512³, where the stall would cost an estimated 20–50 thousand core-hours (projected from the 64³ and 128³
    timings: 90–190 thousand cell-steps per rank-second, 105 N steps to the cap; the two 512³ runs are nearly all of it).
19. **An AthenaK run whose job was stopped after the solver finished, while its output was being converted, could
    never be converted (found 2026-10-01 by `audit`: `current_sheet` and `brio_wu`, matched, 512³).** The resumed job
    restarted AthenaK from its last restart file, which is at the end time; AthenaK then only wrote the final output
    once more under the next number, and the converter stopped on `Duplicate output at frame N`. Every retry did the
    same. Reproduced on the Mac with the real build. The driver now recognises a finished solver (AthenaK: its
    "Terminating on time limit" and every output file present; RAMSES: every output complete) and converts what is
    there without running the code again; the states and diagnostics equal those of an uninterrupted run.
20. Per-output diagnostics copied the whole state to the host in Float64 (~50 GB of RAM and longer than the solve at
   512³); they now run on the GPU. The uniform-state check no longer downloads the full cube each output.

21. **A code killed from outside was recorded as a failed run, and a failed run is never resumed (found 2026-10-04,
    when the sweep ended with eight RAMSES 512³ runs "failed").** None had failed by itself: on the owner nodes their
    16-node jobs were preempted 7–24 times each, three lost a node's file system (`SIGBUS`, errno 108 from Lustre), one
    aborted in the MPI layer at start-up, one was requeued by hand. Slurm kills the code first, and the driver often
    lived long enough to write `status = "failed"` (`exit code 143` / `137`, srun's "tasks …: Terminated"); the
    requeued job then found a failed run, kept it and ended. Two of the eight also resumed from an output that an
    earlier stop had cut short, and RAMSES could not read it. Reproduced on the Mac with the real RAMSES build and a
    stand-in `srun` that kills the code as Slurm does: the patch-9 driver records the failure and never resumes. Now
    (`codes/external.py`): a code that was killed (exit 129/130/137/143, or srun reporting killed or terminated tasks;
    out-of-memory kills excepted) records nothing and the driver exits 76, so the next attempt continues; a part an
    earlier driver recorded that way is continued instead of kept; `audit` lists such runs and counts their jobs
    among those to hand back (`redo --audit` hands them back without deleting anything). RAMSES outputs are checked
    for size (each must hold as much data as the one before it; a rule that can reject one output at most), and an
    output RAMSES cannot read is removed so that the next attempt resumes from the one before.
    **Restart points.** RAMSES resumes only from its outputs, which at 512³ are 2–7 hours apart; an attempt preempted
    before the next one is lost (79.8k core-hours were used where 60k were planned). An attempt that resumes an
    interrupted run now also writes an output every ~20 minutes (`foutput`, which does not change the timestep) and
    keeps only the newest; outputs are matched to frames by their time, not their number. First attempts write none,
    so their timing stays comparable.

## Not checked here (the Sherlock gates check them)

- The GPU driver on CUDA with the new outputs (spectra, slices, block averages, device diagnostics): `smoke` runs every
  config and method at 32³; `resume-check` compares interrupted and uninterrupted runs in 2D and 3D, all six methods.
- AthenaK (CUDA build), RAMSES and Athena++ with MPI on Sherlock, and their post-processing there: the pilots.
- GPU memory at 512³ (CT with spectra ~40 GB; AthenaK fp64 an estimated 55–60 GB of 80 GB).
