#!/bin/bash
# External-code job (AthenaK on a GPU; RAMSES and Athena++ on CPUs): runs the run list in $MHD_TASK_FILE (written
# by tools/sweep.py; lines RUNDIR TEST CODE VARIANT MODE N SEED GEOMETRY EXTRA) one after another, packing each.
# Run directories are named by what they hold, so a requeued or resubmitted job resumes (codes/external.py --resume):
# finished runs are kept, an interrupted one continues from its restart point. A failed run does not stop the job.
# A run directory in use by another running job (RUNDIR/data.lock) is skipped untouched: one job per run at a time.
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${MHD_TASK_FILE:?MHD_TASK_FILE is not set; submit with tools/sweep.py}"
source jobs/common.sh
prepare_environment
prepare_codes
if grep -q ' athenak ' "$MHD_TASK_FILE"; then
    load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
else
    load_modules "${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
fi
device="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
status=0; count=0; skipped=0
# fd 3 for the list: srun and the codes read stdin, which must not swallow the remaining runs.
while read -r rundir test code variant mode n seed geometry extra <&3; do
    [[ -z "${rundir:-}" || "$rundir" == \#* ]] && continue
    [[ "$extra" == - ]] && extra=""
    run_ranks="${SLURM_NTASKS:-1}"
    case "$code" in
        athenak)
            exe="$MHD_CODES/athenak/build-$(gpu_arch)-fp64/src/athena"; launcher=""; kind=gpu ;;
        ramses)
            ndim=3; [[ "$geometry" == plane ]] && ndim=2
            exe="$MHD_CODES/ramses/bin/ramses_abel${ndim}d"; launcher="srun -n $run_ranks"; kind=cpu ;;
        athenapp)
            flux=hlld; [[ "$variant" == matched ]] && flux=hlle     # Athena++ fixes the Riemann solver at build time
            # With a minimum 8-cell block, small grids cannot occupy every allocated rank.
            blocks_per_axis=$((n / 8)); max_ranks=$((blocks_per_axis * blocks_per_axis))
            [[ "$geometry" == cube ]] && max_ranks=$((max_ranks * blocks_per_axis))
            (( run_ranks > max_ranks )) && run_ranks="$max_ranks"
            exe="$MHD_CODES/athena/bin/athena_$flux"; launcher="srun -n $run_ranks"; kind=cpu ;;
        *) printf 'Unknown code %s in %s\n' "$code" "$MHD_TASK_FILE" >&2; exit 2 ;;
    esac
    [[ -x "$exe" ]] || { printf 'Missing executable %s: run the build job for this code (and GPU model)\n' "$exe" >&2; exit 2; }
    run_dir="$MHD_RESULTS/$rundir"
    if run_in_use "$run_dir"; then
        printf '=== %s: in use by another running job, skipped\n' "$rundir"; skipped=$((skipped + 1)); continue
    fi
    mkdir -p "$run_dir"
    capture_environment "$run_dir"
    dev="${device:-CPU $(hostname) x ${run_ranks} ranks}"; [[ "$kind" == cpu ]] && dev="CPU $(hostname) x ${run_ranks} ranks"
    args=("$code" "$variant" "configs/$test.toml" "$n" "$run_dir/data" --seed "$seed" --geometry "$geometry" --exe "$exe"
          --device "$dev" --device-kind "$kind" --resume)
    [[ -n "$launcher" ]] && args+=(--mpi "$launcher")
    [[ "$mode" == reference ]] && args+=(--repeats 1 --restrict-to "$extra")
    printf '=== %s (%s)\n' "$rundir" "$(date -u +%H:%M:%S)"
    set +e
    python3 codes/external.py "${args[@]}" < /dev/null 2>&1 | tee -a "$run_dir/driver.log"
    rc=${PIPESTATUS[0]}
    set -e
    if (( rc == 75 )); then skipped=$((skipped + 1)); continue; fi     # codes/external.py found it in use: untouched
    printf '%s\n' "$rc" > "$run_dir/exit_code.txt"
    bash tools/pack_results.sh "$run_dir" < /dev/null || true
    (( rc == 0 )) || status=1
    count=$((count + 1))
done 3< "$MHD_TASK_FILE"
printf 'Job %s: %s runs, %s skipped (in use by another job), status %s\n' "${SLURM_JOB_ID:-local}" "$count" "$skipped" "$status"
exit "$status"
