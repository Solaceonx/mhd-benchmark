#!/bin/bash
# Submit afterok on a separately validated AthenaK pilot. This coordinator uses
# one small CPU allocation and keeps resuming submission when Sherlock's
# per-user submitted-job limit is full. Simulation jobs retain their own gates.
#SBATCH --job-name=gem-auto-submit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

gpu_feature="${MHD_GPU_FEATURE:-GPU_SKU:H100_SXM5}"
attempt_log="logs/movie-auto-submit-${SLURM_JOB_ID}.campaign.log"

export JULIA_MODULE="${JULIA_MODULE:-julia/1.12.6}"
export MHD_PYTHON_MODULES="${MHD_PYTHON_MODULES:-python/3.12.1 py-numpy/2.2.6_py312}"
export MHD_GPU_BUILD_MODULES="${MHD_GPU_BUILD_MODULES:-devel cmake/3.31.4 cuda/12.4.0 gcc/12.4.0}"
export MHD_GPU_RUN_MODULES="${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"

run_campaign() {
    bash tools/submit_campaign.sh --gpu-only --gpu-feature="$gpu_feature" "$@" 2>&1 | tee -a "$attempt_log"
    return "${PIPESTATUS[0]}"
}

printf 'Starting automatic GPU campaign submission at %s\n' "$(date -u)" | tee "$attempt_log"
ledger="${MHD_CAMPAIGN_LEDGER:-}"
if [[ -n "$ledger" ]]; then
    ledger="$(realpath "$ledger")"
    run_campaign --resume "$ledger"
    status=$?
else
    run_campaign
    status=$?
    ledger="$(sed -n 's/^Job ledger: //p' "$attempt_log" | tail -n 1)"
fi
[[ -n "$ledger" && -f "$ledger" ]] || {
    printf 'Could not identify the new campaign ledger.\n' >&2
    exit "${status:-1}"
}
matrix_commands="$(sed -n 's/^Campaign: \([0-9][0-9]*\) matrix commands.*/\1/p' "$attempt_log" | head -n 1)"
[[ -n "$matrix_commands" ]] || { printf 'Could not read the matrix size from %s.\n' "$attempt_log" >&2; exit 1; }
expected_lines=$((1 + 4 + matrix_commands)) # header + 4 GPU-only gates + matrix commands

while (( $(wc -l < "$ledger") < expected_lines )); do
    if (( status != 0 )) && ! tail -n 30 "$attempt_log" | grep -q 'QOSMaxSubmitJobPerUserLimit'; then
        printf 'Submission stopped for a reason other than the QOS cap; refusing to retry.\n' >&2
        exit "$status"
    fi
    printf 'Ledger has %s/%s lines; retrying in 5 minutes.\n' "$(wc -l < "$ledger")" "$expected_lines" | tee -a "$attempt_log"
    sleep 300
    run_campaign --resume "$ledger"
    status=$?
done

printf 'All GPU-only campaign commands submitted. Ledger: %s\n' "$ledger" | tee -a "$attempt_log"
