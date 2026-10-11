#!/bin/bash
# GPU-driver job (GLM kernels and CT): runs the run list in $MHD_TASK_FILE (written by tools/sweep.py; lines
# RUNDIR TEST CODE VARIANT MODE N SEED GEOMETRY METHODS) in one Julia process, packing each run as it finishes.
# Run directories are named by what they hold, so a requeued or resubmitted job resumes: finished runs are kept,
# an interrupted one continues from its checkpoint. A run directory in use by another running job (RUNDIR/data.lock)
# is skipped untouched by src/run_suite.jl: one job per run at a time.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${MHD_TASK_FILE:?MHD_TASK_FILE is not set; submit with tools/sweep.py}"
source jobs/common.sh
prepare_environment
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
list="$MHD_RESULTS/job-${SLURM_JOB_ID}.runs"
: > "$list"
while read -r rundir test code variant mode n seed geometry methods <&3; do
    [[ -z "${rundir:-}" || "$rundir" == \#* ]] && continue
    [[ "$code" == gpu ]] || { printf 'Not a GPU-driver run: %s\n' "$rundir" >&2; exit 2; }
    mkdir -p "$MHD_RESULTS/$rundir"
    run_in_use "$MHD_RESULTS/$rundir" || capture_environment "$MHD_RESULTS/$rundir"
    printf '%s %s %s %s %s\n' "configs/$test.toml" "$MHD_RESULTS/$rundir/data" "$n" "$seed" "$methods" >> "$list"
done 3< "$MHD_TASK_FILE"
printf 'Job %s: %s runs from %s\n' "$SLURM_JOB_ID" "$(wc -l < "$list" | tr -d ' ')" "$MHD_TASK_FILE"
srun "$MHD_JULIA" --project="$MHD_BUNDLE/env" --threads="$JULIA_NUM_THREADS" src/run_suite.jl --runs "$list"
