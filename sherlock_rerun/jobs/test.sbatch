#!/bin/bash
# Generic physics-test array job. Submit through tools/submit.sh, which sets MHD_TEST,
# the job name and the array range from configs/$MHD_TEST.toml.
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
: "${MHD_TEST:?MHD_TEST is not set; submit with tools/submit.sh TEST}"
source jobs/common.sh
prepare_environment
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
run_id="$MHD_TEST-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
run_dir="$MHD_RESULTS/$run_id"
mkdir -p "$run_dir"
capture_environment "$run_dir"
set +e
srun "$MHD_JULIA" --project="$MHD_BUNDLE/env" --threads="$JULIA_NUM_THREADS" src/run_suite.jl "configs/$MHD_TEST.toml" "$run_dir/data" "${SLURM_ARRAY_TASK_ID:-0}" 2>&1 | tee "$run_dir/driver.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$run_dir/exit_code.txt"
bash tools/pack_results.sh "$run_dir"
exit "$run_status"
