#!/bin/bash
#SBATCH --job-name=gem-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/common.sh
prepare_environment
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
run_dir="$MHD_RESULTS/smoke-$SLURM_JOB_ID"
mkdir -p "$run_dir"
capture_environment "$run_dir"
set +e
srun "$MHD_JULIA" --project=env --threads="$JULIA_NUM_THREADS" src/smoke.jl "$run_dir/data" 2>&1 | tee "$run_dir/driver.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$run_dir/exit_code.txt"
bash tools/pack_results.sh "$run_dir"
exit "$run_status"
