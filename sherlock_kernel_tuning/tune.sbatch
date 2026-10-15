#!/bin/bash
#SBATCH --job-name=mhd-kernel-tune
#SBATCH --partition=gpu,owners
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
tune_bundle="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE"
source jobs/common.sh
prepare_environment
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
cd "$tune_bundle"
tune_results="${TUNE_OUTPUT:-$SCRATCH/mhd-results-kernel-tuning/job-$SLURM_JOB_ID}"
nvidia-smi
TUNE_OUTPUT="$tune_results/smoke" TUNE_GRIDS=48 TUNE_SAMPLES=3 TUNE_STEPS=2 TUNE_REQUIRE_ALL=1 \
    srun "$MHD_JULIA" --project="$SWEEP_BUNDLE/env" --threads=4 tune.jl
export TUNE_OUTPUT="$tune_results/full"
srun "$MHD_JULIA" --project="$SWEEP_BUNDLE/env" --threads=4 tune.jl
