#!/bin/bash
# Build AthenaK (CUDA, double precision) for the GPU model this job lands on. Use the same
# --constraint here as for the timed runs so the architecture matches.
#SBATCH --job-name=sweep-build-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/common.sh
prepare_environment
prepare_codes
load_modules "${MHD_GPU_BUILD_MODULES:-devel cmake/3.31.4 cuda/12.4.0 gcc/12.4.0}"
bash codes/build.sh athenak fp64
