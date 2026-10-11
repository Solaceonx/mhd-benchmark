#!/bin/bash
# Build AthenaK (CUDA, double precision) for the GPU model this job lands on. Use the same
# --constraint here as for the timed runs so the architecture matches.
#SBATCH --job-name=mhd-build-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/common.sh
load_site_profile
prepare_environment
prepare_codes
load_modules "${MHD_GPU_BUILD_MODULES:-nvhpc mps}"
command -v cmake >/dev/null || { printf 'cmake is unavailable; set MHD_GPU_BUILD_MODULES to include the Marlowe cmake module\n' >&2; exit 2; }
bash codes/build.sh athenak fp64
