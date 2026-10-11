#!/bin/bash
# Build RAMSES (MPI, 3D MHD) and Athena++ (MPI, HLLD) with the suite initial conditions.
#SBATCH --job-name=sweep-build-cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/common.sh
prepare_environment
prepare_codes
load_modules "${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
bash codes/build.sh ramses
bash codes/build.sh athenapp
