#!/bin/bash -l
# Build the four grid codes (build.sh) on a CPU node: the GPU builds need the CUDA and NVHPC compilers, not a GPU.
#SBATCH --job-name=newcodes-build
#SBATCH --partition=normal,owners
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
NEWCODES_BUILD_JOBS="$SLURM_CPUS_PER_TASK" bash "$here/build.sh" ${NEWCODES:-athenapk cholla pluto enzo}
