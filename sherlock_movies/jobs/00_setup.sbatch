#!/bin/bash
#SBATCH --job-name=movie-setup
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/common.sh
prepare_environment
srun "$MHD_JULIA" --project=env --threads="$JULIA_NUM_THREADS" tools/setup.jl
