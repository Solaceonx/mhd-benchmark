#!/bin/bash
#SBATCH --job-name=mhd-setup
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
load_site_profile
prepare_environment
# CUDA.jl uses the system toolkit (env/LocalPreferences.toml). Precompilation records the
# CUDA runtime version, so the toolkit modules must be loaded here, not only in GPU jobs.
load_modules "${MHD_GPU_RUN_MODULES:-nvhpc mps}"
srun "$MHD_JULIA" --project=env --threads="$JULIA_NUM_THREADS" tools/setup.jl
