#!/bin/bash
# Tom Abel's kernels in the driven-turbulence pilot (README.md): every run of the [gpu] table of the configuration
# file ($TURB_CONFIG, default config.toml), one after
# another in one Julia process, after a short trial of each kernel. Needs $TURB_RESULTS/forcing.toml (the calibration
# job writes it; check.sh submits this job to start after that one).
#   sbatch gpu.sbatch
# The same GPU model as the sweep, so the speeds it prints compare with the sweep's (check.sh: TURB_GPU_CONSTRAINT).
# Requeued or submitted again, it keeps the finished runs.
#SBATCH --job-name=turb-gpu
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${TURB_RESULTS:-$SCRATCH/mhd-results-turb}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
[[ -s "$results/forcing.toml" ]] || { echo "no $results/forcing.toml: the calibration has not finished"; exit 2; }
echo "== $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1), $(date -u)"
config="${TURB_CONFIG:-config.toml}"; echo "== configuration $config"
srun "$MHD_JULIA" --project="$SWEEP_BUNDLE/env" --threads="$JULIA_NUM_THREADS" "$here/turb_drive.jl" "$here/$config" "$results" ${TURB_DRIVE_OPTIONS:-}; rc=$?
echo "== exit code $rc, $(date -u)"
exit "$rc"
