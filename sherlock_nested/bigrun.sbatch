#!/bin/bash
# A long nested run that stops and resumes (bigrun.jl): checkpoints every 30 min; a requeued or resubmitted job resumes.
#   cd $SCRATCH/mhd-study/sherlock_nested && sbatch bigrun.sbatch deep192        (~10 h on one H100; resubmit to continue)
#SBATCH --job-name=nested-big
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --open-mode=append
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; cd "$here"
module load "${JULIA_MODULE:-julia/1.12.6}" ${NESTED_CUDA_MODULES:-devel cuda/12.4.0}   # env/LocalPreferences.toml: the local toolkit
export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
nested="${NESTED_DIR:-$SCRATCH/mhd-study/NestedTurbulence-cap}"
out="${NESTED_RESULTS:-$SCRATCH/mhd-results-nested}/big"
mkdir -p "$out"
echo "== $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1), $(date -u)"
julia --project=env bigrun.jl "$nested" "$out" "$@"; rc=$?
echo "== exit code $rc, $(date -u)"
exit "$rc"
