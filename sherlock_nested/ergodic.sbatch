#!/bin/bash
# The 3-D ergodicity ensemble (ergodic.jl): nest 3 x 64^3 to t = 1 for each seed, a probe at the box centre; no cap, as Tom Abel's 2-D runs.
#   sbatch ergodic.sbatch 1 2 3 4 5 6 7 8          (about 5 min a seed)
#SBATCH --job-name=nested-ergodic
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; cd "$here"
module load "${JULIA_MODULE:-julia/1.12.6}" ${NESTED_CUDA_MODULES:-devel cuda/12.4.0}   # env/LocalPreferences.toml: the local toolkit
export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
nested="${NESTED_DIR:-$SCRATCH/mhd-study/NestedTurbulence-cap}"
out="${NESTED_RESULTS:-$SCRATCH/mhd-results-nested}/ergodic"
mkdir -p "$out"
echo "== $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1), $(date -u)"
julia --project=env ergodic.jl "$nested" "$out" "$@"; rc=$?
echo "== exit code $rc, $(date -u)"
exit "$rc"
