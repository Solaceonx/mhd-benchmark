#!/bin/bash
# Movie frames of the driven nest (movie.jl): mid-plane slices of every level every 0.002, nest64 then nest128.
#   cd $SCRATCH/mhd-study/sherlock_nested && sbatch movie.sbatch nest64 nest128         (~5 min + ~2.2 h)
#SBATCH --job-name=nested-movie
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; cd "$here"
module load "${JULIA_MODULE:-julia/1.12.6}" ${NESTED_CUDA_MODULES:-devel cuda/12.4.0}   # env/LocalPreferences.toml: the local toolkit
export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
nested="${NESTED_DIR:-$SCRATCH/mhd-study/NestedTurbulence-gpu3d}"
out="${NESTED_RESULTS:-$SCRATCH/mhd-results-nested}/movie"
mkdir -p "$out"
export NESTED_CASE_WALL="${NESTED_CASE_WALL:-10800}"
echo "== $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1), $(date -u)"
julia --project=env movie.jl "$nested" "$out" "$@"; rc=$?
echo "== exit code $rc, $(date -u)"
exit "$rc"
