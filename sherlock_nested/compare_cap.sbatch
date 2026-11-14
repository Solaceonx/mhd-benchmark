#!/bin/bash
# Nested vs uniform with the same Alfvén-speed cap (compare.jl, NESTED_VA_MAX=30; NestedTurbulence 1164c73): the fair comparison.
#   sbatch compare_cap.sbatch nest64 uni256        (Float64, to t = 1)     sbatch compare_cap.sbatch nest128 uni512   (Float32, to t = 0.5)
#SBATCH --job-name=nested-cmpcap
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; cd "$here"
module load "${JULIA_MODULE:-julia/1.12.6}" ${NESTED_CUDA_MODULES:-devel cuda/12.4.0}   # env/LocalPreferences.toml: the local toolkit
export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
nested="${NESTED_DIR:-$SCRATCH/mhd-study/NestedTurbulence-cap}"
out="${NESTED_RESULTS:-$SCRATCH/mhd-results-nested}/compare_cap"
mkdir -p "$out"
export NESTED_CASE_WALL="${NESTED_CASE_WALL:-25200}" NESTED_VA_MAX="${NESTED_VA_MAX:-30}"
echo "== $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1), $(date -u)"
julia --project=env compare.jl "$nested" "$out" "$@"; rc=$?
echo "== exit code $rc, $(date -u)"
exit "$rc"
