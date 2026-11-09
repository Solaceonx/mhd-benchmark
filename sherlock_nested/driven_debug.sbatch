#!/bin/bash
# Why the driven Mach-10 3-D nest stopped in gpu_check (job 51686178) and whether Courant 0.25 cures it: driven_debug.jl
# runs a 3 x 32^3 nest and a uniform 128^3 at Courant 0.4 and 0.25, then the gpu_check nest (3 x 64^3) at 0.25, each
# capped at NESTED_CASE_WALL seconds, with a watchdog that saves the state when a CFL step collapses.
#   cd $SCRATCH/mhd-study/sherlock_nested && sbatch driven_debug.sbatch
#SBATCH --job-name=nested-driven-debug
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; cd "$here"
module load "${JULIA_MODULE:-julia/1.12.6}" ${NESTED_CUDA_MODULES:-devel cuda/12.4.0}   # env/LocalPreferences.toml: the local toolkit
export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
nested="${NESTED_DIR:-$SCRATCH/mhd-study/NestedTurbulence}"
out="${NESTED_RESULTS:-$SCRATCH/mhd-results-nested}/driven_debug"
mkdir -p "$out"
export NESTED_CASE_WALL="${NESTED_CASE_WALL:-900}"
echo "== $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1), $(date -u)"
julia --project=env driven_debug.jl "$nested" "$out" "$@"; rc=$?
echo "== exit code $rc, $(date -u)"
exit "$rc"
