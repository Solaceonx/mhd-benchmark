#!/bin/bash
# Installs and precompiles env/ (CUDA.jl with Sherlock's CUDA 12.4 module, LocalPreferences.toml) on a CPU node, so a
# GPU job queued behind it (afterok) only loads it.
#   cd $SCRATCH/mhd-study/sherlock_nested && sbatch precompile.sbatch
#SBATCH --job-name=nested-precompile
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load "${JULIA_MODULE:-julia/1.12.6}" ${NESTED_CUDA_MODULES:-devel cuda/12.4.0}
export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
julia --project=env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA, KernelAbstractions; println("== loaded CUDA.jl ", pkgversion(CUDA), ", local toolkit ", CUDA.CUDACore.local_toolkit)'
