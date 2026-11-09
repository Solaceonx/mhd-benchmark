#!/bin/bash
# First GPU runs of Tom Abel's NestedMHD (gpu_check.jl): CPU/GPU agreement, speed, one crossing time of driven
# Mach-10 turbulence nested and uniform. Installs and precompiles env/ first (CUDA.jl 6.5.1, KernelAbstractions
# 0.10, which Tom's kernels need) into the persistent depot $HOME/.julia-mhd.
#   cd $SCRATCH/mhd-study/sherlock_nested && sbatch gpu_check.sbatch
#SBATCH --job-name=nested-gpu-check
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; cd "$here"
module load "${JULIA_MODULE:-julia/1.12.6}"
export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
nested="${NESTED_DIR:-$SCRATCH/mhd-study/NestedTurbulence}"
out="${NESTED_RESULTS:-$SCRATCH/mhd-results-nested}/gpu_check"
mkdir -p "$out"
echo "== $(hostname), $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1), $(date -u)"
julia --project=env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || { echo "== the environment could not be installed"; exit 2; }
julia --project=env -e 'using CUDA; CUDA.versioninfo()' || { echo "== CUDA.jl cannot use this GPU"; exit 2; }
julia --project=env --threads="$SLURM_CPUS_PER_TASK" gpu_check.jl "$nested" "$out"; rc=$?
echo "== exit code $rc, $(date -u)"
exit "$rc"
