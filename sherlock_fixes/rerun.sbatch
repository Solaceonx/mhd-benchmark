#!/bin/bash
# One array task of the full rerun (README.md, "The full rerun"): its share of `fixes.py --plan rerun` for the kernel
# group in $KERNELS (glm or ct), the shares chosen so the $NTASKS tasks take about the same time. rerun.sh submits it.
# Run again, it keeps the finished runs and continues an interrupted one from its checkpoint.
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${RERUN_RESULTS:-$SCRATCH/mhd-results-rerun}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
cd "$here" || exit 2
mkdir -p "$results"; cp vendor/BRANCH.json "$results/" 2>/dev/null
echo "== $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1), $(date -u)"
echo "== kernels $KERNELS, task $SLURM_ARRAY_TASK_ID of $NTASKS; $(tr -d '\n' < vendor/BRANCH.json | cut -c1-120)"
list="$results/task-$KERNELS-$SLURM_ARRAY_TASK_ID-of-$NTASKS.runs"
python3 fixes.py --plan rerun runs "$results" --kernels "$KERNELS" --task "$SLURM_ARRAY_TASK_ID" --tasks "$NTASKS" > "$list" || { echo "no run list"; exit 2; }
echo "   $(wc -l < "$list" | tr -d ' ') runs"
srun "$MHD_JULIA" --project="$SWEEP_BUNDLE/env" --threads="$JULIA_NUM_THREADS" src/run_suite.jl --runs "$list"; rc=$?
echo "== driver exit code $rc (1 when some run stopped: that is a result), $(date -u)"
exit 0
