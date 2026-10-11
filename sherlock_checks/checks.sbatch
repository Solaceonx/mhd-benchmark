#!/bin/bash
# Courant-number and single-precision checks of Tom Abel's kernels (README.md): a trial of the new kernel variants,
# then every run of `checks.py plan`, one after another in one Julia process, cheapest first.
#   sbatch checks.sbatch          (check.sh submits it, on the sweep's GPU model)
# Requeued or submitted again, it keeps the finished runs and continues the interrupted one from its checkpoint.
#SBATCH --job-name=mhd-checks
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${CHECK_RESULTS:-$SCRATCH/mhd-results-checks}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
cd "$here" || exit 2
mkdir -p "$results"
echo "== $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1), $(date -u)"
drive() { srun "$MHD_JULIA" --project="$SWEEP_BUNDLE/env" --threads="$JULIA_NUM_THREADS" src/run_suite.jl --runs "$1"; }

if [[ ! -s "$results/trial_ok.txt" ]]; then
    echo "== 1. trial: Orszag-Tang to t = 0.05 on 32^3 and 48^3"
    rm -rf "$results"/trial-N*          # a trial cut short is repeated from the start
    python3 checks.py runs "$results" --trial > "$results/trial.runs"
    drive "$results/trial.runs"; echo "   driver exit code $? (a kernel that cannot run is reported below, not here)"
    python3 checks.py verdict "$results" || { echo "== TRIAL FAILED: no run was started"; rm -f "$results/trial_ok.txt"; exit 1; }
else
    echo "== 1. trial: passed in an earlier attempt ($(tr '\n' ';' < "$results/trial_ok.txt"))"
fi
echo "== 2. the runs"
python3 checks.py runs "$results" --ok "$results/trial_ok.txt" > "$results/job-$SLURM_JOB_ID.runs"
echo "   $(wc -l < "$results/job-$SLURM_JOB_ID.runs" | tr -d ' ') run directories"
drive "$results/job-$SLURM_JOB_ID.runs"; rc=$?
echo "== driver exit code $rc (1 when some kernel stopped on some test: that is a result), $(date -u)"
python3 checks.py summary "$results" "${MHD_RESULTS:-$SCRATCH/mhd-results-sweep}" > "$results/summary.txt" 2>&1
echo "== summary: $results/summary.txt"
exit 0
