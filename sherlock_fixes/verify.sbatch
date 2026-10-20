#!/bin/bash
# The checks of the sweep-fixes kernels (README.md, "Checking the fixes"): a trial, then every run of
# `fixes.py --plan verify plan`, one after another in one Julia process, cheapest first; at the end the control
# (originals on the branch against the sweep, digit by digit) and the table.
#   sbatch verify.sbatch          (verify.sh submits it, on the sweep's GPU model)
# Requeued or submitted again, it keeps the finished runs and continues the interrupted one from its checkpoint.
#SBATCH --job-name=mhd-verify
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${VERIFY_RESULTS:-$SCRATCH/mhd-results-verify}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
cd "$here" || exit 2
mkdir -p "$results"; cp vendor/BRANCH.json "$results/" 2>/dev/null
echo "== $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1), $(date -u)"
echo "== kernels: $(tr -d '\n' < vendor/BRANCH.json | cut -c1-200)"
drive() { srun "$MHD_JULIA" --project="$SWEEP_BUNDLE/env" --threads="$JULIA_NUM_THREADS" src/run_suite.jl --runs "$1"; }

if [[ ! -s "$results/trial_ok.txt" ]]; then
    echo "== 1. trial: Orszag-Tang to t = 0.05 on 32^3 and 48^3"
    rm -rf "$results"/trial-N*          # a trial cut short is repeated from the start
    python3 fixes.py --plan verify runs "$results" --trial > "$results/trial.runs"
    drive "$results/trial.runs"; echo "   driver exit code $? (a kernel that cannot run is reported below, not here)"
    python3 fixes.py --plan verify verdict "$results" || { echo "== TRIAL FAILED: no new kernel ran; no run was started"; rm -f "$results/trial_ok.txt"; exit 1; }
else
    echo "== 1. trial: passed in an earlier attempt ($(cat "$results/trial_ok.txt"))"
fi
echo "== 2. the runs"
python3 fixes.py --plan verify runs "$results" --ok "$results/trial_ok.txt" > "$results/job-$SLURM_JOB_ID.runs"
echo "   $(wc -l < "$results/job-$SLURM_JOB_ID.runs" | tr -d ' ') run directories"
drive "$results/job-$SLURM_JOB_ID.runs"; rc=$?
echo "== driver exit code $rc (1 when some kernel stopped on some test: that is a result), $(date -u)"
python3 fixes.py --plan verify summary "$results" "${MHD_RESULTS:-$SCRATCH/mhd-results-sweep}" > "$results/summary.txt" 2>&1
python3 fixes.py --plan verify control "$results" "${MHD_RESULTS:-$SCRATCH/mhd-results-sweep}" > "$results/control.txt" 2>&1
echo "== summary: $results/summary.txt; control: $results/control.txt"
exit 0
