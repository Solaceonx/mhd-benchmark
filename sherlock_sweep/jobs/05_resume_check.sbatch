#!/bin/bash
# Gate: an interrupted-and-resumed run must match an uninterrupted one output for output.
#SBATCH --job-name=sweep-resume-check
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/common.sh
prepare_environment
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
base="$MHD_RESULTS/resume-check-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"   # a requeued gate starts clean
run() { srun "$MHD_JULIA" --project="$MHD_BUNDLE/env" --threads="$JULIA_NUM_THREADS" src/run_suite.jl "$@"; }
# check CONFIG MODE: every method is stopped once (exit 75) and resumed by the next attempt. MODE 1 stops right
# after the first checkpoint; MODE output stops at the first output written after it, so the resumed run
# redoes that output (it must not be recorded twice).
check() {
    local config="$1" mode="$2" name;name="$(basename "$1" .toml)"
    local a="$base/$name-resumed" b="$base/$name-straight" stops=0 rc=1
    for attempt in $(seq 1 12); do
        set +e;MHD_STOP_AFTER_CHECKPOINT="$mode" run "$config" "$a" 0;rc=$?;set -e
        [[ "$rc" -eq 0 ]] && break
        [[ "$rc" -eq 75 ]] || { echo "$name attempt $attempt failed with $rc" >&2; exit 1; }
        stops=$((stops+1))
    done
    [[ "$rc" -eq 0 && "$stops" -ge 6 ]] || { echo "$name: expected 6 interrupted methods, got $stops (rc $rc)" >&2; exit 1; }
    run "$config" "$b" 0            # uninterrupted
    "$MHD_JULIA" --project="$MHD_BUNDLE/env" --startup-file=no tools/compare_runs.jl "$a" "$b"
}
check configs/gate_resume.toml 1
check configs/gate_resume_3d.toml output
echo "RESUME GATE PASSED"
