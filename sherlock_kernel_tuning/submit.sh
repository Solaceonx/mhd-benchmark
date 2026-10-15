#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
active_tuning_jobs="$(squeue -h -u "$USER" -n mhd-kernel-tune -o %i)"
if [[ -n "$active_tuning_jobs" ]]; then
    echo "A kernel-tuning job is already queued or running; nothing submitted."
    squeue -u "$USER" -n mhd-kernel-tune
    exit 0
fi
sbatch --constraint="${TUNE_GPU_CONSTRAINT:-GPU_SKU:H100_SXM5}" tune.sbatch
