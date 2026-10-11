#!/bin/bash
# Sourced by job scripts after cd to the submitted bundle directory.
prepare_environment() {
    export MHD_BUNDLE="$PWD"
    if [[ -n "${JULIA_BIN:-}" ]]; then
        MHD_JULIA="$JULIA_BIN"
    else
        module load "${JULIA_MODULE:-julia}"
        MHD_JULIA="$(command -v julia)"
    fi
    export MHD_JULIA
    export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export OPENBLAS_NUM_THREADS=1
    # Keep packages persistent, while job output goes to scratch.
    export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
    export JULIA_PKG_PRECOMPILE_AUTO=0
    export MHD_RESULTS="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set; set MHD_RESULTS explicitly}/mhd-results-rerun}"
    # A job requeued after preemption (owners partition) restarts from t=0, and the writers
    # reject nonempty run directories. Each restart writes under its own subdirectory; the
    # interrupted attempt is kept, not resumed. Sourced at run time, so queued jobs get it too.
    if [[ "${SLURM_RESTART_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
        export MHD_RESULTS="$MHD_RESULTS/restart$SLURM_RESTART_COUNT"
    fi
    mkdir -p "$MHD_RESULTS" "$MHD_BUNDLE/returns"
}
capture_environment() {
    local destination="$1"
    date -u > "$destination/job_started_utc.txt"
    hostname > "$destination/hostname.txt"
    module list > "$destination/modules.txt" 2>&1 || true
    nvidia-smi -q > "$destination/nvidia-smi.txt" 2>&1 || true
    scontrol show job "${SLURM_JOB_ID:-0}" > "$destination/slurm_job.txt" 2>&1 || true
    cp vendor/PROVENANCE.json "$destination/vendor_provenance.json"
}
# Load a space-separated module list from a variable (override on Sherlock after `module spider`).
load_modules() {
    local m
    for m in $1; do module load "$m"; done
}
# External-code sources and builds (tools/fetch_codes.sh, codes/build.sh).
prepare_codes() {
    export MHD_CODES="${MHD_CODES:-${SCRATCH:?SCRATCH is not set; set MHD_CODES}/mhd-codes-rerun}"
    # The readers in codes/external.py import the visualization modules from the
    # same pinned source trees that the build jobs use.
    export ATHENAK_DIR="$MHD_CODES/athenak" ATHENAPP_DIR="$MHD_CODES/athena"
    load_modules "${MHD_PYTHON_MODULES:-python/3.12 py-numpy}"
    python3 -c 'import numpy' || { printf 'python3 with numpy is required; set MHD_PYTHON_MODULES\n' >&2; return 1; }
}
# Kokkos architecture name of the first visible GPU.
gpu_arch() {
    case "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')" in
        70) echo VOLTA70 ;; 75) echo TURING75 ;; 80) echo AMPERE80 ;; 86) echo AMPERE86 ;;
        89) echo ADA89 ;; 90) echo HOPPER90 ;; *) echo UNKNOWN ;;
    esac
}
