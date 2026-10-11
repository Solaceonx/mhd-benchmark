#!/bin/bash
# Sourced by job scripts after cd to the submitted bundle directory.
prepare_environment() {
    export MHD_BUNDLE="$PWD"
    if [[ -n "${JULIA_BIN:-}" ]]; then
        MHD_JULIA="$JULIA_BIN"
    else
        module load "${JULIA_MODULE:-julia/1.12.6}"
        MHD_JULIA="$(command -v julia)"
    fi
    export MHD_JULIA
    export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export OPENBLAS_NUM_THREADS=1
    # Keep packages persistent, while job output goes to scratch.
    export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
    export JULIA_PKG_PRECOMPILE_AUTO=0
    # A job submitted from inside another job (the sweep coordinator) may inherit that job's SLURM_MEM_PER_NODE;
    # srun refuses to start when it meets this job's own SLURM_MEM_PER_CPU (the CPU jobs use --mem-per-cpu).
    if [[ -n "${SLURM_MEM_PER_CPU:-}" && -n "${SLURM_MEM_PER_NODE:-}" ]]; then unset SLURM_MEM_PER_NODE; fi
    export MHD_RESULTS="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set; set MHD_RESULTS explicitly}/mhd-results-sweep}"
    # Run directories are named by what they run (test, code, task index), not by Slurm job: a job
    # requeued after preemption, or resubmitted after a time limit, resumes in the same directory
    # (GPU driver from its checkpoint; AthenaK from its restart file; RAMSES and Athena++ from their
    # last output). Finished runs in it are kept, not redone.
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
# Whether another running Slurm job holds this run directory: DIR/data.lock names the job working in it (written by
# codes/external.py and src/run_suite.jl). Two jobs must never work in one run directory. A waiting or ended holder,
# no lock, or no answer from squeue all count as free: the drivers make the careful check when they start a run.
run_in_use() {
    local lock="$1/data.lock" owner state
    [[ -n "${SLURM_JOB_ID:-}" && -s "$lock" ]] || return 1
    owner="$(awk 'NR==1{print $1}' "$lock" 2>/dev/null)" || return 1
    [[ -n "$owner" && "$owner" != "$SLURM_JOB_ID" ]] || return 1
    state="$(squeue -h -j "$owner" -o %T 2>/dev/null | head -n1)" || return 1
    case "$state" in ""|PENDING|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESV_DEL_HOLD|SPECIAL_EXIT|CONFIGURING) return 1 ;; esac
    return 0
}
# Load a space-separated module list from a variable (override on Sherlock after `module spider`).
load_modules() {
    local m
    for m in $1; do module load "$m"; done
}
# External-code sources and builds (tools/fetch_codes.sh, codes/build.sh).
prepare_codes() {
    export MHD_CODES="${MHD_CODES:-${SCRATCH:?SCRATCH is not set; set MHD_CODES}/mhd-codes-sweep}"
    # The readers in codes/external.py import the visualization modules from the
    # same pinned source trees that the build jobs use.
    export ATHENAK_DIR="$MHD_CODES/athenak" ATHENAPP_DIR="$MHD_CODES/athena"
    load_modules "${MHD_PYTHON_MODULES:-python/3.12.1 py-numpy/2.2.6_py312}"   # unversioned py-numpy is the Python 2.7 build
    python3 -c 'import numpy' || { printf 'python3 with numpy is required; set MHD_PYTHON_MODULES\n' >&2; return 1; }
}
# Kokkos architecture name of the first visible GPU.
gpu_arch() {
    case "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')" in
        70) echo VOLTA70 ;; 75) echo TURING75 ;; 80) echo AMPERE80 ;; 86) echo AMPERE86 ;;
        89) echo ADA89 ;; 90) echo HOPPER90 ;; *) echo UNKNOWN ;;
    esac
}
