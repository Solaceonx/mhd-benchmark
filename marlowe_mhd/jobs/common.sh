#!/bin/bash
# Sourced by job scripts after cd to the submitted bundle directory.
prepare_environment() {
    export MHD_BUNDLE="$PWD"
    [[ "${MHD_SITE:-}" != marlowe ]] || validate_marlowe_profile
    if [[ -n "${JULIA_BIN:-}" ]]; then
        [[ -x "$JULIA_BIN" ]] || { printf 'JULIA_BIN is not executable: %s\n' "$JULIA_BIN" >&2; return 2; }
        MHD_JULIA="$JULIA_BIN"
    else
        if [[ -n "${JULIA_MODULE:-}" ]]; then
            module load "$JULIA_MODULE"
        elif ! command -v julia >/dev/null; then
            printf 'Set JULIA_MODULE or JULIA_BIN after checking Julia availability on this site.\n' >&2
            return 2
        fi
        MHD_JULIA="$(command -v julia)"
    fi
    export MHD_JULIA
    export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export OPENBLAS_NUM_THREADS=1
    # Keep packages persistent, while job output goes to scratch.
    export JULIA_DEPOT_PATH="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}:"
    export JULIA_PKG_PRECOMPILE_AUTO=0
    export MHD_RESULTS="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set; set MHD_RESULTS explicitly}/mhd-results}"
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
# Empty values are intentional when software is already on PATH.
load_modules() {
    local m
    for m in $1; do module load "$m" || return; done
}

# Load the site profile before calling prepare_environment.  Keeping scheduler,
# module and path choices here lets the physics configs remain byte-identical to
# the Sherlock campaign.
load_site_profile() {
    local site="${MHD_SITE:-marlowe}"
    case "$site" in
        marlowe)
            local profile="${MHD_BUNDLE:-$PWD}/site/marlowe.sh"
            [[ -f "$profile" ]] || { printf 'Missing Marlowe profile: %s\n' "$profile" >&2; return 2; }
            # Export the selected settings with the job. Queued jobs must not
            # silently switch projects if this file is later edited.
            if [[ "${MHD_SITE_CONFIG_LOADED:-}" != 1 ]]; then
                if [[ -f "${profile%.sh}.env" ]]; then
                    source "${profile%.sh}.env"
                fi
                export MHD_SITE_CONFIG_LOADED=1
            fi
            # shellcheck disable=SC1090
            source "$profile"
            ;;
        sherlock) ;;
        *) printf 'Unknown MHD_SITE=%s\n' "$site" >&2; return 2 ;;
    esac
}
# External-code sources and builds (tools/fetch_codes.sh, codes/build.sh).
prepare_codes() {
    export MHD_CODES="${MHD_CODES:-${SCRATCH:?SCRATCH is not set; set MHD_CODES}/mhd-codes}"
    # The readers in codes/external.py import the visualization modules from the
    # same pinned source trees that the build jobs use.
    export ATHENAK_DIR="$MHD_CODES/athenak" ATHENAPP_DIR="$MHD_CODES/athena"
    load_modules "${MHD_PYTHON_MODULES-}"
    python3 -c 'import numpy, tomllib' || { printf 'Python 3.11+ with NumPy is required; set MHD_PYTHON_MODULES or activate its environment.\n' >&2; return 1; }
}

# Preemption requeues a job from the start, and our writers reject nonempty
# output directories. Preserve previous attempts without claiming a checkpoint.
attempt_suffix() {
    local count="${SLURM_RESTART_COUNT:-0}"
    [[ "$count" =~ ^[0-9]+$ ]] || { printf 'Invalid SLURM_RESTART_COUNT\n' >&2; return 2; }
    if (( count > 0 )); then printf -- '-restart%s' "$count"; fi
}
# Kokkos architecture name of the first visible GPU.
gpu_arch() {
    case "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')" in
        70) echo VOLTA70 ;; 75) echo TURING75 ;; 80) echo AMPERE80 ;; 86) echo AMPERE86 ;;
        89) echo ADA89 ;; 90) echo HOPPER90 ;; *) echo UNKNOWN ;;
    esac
}
