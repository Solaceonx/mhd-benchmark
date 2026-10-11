#!/bin/bash
# Marlowe site profile. Source this on a Marlowe login node after setting
# MARLOWE_PROJECT to the project directory name, for example abc123.

export MHD_SITE=marlowe

if [[ -n "${MARLOWE_PROJECT:-}" ]]; then
    export MARLOWE_ACCOUNT="${MARLOWE_ACCOUNT:-marlowe-${MARLOWE_PROJECT}}"
    export MARLOWE_SCRATCH="${MARLOWE_SCRATCH:-/scratch/${MARLOWE_PROJECT}}"
    export MHD_RESULTS="${MHD_RESULTS:-$MARLOWE_SCRATCH/mhd-results}"
    export MHD_CODES="${MHD_CODES:-$MARLOWE_SCRATCH/mhd-codes}"
    export MHD_JULIA_DEPOT="${MHD_JULIA_DEPOT:-$MARLOWE_SCRATCH/julia-mhd}"
fi

export MARLOWE_PARTITION="${MARLOWE_PARTITION:-preempt}"
export MHD_SCHEDULER_MODULES="${MHD_SCHEDULER_MODULES-slurm}"
export MHD_GPU_RUN_MODULES="${MHD_GPU_RUN_MODULES-nvhpc mps}"
export MHD_GPU_BUILD_MODULES="${MHD_GPU_BUILD_MODULES-nvhpc mps}"
# Set this after `module spider python`; leave it empty when python3 and NumPy
# are already present in the base environment.
export MHD_PYTHON_MODULES="${MHD_PYTHON_MODULES:-}"

validate_marlowe_profile() {
    [[ "${MARLOWE_PROJECT:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ &&
       "$MARLOWE_PROJECT" != *REPLACE* && "$MARLOWE_PROJECT" != PROJECT_ID ]] || {
        printf 'Set MARLOWE_PROJECT to your real project ID in site/marlowe.env.\n' >&2; return 2;
    }
    [[ "${MARLOWE_ACCOUNT:-}" == "marlowe-$MARLOWE_PROJECT" ||
       "${MARLOWE_ACCOUNT:-}" =~ ^marlowe-${MARLOWE_PROJECT}-p[ml][0-9]+$ ]] || {
        printf 'MARLOWE_ACCOUNT must match this project and its allocation suffix.\n' >&2; return 2;
    }
    case "$MARLOWE_PARTITION" in
        preempt) ;;
        batch) [[ "$MARLOWE_ACCOUNT" =~ -pm[0-9]+$ ]] || {
            printf 'The batch partition requires a medium-allocation account (pm suffix).\n' >&2; return 2; } ;;
        hero) [[ "$MARLOWE_ACCOUNT" =~ -pl[0-9]+$ ]] || {
            printf 'The hero partition requires a large-allocation account (pl suffix).\n' >&2; return 2; } ;;
        *) printf 'Unknown Marlowe partition: %s\n' "$MARLOWE_PARTITION" >&2; return 2 ;;
    esac
    local key value
    for key in MARLOWE_SCRATCH MHD_RESULTS MHD_CODES MHD_JULIA_DEPOT; do
        value="${!key}"
        [[ "$value" == /* && "$value" != / ]] || {
            printf '%s must be an absolute directory other than /.\n' "$key" >&2; return 2;
        }
    done
}
