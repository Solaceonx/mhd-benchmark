#!/bin/bash
# One-command Sherlock four-test rerun; jobs are gated by Slurm afterok dependencies.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JULIA_MODULE="${JULIA_MODULE:-julia/1.12.6}"
export MHD_PYTHON_MODULES="${MHD_PYTHON_MODULES:-python/3.12.1 py-numpy/2.2.6_py312}"
export MHD_GPU_BUILD_MODULES="${MHD_GPU_BUILD_MODULES:-devel cmake/3.31.4 cuda/12.4.0 gcc/12.4.0}"
export MHD_GPU_RUN_MODULES="${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
export MHD_CPU_BUILD_MODULES="${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
source jobs/common.sh
prepare_environment
prepare_codes
# The rerun rebuilds AthenaK with the corrected problem header; never in the first campaign's tree.
for dir in "$MHD_CODES" "$MHD_RESULTS"; do
    case "${dir%/}" in
        "${SCRATCH%/}/mhd-codes"|"${SCRATCH%/}/mhd-results")
            printf 'Refusing to use the first campaign directory %s; unset MHD_CODES and MHD_RESULTS.\n' "$dir" >&2
            exit 2 ;;
    esac
done
printf 'Codes: %s\nResults: %s\n' "$MHD_CODES" "$MHD_RESULTS"
"$MHD_JULIA" --version
python3 -c 'import sys, numpy, tomllib; assert sys.version_info >= (3,11); print("Python",sys.version.split()[0],"NumPy",numpy.__version__)'
exec python3 tools/submit_campaign.py "$@"
