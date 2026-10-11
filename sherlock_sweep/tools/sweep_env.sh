#!/bin/bash
# Environment for the sweep's submissions (sourced by tools/start_sweep.sh and tools/restart_coordinator.sh).
# Every job is submitted with --export=ALL, so these reach the jobs.
export JULIA_MODULE="${JULIA_MODULE:-julia/1.12.6}"
export MHD_PYTHON_MODULES="${MHD_PYTHON_MODULES:-python/3.12.1 py-numpy/2.2.6_py312}"
export MHD_GPU_BUILD_MODULES="${MHD_GPU_BUILD_MODULES:-devel cmake/3.31.4 cuda/12.4.0 gcc/12.4.0}"
export MHD_GPU_RUN_MODULES="${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
export MHD_CPU_BUILD_MODULES="${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
# The sweep's own directories, never an earlier campaign's (values inherited from a shell are ignored).
export MHD_CODES="${SCRATCH:?SCRATCH is not set}/mhd-codes-sweep" MHD_RESULTS="$SCRATCH/mhd-results-sweep"
export MHD_PACK_MAX_FILE_MB="${MHD_PACK_MAX_FILE_MB:-50}"
mkdir -p logs returns
source jobs/common.sh
prepare_environment
prepare_codes
