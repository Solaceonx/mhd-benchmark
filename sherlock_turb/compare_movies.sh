#!/bin/bash
# On Sherlock: bash -l compare_movies.sh [--submit] [--mach 30 50]
# Default prints a plan. --submit submits only missing dense-output runs.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE"
source jobs/common.sh
prepare_environment
prepare_codes
cd "$here"
python3 compare_movies.py "$@"
