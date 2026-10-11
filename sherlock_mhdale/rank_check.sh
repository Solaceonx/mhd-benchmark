#!/bin/bash
# Does the mean-field check of the fork fail with the grid or with the number of MPI ranks?  bash -l rank_check.sh
# In the production runs every test with a mean field passed at 128^2 (8 ranks, error 1e-14) and stopped at 256^2 and
# 512^2 (16 ranks, error 1e-8 to 6e-6; "Mean-field potential decomposition failed its native flux check"). Two short
# runs of Brio-Wu separate the two: 256^2 on 8 ranks, and 128^2 on 16 ranks. Results go to a directory of their own.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
checks="${SCRATCH:?SCRATCH is not set}/mhd-results-mhdale-checks"
MHDALE_ONLY=brio_wu MHDALE_GRIDS=256 MHDALE_RANKS=8 MHD_RESULTS="$checks/n256_ranks8" \
    sbatch --job-name=mhdale-check --array=1 --time=02:00:00 mhdale.sbatch
MHDALE_ONLY=brio_wu MHDALE_GRIDS=128 MHDALE_RANKS=16 MHD_RESULTS="$checks/n128_ranks16" \
    sbatch --job-name=mhdale-check --array=1 --time=02:00:00 mhdale.sbatch
echo "Afterwards: bash -l $here/rank_check_status.sh"
