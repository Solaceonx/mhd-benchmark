#!/bin/bash
# Run again the array tasks that stopped before all their runs were recorded, and two check runs on what stopped them:
#   bash -l rerun_failed.sh
# Tasks 9, 11, 17, 21 of job 46813282 (static mesh; field_loop, field_loop_novz, orszag_tang, current_sheet): AREPO
# ended its 256^2 run itself ("time-step of integer size 1 not allowed": its timestep had fallen to the smallest its
# time line holds), the driver took that for a job being stopped, recorded nothing, and 512^2 never ran. arepo_run.py
# now records such a run as failed, with the time it reached, and the task goes on to the next grid.
# The checks repeat Orszag-Tang 256^2 on the static mesh on one rank, and on 8 ranks with half the Courant factor, in
# a directory of their own ($SCRATCH/mhd-results-arepo-checks): does the collapse depend on the number of ranks, or
# on the timestep?
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
if [[ -n "$(squeue -h -u "$USER" -n arepo,arepo-check 2>/dev/null)" ]]; then echo "arepo jobs are still in the queue: nothing submitted"; exit 0; fi
sbatch --array="${AREPO_RERUN_TASKS:-9,11,17,21}" arepo.sbatch
checks="${SCRATCH:?SCRATCH is not set}/mhd-results-arepo-checks"
AREPO_ONLY="orszag_tang static 256" AREPO_RANKS=1 MHD_RESULTS="$checks/ranks1" \
    sbatch --job-name=arepo-check --array=101 --ntasks=1 --time=03:00:00 arepo.sbatch
AREPO_ONLY="orszag_tang static 256" AREPO_COURANT=0.15 MHD_RESULTS="$checks/courant015" \
    sbatch --job-name=arepo-check --array=102 --ntasks=8 --time=03:00:00 arepo.sbatch
