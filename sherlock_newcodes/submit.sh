#!/bin/bash
# On a login node: fetch the sources (fetch.sh), then submit the build (build.sbatch) and the trial (trial.sbatch),
# which waits for the build to succeed. Production comes after the trial (it measures each code's speed first).
#   bash -l submit.sh            NEWCODES="athenapk cholla" bash -l submit.sh   (a subset)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
if squeue -h -u "$USER" -o %j | grep -q '^newcodes-'; then
    echo "newcodes jobs are already in the queue; nothing submitted:"; squeue -u "$USER" -o "%.12i %.16j %.9T %.10M %R" | grep -E 'JOBID|newcodes'; exit 0
fi
echo "== fetching sources"; bash "$here/fetch.sh" || { echo "FETCH FAILED: nothing submitted"; exit 1; }
build="$(sbatch --parsable --export=ALL build.sbatch)" || { echo "the build could not be submitted"; exit 1; }; build="${build%%;*}"
trial="$(sbatch --parsable --dependency=afterok:"$build" trial.sbatch)" || { echo "the trial could not be submitted"; exit 1; }
echo "build job $build, trial job ${trial%%;*} (waits for the build). Then: bash -l $here/status.sh"
