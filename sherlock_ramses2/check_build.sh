#!/bin/bash
# Try the RAMSES II build without a GPU, then submit the speed job if it works:
#   bash -l check_build.sh
# The H100 queue can take hours, so a build that cannot work should show here, not there. The trial runs on a node of
# the dev partition if one is free within two minutes, else on this login node (RAMSES2_CHECK_ON=login: always here).
# It goes to its own directory ($MHD_CODES/ramses2-build-check) and is not used by the job, which builds on its own
# node: the compiler tunes the code to the processor it runs on, and these nodes have other processors.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs; status="$here/logs/trial_status.txt"; rm -f "$status"
if [[ "${RAMSES2_CHECK_ON:-dev}" == dev ]]; then
    echo "== asking for a node of the dev partition (waits up to 2 minutes)"
    srun --partition=dev --cpus-per-task=4 --mem=12G --time=00:45:00 --immediate=120 --job-name=ramses2-check bash -l "$here/trial_build.sh" \
        || echo "   (srun ended with an error)"
fi
[[ -s "$status" ]] || { echo "== no dev node: building on $(hostname)"; bash -l "$here/trial_build.sh"; }
state=""; chosen=""; read -r state chosen < "$status" || true
[[ "$state" == OK && -n "$chosen" ]] || { echo "TRIAL BUILD ${state:-NOT RUN}: the speed job was not submitted"; exit 1; }
echo "TRIAL BUILD OK with $chosen"
if [[ -n "$(squeue -h -u "$USER" -n ramses2-speed 2>/dev/null)" ]]; then echo "a ramses2-speed job is already in the queue: not submitting another"; exit 0; fi
RAMSES2_NVHPC_MODULE="$chosen" sbatch speed.sbatch          # the job loads the same compiler
