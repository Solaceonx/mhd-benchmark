#!/bin/bash
# Carry on from wherever check_build.sh was left (a dropped connection, for instance), without doing anything twice:
#   bash -l resume.sh
# Trial build running or speed job queued: nothing. Trial build interrupted or never run: check_build.sh again.
# Trial build passed but the job was not submitted: submit it. Trial build failed: nothing (status.sh shows why).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs; status="$here/logs/trial_status.txt"
queued() { [[ -n "$(squeue -h -u "$USER" -n "$1" 2>/dev/null)" ]]; }
if queued ramses2-check; then echo "== the trial build is still running: nothing to do"; exit 0; fi
if queued ramses2-speed; then echo "== the speed job is in the queue: nothing to do"; exit 0; fi
state=""; chosen=""; [[ -s "$status" ]] && read -r state chosen < "$status"
case "$state" in
    OK) since="$(date -d "@$(stat -c %Y "$status")" +%Y-%m-%dT%H:%M:%S)"
        if [[ -n "$(sacct -X -n -u "$USER" --name=ramses2-speed -S "$since" -o JobID 2>/dev/null | tr -d ' ')" ]]; then
            echo "== the speed job was submitted after the trial build passed, and has ended: nothing to do"
        else
            echo "== the trial build passed with $chosen but no speed job followed: submitting it"
            RAMSES2_NVHPC_MODULE="$chosen" sbatch speed.sbatch
        fi ;;
    FAILED) echo "== the trial build failed: nothing submitted" ;;
    *) echo "== the trial build did not finish (${state:-never started}): running it again"; bash -l "$here/check_build.sh" ;;
esac
