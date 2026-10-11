#!/bin/bash
# State of the checks: the queue, the trial's verdict, and the table of runs so far.
#   bash -l status.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${CHECK_RESULTS:-$SCRATCH/mhd-results-checks}"
( cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_codes ) >/dev/null 2>&1
module load python/3.12.1 >/dev/null 2>&1 || true
echo "== queue"; squeue -u "$USER" -n mhd-checks -o "%.12i %.12j %.10T %.10M %.10l %R"
echo "== jobs that ended (sacct, last 3 days)"
sacct -u "$USER" --name=mhd-checks -S now-3days -X -n -o JobID%12,State%12,Elapsed,ExitCode | grep -v -E "PENDING|RUNNING" || true
log="$(ls -t logs/mhd-checks-*.out 2>/dev/null | head -n1)"
if [[ -n "$log" ]]; then
    echo "== $log"; grep -E "^== |^   N=|CONTROL|TRIAL|run directories|^START|RUN FAILED|ERROR|error:" "$log" | cut -c1-200 | tail -n 40
fi
[[ -d "$results" ]] && { echo "== runs ($(du -sh "$results" 2>/dev/null | cut -f1) on scratch)"; python3 checks.py summary "$results" "${MHD_RESULTS:-$SCRATCH/mhd-results-sweep}"; }
