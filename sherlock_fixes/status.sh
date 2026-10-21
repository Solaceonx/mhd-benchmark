#!/bin/bash
# State of the sweep-fixes runs: the queue, the log, and the table (the rerun: counts per kernel; its table is written
# to $SCRATCH/mhd-results-rerun/summary.txt).
#   bash -l status.sh [fixes|verify|emfcheck|rerun]      (default fixes)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
plan="${1:-fixes}"; case "$plan" in fixes|verify|emfcheck|rerun) ;; *) echo "usage: status.sh [fixes|verify|emfcheck|rerun]"; exit 1;; esac
results="$SCRATCH/mhd-results-$plan"
jobs="mhd-$plan"; [[ "$plan" == rerun ]] && jobs="mhd-rerun-glm,mhd-rerun-ct"
( cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_codes ) >/dev/null 2>&1
module load python/3.12.1 >/dev/null 2>&1 || true
echo "== queue"; squeue -u "$USER" -n "$jobs" -o "%.18i %.16j %.10T %.10M %.10l %R"
echo "== jobs that ended (sacct, last 3 days)"
sacct -u "$USER" --name="$jobs" -S now-3days -X -n -o JobID%18,JobName%16,State%12,Elapsed,ExitCode | grep -v -E "PENDING|RUNNING" || true
log="$(ls -t logs/mhd-$plan*.out 2>/dev/null | head -n1)"
if [[ -n "$log" ]]; then
    echo "== $log"; grep -E "^== |^   N=|^   kernels|TRIAL|run directories|runs$|^START|RUN FAILED|ERROR|error:" "$log" | cut -c1-200 | tail -n 25
fi
[[ -d "$results" ]] || exit 0
echo "== runs ($(du -sh "$results" 2>/dev/null | cut -f1) on scratch)"
if [[ "$plan" == rerun ]]; then
    python3 fixes.py --plan rerun progress "$results"
    python3 fixes.py --plan rerun summary "$results" "${MHD_RESULTS:-$SCRATCH/mhd-results-sweep}" > "$results/summary.txt" 2>&1 && echo "== table: $results/summary.txt"
else
    python3 fixes.py --plan "$plan" summary "$results" "${MHD_RESULTS:-$SCRATCH/mhd-results-sweep}"
fi
[[ "$plan" == verify ]] && { echo "== control: the originals on the branch against the sweep"; python3 fixes.py --plan verify control "$results" "${MHD_RESULTS:-$SCRATCH/mhd-results-sweep}"; }
true
