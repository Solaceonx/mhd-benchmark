#!/bin/bash
# Where the order x solver study stands: its jobs, the pilots' verdicts, and every run.
#   bash -l status.sh [DAYS]
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; days="${1:-4}"
module load python/3.12.1 >/dev/null 2>&1 || true         # order.py needs tomllib (Python 3.11+); the login node's python3 is 3.6
results="${ORDER_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-order}"
echo "== order-* jobs in the last $days days: tasks by name and state"
sacct -X -n -P -u "$USER" -S "now-${days}days" -o JobID,JobName,State | awk -F'|' '$2 ~ /^order-/ { n[$2 " " $3]++ } END { for (k in n) printf "   %4d  %s\n", n[k], k }' | sort -k2
echo "== in the queue now"; squeue -h -u "$USER" -o "   %.18i %.14j %.9T %.10M %R" | grep ' order-' | head -n 12
for kind in glm athenak athenapp; do
    log="$(ls -t logs/order-$kind-*_0.out 2>/dev/null | xargs -r grep -l 'pilot on' 2>/dev/null | head -n 1)"
    [[ -n "$log" ]] && echo "== $kind pilot ($log): $(grep -E 'PILOT (OK|FAILED)' "$log" | tail -n 1)"
    [[ -n "$log" ]] && grep -E '^WRONG' "$log" | head -n 5
done
echo "== runs"; python3 order.py progress "$results"
echo "== newest problems in the logs"
grep -hE 'exit code [1-9]|ERROR|Error|no AthenaK|no Athena\+\+|CANCELLED|TIME LIMIT' $(ls -t logs/order-*.out 2>/dev/null | head -n 20) 2>/dev/null | tail -n "${ORDER_STATUS_LINES:-12}"
