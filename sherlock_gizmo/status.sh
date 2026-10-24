#!/bin/bash
# Where the GIZMO runs stand: the jobs, the pilot's summary and log, and every run on scratch.
#   bash -l status.sh [DAYS]
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; days="${1:-3}"
module load python/3.12.1 >/dev/null 2>&1 || true         # summarize.py needs tomllib (Python 3.11+); the login node's python3 is 3.6
results="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-sweep}"
echo "== jobs named gizmo in the last $days days: pilots (task 0) and tasks that did not complete"
tmp="$(mktemp)"
sacct -X -n -P -u "$USER" --name=gizmo -S "now-${days}days" -o JobID,State,ExitCode,Elapsed,End > "$tmp" 2>&1 || cat "$tmp"
awk -F'|' '$1 ~ /_0$/ || $2 !~ /^(COMPLETED|PENDING|RUNNING)/ { printf "   %-18s %-28s %-6s %-10s %s\n", $1, $2, $3, $4, $5 }' "$tmp" | head -n 30
echo "== tasks by job and state"; awk -F'|' '{ split($1, a, "_"); n[a[1] " " $2]++ } END { for (k in n) printf "   %4d  %s\n", n[k], k }' "$tmp" | sort -k2
echo "== in the queue now"; squeue -h -u "$USER" -n gizmo -o "   %.18i %.9T %.10M %R" | head -n 10
echo "== pilot summary"; cat logs/pilot_summary.txt 2>/dev/null || echo "   (none written)"
pilot="$(ls -t logs/gizmo-*_0.out 2>/dev/null | head -n 1)"
echo "== end of the newest pilot log: ${pilot:-none}"; [[ -n "$pilot" ]] && grep -v '^ *$' "$pilot" | tail -n "${GIZMO_STATUS_LINES:-25}"
echo "== GIZMO's own log, pilot runs (end)"
for d in "$results"/orszag_tang-gizmo-*-run-N32-s42/data/gizmo_*_r1; do
    [[ -f "$d/gizmo.log" ]] && { echo "-- ${d#$results/}"; grep -v '^ *$' "$d/gizmo.log" | tail -n 5 | cut -c1-200; }
done
echo "== runs: $(python3 summarize.py "$results" | awk '{ print $2 }' | sort | uniq -c | tr '\n' ' ')"
python3 summarize.py "$results" | grep -v ' complete ' | head -n "${GIZMO_STATUS_RUNS:-20}"
rm -f "$tmp"
