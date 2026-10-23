#!/bin/bash
# Where the mini-ramses-ism runs stand: the jobs, the pilot's summary and log, and every run on scratch.
#   bash -l status.sh [DAYS]
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; days="${1:-3}"
results="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-sweep}"
echo "== jobs named miniramses in the last $days days"
sacct -X -n -P -u "$USER" --name=miniramses -S "now-${days}days" -o JobID,State,ExitCode,Elapsed,End 2>&1 | awk -F'|' '{ printf "   %-18s %-24s %-6s %-10s %s\n", $1, $2, $3, $4, $5 }' | tail -n 12
echo "== in the queue now"; squeue -h -u "$USER" -n miniramses -o "   %.18i %.9T %.10M %R" | head -n 8
echo "== pilot summary"; cat logs/pilot_summary.txt 2>/dev/null || echo "   (none written)"
pilot="$(ls -t logs/miniramses-*_0.out 2>/dev/null | head -n 1)"
echo "== end of the newest pilot log: ${pilot:-none}"; [[ -n "$pilot" ]] && grep -v '^ *$' "$pilot" | tail -n "${MINIRAMSES_STATUS_LINES:-25}"
echo "== runs: $(python3 summarize.py "$results" | awk '{ print $2 }' | sort | uniq -c | tr '\n' ' ')"
python3 summarize.py "$results" | grep -v ' complete ' | head -n "${MINIRAMSES_STATUS_RUNS:-20}"
