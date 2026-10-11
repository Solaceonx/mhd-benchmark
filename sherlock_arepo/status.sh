#!/bin/bash
# Where the AREPO runs stand: Slurm's record of the jobs, the pilot's summary and log, the runs on scratch.
#   bash -l status.sh [DAYS]
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; days="${1:-3}"
results="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-sweep}"
echo "== jobs named arepo in the last $days days: pilots (task 0) and tasks that failed"
tmp="$(mktemp)"
sacct -X -n -P -u "$USER" --name=arepo -S "now-${days}days" -o JobID,State,ExitCode,Elapsed,End > "$tmp" 2>&1 || cat "$tmp"
awk -F'|' '$1 ~ /_0$/ || $2 !~ /^(COMPLETED|PENDING|RUNNING|CANCELLED)/ { printf "   %-18s %-28s %-6s %-10s %s\n", $1, $2, $3, $4, $5 }' "$tmp" | head -n 45
echo "== tasks by job and state"
awk -F'|' '{ split($1, a, "_"); n[a[1] " " $2]++ } END { for (k in n) printf "   %4d  %s\n", n[k], k }' "$tmp" | sort -k2
echo "== in the queue now"; squeue -h -u "$USER" -n arepo -o "   %.18i %.9T %.10M %R" | head -n 12
echo "== Slurm's default MPI plugin: $(scontrol show config 2>/dev/null | grep -i '^MpiDefault' | tr -s ' ')"
echo "== pilot summary"; cat logs/pilot_summary.txt 2>/dev/null || echo "   (none written)"
pilot="$(ls -t logs/arepo-*_0.out 2>/dev/null | head -n 1)"
echo "== end of the newest pilot log: ${pilot:-none}"; [[ -n "$pilot" ]] && tail -n "${AREPO_STATUS_LINES:-45}" "$pilot"
echo "== AREPO's own log, pilot runs (end)"
for d in "$results"/orszag_tang-arepo-*-run-N32-s42/data/arepo_*_r1; do
    [[ -f "$d/arepo.log" ]] || continue
    echo "-- ${d#$results/}: $(grep -m1 -i 'MPI' "$d/arepo.log" | cut -c1-90)"; grep -v '^ *$' "$d/arepo.log" | tail -n 6 | cut -c1-200
done
echo "== logs of the tasks that failed (newest ${AREPO_STATUS_TASKS:-6})"
for job in $(awk -F'|' '$1 !~ /_0$/ && $2 !~ /^(COMPLETED|PENDING|RUNNING|CANCELLED)/ { print $1 }' "$tmp" | sort -t_ -k1,1n -k2,2n | tail -n "${AREPO_STATUS_TASKS:-6}"); do
    [[ -f "logs/arepo-$job.out" ]] || { echo "-- $job: no log file"; continue; }
    echo "-- $job: $(grep -m1 '^== task' "logs/arepo-$job.out")"; grep -v '^ *$' "logs/arepo-$job.out" | tail -n 14 | cut -c1-220
done
echo "== runs without a finished record (first ${AREPO_STATUS_RUNS:-6}): AREPO's own log"
shown=0
for d in "$results"/*-arepo-*-run-N*-s42; do
    [[ -d "$d" ]] || continue
    grep -qs 'status = "complete"' "$d"/data/arepo_*_r1/run.toml && continue
    shown=$((shown + 1)); [[ $shown -le "${AREPO_STATUS_RUNS:-6}" ]] || continue
    echo "-- ${d#$results/}: $(ls "$d"/data/arepo_*_r1 2>/dev/null | tr '\n' ' ' | cut -c1-150)"
    for log in "$d"/data/arepo_*_r1/arepo.log; do
        [[ -f "$log" ]] || continue
        echo "   $(grep -m1 -i 'MPI tasks' "$log" | cut -c1-80); last sync point: $(grep 'Sync-Point' "$log" | tail -n 1 | cut -c1-70)"
        grep -n -m 6 -E 'TERMINATE|[Aa]bort|[Kk]illed|negative|srun: error|slurmstepd|Segmentation' "$log" | cut -c1-230
        echo "   .. the end:"; grep -v '^ *$' "$log" | tail -n 10 | cut -c1-230
    done
done
echo "   runs without a finished record: $shown"
count="$(ls -d "$results"/*-arepo-*-run-N*-s42 2>/dev/null | wc -l)"; echo "== run directories on scratch: $count (164 when every run has started)"
module load python/3.12.1 2>/dev/null
if [[ "$count" -gt 0 ]]; then
    python3 "$here/summarize.py" "$results" > "$tmp" 2>&1
    echo "   by status: $(awk 'NR > 1 { n[$2]++ } END { for (k in n) printf "%s %d   ", k, n[k] }' "$tmp")"
    head -n 1 "$tmp"; awk 'NR > 1 && $2 != "complete"' "$tmp" | head -n 25
fi
checks="$SCRATCH/mhd-results-arepo-checks"
if [[ -d "$checks" ]]; then
    echo "== check runs ($checks)"
    for c in "$checks"/*/; do
        [[ -d "$c" ]] || continue
        echo "-- $(basename "$c"): $(python3 "$here/summarize.py" "$c" 2>&1 | tail -n +2 | cut -c1-330)"
        for log in "$c"/*/data/arepo_*_r1/arepo.log; do
            [[ -f "$log" ]] && echo "   $(grep -m1 -i 'MPI tasks' "$log" | cut -c1-60); last: $(grep 'Sync-Point' "$log" | tail -n 1 | cut -c1-80)"
        done
    done
    squeue -h -u "$USER" -n arepo-check -o "   in the queue: %.18i %.9T %.10M %R" | head -n 4
fi
rm -f "$tmp"
