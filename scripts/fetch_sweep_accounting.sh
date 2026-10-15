#!/bin/bash
# Run on lyxia's Mac. Read-only Slurm/status queries; submits and cancels no jobs.
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download="$(mktemp -d "${TMPDIR:-/tmp}/mhd-accounting.XXXXXX")"
trap 'rm -rf "$download"' EXIT
printf 'Fetching sweep status and allocation accounting from Sherlock…\n'
fetch_status=0
ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash -l -s' > "$download/report.tar" <<'REMOTE' || fetch_status=$?
set -euo pipefail
exec 3>&1
cd /scratch/users/lyxia/mhd-study/sherlock_sweep
module load python/3.12.1
report="$(mktemp -d "${TMPDIR:-/tmp}/mhd-accounting.XXXXXX")"
stage='sweep status export'
finish_report() {
    local status=$?
    trap - EXIT
    if [ "$status" -ne 0 ]; then
        printf 'Accounting fetch failed during %s (exit %s). Returning available files.\n' "$stage" "$status" >&2
    fi
    tar -cf - -C "$report" . >&3 || status=$?
    rm -rf "$report"
    exit "$status"
}
trap finish_report EXIT
capture_query() {
    local output=$1 status
    shift
    # A login-shell wrapper may call exit, or print its diagnostic on stdout.
    # Isolate it and keep the archive on its own descriptor in either case.
    if (trap - EXIT; "$@" </dev/null) > "$output" 2> "$output.stderr"; then
        cat "$output.stderr" >&2
    else
        status=$?
        printf '%s failed (exit %s). Query diagnostics:\n' "$1" "$status" >&2
        sed -n '1,25p' "$output.stderr" >&2
        sed -n '1,25p' "$output" >&2
        return "$status"
    fi
}
printf '[1/5] Exporting sweep status…\n' >&2
python3 tools/sweep.py runs --json "$report/export_latest.json" </dev/null >&2
stage='sacct allocation query'
printf '[2/5] Querying sacct allocations since September 1…\n' >&2
printf 'sacct command type: %s\n' "$(type -t sacct)" >&2
query_window() {
    local start=$1 end=$2 chunk status=0 midpoint
    chunk="$(mktemp "$report/window.XXXXXX")"
    printf '  Accounting window %s to %s\n' "$start" "$end" >&2
    capture_query "$chunk" sacct -X --duplicates --array -u "$USER" -S "$start" -E "$end" -P \
      --format=JobIDRaw%64,JobID%64,JobName%120,Account%40,Partition%80,State%40,AllocCPUS,ElapsedRaw,CPUTimeRAW,Submit,Start,End,AllocTRES%300 || status=$?
    # Sherlock may print this error on stdout while returning exit 0.
    if grep -qi 'Too wide of a date range' "$chunk" "$chunk.stderr"; then
        midpoint="$(python3 - "$start" "$end" <<'PY'
from datetime import datetime, timedelta
import sys
a, b = map(datetime.fromisoformat, sys.argv[1:])
if b-a <= timedelta(hours=1):
    raise SystemExit('sacct refuses even a one-hour window; no accounting report was accepted')
print((a+(b-a)/2).replace(microsecond=0).isoformat())
PY
)"
        rm -f "$chunk" "$chunk.stderr"
        printf '  Date range refused; splitting this window.\n' >&2
        query_window "$start" "$midpoint"
        query_window "$midpoint" "$end"
        return
    fi
    if [ "$status" -ne 0 ]; then return "$status"; fi
    python3 - "$chunk" <<'PY'
import csv, sys
required = 'JobIDRaw JobID JobName Account Partition State AllocCPUS ElapsedRaw CPUTimeRAW Submit Start End AllocTRES'.split()
with open(sys.argv[1], newline='') as stream:
    reader = csv.DictReader(stream, delimiter='|')
    if reader.fieldnames != required:
        raise SystemExit('Invalid sacct header; accounting report rejected')
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise SystemExit('Invalid sacct record: '+str(next(iter(row.values()), '')))
PY
    if [ -s "$report/accounting.psv" ]; then
        tail -n +2 "$chunk" >> "$report/accounting.psv"
    else
        cat "$chunk" > "$report/accounting.psv"
    fi
    rm -f "$chunk" "$chunk.stderr"
}
python3 - > "$report/windows.txt" <<'PY'
from datetime import datetime, timedelta
a, end = datetime(2026, 9, 1), datetime.now().replace(microsecond=0)
while a < end:
    b = min(a+timedelta(days=7), end)
    print(a.isoformat(), b.isoformat())
    a = b
PY
while read -r start end; do query_window "$start" "$end"; done < "$report/windows.txt"
rm -f "$report/windows.txt"
stage='sweep ledger copy'
printf '[3/5] Copying sweep ledger…\n' >&2
cp sweep_state/ledger.jsonl "$report/ledger.jsonl"
stage='fairshare query'
printf '[4/5] Querying fairshare…\n' >&2
capture_query "$report/fairshare.txt" sshare -l -u "$USER" || printf 'Fairshare unavailable; allocation accounting is unaffected.\n' >&2
stage='priority configuration query'
printf '[5/5] Querying priority configuration…\n' >&2
if capture_query "$report/priority_config.raw" scontrol show config; then
    awk '/PriorityDecayHalfLife|PriorityFlags|PriorityWeight/ {print}' "$report/priority_config.raw" > "$report/priority_config.txt"
else
    printf 'Priority configuration unavailable; allocation accounting is unaffected.\n' >&2
    : > "$report/priority_config.txt"
fi
stage='query metadata'
python3 - "$report" <<'PY'
import datetime, json, os, pathlib, sys
pathlib.Path(sys.argv[1], 'query.json').write_text(json.dumps(dict(
    fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    since='2026-09-01', until='query time', user=os.environ['USER'],
    scope='User job allocations; no job steps; expanded arrays; includes requeue records',
    window_days=7, subdivide_on_range_limit=True,
    elapsed_scope='Whole recorded allocation duration for jobs overlapping the query window'), indent=2)+'\n')
PY
REMOTE
if ! tar -xf "$download/report.tar" -C "$download"; then
    printf 'No readable report was received (SSH exit %s). See the error above.\n' "$fetch_status" >&2
    exit 1
fi
destination="$project/progress/sweep_status"
mkdir -p "$destination"
if [ -s "$download/export_latest.json" ]; then
    cp "$download/export_latest.json" "$destination/export_latest.json"
    printf 'Saved fresh sweep export in %s/export_latest.json\n' "$destination"
fi
if [ "$fetch_status" -ne 0 ] || [ ! -s "$download/query.json" ]; then
    printf 'Allocation accounting is incomplete; previous accounting files were kept. Paste the errors above.\n' >&2
    exit 1
fi
for file in accounting.psv ledger.jsonl fairshare.txt priority_config.txt query.json; do
    if [ ! -f "$download/$file" ]; then
        printf 'Report is missing %s; previous accounting files were kept.\n' "$file" >&2
        exit 1
    fi
done
mkdir -p "$destination/accounting"
for file in accounting.psv ledger.jsonl fairshare.txt priority_config.txt query.json; do
    cp "$download/$file" "$destination/accounting/$file"
done
printf 'Fetched current sweep status and Slurm allocation accounting. No jobs submitted.\n'
printf 'Saved in %s/accounting/\n' "$destination"
printf 'Allocations reported: '
awk 'END {print NR-1}' "$destination/accounting/accounting.psv"
printf 'The website still needs a local rebuild and publish after this download.\n'
