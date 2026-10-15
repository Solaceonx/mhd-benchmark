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
    tar -cf - -C "$report" . || status=$?
    rm -rf "$report"
    exit "$status"
}
trap finish_report EXIT
printf '[1/5] Exporting sweep status…\n' >&2
python3 tools/sweep.py runs --json "$report/export_latest.json" >&2
stage='sacct allocation query'
printf '[2/5] Querying sacct allocations since September 1…\n' >&2
sacct -X --duplicates --array -u "$USER" -S 2026-09-01 -E now -P \
  --format=JobIDRaw%64,JobID%64,JobName%120,Account%40,Partition%80,State%40,AllocCPUS,ElapsedRaw,CPUTimeRAW,Submit,Start,End,AllocTRES%300 \
  > "$report/accounting.psv"
stage='sweep ledger copy'
printf '[3/5] Copying sweep ledger…\n' >&2
cp sweep_state/ledger.jsonl "$report/ledger.jsonl"
stage='fairshare query'
printf '[4/5] Querying fairshare…\n' >&2
sshare -l -u "$USER" > "$report/fairshare.txt"
stage='priority configuration query'
printf '[5/5] Querying priority configuration…\n' >&2
scontrol show config | awk '/PriorityDecayHalfLife|PriorityFlags|PriorityWeight/ {print}' > "$report/priority_config.txt"
stage='query metadata'
python3 - "$report" <<'PY'
import datetime, json, os, pathlib, sys
pathlib.Path(sys.argv[1], 'query.json').write_text(json.dumps(dict(
    fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    since='2026-09-01', until='query time', user=os.environ['USER'],
    scope='User job allocations; no job steps; expanded arrays; includes requeue records',
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
