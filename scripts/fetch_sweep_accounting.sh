#!/bin/bash
# Run on lyxia's Mac. Read-only Slurm/status queries; submits and cancels no jobs.
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download="$(mktemp -d "${TMPDIR:-/tmp}/mhd-accounting.XXXXXX")"
trap 'rm -rf "$download"' EXIT
ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash -l -s' > "$download/report.tar" <<'REMOTE'
set -euo pipefail
cd /scratch/users/lyxia/mhd-study/sherlock_sweep
module load python/3.12.1
report="$(mktemp -d "${TMPDIR:-/tmp}/mhd-accounting.XXXXXX")"
trap 'rm -rf "$report"' EXIT
python3 tools/sweep.py runs --json "$report/export_latest.json" >&2
sacct -X --duplicates --array -u "$USER" -S 2026-09-01 -E now -P \
  --format=JobIDRaw%64,JobID%64,JobName%120,Account%40,Partition%80,State%40,AllocCPUS,ElapsedRaw,CPUTimeRAW,Submit,Start,End,AllocTRES%300 \
  > "$report/accounting.psv"
cp sweep_state/ledger.jsonl "$report/ledger.jsonl"
sshare -l -u "$USER" > "$report/fairshare.txt"
scontrol show config | awk '/PriorityDecayHalfLife|PriorityFlags|PriorityWeight/ {print}' > "$report/priority_config.txt"
python3 - "$report" <<'PY'
import datetime, json, os, pathlib, sys
pathlib.Path(sys.argv[1], 'query.json').write_text(json.dumps(dict(
    fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    since='2026-09-01', until='query time', user=os.environ['USER'],
    scope='User job allocations; no job steps; expanded arrays; includes requeue records',
    elapsed_scope='Whole recorded allocation duration for jobs overlapping the query window'), indent=2)+'\n')
PY
tar -cf - -C "$report" export_latest.json accounting.psv ledger.jsonl fairshare.txt priority_config.txt query.json
REMOTE
tar -xf "$download/report.tar" -C "$download"
destination="$project/progress/sweep_status"
mkdir -p "$destination/accounting"
cp "$download/export_latest.json" "$destination/export_latest.json"
for file in accounting.psv ledger.jsonl fairshare.txt priority_config.txt query.json; do
    cp "$download/$file" "$destination/accounting/$file"
done
printf 'Fetched current sweep status and Slurm allocation accounting. No jobs submitted.\n'
printf 'Saved in %s/accounting/\n' "$destination"
printf 'Allocations reported: '
awk 'END {print NR-1}' "$destination/accounting/accounting.psv"
