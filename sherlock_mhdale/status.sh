#!/bin/bash
# Where MHD-ALE stands: the trial's report, the production tasks, the runs recorded so far: bash -l status.sh
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
results="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-sweep}"
echo "== mhdale jobs in the queue"; squeue -h -u "$USER" -o "   %.20i %.14j %.9T %.10M %R" | grep mhdale || echo "   none"
echo "== mhdale jobs of the last 3 days, by state"
sacct -X -n -P -u "$USER" -S now-3days -o JobID,JobName,State,ExitCode,Elapsed | grep "|mhdale" | awk -F'|' '{ split($1, a, "_"); n[a[1] " " $2 " " $3]++ } END { for (k in n) printf "   %4d  %s\n", n[k], k }' | sort -k2
if [[ -s logs/trial_report.txt ]]; then echo "== trial report"; cat logs/trial_report.txt; else echo "== no trial report yet"; fi
newest="$(ls -t logs/mhdale-trial-*.out 2>/dev/null | head -n 1)"
if [[ -n "$newest" ]] && ! grep -q "TRIAL" logs/trial_report.txt 2>/dev/null; then echo "== end of $newest"; tail -n "${MHDALE_STATUS_LINES:-25}" "$newest" | cut -c1-240; fi
count="$(ls -d "$results"/*-mhdale-ale-run-N*-s42 2>/dev/null | wc -l)"; echo "== production runs on scratch: $count directories"
if [[ "$count" -gt 0 ]]; then
    module load python/3.12.1 2>/dev/null
    python3 - "$results" <<'PY'
import sys, tomllib
from pathlib import Path
rows = []
for d in sorted(Path(sys.argv[1]).glob("*-mhdale-ale-run-N*-s42"), key=lambda p: (p.name.rsplit("-N", 1)[0], int(p.name.rsplit("-N", 1)[1].split("-")[0]))):
    r = d/"data"/"mhdale_ale_r1"/"run.toml"
    if not r.is_file():rows.append(f"   {d.name:46s} running or not started");continue
    m = tomllib.loads(r.read_text())
    rows.append(f"   {d.name:46s} {m.get('status', '?'):9s} t = {m.get('time_reached', 0):<8.4g} {m.get('steps', 0):7d} steps {m.get('solve_seconds', 0):9.1f} s  remaps {m.get('remaps', '-')}"
                + ("" if m.get("status") == "complete" else "   " + ((m.get("error") or "").splitlines() or [""])[-1][:140]))
print("\n".join(rows))
PY
    for log in $(ls -t logs/mhdale-*_*.out 2>/dev/null | head -n 2); do echo "== end of $log"; grep -v '^ *$' "$log" | tail -n 6 | cut -c1-220; done
fi
