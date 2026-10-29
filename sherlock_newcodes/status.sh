#!/bin/bash
# Where the new codes stand: their jobs, the build's results, the trial's report: bash -l status.sh
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
echo "== newcodes jobs in the queue"; squeue -h -u "$USER" -o "   %.14i %.16j %.9T %.10M %R" | grep newcodes || echo "   none"
b="$(ls -t logs/newcodes-build-*.out 2>/dev/null | head -n 1)"
if [[ -n "$b" ]]; then echo "== build ($b)"; grep -E "built in|BUILD FAILED|^   [0-9]+:|error" "$b" | head -n 40 | cut -c1-220; fi
if [[ -s logs/trial_report.txt ]]; then echo "== trial report"; cut -c1-220 logs/trial_report.txt; fi
