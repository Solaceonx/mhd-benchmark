#!/bin/bash
# Where the new codes stand: their jobs, the build's results, the trial's report: bash -l status.sh
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
echo "== newcodes jobs in the queue"; squeue -h -u "$USER" -o "   %.14i %.16j %.9T %.10M %R" | grep newcodes || echo "   none"
b="$(ls -t logs/newcodes-build-*.out 2>/dev/null | head -n 1)"
if [[ -n "$b" ]]; then echo "== build ($b)"; grep -E "built in|BUILD FAILED|^   [0-9]+:|error" "$b" | head -n 40 | cut -c1-220; fi
for r in logs/trial_report*.txt; do [[ -s "$r" ]] && { echo "== $r"; cut -c1-220 "$r"; }; done
r="${NEWCODES_RESULTS:-$SCRATCH/mhd-results-newcodes}"
if [[ -d "$r" ]]; then
    echo "== production runs in $r, by code and set-up: complete / failed / directories"
    for d in $(ls -d "$r"/*-run-N*-s* 2>/dev/null | sed -E 's#.*/[a-z0-9_]+-([a-z]+-[a-z_0-9]+)-run-N.*#\1#' | sort -u); do
        c=$(ls "$r"/*-"$d"-run-N*/data/*_r1/run.toml 2>/dev/null | xargs -r grep -l '^status = "complete"' | wc -l)
        f=$(ls "$r"/*-"$d"-run-N*/data/*_r1/run.toml 2>/dev/null | xargs -r grep -l '^status = "failed"' | wc -l)
        echo "   $d: $c / $f / $(ls -d "$r"/*-"$d"-run-N* | wc -l)"
    done
fi
