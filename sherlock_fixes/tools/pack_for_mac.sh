#!/bin/bash
# Writes to stdout a tar.gz of what the Mac reads of the fixes: the tables, every run record and diagnostics file (and
# spectra) of mhd-results-{fixes,verify,emfcheck,rerun}, and the last saved state of each run when it is a planar one
# (at most 512^2 x 9 Float32, 9.4 MB; 3D states stay on Sherlock). Run without -l (a login shell's messages would
# corrupt the stream); the Mac command in README.md pipes it into tar.
set -uo pipefail
cd "${SCRATCH:-/scratch/users/$USER}" || exit 1
sets=(); for s in fixes verify emfcheck rerun; do [[ -d "mhd-results-$s" ]] && sets+=("mhd-results-$s"); done
{
    ls mhd-results-*/*.txt mhd-results-*/BRANCH.json 2>/dev/null
    find "${sets[@]}" -path '*/data/*' \( -name run.toml -o -name diagnostics.csv -o -name spectra.csv -o -name dense_diagnostics.csv \) 2>/dev/null
    for d in $(find "${sets[@]}" -maxdepth 3 -type d -name '*_r1' -path '*/data/*' 2>/dev/null); do
        f="$(ls "$d"/state_*.bin 2>/dev/null | sort | tail -n 1)"
        [[ -n "$f" && $(wc -c < "$f") -le 9437184 ]] && echo "$f"
    done
} | tar -czf - -T -
