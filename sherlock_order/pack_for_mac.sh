#!/bin/bash
# Writes to stdout a tar.gz of what the Mac reads of the order x solver study: every run's record, diagnostics, spectra
# and data/environment.toml (its launcher: the rank count), and the last saved state of each run when it is a plane
# (at most 512^2 x 9 Float32, 9.4 MB; 3D states stay on Sherlock, their errors are in run.toml). Run without -l (a
# login shell's messages would corrupt the stream); the Mac command in README.md pipes it into tar.
set -uo pipefail
cd "${SCRATCH:-/scratch/users/$USER}" || exit 1
set=mhd-results-order
{
    ls "$set"/BRANCH.json 2>/dev/null
    find "$set" -path '*/data/*' \( -name run.toml -o -name diagnostics.csv -o -name spectra.csv \) 2>/dev/null
    find "$set" -mindepth 3 -maxdepth 3 -path '*/data/environment.toml' 2>/dev/null
    for d in $(find "$set" -maxdepth 3 -type d -name '*_r1' -path '*/data/*' 2>/dev/null); do
        f="$(ls "$d"/state_*.bin 2>/dev/null | sort | tail -n 1)"
        [[ -n "$f" && $(wc -c < "$f") -le 9437184 ]] && echo "$f"
    done
} | tar -czf - -T -
