#!/bin/bash
# Writes to stdout a tar.gz of what the Mac reads of the mini-ramses-ism runs: every record, diagnostics and spectra
# file, the native logs and namelists, and the last saved state of each planar run (at most 512^2 x 9 Float32, 9.4 MB;
# 3D states stay on Sherlock). Run without -l (a login shell's messages would corrupt the stream).
set -uo pipefail
cd "${MHD_RESULTS:-${SCRATCH:-/scratch/users/$USER}/mhd-results-sweep}" || exit 1
{
    find . -maxdepth 1 -type d -name '*-miniramses-*' | while read -r d; do
        find "$d" -maxdepth 1 -name '*.toml'
        find "$d/data" \( -name run.toml -o -name diagnostics.csv -o -name spectra.csv -o -name miniramses.log -o -name abel.nml \) 2>/dev/null
        for r in "$d"/data/miniramses_*_r1; do
            f="$(ls "$r"/state_*.bin 2>/dev/null | sort | tail -n 1)"
            [[ -n "$f" && $(wc -c < "$f") -le 9437184 ]] && echo "$f"
        done
    done
} | tar -czf - -T -
