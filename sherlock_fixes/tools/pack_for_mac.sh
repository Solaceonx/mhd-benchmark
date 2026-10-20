#!/bin/bash
# Writes to stdout a tar.gz of what the Mac reads of the fixes (scripts/fix_accuracy.py, the write-up): the tables,
# every run record and diagnostics file of mhd-results-fixes and mhd-results-verify, and the last saved state of the
# planar shock, leak and field-loop runs (a few hundred MB at most; no 3D states). Run without -l (a login shell's
# messages would corrupt the stream); the Mac command in README.md pipes it into tar.
set -uo pipefail
cd "${SCRATCH:-/scratch/users/$USER}" || exit 1
{
    ls mhd-results-fixes/*.txt mhd-results-verify/*.txt mhd-results-fixes/BRANCH.json 2>/dev/null
    find mhd-results-fixes mhd-results-verify -path '*/data/*' \( -name run.toml -o -name diagnostics.csv \) 2>/dev/null
    for d in mhd-results-fixes/*/data/*_r1 mhd-results-verify/*/data/*_r1; do
        case "$d" in
            */orszag_tang-N*|*/brio_wu-N*|*/brio_wu_rotated-N*|*/rotor-N*|*/kh_localized-N*|*/field_loop-N*|*/field_loop_novz-N*)
                ls "$d"/state_*.bin 2>/dev/null | sort | tail -n 1;;
        esac
    done
} | tar -czf - -T -
