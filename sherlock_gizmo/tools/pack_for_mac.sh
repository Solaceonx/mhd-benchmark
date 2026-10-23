#!/bin/bash
# Writes to stdout a tar.gz of what the Mac reads of the GIZMO runs: every record, diagnostics file, GIZMO's own
# divergence table, log and parameter file, and the saved states (planar: at most 512^2 x 9 Float32, 9.4 MB each).
# Run without -l (a login shell's messages would corrupt the stream).
set -uo pipefail
cd "${MHD_RESULTS:-${SCRATCH:-/scratch/users/$USER}/mhd-results-sweep}" || exit 1
find . -maxdepth 1 -type d -name '*-gizmo-*' | while read -r d; do
    find "$d" -maxdepth 1 -name '*.toml'
    find "$d/data" \( -name run.toml -o -name diagnostics.csv -o -name gizmo_divergence.csv -o -name gizmo.log -o -name params.txt -o -name 'state_*.bin' \) 2>/dev/null
done | tar -czf - -T -
