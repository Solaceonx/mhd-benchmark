#!/bin/bash
# Writes to stdout a tar.gz of what the Mac reads of the five new codes: every run's record, diagnostics, spectra,
# data/input.toml and data/environment.toml (its launcher: the rank count); the saved states and z-midplane slices up to
# 256^2 x 9 Float32 (2.4 MB: planar grids to 256, 3D states only at 32^3); and of each run the last state and slice up to
# 512^2 (9.4 MB). 3D states stay on Sherlock, their errors are in run.toml. Run it in a job, without -l (a login shell's
# messages would corrupt the stream):
#   sbatch --time=01:00:00 --wrap "bash pack_for_mac.sh > $SCRATCH/newcodes_for_mac.tgz"
set -uo pipefail
cd "${SCRATCH:-/scratch/users/$USER}" || exit 1
set=mhd-results-newcodes
{
    find "$set" -path '*/data/*' \( -name run.toml -o -name diagnostics.csv -o -name spectra.csv \) 2>/dev/null
    find "$set" -mindepth 3 -maxdepth 3 -path '*/data/*' \( -name environment.toml -o -name input.toml \) 2>/dev/null
    for d in $(find "$set" -maxdepth 3 -type d -name '*_r1' -path '*/data/*' 2>/dev/null); do
        for f in "$d"/state_*.bin "$d"/slice_*.bin; do
            [[ -f "$f" && $(stat -c %s "$f") -le 2654208 ]] && echo "$f"
        done
        for k in state slice; do
            f="$(ls "$d"/${k}_*.bin 2>/dev/null | sort | tail -n 1)"
            [[ -n "$f" && $(stat -c %s "$f") -le 9437184 ]] && echo "$f"
        done
    done
    true            # the list's exit status is not the archive's (pipefail)
} | sort -u | tar -czf - -T -
