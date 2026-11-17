#!/bin/bash
# The MHD-ALE pilot of 2026-11-17 (Tom: "supposed to have low error at least for a fixed number of zones"): set-ups
# ale_l2 (projected initial state) and ale_l2_q4 (the same, degree 4/3 on elements four cells wide) on eight planar
# tests at 64^2, 128^2, 256^2 (kh_uniform to 128^2), compared with the published ale runs at the same grids.
#   bash -l pilot_l2.sh          (login node, in the deployed sherlock_mhdale)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
exe="/scratch/users/lyxia/mhd-codes-sweep/mhdale-yipihey/build-abel-l2/MHD"
results="$SCRATCH/mhd-results-mhdale-l2"
b="$(sbatch --parsable build_l2.sbatch)"; echo "build + smoke: job $b"
tests="entropy field_loop_novz brio_wu orszag_tang rotor blast_2d_b50 gem"
for v in ale_l2 ale_l2_q4; do
    a="$(sbatch --parsable --dependency=afterok:"$b" --array=1-7 --time=24:00:00 --job-name="mhdale-$v" \
        --export=ALL,MHDALE_VARIANT="$v",MHDALE_EXE="$exe",MHD_RESULTS="$results",MHDALE_ARRAY_TESTS="$tests",MHDALE_GRIDS="64 128 256" mhdale.sbatch)"
    k="$(sbatch --parsable --dependency=afterok:"$b" --array=1 --time=24:00:00 --job-name="mhdale-$v-kh" \
        --export=ALL,MHDALE_VARIANT="$v",MHDALE_EXE="$exe",MHD_RESULTS="$results",MHDALE_ARRAY_TESTS=kh_uniform,MHDALE_GRIDS="64 128" mhdale.sbatch)"
    echo "$v: array $a (7 tests), kh_uniform $k"
done
