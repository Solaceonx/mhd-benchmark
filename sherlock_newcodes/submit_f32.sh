#!/bin/bash
# The GPU codes in single precision (lyxia, 2026-11-12): build (CPU node) -> pilot (one H100) -> production arrays.
#   bash -l submit_f32.sh        (login node, in the deployed sherlock_newcodes)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs tasks
# Cholla is not in: its source stops a PRECISION=1 build ("Single precision is not currently supported", job 52114552).
# A build that is already there is kept (NEWCODES_F32_BUILD chooses what to build again).
b="$(sbatch --parsable --export=ALL,NEWCODES="${NEWCODES_F32_BUILD:-athenak-f32 athenapk-f32}" build.sbatch)"
p="$(sbatch --parsable --dependency=afterok:"$b" pilot_f32.sbatch)"
echo "build: job $b; pilot: job $p (after the build)"
bash -l production.sh "$p" athenakf32 athenapkf32
