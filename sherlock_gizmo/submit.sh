#!/bin/bash
# Get the source and submit the pilot and, depending on it, the 34 tasks (17 planar tests x mfm, mfv). Refuses while a
# job named gizmo is still in the queue (a requeued or waiting task keeps its place; submitting again would run it twice).
#   bash -l submit.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
if squeue -h -u "$USER" -n gizmo | grep -q .; then echo "gizmo jobs are still in the queue: not submitting"; squeue -u "$USER" -n gizmo; exit 1; fi
bash fetch.sh
mkdir -p logs
p="$(sbatch --parsable --array=0 gizmo.sbatch)"
a="$(sbatch --parsable --dependency=afterok:"$p" --array=1-34 gizmo.sbatch)"
echo "pilot: job $p; tasks 1-34: job $a (they start when the pilot has passed)"
