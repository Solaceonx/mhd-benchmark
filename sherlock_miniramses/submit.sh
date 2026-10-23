#!/bin/bash
# Get the source and submit the pilot and, depending on it, the three tasks of runs.json. Refuses while a job named
# miniramses is still in the queue (a requeued or waiting task keeps its place; submitting again would run it twice).
#   bash -l submit.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
if squeue -h -u "$USER" -n miniramses | grep -q .; then echo "miniramses jobs are still in the queue: not submitting"; squeue -u "$USER" -n miniramses; exit 1; fi
bash -l fetch.sh
mkdir -p logs
tasks="$(python3 -c "import json; print(len(json.load(open('runs.json'))['tasks']))")"
p="$(sbatch --parsable --array=0 miniramses.sbatch)"
a="$(sbatch --parsable --dependency=afterok:"$p" --array=1-"$tasks" miniramses.sbatch)"
echo "pilot: job $p; tasks 1-$tasks: job $a (they start when the pilot has passed)"
