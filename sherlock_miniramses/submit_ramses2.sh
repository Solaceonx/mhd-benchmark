#!/bin/bash
# Get the official RAMSES II and submit the pilot and, depending on it, the tasks of runs_ramses2.json. Refuses while a
# job named ramses2 is still in the queue.   bash -l submit_ramses2.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
if squeue -h -u "$USER" -n ramses2 | grep -q .; then echo "ramses2 jobs are still in the queue: not submitting"; squeue -u "$USER" -n ramses2; exit 1; fi
bash -l fetch_ramses2.sh
mkdir -p logs
tasks="$(python3 -c "import json; print(len(json.load(open('runs_ramses2.json'))['tasks']))")"
p="$(sbatch --parsable --array=0 ramses2.sbatch)"
a="$(sbatch --parsable --dependency=afterok:"$p" --array=1-"$tasks" ramses2.sbatch)"
echo "pilot: job $p; tasks 1-$tasks: job $a (they start when the pilot has passed)"
