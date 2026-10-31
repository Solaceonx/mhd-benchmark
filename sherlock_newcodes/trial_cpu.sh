#!/bin/bash
# The trial of CPU codes (enzo, phantom) on CPU nodes, so that they do not wait for an H100; prints the job id:
#   bash -l trial_cpu.sh "CODES" [sbatch options...]
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
codes="${1:?usage: bash -l trial_cpu.sh \"CODES\" [sbatch options...]}"; shift
sed -e '/^#SBATCH -G /d' -e '/^#SBATCH --constraint=/d' -e 's/^#SBATCH --partition=.*/#SBATCH --partition=normal,owners/' trial.sbatch \
    | sbatch --parsable --export=ALL,NEWCODES_TRIAL="$codes",NEWCODES_TRIAL_DIR="$SCRATCH/mhd-results-newcodes-trial-cpu" "$@"
