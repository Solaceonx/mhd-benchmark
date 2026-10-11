#!/bin/bash
# Fetch the sources on this login node, submit the trial (trial.sbatch) and, to start only if it passes, the
# production runs (mhdale.sbatch: 13 tests, every grid): bash -l check.sh
# A trial job of the earlier bundle (for the code before T. Abel's fork) that is still queued or running is
# cancelled first: it would use these files for the old source.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
old="$(squeue -h -u "$USER" -n mhdale-trial,mhdale -o %i 2>/dev/null | tr '\n' ' ')"
if [[ -n "${old// /}" ]]; then echo "== cancelling earlier mhdale jobs: $old"; scancel $old; sleep 5; fi
echo "== fetching sources (hypre, METIS, gslib, the MFEM fork, T. Abel's fork of MHD-ALE)"
bash "$here/fetch.sh" || { echo "FETCH FAILED: nothing submitted"; exit 1; }
rm -f logs/trial_report.txt
trial="$(sbatch --parsable --partition=dev trial.sbatch 2> logs/sbatch_dev.err)" || trial=""
if [[ -z "$trial" ]]; then echo "   (dev partition refused the job: $(head -n 1 logs/sbatch_dev.err)); trying normal"; trial="$(sbatch --parsable --partition=normal,owners --requeue trial.sbatch)" || { echo "the trial could not be submitted"; exit 1; }; fi
trial="${trial%%;*}"; echo "Submitted the trial as job $trial"
source "$here/options.sh"; count=$(echo $MHDALE_TESTS | wc -w)
sbatch --dependency=afterok:"$trial" --array=1-"$count" mhdale.sbatch && echo "The $count production tasks wait for the trial to pass; if it does not, Slurm drops them."
echo "The trial takes about an hour (half of it building). Then: bash -l $here/status.sh"
