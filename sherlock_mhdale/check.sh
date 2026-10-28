#!/bin/bash
# Fetch the sources on this login node, submit the trial (trial.sbatch) and, to start only if it passes, the
# production runs (mhdale.sbatch: 13 tests, every grid): bash -l check.sh
#   bash -l check.sh trial         the trial only (it measures the speed at 256^2 and 512^2 before the runs are sized)
#   bash -l check.sh production    the production runs only, after a trial that passed
# A trial job of the earlier bundle (for the code before T. Abel's fork) that is still queued or running is
# cancelled first: it would use these files for the old source.
# The runs of T. Abel's fork with our two fixes go to a results directory of their own (those of the fork before them,
# 2e900cd, stay in mhd-results-sweep).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
mode="${1:-both}"; [[ "$mode" == both || "$mode" == trial || "$mode" == production ]] || { echo "usage: bash -l check.sh [trial|production]"; exit 1; }
export MHD_RESULTS="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-mhdale-fix}"
source "$here/options.sh"; count=$(echo $MHDALE_TESTS | wc -w)
if [[ "$mode" == production ]]; then
    grep -q "TRIAL PASSED" logs/trial_report.txt 2>/dev/null || { echo "no trial has passed (logs/trial_report.txt): nothing submitted"; exit 1; }
    squeue -h -u "$USER" -n mhdale -o %i | grep -q . && { echo "mhdale jobs are already queued: nothing submitted"; exit 1; }
    sbatch --array=1-"$count" mhdale.sbatch && echo "Then: bash -l $here/status.sh (runs go to $MHD_RESULTS)"; exit
fi
old="$(squeue -h -u "$USER" -n mhdale-trial,mhdale -o %i 2>/dev/null | tr '\n' ' ')"
if [[ -n "${old// /}" ]]; then echo "== cancelling earlier mhdale jobs: $old"; scancel $old; sleep 5; fi
echo "== fetching sources (hypre, METIS, gslib, the MFEM fork, T. Abel's fork of MHD-ALE)"
bash "$here/fetch.sh" || { echo "FETCH FAILED: nothing submitted"; exit 1; }
rm -f logs/trial_report.txt
trial="$(sbatch --parsable --partition=dev trial.sbatch 2> logs/sbatch_dev.err)" || trial=""
if [[ -z "$trial" ]]; then echo "   (dev partition refused the job: $(head -n 1 logs/sbatch_dev.err)); trying normal"; trial="$(sbatch --parsable --partition=normal,owners --requeue trial.sbatch)" || { echo "the trial could not be submitted"; exit 1; }; fi
trial="${trial%%;*}"; echo "Submitted the trial as job $trial"
[[ "$mode" == trial ]] && { echo "The trial takes about an hour. Then: bash -l $here/status.sh; production: bash -l $here/check.sh production"; exit 0; }
sbatch --dependency=afterok:"$trial" --array=1-"$count" mhdale.sbatch && echo "The $count production tasks wait for the trial to pass; if it does not, Slurm drops them."
echo "The trial takes about an hour. Then: bash -l $here/status.sh (runs go to $MHD_RESULTS)"
