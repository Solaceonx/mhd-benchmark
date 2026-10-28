#!/bin/bash
# Fetch the sources on this login node, submit the trial (trial.sbatch) and, to start only if it passes, the
# production runs (mhdale.sbatch: 13 tests, every grid): bash -l check.sh
#   bash -l check.sh trial         the trial only (it measures the speed at 256^2 and 512^2 before the runs are sized)
#   bash -l check.sh production [TRIAL_JOB]   the production runs only: after a trial that passed, or to start when the
#                                  queued trial TRIAL_JOB passes
# Production is sized by grid (the code has no restart; normal jobs end at 48 h and a user has at most 512 cores on
# normal at once; the old runs at 16 ranks took up to 21 h at 512^2, and Kelvin-Helmholtz and the current sheet
# would need 70 to 110 h): 32^2-128^2 in turn on 8 ranks (normal or owners, requeued), 256^2 on 16 ranks, 512^2 on 64
# ranks and on 128 for the two Kelvin-Helmholtz tests and the current sheet (normal only: a preempted run is lost).
# A trial job of the earlier bundle (for the code before T. Abel's fork) that is still queued or running is
# cancelled first: it would use these files for the old source.
# The runs of T. Abel's fork with our two fixes go to a results directory of their own (those of the fork before them,
# 2e900cd, stay in mhd-results-sweep).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs
mode="${1:-both}"; [[ "$mode" == both || "$mode" == trial || "$mode" == production ]] || { echo "usage: bash -l check.sh [trial|production [TRIAL_JOB]]"; exit 1; }
export MHD_RESULTS="${MHD_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-mhdale-fix}"
source "$here/options.sh"; count=$(echo $MHDALE_TESTS | wc -w)
production() {      # [dependency]: the four arrays
    local dep="${1:-}" long="kh_uniform kh_localized current_sheet" rest="" t
    for t in $MHDALE_TESTS; do [[ " $long " == *" $t "* ]] || rest="$rest $t"; done
    submit() {       # label, tests, grids, ranks (empty: 1, 4 or 8 by grid), cores, time, partition options...
        local label="$1" tests="$2" grids="$3" ranks="$4" cores="$5" time="$6"; shift 6
        local n id; n=$(echo $tests | wc -w | tr -d " ")
        id="$(MHDALE_ARRAY_TESTS="$tests" MHDALE_GRIDS="$grids" MHDALE_RANKS="$ranks" sbatch --parsable ${dep:+--dependency=afterok:$dep} \
              --array=1-"$n" --ntasks="$cores" --time="$time" "$@" mhdale.sbatch)" || { echo "   $label: NOT SUBMITTED"; return 1; }
        echo "   $label: array ${id%%;*}, $n tasks of $cores cores, $time"
    }
    echo "== production$( [[ -n "$dep" ]] && echo ", to start when trial $dep passes"), runs to $MHD_RESULTS"
    submit "32-128" "$MHDALE_TESTS" "32 64 128" "" 8 12:00:00 --partition=normal,owners --requeue
    submit "256" "$MHDALE_TESTS" "256" 16 16 24:00:00 --partition=normal --no-requeue
    submit "512, long" "$long" "512" 128 128 48:00:00 --partition=normal --no-requeue
    submit "512" "$rest" "512" 64 64 24:00:00 --partition=normal --no-requeue
    echo "Then: bash -l $here/status.sh"
}
if [[ "$mode" == production ]]; then
    squeue -h -u "$USER" -n mhdale -o %i | grep -q . && { echo "mhdale jobs are already queued: nothing submitted"; exit 1; }
    if [[ -n "${2:-}" ]]; then
        squeue -h -j "$2" -n mhdale-trial -o %i 2>/dev/null | grep -q . || { echo "trial job $2 is not queued: nothing submitted"; exit 1; }
    else
        grep -q "TRIAL PASSED" logs/trial_report.txt 2>/dev/null || { echo "no trial has passed (logs/trial_report.txt): nothing submitted"; exit 1; }
    fi
    production "${2:-}"; exit
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
production "$trial"; echo "The production tasks wait for the trial to pass; if it does not, Slurm keeps them pending (DependencyNeverSatisfied): scancel them."
echo "The trial takes about an hour. Then: bash -l $here/status.sh (runs go to $MHD_RESULTS)"
