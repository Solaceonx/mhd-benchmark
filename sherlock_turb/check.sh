#!/bin/bash
# Driven-turbulence pilot: build, try, submit (README.md). On a Sherlock login node:
#   bash -l sherlock_turb/check.sh
# 1. builds RAMSES with its driving module, 2. runs the whole RAMSES chain once on a 32^3 box, 3. submits what is
# left to do: the calibration (driving strength for each Mach number), then, to start when it is done, the GPU job
# and one job per RAMSES run. Safe to run again at any time: nothing built, calibrated, measured or queued is done
# twice, and a run that was cut short continues.
#   TURB_CONFIG           the set of runs: config.toml (default; Mach 3 and 10) or config_highmach.toml (Mach 30 and 50)
#   TURB_GPU_CONSTRAINT   GPU model the GPU job asks for (default: the sweep's, GPU_SKU:H100_SXM5; "none": any GPU)
#   TURB_SKIP_TRIAL=1     submit without the trial
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; mkdir -p "$here/logs"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
export TURB_RESULTS="${TURB_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-turb}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the environment of the sweep bundle could not be set up"; exit 2; }
load_modules "${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
cd "$here"; mkdir -p "$TURB_RESULTS"
export TURB_CONFIG="${TURB_CONFIG:-config.toml}"; [[ -f "$TURB_CONFIG" ]] || { echo "no configuration file $here/$TURB_CONFIG"; exit 2; }
echo "== configuration $TURB_CONFIG"
echo "== 1. RAMSES with its driving module: as released, and with the projection of the driving corrected"
bash build_ramses.sh released || exit 1
bash build_ramses.sh fixed || exit 1
echo "== 2. trial: 32^3, half a driving time, through RAMSES (corrected build), the conversion and the measurements"
if [[ "${TURB_SKIP_TRIAL:-0}" == 1 ]]; then echo "   skipped (TURB_SKIP_TRIAL=1)"
else bash trial.sh "$MHD_CODES/ramses/bin-turb-fixed/ramses_turb3d" fixed || { echo "nothing was submitted"; exit 1; }; fi
echo "== 3. jobs (results in $TURB_RESULTS)"
queued() { squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null | head -n1; }
submit() {      # name, then sbatch options and the script with its arguments; prints the job id
    local name="$1" id; shift
    id="$(sbatch --parsable --export=ALL,TURB_CONFIG="$TURB_CONFIG" --job-name="$name" "$@")" || { echo "   sbatch refused $name" >&2; return 1; }
    echo "${id%%;*}"
}
after=(); count=0
# fd 3 for the list, so that nothing in the loop can read it away
while read -r kind a b c d e f g h i <&3; do
    case "$kind" in
    calibrate)      # state ranks per_node time projection jobname
        if [[ "$a" == done ]]; then echo "   driving strengths: already in forcing.toml"
        else
            id="$(queued "$f")"
            if [[ -n "$id" ]]; then echo "   calibration: job $id is already in the queue"
            else id="$(submit "$f" --ntasks="$b" --ntasks-per-node="$c" --time="$d" ramses.sbatch calibrate "$e")" || exit 1
                echo "   calibration ($e projection): job $id ($b ranks, up to $d)"; count=$((count + 1)); fi
            after=(--dependency="afterok:$id")
        fi ;;
    gpu)            # state jobname
        if [[ "$a" == done ]]; then echo "   GPU runs: all there"
        elif [[ -n "$(queued "$b")" ]]; then echo "   GPU runs: job $(queued "$b") is already in the queue"
        else
            feature=(); want="${TURB_GPU_CONSTRAINT:-GPU_SKU:H100_SXM5}"; [[ "$want" == none ]] || feature=(--constraint="$want")
            id="$(submit "$b" ${after[@]+"${after[@]}"} ${feature[@]+"${feature[@]}"} gpu.sbatch)" || exit 1
            echo "   GPU runs: job $id (${want})"; count=$((count + 1))
        fi ;;
    run)            # variant mach n state ranks per_node time name projection
        name="turb-${h%-s*}"
        if [[ "$d" == done ]]; then echo "   $h: measured"
        elif [[ -n "$(queued "$name")" ]]; then echo "   $h: job $(queued "$name") is already in the queue"
        else
            id="$(submit "$name" ${after[@]+"${after[@]}"} --ntasks="$e" --ntasks-per-node="$f" --time="$g" ramses.sbatch run "$a" "$b" "$c" "$i")" || exit 1
            echo "   $h: job $id ($e ranks, up to $g)"; count=$((count + 1))
        fi ;;
    esac
done 3< <(python3 ramses_turb.py plan "$TURB_CONFIG" "$TURB_RESULTS")
echo "== $count jobs submitted. Progress and results:  bash -l $here/status.sh"
