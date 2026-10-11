#!/bin/bash
# Where the driven-turbulence pilot stands:  bash -l sherlock_turb/status.sh
# The queue, the driving strengths, a table of every run (summarize.py), and the end of any log that reports trouble.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
export TURB_RESULTS="${TURB_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-turb}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_codes > /dev/null 2>&1 || { echo "the environment of the sweep bundle could not be set up"; exit 2; }
cd "$here"
config="${TURB_CONFIG:-config.toml}"
echo "== queue"
squeue -u "$USER" -o "%.12i %.30j %.9T %.11M %.11l %.5C %R" 2>/dev/null | awk 'NR==1 || /turb-/' || echo "   (no answer from squeue)"
echo "== jobs that ended (sacct, last 3 days)"
sacct -u "$USER" -S now-3days -X -n -o JobID%12,JobName%30,State%14,Elapsed%11,ExitCode 2>/dev/null | awk '/turb-/ && !/RUNNING|PENDING/' | tail -n 30
echo "== driving strengths"
if [[ -s "$TURB_RESULTS/forcing.toml" ]]; then
    python3 - "$TURB_RESULTS/forcing.toml" <<'PY'
import sys, tomllib
for key, f in tomllib.load(open(sys.argv[1], "rb")).items():
    tries = ", ".join(f"{t['a_rms']:.4g} -> Mach {t['mach']:.2f}" for t in f.get("tries", []))
    print(f"   target Mach {f.get('mach_target', key)}: a_rms = {f['a_rms']:.6g} gave Mach {f.get('mach_measured', float('nan')):.3f}"
          f"{'' if f.get('converged', True) else '  (NOT within tolerance)'}   tries: {tries}")
PY
else echo "   not yet (the calibration job writes forcing.toml)"; grep -h "try\|FAILED\|killed" logs/turb-cal*.out 2>/dev/null | tail -n 12; fi
# Logs stay after their job: say when a line below comes from a job that a later one of the same name replaces.
waiting() { squeue -h -u "$USER" -n "$1" -o "%i %T" 2>/dev/null | head -n1; }
echo "== GPU kernels: their trial"
newest="$(ls -t logs/turb-gpu*.out 2>/dev/null | head -n1)"
if [[ -z "$newest" ]]; then echo "   no GPU job has run yet"
else
    id="${newest##*-}"; id="${id%.out}"; name="$(basename "$newest")"; name="${name%-*}"; later="$(waiting "$name")"
    if [[ -n "$later" && "${later%% *}" != "$id" ]]; then echo "   job ${later%% *} is ${later##* } and has written nothing yet; this is the EARLIER job $id:"
    else echo "   job $id ($name), the newest GPU job that ran:"; fi
    grep -h "^smoke\|^== driven" "$newest" | cut -c1-240 | tail -n 8
fi
echo "== runs"
python3 summarize.py "$TURB_RESULTS" --config "$config"
echo "== logs that report trouble"
found=0; trouble='Traceback\|ERROR\|FAILED\|UNMEASURED\|^failed \|^killed \|exit code [1-9]\|could not be set up\|CANCELLED\|TIME LIMIT'
seen=" "
for f in $(ls -t logs/*.out 2>/dev/null | head -n 40); do
    name="$(basename "$f")"; name="${name%-*}"; id="${f##*-}"; id="${id%.out}"
    [[ "$seen" == *" $name "* ]] && continue        # only the newest log of each job name
    seen="$seen$name "
    if grep -q "$trouble" "$f"; then
        found=1; later="$(waiting "$name")"; note=""
        [[ -n "$later" && "${later%% *}" != "$id" ]] && note="   (an earlier job: job ${later%% *} of this name is ${later##* })"
        echo "--- $f$note"; grep -n "$trouble" "$f" | head -n 6 | cut -c1-300; echo "    ..."; tail -n 6 "$f" | cut -c1-300
    fi
done
[[ $found -eq 1 ]] || echo "   none"
