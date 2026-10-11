#!/bin/bash
# Where the RAMSES II work stands: the trial build, the speed job, its summary.
#   bash -l status.sh
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
codes="${MHD_CODES:-${SCRATCH:?SCRATCH is not set}/mhd-codes-sweep}"; results="${RAMSES2_RESULTS:-$SCRATCH/mhd-results-ramses2}"
check="$codes/ramses2-build-check/ot_ns1_p8"
echo "== trial build: $(cat logs/trial_status.txt 2>/dev/null || echo 'not run') ($(ls "$check"/bin/*.o 2>/dev/null | wc -l) objects compiled, about 180 in all)"
ls "$check"/retry_*.log >/dev/null 2>&1 && grep -H -c "Internal compiler error" "$check"/retry_*.log | sed "s|$check/||"
[[ -f "$check/BUILD_INFO.txt" ]] && sed -n 4,5p "$check/BUILD_INFO.txt"
if [[ "$(cut -d' ' -f1 logs/trial_status.txt 2>/dev/null)" == FAILED ]]; then
    grep -n -i "error\|fatal\|undefined\|killed" "$check/build.log" | head -n 12 | cut -c1-220; tail -n 6 "$check/build.log" | cut -c1-220
fi
echo "== jobs"; squeue -h -u "$USER" -o "   %.18i %.16j %.9T %.10M %.20S %R" | grep -E "ramses2" || echo "   none in the queue"
sacct -X -n -u "$USER" -S now-3days -o JobID%14,JobName%16,State%22,ExitCode,Elapsed,End | grep ramses2
if [[ -f "$results/summary.txt" ]]; then echo "== summary of $(date -u -r "$results/summary.txt" +%Y-%m-%dT%H:%MZ)"; cat "$results/summary.txt"; fi
newest="$(ls -t logs/ramses2-speed-*.out 2>/dev/null | head -n 1)"
[[ -n "$newest" ]] && { echo "== $newest (lines that matter)"; grep -E "^== |^   (ran|skipping|done already|[a-z_0-9]+\.o:)|built |BUILD FAILED|NOTE|NO COMPILER" "$newest" | tail -n 45 | cut -c1-200; }
