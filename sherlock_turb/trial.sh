#!/bin/bash
# The RAMSES chain once on a 32^3 box before anything is queued: RAMSES with its driving (as an MPI program), the
# conversion of its output, and the measurements in Julia.   bash trial.sh EXECUTABLE [released|fixed]
# RAMSES runs on a node of the dev partition if one is free within two minutes, otherwise as one process on this
# node; the conversion and the measurements run here. Ends with TRIAL OK or says what went wrong.
set -uo pipefail
exe="$1"; projection="${2:-released}"; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prefix=ramses; [[ "$projection" == fixed ]] && prefix=ramsesfix
results="${TURB_RESULTS:?TURB_RESULTS is not set}"; trial="$results/trial"; run="$trial/$prefix-matched-M3-N32-s1"
attempt() {     # launcher
    rm -rf "$trial"; mkdir -p "$trial"; printf '[mach_3]\na_rms = 16.0\n' > "$trial/forcing.toml"
    python3 "$here/ramses_turb.py" run "$here/trial_config.toml" "$trial" matched 3 32 --projection "$projection" --exe "$exe" --mpi "$1" < /dev/null
}
verdict() {
    python3 - "$run" <<'EOF'
import csv, sys, tomllib
from pathlib import Path
run = Path(sys.argv[1])
try:
    meta = tomllib.loads((run/"run.toml").read_text());rows = list(csv.DictReader((run/"series.csv").open()))
except OSError as err:
    print(f"   no result: {err}");sys.exit(1)
if meta.get("status") != "complete" or not meta.get("measured") or len(rows) != 3:
    print(f"   status {meta.get('status')}, measured {meta.get('measured')}, {len(rows)} of 3 frames; {meta.get('reason', '')[:500]}");sys.exit(1)
last = rows[-1];mach = float(last["mach_vol"]);mass = float(last["mass"])
ok = 0.5 < mach < 3.5 and abs(mass - 1) < 1e-6 and last["finite"] == "true" and all((run/f).is_file() for f in ("spectra.csv", "pdf.csv", "state_final.bin"))
print(f"   {meta['device']}: Mach {mach:.3f} after half a driving time (about 2.0 in the same run on a laptop), mean density {mass:.8f}, "
      f"sound speed squared between {float(last['cs2_min']):.6f} and {float(last['cs2_max']):.6f}")
sys.exit(0 if ok else 1)
EOF
}
echo "   RAMSES on a node of the dev partition (waits up to two minutes for one)"
attempt "srun --partition=dev --ntasks=4 --cpus-per-task=1 --mem-per-cpu=2G --time=00:10:00 --immediate=120 --job-name=turb-trial"
if verdict; then echo "TRIAL OK"; exit 0; fi
echo "   no good result that way; the end of its log:"; tail -n 8 "$run/ramses.log" 2>/dev/null | cut -c1-200
echo "   RAMSES as one process on $(hostname)"
attempt ""
if verdict; then echo "TRIAL OK (one process here; the MPI launch is first tried by the calibration job)"; exit 0; fi
echo "   the end of its log:"; tail -n 15 "$run/ramses.log" 2>/dev/null | cut -c1-200
echo "TRIAL FAILED"; exit 1
