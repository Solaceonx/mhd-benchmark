#!/bin/bash
# RAMSES II (official, ramses2 develop 04abdd9, the build of the RAMSES II campaign's source) in the driven-turbulence
# set-up (README.md, "RAMSES II"): UCT-HLLD, double precision, Mach 3 and 10 (config.toml) and 30 and 50
# (config_highmach.toml), 128^3 and 256^3, seeds 1 and 2: 16 runs, one after another on one H100 (ramses2_turb.py).
# First the build with the driving module (sherlock_miniramses/build.sh, MINIRAMSES_TURB=1) and a 32^3 trial of one
# driving time with UCT-HLLD; the runs only start if the trial completed and was measured. A run already measured is
# kept, so the job can simply run again (requeued, or after a failure).        cd sherlock_turb && sbatch ramses2.sbatch
#SBATCH --job-name=turb-ramses2
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${TURB_RESULTS:-$SCRATCH/mhd-results-turb}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
cd "$here" || exit 2
[[ -s "$results/forcing.toml" ]] || { echo "no $results/forcing.toml: the driving strengths come from the pilot's calibration"; exit 2; }
module load devel 2>/dev/null || true          # the newer nvhpc modules are visible only with devel (job 49282778)
source "$here/../sherlock_miniramses/compiler.sh"
export MINIRAMSES_SRC="${MINIRAMSES_SRC:-$MHD_CODES/ramses2-04abdd9}" MINIRAMSES_NSUBGRID=2
export MINIRAMSES_BUILD_ROOT="${MINIRAMSES_BUILD_ROOT:-$MHD_CODES/ramses2-04abdd9-build}"
use_nvhpc "$(cat "$MINIRAMSES_BUILD_ROOT/nvhpc_module.txt")" || { echo "the RAMSES II campaign's compiler module did not load"; exit 2; }
MINIRAMSES_TURB=1 bash "$here/../sherlock_miniramses/build.sh" 3 8 || { echo "BUILD FAILED"; exit 1; }
exe="$MINIRAMSES_BUILD_ROOT/nd3_p8_turb/bin/ramses3d"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
export TURB_DEVICE="$(hostname) $gpu" TURB_PRECISION="double (NPRE=8); outputs Float32"
echo "== $(hostname), $gpu, $(date -u); source $(cd "$MINIRAMSES_SRC" && git rev-parse --short HEAD); results $results"

# Trial: 32^3, Mach 3, one driving time, UCT-HLLD on the GPU, converted and measured.
trial="$results/smoke-ramses2"; mkdir -p "$trial"; cp "$results/forcing.toml" "$trial/"
if ! grep -q 'measured = true' "$trial/smoke-uct/run.toml" 2>/dev/null; then
    python3 ramses2_turb.py run config.toml "$trial" 3 32 1 --exe "$exe" --tend-T 1 --name smoke-uct < /dev/null; rc=$?
    [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
fi
python3 - "$trial/smoke-uct" <<'PY' || { echo "TRIAL FAILED: nothing else was run (see $trial/smoke-uct/ramses2.log)"; exit 1; }
import csv, sys, tomllib
from pathlib import Path
d = Path(sys.argv[1]);m = tomllib.loads((d/"run.toml").read_text())
rows = list(csv.DictReader(open(d/"series.csv")))
mach = float(rows[-1]["mach_vol"]);print(f"   trial: {m['status']}, measured {m.get('measured')}, {len(rows)} frames, Mach {mach:.2f} at 1 T, {m['steps']} steps")
sys.exit(0 if m["status"] == "complete" and m.get("measured") and mach > 1.0 else 1)
PY

for config in config.toml config_highmach.toml; do
    while read -r kind mach n seed state name <&3; do
        [[ "$kind" == run ]] || continue
        if [[ "$state" == done ]]; then echo "   $name: measured"; continue; fi
        echo "== $name ($config), $(date -u)"
        python3 ramses2_turb.py run "$config" "$results" "$mach" "$n" "$seed" --exe "$exe" < /dev/null; rc=$?
        [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
        [[ $rc -ne 0 ]] && echo "   $name: exit code $rc"
    done 3< <(python3 ramses2_turb.py plan "$config" "$results")
done
python3 summarize.py "$results" --config config.toml 2>/dev/null | grep -E "^ *(run|ramses2-)" || true
echo "== done, $(date -u)"
