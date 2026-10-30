#!/bin/bash
# Submit the production runs of the given codes (login node), to start when trial job TRIAL passes:
#   bash -l production.sh TRIAL CODE ...        (TRIAL "none": no dependency, for codes whose trial already passed)
# GPU codes: one H100 per (set-up, test, seed), all its grids in turn (gpu,owners, requeued; finished runs are kept).
# Enzo: planar tests on 16 ranks; 3D tests to 256^3 on 32 ranks, 512^3 on 128 (normal: a preempted run is lost).
# Phantom: 32 OpenMP threads per task; 3D tests to 256^3. Results: $SCRATCH/mhd-results-newcodes.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs tasks
trial="${1:?usage: bash -l production.sh TRIAL CODE ...}"; shift
dep=(); [[ "$trial" == none ]] || dep=(--dependency=afterok:"$trial")
module load python/3.12.1 >/dev/null 2>&1
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
submit() {      # label, tasks file, sbatch options...
    local label="$1" file="$2"; shift 2; local n; n=$(wc -l < "$file" | tr -d ' ')
    [[ $n -gt 0 ]] || return 0
    local id; id="$(sbatch --parsable ${dep[@]+"${dep[@]}"} --array=1-"$n" --export=ALL,NEWCODES_TASKS="$here/$file" "$@" production.sbatch)" \
        && echo "   $label: array ${id%%;*}, $n tasks" || echo "   $label: NOT SUBMITTED"
}
gpu=(--partition=gpu,owners --requeue -G 1 --constraint=GPU_SKU:H100_SXM5 --ntasks=1 --cpus-per-task=4 --mem=64G --time=12:00:00)
for code in "$@"; do
    case $code in
        athenapk|cholla) python3 tasks.py "$code" > "tasks/$code.txt" && submit "$code" "tasks/$code.txt" "${gpu[@]}" ;;
        pluto) for d in ct glm; do python3 tasks.py pluto ${d}_matched ${d}_best > "tasks/pluto_$d.txt" && submit "pluto $d" "tasks/pluto_$d.txt" "${gpu[@]}"; done ;;
        enzo) NEWCODES_DIM=planar python3 tasks.py enzo > tasks/enzo_2d.txt && submit "enzo 2D" tasks/enzo_2d.txt --partition=normal,owners --requeue --ntasks=16 --mem-per-cpu=4G --time=24:00:00
              NEWCODES_DIM=3d NEWCODES_GRIDS=32,64,128,256 python3 tasks.py enzo > tasks/enzo_3d.txt && submit "enzo 3D to 256" tasks/enzo_3d.txt --partition=normal --ntasks=32 --mem-per-cpu=4G --time=24:00:00
              NEWCODES_DIM=3d NEWCODES_GRIDS=512 python3 tasks.py enzo > tasks/enzo_3d_512.txt && submit "enzo 3D 512" tasks/enzo_3d_512.txt --partition=normal --ntasks=128 --mem-per-cpu=6G --time=48:00:00 ;;
        phantom) python3 tasks.py phantom > tasks/phantom.txt && submit phantom tasks/phantom.txt --partition=normal,owners --requeue --ntasks=1 --cpus-per-task=32 --mem=128G --time=24:00:00 ;;
        *) echo "unknown code $code" ;;
    esac
done
echo "Then: bash -l $here/status.sh"
