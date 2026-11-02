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
module load python/3.12.1 py-numpy/2.2.6_py312 >/dev/null 2>&1
py=python3; "$py" -c "import numpy" || { echo "no numpy for $py (module py-numpy)"; exit 1; }
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
submit() {      # label, tasks file, sbatch options...
    local label="$1" file="$2"; shift 2; local n; n=$(wc -l < "$file" | tr -d ' ')
    [[ $n -gt 0 ]] || return 0
    local g="${LINES_PER_TASK:-1}"; local k=$(( (n + g - 1)/g ))
    local id; id="$(sbatch --parsable ${dep[@]+"${dep[@]}"} --array=1-"$k" --export=ALL,NEWCODES_TASKS="$here/$file",NEWCODES_GROUP="$g" "$@" production.sbatch)" \
        && echo "   $label: array ${id%%;*}, $k tasks of up to $g lines ($n lines)" || echo "   $label: NOT SUBMITTED"
}
# GPU: five lines per task (Sherlock's gpu QOS takes 100 queued jobs per user, 16 GPUs at once)
gpu=(--partition=gpu,owners --requeue -G 1 --constraint=GPU_SKU:H100_SXM5 --ntasks=1 --cpus-per-task=4 --mem=64G --time=24:00:00)
for code in "$@"; do
    case $code in
        athenapk|cholla) "$py" tasks.py "$code" > "tasks/$code.txt" && LINES_PER_TASK=5 submit "$code" "tasks/$code.txt" "${gpu[@]}" ;;
        pluto) for d in ct glm; do "$py" tasks.py pluto ${d}_matched ${d}_best > "tasks/pluto_$d.txt" && LINES_PER_TASK=5 submit "pluto $d" "tasks/pluto_$d.txt" "${gpu[@]}"; done ;;
        # Enzo's rank 0 holds the whole 3D grid while it starts (4 GB at 256^3, 8 times that at 512^3): full nodes, or the
        # node where it has few CPUs kills it for memory
        enzo) NEWCODES_DIM=planar "$py" tasks.py enzo > tasks/enzo_2d.txt && submit "enzo 2D" tasks/enzo_2d.txt --partition=normal,owners --requeue --ntasks=16 --mem-per-cpu=4G --time=24:00:00
              NEWCODES_DIM=3d NEWCODES_GRIDS=32,64,128,256 "$py" tasks.py enzo > tasks/enzo_3d.txt && submit "enzo 3D to 256" tasks/enzo_3d.txt --partition=normal --ntasks=32 --ntasks-per-node=32 --mem-per-cpu=4G --time=24:00:00
              NEWCODES_DIM=3d NEWCODES_GRIDS=512 "$py" tasks.py enzo > tasks/enzo_3d_512.txt && submit "enzo 3D 512" tasks/enzo_3d_512.txt --partition=normal --ntasks=128 --ntasks-per-node=32 --mem-per-cpu=6G --time=48:00:00 ;;
        # The 3D tasks that stopped: tasks/enzo_3d_redo.txt and enzo_3d_512_redo.txt, lines of the two files above
        enzo-redo) submit "enzo 3D to 256, again" tasks/enzo_3d_redo.txt --partition=normal --ntasks=32 --ntasks-per-node=32 --mem-per-cpu=4G --time=24:00:00
              submit "enzo 3D 512, again" tasks/enzo_3d_512_redo.txt --partition=normal --ntasks=128 --ntasks-per-node=32 --mem-per-cpu=6G --time=48:00:00 ;;
        phantom) "$py" tasks.py phantom > tasks/phantom.txt && submit phantom tasks/phantom.txt --partition=normal,owners --requeue --ntasks=1 --cpus-per-task=32 --mem=128G --time=24:00:00 ;;
        *) echo "unknown code $code" ;;
    esac
done
echo "Then: bash -l $here/status.sh"
