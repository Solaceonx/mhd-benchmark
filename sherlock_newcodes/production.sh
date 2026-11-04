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
[[ "$trial" == after* ]] && dep=(--dependency="$trial")         # or a dependency as is: afterany:JOB, to start when an array ends
module load python/3.12.1 py-numpy/2.2.6_py312 >/dev/null 2>&1
py=python3; "$py" -c "import numpy" || { echo "no numpy for $py (module py-numpy)"; exit 1; }
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
submit() {      # label, tasks file, sbatch options...
    local label="$1" file="$2"; shift 2; local n; n=$(wc -l < "$file" | tr -d ' ')
    [[ $n -gt 0 ]] || return 0
    local g="${LINES_PER_TASK:-1}"; local k=$(( (n + g - 1)/g ))
    local id; id="$(sbatch --parsable ${dep[@]+"${dep[@]}"} --array=1-"$k" --export=ALL,NEWCODES_TASKS="$here/$file",NEWCODES_GROUP="$g",NEWCODES_RETRY="${RETRY:-0}" "$@" production.sbatch)" \
        && echo "   $label: array ${id%%;*}, $k tasks of up to $g lines ($n lines)" || echo "   $label: NOT SUBMITTED"
}
# GPU: five lines per task (Sherlock's gpu QOS takes 100 queued jobs per user, 16 GPUs at once)
gpu=(--partition=gpu,owners --requeue -G 1 --constraint=GPU_SKU:H100_SXM5 --ntasks=1 --cpus-per-task=4 --mem=64G --time=24:00:00)
# Two H100s on one node, a rank each (3D 512^3 in Cholla and gPLUTO). The driver holds a 512^3 frame and its postprocessing
# took 128 GB in one job, so 192
gpu2=(--partition=gpu,owners --requeue -G 2 --constraint=GPU_SKU:H100_SXM5 --nodes=1 --ntasks=2 --cpus-per-task=4 --mem=192G --time=12:00:00)
for code in "$@"; do
    case $code in
        athenapk|cholla) "$py" tasks.py "$code" > "tasks/$code.txt" && LINES_PER_TASK=5 submit "$code" "tasks/$code.txt" "${gpu[@]}" ;;
        # Lines of tasks/athenapk.txt with the grids that aborted on blocks of 256, written by hand
        athenapk-redo) LINES_PER_TASK=5 submit "athenapk, again" tasks/athenapk_redo.txt "${gpu[@]}" ;;
        # Lines of tasks/cholla.txt whose task stopped (Cholla's own stops were read as lost jobs), 3D grids to 256; and the
        # 3D tests at 512^3, about 100 GB: two H100s, a rank each
        cholla-redo) LINES_PER_TASK=5 submit "cholla, again" tasks/cholla_redo.txt "${gpu[@]}" ;;
        cholla-512) LINES_PER_TASK=2 RETRY='does not fit\|illegal memory access' submit "cholla 3D 512 on two GPUs" tasks/cholla_512.txt "${gpu2[@]}" ;;
        # The 512^3 runs whose stop on two ranks was read as a kill (srun returned 137)
        cholla-512-redo) LINES_PER_TASK=1 submit "cholla 3D 512 on two GPUs, again" tasks/cholla_512_redo.txt "${gpu2[@]}" ;;
        # 3D 512^3 does not fit one GPU in gPLUTO either ("Memory Sturated!", about 125 GB): its lines to 256 on one GPU (the
        # tasks stopped at 512, and the lines after it never ran), and 512 on two
        pluto-redo2) for d in ct glm; do LINES_PER_TASK=5 submit "pluto $d to 256" "tasks/pluto_${d}_redo.txt" "${gpu[@]}"; done ;;
        pluto-512) LINES_PER_TASK=2 RETRY='Memory Sturated' submit "pluto 3D 512 on two GPUs" tasks/pluto_512.txt "${gpu2[@]}" ;;
        # gPLUTO writes its stops (QuitPLUTO from ConsToPrim3D on decay_supersonic) to pluto.0.log, which run_pluto did not read:
        # each read as a kill and the task's later lines (entropy, field_loop) never ran. The best set-ups' lines again.
        pluto-best-redo) LINES_PER_TASK=4 submit "pluto best, again" tasks/pluto_best_redo.txt "${gpu[@]}"
              LINES_PER_TASK=1 submit "pluto best 3D 512 on two GPUs, again" tasks/pluto_best_512_redo.txt "${gpu2[@]}" ;;
        # On two ranks the stop is in the stopping rank's log (pluto.1.log): its first line, read as a kill
        pluto-best-512-again) sed -n 1p tasks/pluto_best_512_redo.txt > tasks/pluto_best_512_again.txt
              submit "pluto best 3D 512 on two GPUs, the first line again" tasks/pluto_best_512_again.txt "${gpu2[@]}" ;;
        # Phantom's 3D 256^3 runs: phantomsetup allocated 5.2 million particles (now --maxp)
        phantom-256) submit "phantom 3D 256" tasks/phantom_256.txt --partition=normal,owners --requeue --ntasks=1 --cpus-per-task=32 --mem=128G --time=24:00:00 ;;
        # The lines that stopped on Parthenon's doubled last output
        athenapk-redo2) LINES_PER_TASK=5 submit "athenapk, again (doubled last output)" tasks/athenapk_redo2.txt "${gpu[@]}" ;;
        # Every pluto line again: runs recorded before dbl.out's 7-digit times were snapped to the output times (resume skips the rest)
        pluto-redo) for d in ct glm; do LINES_PER_TASK=5 RETRY='missing output times' submit "pluto $d, again" "tasks/pluto_$d.txt" "${gpu[@]}"; done ;;
        pluto) for d in ct glm; do "$py" tasks.py pluto ${d}_matched ${d}_best > "tasks/pluto_$d.txt" && LINES_PER_TASK=5 submit "pluto $d" "tasks/pluto_$d.txt" "${gpu[@]}"; done ;;
        # Enzo's rank 0 holds the whole 3D grid while it starts (4 GB at 256^3, 8 times that at 512^3): full nodes, or the
        # node where it has few CPUs kills it for memory
        enzo) NEWCODES_DIM=planar "$py" tasks.py enzo > tasks/enzo_2d.txt && submit "enzo 2D" tasks/enzo_2d.txt --partition=normal,owners --requeue --ntasks=16 --mem-per-cpu=4G --time=24:00:00
              NEWCODES_DIM=3d NEWCODES_GRIDS=32,64,128,256 "$py" tasks.py enzo > tasks/enzo_3d.txt && submit "enzo 3D to 256" tasks/enzo_3d.txt --partition=normal --ntasks=32 --ntasks-per-node=32 --mem-per-cpu=4G --time=24:00:00
              NEWCODES_DIM=3d NEWCODES_GRIDS=512 "$py" tasks.py enzo > tasks/enzo_3d_512.txt && submit "enzo 3D 512" tasks/enzo_3d_512.txt --partition=normal --ntasks=128 --ntasks-per-node=32 --mem-per-cpu=6G --time=48:00:00 ;;
        # The 3D tasks that stopped: tasks/enzo_3d_redo.txt and enzo_3d_512_redo.txt, lines of the two files above
        # (also owners, requeued: they waited two days in normal behind our fair share, and a run takes 1-3.4 hours)
        enzo-redo) submit "enzo 3D to 256, again" tasks/enzo_3d_redo.txt --partition=normal,owners --requeue --ntasks=32 --ntasks-per-node=32 --mem-per-cpu=4G --time=24:00:00
              submit "enzo 3D 512, again" tasks/enzo_3d_512_redo.txt --partition=normal,owners --requeue --ntasks=128 --ntasks-per-node=32 --mem-per-cpu=6G --time=48:00:00 ;;
        # decay_supersonic dedner 512^3 seed 43 ended, then its driver was killed for memory (25 GB on a node with few of the
        # 128 ranks, 50676336_8) before it wrote the record: on full nodes, again
        enzo-512-again) grep "^enzo|dedner|decay_supersonic|43|512|" tasks/enzo_3d_512.txt > tasks/enzo_3d_512_again.txt
              submit "enzo 3D 512, the run killed in postprocessing" tasks/enzo_3d_512_again.txt --partition=normal,owners --requeue --ntasks=128 --ntasks-per-node=32 --mem-per-cpu=6G --time=48:00:00 ;;
        phantom) "$py" tasks.py phantom > tasks/phantom.txt && submit phantom tasks/phantom.txt --partition=normal,owners --requeue --ntasks=1 --cpus-per-task=32 --mem=128G --time=24:00:00 ;;
        *) echo "unknown code $code" ;;
    esac
done
echo "Then: bash -l $here/status.sh"
