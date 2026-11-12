#!/bin/bash
# The official RAMSES II (github.com/ramses-organisation/ramses2, develop 04abdd9, fetch_ramses2.sh) on the sweep's
# tests, through the campaign's driver (miniramses_run.py): set-ups matched (HLL), best (HLLD), uct (UCT-HLLD), each in
# double (code ramses2, NPRE 8) and single precision (ramses2f32, NPRE 4); NSUBGRID 2 as in the mini-ramses campaign.
# Task 0 is the pilot: compiler, both builds, the field loop at 64 in all six, 3D Orszag-Tang at 64 with UCT in both
# precisions, and the 512^3 speed test with UCT in double precision (the largest GPU memory). Task k >= 1 runs task k of
# runs_ramses2.json (make_tasks_ramses2.py). A requeued task keeps its finished runs.   submit_ramses2.sh submits.
#SBATCH --job-name=ramses2
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; task="${SLURM_ARRAY_TASK_ID:-0}"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
cd "$here" || exit 2
module load devel 2>/dev/null || true          # the newer nvhpc modules are visible only with devel (job 49282778)
source "$here/compiler.sh"
export MINIRAMSES_SRC="${MINIRAMSES_SRC:-$MHD_CODES/ramses2-04abdd9}" MINIRAMSES_NSUBGRID="${MINIRAMSES_NSUBGRID:-2}"
export MINIRAMSES_BUILD_ROOT="${MINIRAMSES_BUILD_ROOT:-$MHD_CODES/ramses2-04abdd9-build}"; root="$MINIRAMSES_BUILD_ROOT"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "== task $task on $(hostname), $gpu, $(date -u); source $(cd "$MINIRAMSES_SRC" && git rev-parse --short HEAD)"
if [[ "$task" == 0 ]]; then
    MINIRAMSES_NPRE=8 choose_compiler_and_build || { echo "NO COMPILER BUILT RAMSES II"; exit 1; }
else
    use_nvhpc "$(cat "$root/nvhpc_module.txt")" || { echo "the pilot's compiler module did not load"; exit 2; }
fi
bash "$here/build.sh" 3 8 && bash "$here/build.sh" 3 4 || { echo "BUILD FAILED"; exit 1; }

run() {     # test code variant n seed geometry repeats
    local test="$1" code="$2" variant="$3" n="$4" seed="$5" geometry="$6" reps="$7" npre=8 rc
    [[ "$code" == ramses2f32 ]] && npre=4
    local out="$MHD_RESULTS/$test-$code-$variant-run-N$n-s$seed/data"
    mkdir -p "${out%/data}"
    python3 "$here/miniramses_run.py" "$code" "$variant" "$SWEEP_BUNDLE/configs/$test.toml" "$n" "$out" \
        --exe "$root/nd3_p$npre/bin/ramses3d" --geometry "$geometry" --seed "$seed" --repeats "$reps" --resume \
        --device "$(hostname) $gpu" --device-kind gpu; rc=$?
    [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
    [[ $rc -ne 0 ]] && echo "   $test $code $variant N=$n s$seed: exit code $rc (see its run.toml and miniramses.log)"
    return 0
}

if [[ "$task" == 0 ]]; then
    for code in ramses2 ramses2f32; do for v in matched best uct; do run field_loop_novz "$code" "$v" 64 42 cube 1; done; done
    run orszag_tang_3d ramses2 uct 64 42 cube 1; run orszag_tang_3d ramses2f32 uct 64 42 cube 1
    run throughput_ot3d ramses2 uct 512 42 cube 1        # memory: UCT keeps ~2 KB per oct in double precision
    python3 "$here/summarize.py" "$MHD_RESULTS" "*-ramses2*-run-N*-s42" | tee "$here/logs/pilot_summary_ramses2.txt"
    [[ "$(grep -cE "^(field_loop_novz-ramses2(f32)?-(matched|best|uct)-run-N64|orszag_tang_3d-ramses2(f32)?-uct-run-N64|throughput_ot3d-ramses2-uct-run-N512)-s42 +complete" "$here/logs/pilot_summary_ramses2.txt")" -eq 9 ]] \
        || { echo "PILOT FAILED: not all nine pilot runs are complete"; exit 1; }
else
    while read -r test code variant n seed geometry reps; do run "$test" "$code" "$variant" "$n" "$seed" "$geometry" "$reps"; done < <(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for i in d['tasks'][int(sys.argv[2]) - 1]:
    r = d['runs'][i];print(r['test'], r['code'], r['variant'], r['n'], r['seed'], r['geometry'], r['repeats'])" "$here/runs_ramses2.json" "$task")
fi
echo "== task $task finished $(date -u)"
