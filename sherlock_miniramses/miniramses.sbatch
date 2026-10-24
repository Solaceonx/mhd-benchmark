#!/bin/bash
# mini-ramses-ism on the sweep's tests (README.md). Task 0 is the pilot: it chooses the compiler, builds the code (3D:
# its GPU path is 3D only; planar tests run on N^3 cells that are the same at every z) and runs the field loop at 64
# (both set-ups) and 3D Orszag-Tang at 64^3; task k >= 1 runs task k of runs.json (make_tasks.py), longest runs first.
#   submit.sh submits the pilot and, depending on it, the tasks.
# A requeued task keeps its finished runs and runs the interrupted one again.
#SBATCH --job-name=miniramses
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; task="${SLURM_ARRAY_TASK_ID:-0}"; npre="${MINIRAMSES_NPRE:-8}"; export MINIRAMSES_NPRE="$npre"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
cd "$here" || exit 2
# The newer nvhpc modules are visible only with `devel` loaded: without it the job saw nvhpc/23.3 alone, whose nvfortran
# does not know -gpu=sm_90 (job 49282778). The RAMSES II speed job loaded it and built with nvhpc/25.1.
module load devel 2>/dev/null || true
source "$here/compiler.sh"
root="${MINIRAMSES_BUILD_ROOT:-$MHD_CODES/miniramses-build}"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "== task $task on $(hostname), $gpu, $(date -u); NPRE=$npre"
if [[ "$task" == 0 ]]; then
    choose_compiler_and_build || { echo "NO COMPILER BUILT mini-ramses-ism"; exit 1; }
else
    use_nvhpc "$(cat "$root/nvhpc_module.txt")" || { echo "the pilot's compiler module did not load"; exit 2; }
fi
bash "$here/build.sh" 3 "$npre" || { echo "BUILD FAILED"; exit 1; }

run() {     # test variant n seed geometry repeats
    local test="$1" variant="$2" n="$3" seed="$4" geometry="$5" reps="$6" ndim=3 rc      # geometry: cube (3D build only)
    local out="$MHD_RESULTS/$test-miniramses-$variant-run-N$n-s$seed/data"
    mkdir -p "${out%/data}"                   # the driver's run guard writes its lock file beside data/
    python3 "$here/miniramses_run.py" miniramses "$variant" "$SWEEP_BUNDLE/configs/$test.toml" "$n" "$out" \
        --exe "$root/nd${ndim}_p$npre/bin/ramses${ndim}d" --geometry "$geometry" --seed "$seed" --repeats "$reps" --resume \
        --device "$(hostname) $gpu" --device-kind gpu; rc=$?
    [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
    [[ $rc -ne 0 ]] && echo "   $test $variant N=$n s$seed: exit code $rc (see its run.toml and miniramses.log)"
    return 0
}

if [[ "$task" == 0 ]]; then
    run field_loop_novz matched 64 42 cube 1; run field_loop_novz best 64 42 cube 1; run orszag_tang_3d matched 64 42 cube 1
    python3 "$here/summarize.py" "$MHD_RESULTS" | tee "$here/logs/pilot_summary.txt"
    # The array waits for this task: it must fail unless all three pilot runs finished.
    [[ "$(grep -cE "^(field_loop_novz-miniramses-(matched|best)-run-N64|orszag_tang_3d-miniramses-matched-run-N64)-s42 +complete" "$here/logs/pilot_summary.txt")" -eq 3 ]] \
        || { echo "PILOT FAILED: not all three pilot runs are complete"; exit 1; }
else
    while read -r test variant n seed geometry reps; do run "$test" "$variant" "$n" "$seed" "$geometry" "$reps"; done < <(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for i in d['tasks'][int(sys.argv[2]) - 1]:
    r = d['runs'][i];print(r['test'], r['variant'], r['n'], r['seed'], r['geometry'], r['repeats'])" "$here/runs.json" "$task")
fi
echo "== task $task finished $(date -u)"
