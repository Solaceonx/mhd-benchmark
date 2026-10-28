#!/bin/bash
# One array task of the order x solver study on a GPU (README.md): KIND=glm (Abel et al.'s kernels, through the sweep's
# GPU driver with this bundle's src/ and vendor/) or KIND=athenak, its share of `order.py runs`. submit.sh submits it.
# Run again, it keeps the finished runs and continues an interrupted one. PILOT=1: the pilot runs only, then the
# verdict, which fails the job (and so holds the array that waits on it) unless every pilot run is complete.
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; task="${SLURM_ARRAY_TASK_ID:-0}"
: "${KIND:?KIND is not set; submit with submit.sh}"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${ORDER_RESULTS:-$SCRATCH/mhd-results-order}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
cd "$here" || exit 2
mkdir -p "$results"
device="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "== $KIND ${PILOT:+pilot}${PILOT:-task $task of $NTASKS} on $(hostname), $device, $(date -u)"
if [[ -n "${PILOT:-}" ]]; then
    list="$results/$KIND-pilot.runs"; python3 order.py runs "$KIND" "$results" --pilot > "$list" || { echo "no run list"; exit 2; }
else
    list="$results/$KIND-task-$task-of-$NTASKS.runs"
    python3 order.py runs "$KIND" "$results" --task "$task" --tasks "$NTASKS" > "$list" || { echo "no run list"; exit 2; }
fi
echo "   $(wc -l < "$list" | tr -d ' ') runs"
if [[ "$KIND" == glm ]]; then
    cp vendor/BRANCH.json "$results/" 2>/dev/null
    srun "$MHD_JULIA" --project="$SWEEP_BUNDLE/env" --threads="$JULIA_NUM_THREADS" src/run_suite.jl --runs "$list"
    echo "== driver exit code $? (1 when some run stopped: that is a result)"
else
    exe="$MHD_CODES/athenak/build-$(gpu_arch)-fp64/src/athena"
    [[ -x "$exe" ]] || { echo "no AthenaK build at $exe"; exit 2; }
    # fd 3 for the list: the code reads stdin, which must not swallow the remaining runs
    while read -r test variant n ranks geometry out <&3; do
        mkdir -p "${out%/data}"; echo "=== $test athenak_$variant N=$n ($(date -u +%H:%M:%S))"
        python3 order.py external athenak "$variant" "$SWEEP_BUNDLE/configs/$test.toml" "$n" "$out" --seed 42 --repeats 1 \
            --geometry "$geometry" --exe "$exe" --device "$device" --device-kind gpu --resume < /dev/null 2>&1 | tee -a "${out%/data}/driver.log"
        rc=${PIPESTATUS[0]}; (( rc == 0 || rc == 75 )) || echo "   exit code $rc (see ${out%/data}/driver.log)"
    done 3< "$list"
fi
if [[ -n "${PILOT:-}" ]]; then python3 order.py verdict "$results" "$KIND"; exit $?; fi
echo "== task $task finished $(date -u)"
