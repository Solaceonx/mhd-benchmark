#!/bin/bash
# Public RAMSES II (MINIRAMSES_SRC, default $MHD_CODES/ramses2 at develop ec61647 of 2026-11-08, with E. Moseley's
# fused-HLLD GPU work) through the same driver, timer and records as the mini-ramses-ism campaign, in fp32 and fp64
# (NPRE 4 and 8), NSUBGRID 2, both set-ups (matched: HLL faces, 2D-HLL edge EMFs, Courant 0.5; best: HLLD, 2D-HLLD, 0.8):
#   - the suite's speed test throughput_ot3d at 64^3-512^3, five repeats, as in the campaign: its rate over the
#     campaign's on the same test scales RAMSES II's time to accuracy (the accuracy runs are not repeated);
#   - accuracy checks that must reproduce the campaign's errors: alfven, entropy at 128, orszag_tang, brio_wu at 256.
# Checked on the Mac (CPU build of ec61647 with prepare_patch.py, fp32): alfven 32 gives the campaign's steps and error.
#   cd $SCRATCH/mhd-study/sherlock_miniramses && mkdir -p logs && sbatch timing.sbatch
#SBATCH --job-name=ramses2pub-timing
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
cd "$here" || exit 2
module load devel 2>/dev/null || true
source "$here/compiler.sh"
export MINIRAMSES_SRC="${MINIRAMSES_SRC:-$MHD_CODES/ramses2}" MINIRAMSES_NSUBGRID="${MINIRAMSES_NSUBGRID:-2}"
tag="$(cd "$MINIRAMSES_SRC" && git rev-parse --short=7 HEAD)"
export MINIRAMSES_BUILD_ROOT="${MINIRAMSES_BUILD_ROOT:-$MHD_CODES/ramses2pub-build-$tag}"
results="${RAMSES2PUB_RESULTS:-$SCRATCH/mhd-results-ramses2pub-$tag}"
use_nvhpc "${MINIRAMSES_NVHPC_MODULE:-nvhpc/25.1}" || { echo "the compiler module did not load"; exit 2; }
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "== $(hostname), $gpu, $(date -u); source $MINIRAMSES_SRC at $tag, NSUBGRID $MINIRAMSES_NSUBGRID; results $results"

run() {     # npre test variant n seed repeats
    local npre="$1" test="$2" variant="$3" n="$4" seed="$5" reps="$6" rc
    local out="$results/p$npre/$test-miniramses-$variant-run-N$n-s$seed/data"
    mkdir -p "${out%/data}"
    MINIRAMSES_NPRE="$npre" python3 "$here/miniramses_run.py" miniramses "$variant" "$SWEEP_BUNDLE/configs/$test.toml" "$n" "$out" \
        --exe "$MINIRAMSES_BUILD_ROOT/nd3_p$npre/bin/ramses3d" --geometry cube --seed "$seed" --repeats "$reps" --resume \
        --device "$(hostname) $gpu" --device-kind gpu; rc=$?
    [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
    [[ $rc -ne 0 ]] && echo "   p$npre $test $variant N=$n: exit code $rc (see its run.toml and miniramses.log)"
    return 0
}

for npre in 4 8; do
    bash "$here/build.sh" 3 "$npre" || { echo "BUILD FAILED for NPRE=$npre"; continue; }
    for variant in best matched; do
        for n in 64 128 256 512; do run "$npre" throughput_ot3d "$variant" "$n" 42 5; done
        for t in alfven entropy; do run "$npre" "$t" "$variant" 128 42 1; done
        for t in orszag_tang brio_wu; do run "$npre" "$t" "$variant" 256 42 1; done
    done
    python3 "$here/summarize.py" "$results/p$npre" | tee "$results/summary_p$npre.txt"
    cp "$MINIRAMSES_BUILD_ROOT/nd3_p$npre/BUILD_INFO.txt" "$results/BUILD_INFO_p$npre.txt" 2>/dev/null
done
echo "== finished $(date -u)"
