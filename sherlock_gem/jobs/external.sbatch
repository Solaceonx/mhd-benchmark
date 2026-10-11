#!/bin/bash
# One external-code run per array task. Submit through tools/submit.sh, which sets MHD_TEST,
# MHD_CODE, MHD_VARIANT, MHD_MODE (run | reference) and the partition/resources for the code.
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${MHD_TEST:?}" "${MHD_CODE:?}" "${MHD_VARIANT:?}" "${MHD_MODE:=run}"
source jobs/common.sh
prepare_environment
prepare_codes
config="configs/$MHD_TEST.toml"
index="${SLURM_ARRAY_TASK_ID:-0}"
read -r n seed geometry restrict < <(python3 codes/plan.py task "$config" "$MHD_CODE" "$MHD_MODE" "$index")
[[ "$restrict" == - ]] && restrict=""
run_ranks="${SLURM_NTASKS:-1}"
case "$MHD_CODE" in
    athenak)
        load_modules "${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
        exe="$MHD_CODES/athenak/build-$(gpu_arch)-fp64/src/athena"; launcher="" ;;
    ramses)
        load_modules "${MHD_CPU_BUILD_MODULES:-gcc openmpi}"
        exe="$MHD_CODES/ramses/bin/ramses_abel3d"; launcher="srun -n ${SLURM_NTASKS:-1}" ;;
    athenapp)
        load_modules "${MHD_CPU_BUILD_MODULES:-gcc openmpi}"
        # With a minimum 8-cell block, small planar grids cannot occupy every
        # allocated rank. Launch only ranks that can own at least one block.
        blocks_per_axis=$((n / 8)); max_ranks=$((blocks_per_axis * blocks_per_axis))
        [[ "$geometry" == cube ]] && max_ranks=$((max_ranks * blocks_per_axis))
        (( run_ranks > max_ranks )) && run_ranks="$max_ranks"
        exe="$MHD_CODES/athena/bin/athena"; launcher="srun -n $run_ranks" ;;
    *) printf 'Unknown code %s\n' "$MHD_CODE" >&2; exit 2 ;;
esac
[[ -x "$exe" ]] || { printf 'Missing executable %s: run the build job for this code (and GPU model)\n' "$exe" >&2; exit 2; }
run_id="$MHD_TEST-$MHD_CODE-$MHD_VARIANT-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-$index"
run_dir="$MHD_RESULTS/$run_id"
mkdir -p "$run_dir"
capture_environment "$run_dir"
device="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
device="${device:-CPU $(hostname) x ${run_ranks} ranks}"
kind=cpu; [[ "$MHD_CODE" == athenak ]] && kind=gpu
args=("$MHD_CODE" "$MHD_VARIANT" "$config" "$n" "$run_dir/data" --seed "$seed" --geometry "$geometry" --exe "$exe" --device "$device" --device-kind "$kind")
[[ -n "$launcher" ]] && args+=(--mpi "$launcher")
[[ "$MHD_MODE" == reference ]] && args+=(--repeats 1 --restrict-to "$restrict")
set +e
python3 codes/external.py "${args[@]}" 2>&1 | tee "$run_dir/driver.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$run_dir/exit_code.txt"
bash tools/pack_results.sh "$run_dir"
exit "$run_status"
