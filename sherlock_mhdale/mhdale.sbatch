#!/bin/bash
# MHD-ALE (T. Abel's fork) on the suite's planar tests in the periodic box (see README.md). One array task per test,
# running that test's grids in turn (32^2 to 512^2); submitted by check.sh to start only after the trial passed:
#   sbatch --dependency=afterok:<trial job> --array=1-13 mhdale.sbatch
# MHD-ALE has no restart: a requeued task keeps its finished runs and starts the interrupted one again; a run that
# does not fit in the time limit ends the task there.
#SBATCH --job-name=mhdale
#SBATCH --partition=normal,owners
#SBATCH --requeue
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; task="${SLURM_ARRAY_TASK_ID:-1}"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHDALE_COMPILER_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
source "$here/options.sh"; tests=($MHDALE_TESTS)
exe="$MHD_CODES/mhdale-yipihey/build-abel/MHD"; [[ -x "$exe" ]] || { echo "NO EXECUTABLE: $exe (the trial builds it)"; exit 2; }
test="${MHDALE_ONLY:-${tests[$((task - 1))]}}"; config="$SWEEP_BUNDLE/configs/$test.toml"
echo "== task $task: $test on $(hostname), $(date -u)"; module list 2>&1 | tail -n +2
for n in ${MHDALE_GRIDS:-$(python3 -c "import tomllib, sys; print(*tomllib.load(open(sys.argv[1], 'rb'))['grids'])" "$config")}; do
    ranks=1; [[ $n -ge 64 ]] && ranks=4; [[ $n -ge 128 ]] && ranks=8; [[ $n -ge 256 ]] && ranks=16
    ranks="${MHDALE_RANKS:-$ranks}"
    out="$MHD_RESULTS/$test-mhdale-ale-run-N$n-s42/data"
    python3 "$here/mhdale_run.py" mhdale ale "$config" "$n" "$out" --exe "$exe" --mpi "srun -n $ranks" --geometry plane \
        --repeats 1 --resume --device "$(hostname)" --device-kind cpu; rc=$?
    [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
    [[ $rc -ne 0 ]] && echo "   $test N=$n: exit code $rc (see its run.toml and mhdale.log)"
done
echo "== task $task finished $(date -u)"
