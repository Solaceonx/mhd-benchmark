#!/bin/bash
# One array task of the order x solver study for Athena++ (README.md): its share of the runs that use RANKS MPI ranks
# (the sweep's count for each test and grid), on CPUs. submit.sh submits one array per rank count, with --ntasks=RANKS.
# Run again, it keeps the finished runs and continues an interrupted one from its last output. PILOT=1: the pilot runs
# only, then the verdict (exit 1 unless every pilot run is complete).
#SBATCH --partition=normal,owners
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; task="${SLURM_ARRAY_TASK_ID:-0}"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${ORDER_RESULTS:-$SCRATCH/mhd-results-order}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
cd "$here" || exit 2
mkdir -p "$results"
echo "== athenapp ${PILOT:+pilot}${PILOT:-ranks $RANKS, task $task of $NTASKS} on $(hostname) x ${SLURM_NTASKS:-1}, $(date -u)"
if [[ -n "${PILOT:-}" ]]; then
    list="$results/athenapp-pilot.runs"; python3 order.py runs athenapp "$results" --pilot > "$list" || { echo "no run list"; exit 2; }
else
    list="$results/athenapp-ranks-$RANKS-task-$task-of-$NTASKS.runs"
    python3 order.py runs athenapp "$results" --ranks "$RANKS" --task "$task" --tasks "$NTASKS" > "$list" || { echo "no run list"; exit 2; }
fi
echo "   $(wc -l < "$list" | tr -d ' ') runs"
while read -r test variant n ranks geometry out <&3; do
    flux=hlle; [[ "$variant" == *hlld* ]] && flux=hlld          # Athena++ fixes the Riemann solver at build time
    exe="$MHD_CODES/athena/bin/athena_$flux"; [[ -x "$exe" ]] || { echo "no Athena++ build at $exe"; exit 2; }
    mkdir -p "${out%/data}"; echo "=== $test athenapp_$variant N=$n on $ranks ranks ($(date -u +%H:%M:%S))"
    python3 order.py external athenapp "$variant" "$SWEEP_BUNDLE/configs/$test.toml" "$n" "$out" --seed 42 --repeats 1 \
        --geometry "$geometry" --exe "$exe" --mpi "srun -n $ranks" --device "CPU $(hostname) x $ranks ranks" --device-kind cpu \
        --resume < /dev/null 2>&1 | tee -a "${out%/data}/driver.log"
    rc=${PIPESTATUS[0]}; (( rc == 0 || rc == 75 )) || echo "   exit code $rc (see ${out%/data}/driver.log)"
done 3< "$list"
if [[ -n "${PILOT:-}" ]]; then python3 order.py verdict "$results" athenapp; exit $?; fi
echo "== task $task finished $(date -u)"
