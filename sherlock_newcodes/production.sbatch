#!/bin/bash -l
# One production task of the new codes: line SLURM_ARRAY_TASK_ID of $NEWCODES_TASKS (tasks.py), every grid of the line
# in turn through newcodes_run.py. Resources, partition and time come from production.sh.
#SBATCH --job-name=newcodes
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
IFS='|' read -r code variant test seed grids geometry <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${NEWCODES_TASKS:?}")"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
export MHD_RESULTS="${NEWCODES_RESULTS:-$SCRATCH/mhd-results-newcodes}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
venv="${NEWCODES_VENV:-$HOME/.venvs/mhd-h5py}"; source "$venv/bin/activate"
root="$MHD_CODES/newcodes"; kind=gpu
case $code in
    athenapk) module load devel gcc/12.4.0 openmpi/4.1.6 cuda/12.4.0 hdf5/1.14.4; exe="$root/athenapk/build/bin/athenaPK"; mpi="srun -n 1" ;;
    cholla) module load devel gcc/12.4.0 openmpi/4.1.6 cuda/12.4.0 hdf5/1.14.4; exe="$root/cholla/$variant/bin/cholla.abel_$variant.sherlock"; mpi="srun -n 1" ;;
    pluto) module load devel nvhpc/24.7; exe="$root/pluto/$variant-2d/pluto"; mpi="" ;;
    enzo) module load devel gcc/12.4.0 openmpi/4.1.6 hdf5/1.14.4; exe="$root/enzo/enzo"; mpi="srun -n $SLURM_NTASKS"; kind=cpu ;;
    phantom) module load devel gcc/12.4.0; exe="$root/phantom/tree/bin/phantom"; mpi=""; kind=cpu ;;
esac
echo "== task $SLURM_ARRAY_TASK_ID: $code $variant $test seed $seed grids $grids on $(hostname), $(date -u)"
for n in ${grids//,/ }; do
    out="$MHD_RESULTS/$test-$code-$variant-run-N$n-s$seed/data"
    "$venv/bin/python3" "$here/newcodes_run.py" "$code" "$variant" "$SWEEP_BUNDLE/configs/$test.toml" "$n" "$out" --exe "$exe" --mpi "$mpi" \
        --geometry "$geometry" --repeats 1 --seed "$seed" --resume --device "$(hostname)" --device-kind "$kind"; rc=$?
    [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
    [[ $rc -ne 0 ]] && echo "   $test N=$n: exit code $rc (see its run.toml and the code's log)"
done
echo "== task $SLURM_ARRAY_TASK_ID finished $(date -u)"
