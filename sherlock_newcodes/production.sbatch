#!/bin/bash -l
# One production task of the new codes: NEWCODES_GROUP lines (default 1) of $NEWCODES_TASKS (tasks.py), task k the
# k-th group, every grid of each line in turn through newcodes_run.py. Resources, partition and time come from production.sh.
#SBATCH --job-name=newcodes
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
export MHD_RESULTS="${NEWCODES_RESULTS:-$SCRATCH/mhd-results-newcodes}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
venv="${NEWCODES_VENV:-$HOME/.venvs/mhd-h5py}"; source "$venv/bin/activate"
root="$MHD_CODES/newcodes"; group="${NEWCODES_GROUP:-1}"         # lines of the tasks file per array task
first=$(( (SLURM_ARRAY_TASK_ID - 1)*group + 1 )); last=$(( SLURM_ARRAY_TASK_ID*group ))
loaded=""
while IFS='|' read -r code variant test seed grids geometry; do
    [[ -n "$code" ]] || continue
    kind=gpu
    case $code in
        athenapk) mods="devel gcc/12.4.0 openmpi/4.1.6 cuda/12.4.0 hdf5/1.14.4"; exe="$root/athenapk/build/bin/athenaPK"; mpi="srun -n 1" ;;
        cholla) mods="devel gcc/12.4.0 openmpi/4.1.6 cuda/12.4.0 hdf5/1.14.4"; exe="$root/cholla/$variant/bin/cholla.abel_$variant.sherlock"; mpi="srun -n $SLURM_NTASKS" ;;   # a rank per GPU
        pluto) mods="devel nvhpc/24.7"; exe="$root/pluto/$variant-2d/pluto"; mpi="srun -n 1" ;;   # an MPI build: started directly it hangs in MPI_Init
        enzo) mods="devel gcc/12.4.0 openmpi/4.1.6 hdf5/1.14.4"; exe="$root/enzo/enzo"; mpi="srun -n $SLURM_NTASKS"; kind=cpu ;;
        phantom) mods="devel gcc/12.4.0"; exe="$root/phantom/tree/bin/phantom"; mpi=""; kind=cpu ;;
    esac
    [[ "$loaded" == "$mods" ]] || { module load $mods; loaded="$mods"; }
    echo "== task $SLURM_ARRAY_TASK_ID: $code $variant $test seed $seed grids $grids on $(hostname), $(date -u)"
    for n in ${grids//,/ }; do
        out="$MHD_RESULTS/$test-$code-$variant-run-N$n-s$seed/data"
        # NEWCODES_RETRY (a grep pattern): a failed run whose record matches it (an attempt on one GPU too few, a reader since
        # fixed) is moved aside and run again
        if [[ -n "${NEWCODES_RETRY:-}" && "${NEWCODES_RETRY:-}" != 0 ]] && grep -qs 'status = "failed"' "$out"/*_r1/run.toml \
            && grep -qs -- "$NEWCODES_RETRY" "$out"/*_r1/run.toml; then
            mkdir -p "$MHD_RESULTS-superseded/$code-retried" && mv "${out%/data}" "$MHD_RESULTS-superseded/$code-retried/" && echo "   moved aside: ${out%/data}"
        fi
        "$venv/bin/python3" "$here/newcodes_run.py" "$code" "$variant" "$SWEEP_BUNDLE/configs/$test.toml" "$n" "$out" --exe "$exe" --mpi "$mpi" \
            --geometry "$geometry" --repeats 1 --seed "$seed" --resume --device "$(hostname)" --device-kind "$kind" < /dev/null; rc=$?
        [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
        [[ $rc -ne 0 ]] && echo "   $test N=$n: exit code $rc (see its run.toml and the code's log)"
    done
done < <(sed -n "${first},${last}p" "${NEWCODES_TASKS:?}")
echo "== task $SLURM_ARRAY_TASK_ID finished $(date -u)"
