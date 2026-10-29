#!/bin/bash -l
# The new codes' trial (trial.py) on one H100: every set-up of AthenaPK, Cholla and gPLUTO, and Enzo's on 4 CPU ranks.
# Ends with exit code 0 only if every check passed. Report: logs/trial_report.txt.
#SBATCH --job-name=newcodes-trial
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH -G 1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; report="$here/logs/trial_report.txt"; : > "$report"; bad=0
say() { echo "$@" | tee -a "$report"; }
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { say "the sweep bundle's environment could not be set up"; exit 2; }
venv="${NEWCODES_VENV:-$HOME/.venvs/mhd-h5py}"; source "$venv/bin/activate" && python3 -c "import h5py" || { say "no h5py in $venv"; exit 2; }
root="$MHD_CODES/newcodes"; trial="${NEWCODES_TRIAL_DIR:-$SCRATCH/mhd-results-newcodes-trial}"; rm -rf "$trial"; mkdir -p "$trial"
say "== new codes trial, job ${SLURM_JOB_ID:-?} on $(hostname), $(date -u)"
cuda_mods="${NEWCODES_CUDA_MODULES:-devel gcc/12.4.0 openmpi/4.1.6 cuda/12.4.0 hdf5/1.14.4}"
nvhpc_mods="${NEWCODES_NVHPC_MODULES:-devel nvhpc/24.7}"
cpu_mods="${NEWCODES_CPU_MODULES:-devel gcc/12.4.0 openmpi/4.1.6 hdf5/1.14.4}"
check() {   # modules, code, variant, executable, launcher
    say ""; say "== $2 $3"
    [[ -x "$4" ]] || { say "   NOT BUILT: $4"; bad=1; return; }
    ( module load $1 >/dev/null 2>&1; "$venv/bin/python3" "$here/trial.py" "$2" "$3" "$4" "$5" "$trial/$2-$3" ) 2>&1 | tee -a "$report"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || bad=1
}
for v in matched best; do check "$cuda_mods" athenapk $v "$root/athenapk/build/bin/athenaPK" "srun -n 1"; done
for v in plmc ppmc; do check "$cuda_mods" cholla $v "$root/cholla/$v/bin/cholla.abel_$v.sherlock" "srun -n 1"; done
for v in ct_matched ct_best glm_matched glm_best; do
    for d in 2 3; do [[ -x "$root/pluto/$v-${d}d/pluto" ]] || { say "   NOT BUILT: pluto $v ${d}D"; bad=1; }; done
    check "$nvhpc_mods" pluto $v "$root/pluto/$v-2d/pluto" ""        # NVHPC's MPI, one rank, started directly
done
for v in dedner ct; do check "$cpu_mods" enzo $v "$root/enzo/enzo" "srun -n 4"; done
say ""
if [[ $bad -eq 0 ]]; then say "== TRIAL PASSED $(date -u)"; exit 0; fi
say "== TRIAL NOT CLEAN $(date -u)"; exit 1
