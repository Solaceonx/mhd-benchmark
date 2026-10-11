#!/bin/bash
# RAMSES in the driven-turbulence pilot (README.md). check.sh submits these with the ranks and time limit of each:
#   sbatch --job-name=turb-cal --ntasks=16 --ntasks-per-node=16 --time=04:00:00 ramses.sbatch calibrate [released|fixed]
#   sbatch --job-name=turb-ramses-matched-M3-N128 --ntasks=32 --ntasks-per-node=16 --time=05:00:00 ramses.sbatch run matched 3 128
#   sbatch --job-name=turb-ramsesfix-matched-M3-N128 ... ramses.sbatch run matched 3 128 fixed      (corrected projection)
# TURB_CONFIG names the configuration file (default config.toml). A job that is requeued (owner nodes) or runs out of
# time continues from RAMSES's newest complete output when it runs again; check.sh submits again whatever is not finished.
#SBATCH --partition=normal,owners
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
results="${TURB_RESULTS:-$SCRATCH/mhd-results-turb}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
config="${TURB_CONFIG:-config.toml}"
if [[ "$1" == run ]]; then projection="${5:-released}"; else projection="${2:-released}"; fi
build=bin-turb; [[ "$projection" == fixed ]] && build=bin-turb-fixed
exe="$MHD_CODES/ramses/$build/ramses_turb3d"
[[ -x "$exe" ]] || { echo "no $exe: run check.sh (it builds it)"; exit 2; }
echo "== $* ($config) on $(hostname), ${SLURM_NTASKS:-1} ranks, $(date -u)"
cd "$here"
if [[ "$1" == run ]]; then
    python3 ramses_turb.py run "$config" "$results" "$2" "$3" "$4" --projection "$projection" --exe "$exe" --mpi "srun -n ${SLURM_NTASKS:-1}" < /dev/null; rc=$?
else
    python3 ramses_turb.py "$1" "$config" "$results" --projection "$projection" --exe "$exe" --mpi "srun -n ${SLURM_NTASKS:-1}" < /dev/null; rc=$?
fi
echo "== exit code $rc, $(date -u)"
if [[ $rc -eq 76 ]]; then echo "this job is being stopped; it continues when it runs again"; exit 1; fi
exit "$rc"
