#!/bin/bash
# AREPO on the sweep's planar tests (see README.md). One array task per test and mesh set-up, running that test's
# grids in turn; task 0 is the pilot: it builds every executable and runs Orszag-Tang at 32^2 and 64^2 on both meshes.
#   p=$(sbatch --parsable --array=0 arepo.sbatch) && sbatch --dependency=afterok:$p --array=1-34 arepo.sbatch
# A requeued task keeps its finished runs and starts the interrupted one again. rerun_failed.sh: the tasks that failed.
#SBATCH --job-name=arepo
#SBATCH --partition=normal,owners
#SBATCH --requeue
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; task="${SLURM_ARRAY_TASK_ID:-0}"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${AREPO_COMPILER_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
# The libraries AREPO links to.
for m in ${AREPO_MODULES:-gsl/2.7 hdf5/1.14.4 gmp/6.2.1}; do module load "$m" 2>/dev/null || echo "module $m: not loaded"; done
# Python: h5py and scipy from pip, in a private environment on scratch. Sherlock's py-h5py module is built for numpy 1
# and does not import with the numpy 2 module the sweep's tools load ("numpy.dtype size changed"), so it is not used.
# Wheels only: left to itself pip took a release without a wheel for this system and failed building it from source.
# In $HOME, not $SCRATCH: Sherlock's rules for automated tools (/etc/agents/AGENTS.md, rule 8) keep environments off the
# purged filesystems; this one is small (h5py, scipy).
venv="${AREPO_VENV:-$HOME/.venvs/mhd-h5py}"; [[ -x "$venv/bin/python3" ]] || python3 -m venv --system-site-packages "$venv"
source "$venv/bin/activate"; python3 -c 'import h5py, scipy' 2>/dev/null || pip install --quiet --only-binary=:all: "h5py>=3.11" "scipy>=1.13"
python3 -c 'import h5py, numpy; print("python:", h5py.__version__, "h5py,", numpy.__version__, "numpy")' || { echo "NO h5py: cannot write AREPO initial conditions"; exit 2; }
echo "== task $task on $(hostname), $(date -u)"; module list 2>&1 | tail -n +2

TESTS=(entropy alfven alfven_oblique uniform field_loop field_loop_novz brio_wu brio_wu_rotated orszag_tang rotor current_sheet
       blast_2d_b0 blast_2d_b50 blast_2d_b100 kh_uniform kh_localized gem)
gamma_of() { python3 -c "import sys; sys.path.insert(0, '$SWEEP_BUNDLE/codes'); import external as X; print(repr(float(X.gamma_of(X.load_config(sys.argv[1])))))" "$1"; }
run_test() {     # test mesh [grids...]
    local test="$1" mesh="$2"; shift 2
    local config="$SWEEP_BUNDLE/configs/$test.toml" gamma exe n ranks out rc
    gamma="$(gamma_of "$config")"; exe="$MHD_CODES/arepo-build/${mesh}_g${gamma}_2d/Arepo"
    [[ -x "$exe" ]] || bash "$here/build.sh" "$mesh" "$gamma" 2 || { echo "NO EXECUTABLE for $test $mesh"; return 1; }
    [[ $# -gt 0 ]] || set -- $(python3 -c "import tomllib, sys; print(*tomllib.load(open(sys.argv[1], 'rb'))['grids'])" "$config")
    for n in "$@"; do
        ranks=1; [[ $n -ge 128 ]] && ranks=4; [[ $n -ge 256 ]] && ranks=8; [[ $n -ge 512 ]] && ranks=16
        ranks="${AREPO_RANKS:-$ranks}"
        out="$MHD_RESULTS/$test-arepo-$mesh-run-N$n-s42/data"
        python3 "$here/arepo_run.py" arepo "$mesh" "$config" "$n" "$out" --exe "$exe" --mpi "srun -n $ranks" --geometry plane \
            --repeats 1 --resume --device "$(hostname)" --device-kind cpu; rc=$?
        [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
        [[ $rc -ne 0 ]] && echo "   $test $mesh N=$n: exit code $rc (see its run.toml and arepo.log)"
    done
}

if [[ -n "${AREPO_ONLY:-}" ]]; then        # a check run: AREPO_ONLY="test mesh grid...", with AREPO_RANKS, AREPO_COURANT, MHD_RESULTS
    echo "== check run: $AREPO_ONLY (ranks ${AREPO_RANKS:-as usual}, Courant ${AREPO_COURANT:-0.3}) -> $MHD_RESULTS"
    run_test $AREPO_ONLY
elif [[ "$task" == 0 ]]; then
    set -e
    for mesh in static moving; do for gamma in 1.6666666666666667 2.0 1.4; do bash "$here/build.sh" "$mesh" "$gamma" 2; done; done
    set +e
    run_test orszag_tang static 32 64; run_test orszag_tang moving 32 64
    python3 "$here/summarize.py" "$MHD_RESULTS" | tee "$here/logs/pilot_summary.txt"
    # The array waits for this task: it must fail unless both meshes ran.
    [[ "$(grep -E "^orszag_tang-arepo-(static|moving)-run-N(32|64)-s42 +complete" "$here/logs/pilot_summary.txt" | wc -l)" -eq 4 ]] \
        || { echo "PILOT FAILED: not all four pilot runs are complete"; exit 1; }
else
    index=$((task - 1)); test="${TESTS[$((index / 2))]}"; mesh=static; [[ $((index % 2)) -eq 1 ]] && mesh=moving
    run_test "$test" "$mesh"
fi
echo "== task $task finished $(date -u)"
