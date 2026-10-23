#!/bin/bash
# GIZMO on the sweep's planar tests (README.md). One array task per test and set-up, running that test's grids in turn;
# task 0 is the pilot: it builds every executable and runs Orszag-Tang at 32^2 and 64^2 with both methods.
#   p=$(sbatch --parsable --array=0 gizmo.sbatch) && sbatch --dependency=afterok:$p --array=1-34 gizmo.sbatch
# A requeued task keeps its finished runs and starts the interrupted one again.
#SBATCH --job-name=gizmo
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
load_modules "${GIZMO_COMPILER_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
for m in ${GIZMO_MODULES:-gsl/2.7 hdf5/1.14.4}; do module load "$m" 2>/dev/null || echo "module $m: not loaded"; done
# Python: h5py and scipy from pip in a private environment on scratch, the one sherlock_arepo made (Sherlock's py-h5py
# module does not import with the numpy 2 module the sweep loads). Wheels only.
venv="${GIZMO_VENV:-$SCRATCH/mhd-venv-arepo}"; [[ -x "$venv/bin/python3" ]] || python3 -m venv --system-site-packages "$venv"
source "$venv/bin/activate"; python3 -c 'import h5py, scipy' 2>/dev/null || pip install --quiet --only-binary=:all: "h5py>=3.11" "scipy>=1.13"
python3 -c 'import h5py, scipy; print("python:", h5py.__version__, "h5py,", scipy.__version__, "scipy")' || { echo "NO h5py/scipy: cannot write GIZMO initial conditions"; exit 2; }
echo "== task $task on $(hostname), $(date -u)"; module list 2>&1 | tail -n +2

TESTS=(entropy alfven alfven_oblique uniform field_loop field_loop_novz brio_wu brio_wu_rotated orszag_tang rotor current_sheet
       blast_2d_b0 blast_2d_b50 blast_2d_b100 kh_uniform kh_localized gem)
gamma_of() { python3 -c "import sys; sys.path.insert(0, '$SWEEP_BUNDLE/codes'); import external as X; print(repr(float(X.gamma_of(X.load_config(sys.argv[1])))))" "$1"; }
run_test() {     # test method [grids...]
    local test="$1" method="$2"; shift 2
    local config="$SWEEP_BUNDLE/configs/$test.toml" gamma exe n ranks out rc
    gamma="$(gamma_of "$config")"; exe="$MHD_CODES/gizmo-build/${method}_g${gamma}_2d/GIZMO"
    [[ -x "$exe" ]] || bash "$here/build.sh" "$method" "$gamma" 2 || { echo "NO EXECUTABLE for $test $method"; return 1; }
    [[ $# -gt 0 ]] || set -- $(python3 -c "import tomllib, sys; print(*tomllib.load(open(sys.argv[1], 'rb'))['grids'])" "$config")
    for n in "$@"; do
        ranks=1; [[ $n -ge 128 ]] && ranks=4; [[ $n -ge 256 ]] && ranks=8; [[ $n -ge 512 ]] && ranks=16
        ranks="${GIZMO_RANKS:-$ranks}"
        out="$MHD_RESULTS/$test-gizmo-$method-run-N$n-s42/data"
        python3 "$here/gizmo_run.py" gizmo "$method" "$config" "$n" "$out" --exe "$exe" --mpi "srun -n $ranks" --geometry plane \
            --repeats 1 --resume --device "$(hostname)" --device-kind cpu; rc=$?
        [[ $rc -eq 76 ]] && { echo "this job is being stopped; it will be requeued"; exit 1; }
        [[ $rc -ne 0 ]] && echo "   $test $method N=$n: exit code $rc (see its run.toml and gizmo.log)"
    done
    return 0
}

if [[ -n "${GIZMO_ONLY:-}" ]]; then        # a check run: GIZMO_ONLY="test method grid...", with GIZMO_RANKS, MHD_RESULTS
    echo "== check run: $GIZMO_ONLY (ranks ${GIZMO_RANKS:-as usual}) -> $MHD_RESULTS"
    run_test $GIZMO_ONLY
elif [[ "$task" == 0 ]]; then
    set -e
    for method in mfm mfv; do for gamma in 1.6666666666666667 2.0 1.4; do bash "$here/build.sh" "$method" "$gamma" 2; done; done
    set +e
    run_test orszag_tang mfm 32 64; run_test orszag_tang mfv 32 64
    python3 "$here/summarize.py" "$MHD_RESULTS" | tee "$here/logs/pilot_summary.txt"
    # The array waits for this task: it must fail unless both methods ran.
    [[ "$(grep -cE "^orszag_tang-gizmo-(mfm|mfv)-run-N(32|64)-s42 +complete" "$here/logs/pilot_summary.txt")" -eq 4 ]] \
        || { echo "PILOT FAILED: not all four pilot runs are complete"; exit 1; }
else
    index=$((task - 1)); test="${TESTS[$((index / 2))]}"; method=mfm; [[ $((index % 2)) -eq 1 ]] && method=mfv
    run_test "$test" "$method"
fi
echo "== task $task finished $(date -u)"
