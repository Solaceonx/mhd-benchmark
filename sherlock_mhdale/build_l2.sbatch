#!/bin/bash
# Build MHD-ALE with the projected initial state (testcase_Abel.cpp, ABEL_IC=l2; 2026-11-17) into build-abel-l2, beside
# build-abel (the published runs' executable, left as it is), then a 32^2 entropy run of each pilot set-up on one rank.
# Exit code 0 only if both ran to the end. Submitted by pilot_l2.sh.
#SBATCH --job-name=mhdale-build-l2
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { echo "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHDALE_COMPILER_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
root="$MHD_CODES/mhdale-yipihey"; build="$root/build-abel-l2"
echo "== building $build on $(hostname), $(date -u)"
rm -rf "$build" && python3 "$here/prepare_patch.py" "$root/src/mhd-ale" "$build" "$SWEEP_BUNDLE/codes/common/abel_problems.hpp" \
    && (cd "$build" && make -j "${SLURM_CPUS_PER_TASK:-8}" MFEM_INSTALL_DIR="$root/mfem-install" MHD > build.log 2>&1) \
    || { echo "BUILD FAILED"; grep -n -i -E "error" "$build/build.log" | head -20; exit 1; }
echo "   built $build/MHD"
smoke="$SCRATCH/mhd-results-mhdale-l2-smoke"; rm -rf "$smoke"; bad=0
for v in ale_l2 ale_l2_q4; do
    out="$smoke/entropy-mhdale-$v-run-N32-s42/data"
    python3 "$here/mhdale_run.py" mhdale "$v" "$SWEEP_BUNDLE/configs/entropy.toml" 32 "$out" --exe "$build/MHD" --mpi "srun -n 1" \
        --geometry plane --repeats 1 --device "$(hostname)" --device-kind cpu --keep-native; rc=$?
    grep -h "initial state" "$out"/mhdale_*_r1/mhdale.log | head -1
    rec="$out/mhdale_${v}_r1/run.toml"; st=$(grep -m1 '^status' "$rec" 2>/dev/null)
    echo "   $v 32^2: exit $rc, $st"; [[ $rc -eq 0 && "$st" == *complete* ]] || bad=1
done
[[ $bad -eq 0 ]] && echo "BUILD AND SMOKE PASSED" || echo "SMOKE FAILED"
exit $bad
