#!/bin/bash
# Submits the checks of the sweep-fixes kernels (README.md, "Checking the fixes"). In the bundle directory on Sherlock:
#   bash -l verify.sh
# vendor/ (the branch) and src/ (the driver) arrive with the bundle; the Julia environment and modules are the sweep
# bundle's own, used in place.
#   VERIFY_GPU_CONSTRAINT   GPU model to ask for (default: the sweep's, GPU_SKU:H100_SXM5; "none": any GPU)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
[[ -s src/run_suite.jl && -s src/problems.jl ]] || { echo "src/ is missing: send the bundle with the command in README.md"; exit 1; }
grep -q glm_fct16 src/run_suite.jl || { echo "src/run_suite.jl does not know the fixed kernels"; exit 1; }
grep -q fct_scratch vendor/glmmhd_turb.jl 2>/dev/null || { echo "vendor/ does not hold the sweep-fixes branch (run prepare.sh on the Mac)"; exit 1; }
[[ -s "$SWEEP_BUNDLE/env/Manifest.toml" ]] || { echo "no sweep bundle at $SWEEP_BUNDLE"; exit 1; }
[[ -e env ]] || ln -s "$SWEEP_BUNDLE/env" env
mkdir -p logs
if squeue -h -u "$USER" -n mhd-verify -o %i | grep -q .; then
    echo "a mhd-verify job is already in the queue; nothing submitted:"; squeue -u "$USER" -n mhd-verify; exit 0
fi
feature=(); want="${VERIFY_GPU_CONSTRAINT:-GPU_SKU:H100_SXM5}"; [[ "$want" == none ]] || feature=(--constraint="$want")
id="$(sbatch --parsable ${feature[@]+"${feature[@]}"} verify.sbatch)"
echo "submitted mhd-verify as job ${id%%;*}. Follow it with: bash -l $PWD/status.sh verify"
