#!/bin/bash
# Submits the Courant-number and single-precision checks (README.md). Run it in the bundle directory on Sherlock:
#   bash -l check.sh
# The driver source in src/ is the sweep's (it arrives with the bundle); the kernels, the Julia environment and the
# modules are the sweep bundle's own, used in place.
#   CHECK_GPU_CONSTRAINT   GPU model to ask for (default: the sweep's, GPU_SKU:H100_SXM5; "none": any GPU)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
[[ -s src/run_suite.jl && -s src/problems.jl ]] || { echo "src/ is missing: send the bundle with the command in README.md"; exit 1; }
grep -q glm_hll32 src/run_suite.jl || { echo "src/run_suite.jl is not the extended driver"; exit 1; }
[[ -s "$SWEEP_BUNDLE/vendor/glmmhd_turb.jl" && -s "$SWEEP_BUNDLE/env/Manifest.toml" ]] || { echo "no sweep bundle at $SWEEP_BUNDLE"; exit 1; }
for d in vendor env; do                 # the driver records and reads these beside its own src/
    [[ -e "$d" ]] || ln -s "$SWEEP_BUNDLE/$d" "$d"
done
mkdir -p logs
if squeue -h -u "$USER" -n mhd-checks -o %i | grep -q .; then
    echo "a mhd-checks job is already in the queue; nothing submitted:"; squeue -u "$USER" -n mhd-checks; exit 0
fi
feature=(); want="${CHECK_GPU_CONSTRAINT:-GPU_SKU:H100_SXM5}"; [[ "$want" == none ]] || feature=(--constraint="$want")
id="$(sbatch --parsable ${feature[@]+"${feature[@]}"} checks.sbatch)"
echo "submitted mhd-checks as job ${id%%;*}. Follow it with: bash -l $PWD/status.sh"
