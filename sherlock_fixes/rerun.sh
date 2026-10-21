#!/bin/bash
# Submits the full rerun of one group of fixed kernels as a Slurm array (README.md, "The full rerun"). On Sherlock:
#   bash -l rerun.sh glm     glm_fix_hll16, glm_fct16, glm_fix_ppm, glm_fix_hlld
#   bash -l rerun.sh ct      ct_fix, once the emfcheck job has passed
# Run it again after a task ran out of time: finished runs are kept and an interrupted one resumes from its checkpoint.
#   RERUN_GPU_CONSTRAINT   GPU model to ask for (default: the sweep's, GPU_SKU:H100_SXM5; "none": any GPU)
#   RERUN_HOURS            planned length of each task (default 8)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
group="${1:-}"; [[ "$group" == glm || "$group" == ct ]] || { echo "usage: bash -l rerun.sh glm|ct"; exit 1; }
SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
[[ -s src/run_suite.jl && -s src/problems.jl ]] || { echo "src/ is missing: send the bundle with the command in README.md"; exit 1; }
grep -q glm_fct16 src/run_suite.jl || { echo "src/run_suite.jl does not know the fixed kernels"; exit 1; }
grep -q ecc_kernel vendor/ct_mhd.jl 2>/dev/null || { echo "vendor/ does not hold the sweep-fixes branch with the upwinded CT (run prepare.sh on the Mac)"; exit 1; }
[[ -s configs_rerun/estimates.json ]] || { echo "configs_rerun/estimates.json is missing: send the bundle again"; exit 1; }
[[ -s "$SWEEP_BUNDLE/env/Manifest.toml" ]] || { echo "no sweep bundle at $SWEEP_BUNDLE"; exit 1; }
[[ -e env ]] || ln -s "$SWEEP_BUNDLE/env" env
module load python/3.12.1 >/dev/null 2>&1 || true
name="mhd-rerun-$group"
if squeue -h -u "$USER" -n "$name" -o %i | grep -q .; then
    echo "a $name job is already in the queue; nothing submitted:"; squeue -u "$USER" -n "$name"; exit 0
fi
n="$(python3 fixes.py --plan rerun tasks "$group" "${RERUN_HOURS:-8}" | head -n 1)"
[[ "$n" =~ ^[0-9]+$ ]] || { echo "could not plan the tasks"; exit 1; }
mkdir -p logs
feature=(); want="${RERUN_GPU_CONSTRAINT:-GPU_SKU:H100_SXM5}"; [[ "$want" == none ]] || feature=(--constraint="$want")
id="$(sbatch --parsable ${feature[@]+"${feature[@]}"} --job-name="$name" --array=0-$((n-1)) --export=ALL,KERNELS="$group",NTASKS="$n" rerun.sbatch)"
echo "submitted $name as array job ${id%%;*}: $n tasks of about ${RERUN_HOURS:-8} GPU hours. Follow it with: bash -l $PWD/status.sh rerun"
