#!/bin/bash
# Submits the order x solver study (README.md). On Sherlock:
#   bash -l submit.sh            the three pilots (Orszag-Tang at 32^2 and 64^2 with every new set-up), and the arrays
#                                that wait on them: each starts only if its pilot passed
#   bash -l submit.sh resume     the arrays again, without pilots (after a task ran out of time; finished runs are kept)
#   ORDER_HOURS          planned length of each array task (default 6)
#   ORDER_GPU_CONSTRAINT GPU model to ask for (default: the sweep's, GPU_SKU:H100_SXM5; "none": any GPU)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-pilot}"; [[ "$mode" == pilot || "$mode" == resume ]] || { echo "usage: bash -l submit.sh [resume]"; exit 1; }
SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"; export SWEEP_BUNDLE
codes="${MHD_CODES:-$SCRATCH/mhd-codes-sweep}"
grep -q glm_fix_ppm_hlld src/run_suite.jl 2>/dev/null || { echo "src/ is missing or does not know the new kernels: send the bundle with the command in README.md"; exit 1; }
grep -q fct_scratch vendor/glmmhd_turb.jl 2>/dev/null || { echo "vendor/ does not hold the sweep-fixes branch (run prepare.sh on the Mac)"; exit 1; }
[[ -s configs/estimates.json ]] || { echo "configs/estimates.json is missing: send the bundle again"; exit 1; }
[[ -s "$SWEEP_BUNDLE/env/Manifest.toml" ]] || { echo "no sweep bundle at $SWEEP_BUNDLE"; exit 1; }
[[ -x "$codes/athenak/build-HOPPER90-fp64/src/athena" ]] || { echo "no AthenaK H100 build in $codes"; exit 1; }
for f in hlle hlld; do [[ -x "$codes/athena/bin/athena_$f" ]] || { echo "no Athena++ build $codes/athena/bin/athena_$f"; exit 1; }; done
[[ -e env ]] || ln -s "$SWEEP_BUNDLE/env" env
module load python/3.12.1 >/dev/null 2>&1 || true
if squeue -h -u "$USER" -o %j | grep -q '^order-'; then
    echo "order-* jobs are already in the queue; nothing submitted:"; squeue -u "$USER" -o "%.18i %.14j %.9T %.10M %R" | grep -E 'JOBID|order-'; exit 0
fi
mkdir -p logs
hours="${ORDER_HOURS:-6}"
gpu=(); want="${ORDER_GPU_CONSTRAINT:-GPU_SKU:H100_SXM5}"; [[ "$want" == none ]] || gpu=(--constraint="$want")
after() { [[ "$mode" == pilot ]] && echo "--dependency=afterok:$1" || true; }
for kind in glm athenak; do
    p=""; [[ "$mode" == pilot ]] && p="$(sbatch --parsable ${gpu[@]+"${gpu[@]}"} --job-name="order-$kind" --array=0 --export=ALL,KIND="$kind",PILOT=1 gpu.sbatch)"
    read -r _ n < <(python3 order.py groups "$kind" "$hours")
    id="$(sbatch --parsable ${gpu[@]+"${gpu[@]}"} $(after "${p%%;*}") --job-name="order-$kind" --array=0-$((n-1)) --export=ALL,KIND="$kind",NTASKS="$n" gpu.sbatch)"
    echo "order-$kind: ${p:+pilot ${p%%;*}, }array ${id%%;*} of $n tasks"
done
p=""; [[ "$mode" == pilot ]] && p="$(sbatch --parsable --ntasks=2 --job-name=order-athenapp --array=0 --export=ALL,PILOT=1 cpu.sbatch)"
ids=()
while read -r ranks n; do
    id="$(sbatch --parsable --ntasks="$ranks" $(after "${p%%;*}") --job-name=order-athenapp --array=0-$((n-1)) --export=ALL,RANKS="$ranks",NTASKS="$n" cpu.sbatch)"
    ids+=("${id%%;*} ($ranks ranks x $n)")
done < <(python3 order.py groups athenapp "$hours")
echo "order-athenapp: ${p:+pilot ${p%%;*}, }arrays ${ids[*]}"
echo "Follow them with: bash -l $PWD/status.sh"
