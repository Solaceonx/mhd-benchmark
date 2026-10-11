#!/bin/bash
# RAMSES II GPU speed test on one H100 (see README.md): 3D Orszag-Tang at 64^3 to 512^3 for every build (nsubgrid 1
# and 2, double and single precision) with the HLL and HLLD solvers. A failed build or run is reported and the job
# goes on; a requeued job skips what is already done. Submit it with check_build.sh, which first tries the build.
#SBATCH --job-name=ramses2-speed
#SBATCH --partition=gpu,owners
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; here="$PWD"
export MHD_CODES="${MHD_CODES:-$SCRATCH/mhd-codes-sweep}"
results="${RAMSES2_RESULTS:-$SCRATCH/mhd-results-ramses2}"; mkdir -p "$results"
echo "== $(date -u) on $(hostname); $(nvidia-smi -L | head -n1)"

# ---- compiler: the newest NVIDIA HPC SDK module that builds the code (compiler.sh)
module load devel 2>/dev/null || true
module load python/3.12.1 2>/dev/null || true
source "$here/compiler.sh"
choose_compiler_and_build && built=1 || { built=0; echo "NO COMPILER BUILT RAMSES II: nothing to run"; }

run_one() {      # variant exe level solver [dump]
    local variant="$1" exe="$2" level="$3" solver="$4" dump="${5:-}"
    local d="$results/$variant/${solver}_L$level${dump:+_dump}"; mkdir -p "$d"
    grep -q "Run completed" "$d/run.log" 2>/dev/null && { echo "   done already: $variant ${solver}_L$level $dump"; return 0; }
    python3 "$here/namelist.py" "$level" "$solver" $dump > "$d/run.nml"
    ( cd "$d"
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > gpu_mem.txt 2>/dev/null & smi=$!
      timeout "${RAMSES2_RUN_LIMIT:-1800}" "$exe" run.nml > run.log 2>&1; rc=$?
      kill "$smi" 2>/dev/null; echo "exit code $rc" >> run.log )
    echo "   ran $variant ${solver}_L$level $dump: $(grep -c 'Time elapsed since last coarse step' "$d/run.log") steps, $(tail -n1 "$d/run.log")"
}

# ---- builds and runs (to t = 0.05, the end of the sweep's throughput test)
{ date -u; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; module list 2>&1; } > "$results/provenance.txt"
if [[ $built -eq 1 ]]; then
    for npre in 8 4; do for ns in 1 2; do
        variant="ot_ns${ns}_p${npre}"; exe="$MHD_CODES/ramses2-build/$variant/bin/ramses_ot3d"
        bash "$here/build.sh" "$ns" "$npre" || { echo "   skipping $variant"; mkdir -p "$results/$variant"; cp "$MHD_CODES/ramses2-build/$variant/build.log" "$results/$variant/" 2>/dev/null; continue; }
        mkdir -p "$results/$variant"; cp "$MHD_CODES/ramses2-build/$variant/BUILD_INFO.txt" "$MHD_CODES/ramses2-build/$variant/build.log" "$results/$variant/" 2>/dev/null
        for solver in hll hlld; do for level in 6 7 8 9; do run_one "$variant" "$exe" "$level" "$solver"; done; done
    done; done
    # One run that also writes the code's snapshots (t = 0 and the end), kept on scratch to check the fields later.
    exe="$MHD_CODES/ramses2-build/ot_ns1_p8/bin/ramses_ot3d"; [[ -x "$exe" ]] && run_one ot_ns1_p8 "$exe" 8 hlld dump
fi
python3 "$here/summarize.py" "$results" --json "$results/summary.json" | tee "$results/summary.txt"
{ echo; echo "compiler: ${nvhpc_chosen:-none}"; grep -H "objects built below -O4" "$results"/*/BUILD_INFO.txt 2>/dev/null | sed "s|$results/||"; } | tee -a "$results/summary.txt"
# Everything small in one file, to copy back: logs, namelists, memory samples, build records, the summary.
( cd "$results" && find . \( -name 'run.log' -o -name 'run.nml' -o -name 'gpu_mem.txt' -o -name 'BUILD_INFO.txt' -o -name 'build.log' -o -name 'summary.*' -o -name 'provenance.txt' \) -print0 \
    | tar --null -czf "$results/ramses2_speed_records.tar.gz" -T - && ls -la "$results/ramses2_speed_records.tar.gz" )

echo "== finished $(date -u). Summary: $results/summary.txt"
