#!/bin/bash -l
# Trial of Castro and Nyx (AMReX-Astro, 2026-11-17) after build.sh castro nyx: small runs of both set-ups of each code on
# one H100, through production.sbatch, into their own results directory; check_amrex.py prints each run's state and its
# end-time error beside other codes' at the same grid. Exit code 0 only if every run completed.
#SBATCH --job-name=newcodes-amrex-trial
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH -G 1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; mkdir -p "$here/tasks"; lines="$here/tasks/amrex_trial.txt"
printf '%s\n' "castro|plm|alfven|42|64|plane" "castro|ppm|alfven|42|64|plane" "castro|plm|orszag_tang|42|64|plane" \
    "castro|ppm|orszag_tang|42|64|plane" "castro|plm|blast_3d_b50|42|32|cube" "castro|ppm|field_loop_3d|42|32|cube" \
    "nyx|plm|entropy|42|64|plane" "nyx|ppm|entropy|42|64|plane" "nyx|ppm|blast_2d_b0|42|64|plane" "nyx|ppm|kh_uniform|42|64|plane" > "$lines"
export NEWCODES_RESULTS="$SCRATCH/mhd-results-newcodes-amrex-trial" NEWCODES_TASKS="$lines" NEWCODES_GROUP=100 SLURM_ARRAY_TASK_ID=1
bash "$here/production.sbatch"
module load python/3.12.1 py-numpy/2.2.6_py312 >/dev/null 2>&1
"${NEWCODES_VENV:-$HOME/.venvs/mhd-h5py}/bin/python3" "$here/check_amrex.py" "$NEWCODES_RESULTS" "$lines" | tee "$here/logs/amrex_trial_report.txt"
exit "${PIPESTATUS[0]}"
