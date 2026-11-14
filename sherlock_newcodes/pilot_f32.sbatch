#!/bin/bash -l
# The single-precision pilot of the GPU codes (after build.sh athenak-f32 athenapk-f32): every set-up of AthenaK and
# AthenaPK in fp32 on the Alfven wave at 64 and the 3D field loop at 32, through production.sbatch, into a
# separate results directory; check_f32.py compares each with its fp64 run. Exit code 0 only if all pass.
#SBATCH --job-name=newcodes-f32-pilot
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
here="$SLURM_SUBMIT_DIR"; mkdir -p "$here/tasks"; lines="$here/tasks/f32_pilot.txt"; : > "$lines"
for cv in athenakf32:matched athenakf32:best athenapkf32:matched athenapkf32:best; do      # Cholla refuses PRECISION=1 (static assertion)
    c="${cv%%:*}"; v="${cv#*:}"; g=plane; [[ "$c" == athenakf32 ]] && g=cube
    printf '%s|%s|alfven|42|64|%s\n%s|%s|field_loop_3d|42|32|cube\n' "$c" "$v" "$g" "$c" "$v" >> "$lines"
done
export NEWCODES_RESULTS="$SCRATCH/mhd-results-newcodes-f32-pilot" NEWCODES_TASKS="$lines" NEWCODES_GROUP=100 SLURM_ARRAY_TASK_ID=1
bash "$here/production.sbatch"
"${NEWCODES_VENV:-$HOME/.venvs/mhd-h5py}/bin/python3" "$here/check_f32.py" "$NEWCODES_RESULTS" "$lines" | tee "$here/logs/f32_pilot_report.txt"
exit "${PIPESTATUS[0]}"
