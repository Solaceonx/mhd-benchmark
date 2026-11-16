#!/bin/bash -l
# AthenaPK with first-order flux correction (newcodes_run.py variants matched_fofc, best_fofc; 2026-11-16) on the Mach 9
# decay: best (PPM/HLLD/RK3) at 256^3, which went NaN by t = 0.02 without it, then matched (PLM/HLLE/VL2) at 512^3 seed 42,
# which gained 180 % total energy at t = 0.054 (dfloor kept the momentum of an emptied cell). Own results directory;
# compare with python3 check_fofc.py afterwards.          cd sherlock_newcodes && sbatch fofc_test.sbatch
#SBATCH --job-name=newcodes-fofc
#SBATCH --partition=gpu,owners
#SBATCH --requeue
#SBATCH -G 1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; mkdir -p "$here/tasks"; lines="$here/tasks/fofc_test.txt"
printf '%s\n' "athenapk|best_fofc|decay_supersonic|42|256|cube" "athenapk|matched_fofc|decay_supersonic|42|512|cube" > "$lines"
export NEWCODES_RESULTS="$SCRATCH/mhd-results-newcodes-fofc" NEWCODES_TASKS="$lines" NEWCODES_GROUP=100 SLURM_ARRAY_TASK_ID=1
bash "$here/production.sbatch"
