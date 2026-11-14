#!/bin/bash
# Spectra of a big nested run's snapshots (spectra.py), on a CPU node:   sbatch spectra.sbatch deep192
#SBATCH --job-name=nested-spectra
#SBATCH --partition=normal,owners
#SBATCH --requeue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
module load python/3.12.1 py-numpy/2.2.6_py312
python3 "$SLURM_SUBMIT_DIR/spectra.py" "${NESTED_RESULTS:-$SCRATCH/mhd-results-nested}/big/$1"
