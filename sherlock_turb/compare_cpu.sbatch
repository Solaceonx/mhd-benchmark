#!/bin/bash
# One dense RAMSES movie per array task; the launcher limits concurrency to one.
#SBATCH --partition=normal,owners
#SBATCH --requeue
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -euo pipefail
here="$SLURM_SUBMIT_DIR"
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE"
source jobs/common.sh
prepare_environment
prepare_codes
load_modules "${MHD_CPU_BUILD_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
cd "$here"
python3 - <<'PY'
import json, os, subprocess, sys
task=json.load(open(os.environ['TURB_COMPARE_TASKS']))[int(os.environ['SLURM_ARRAY_TASK_ID'])]
command=[sys.executable,'ramses_movie.py',task['config'],task['results'],task['variant'],str(task['mach']),str(task['n']),
         '--exe',task['exe'],'--mpi','srun -n '+os.environ['SLURM_NTASKS']]
sys.exit(subprocess.run(command,stdin=subprocess.DEVNULL).returncode)
PY
