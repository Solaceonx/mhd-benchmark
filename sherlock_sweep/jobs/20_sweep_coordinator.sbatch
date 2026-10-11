#!/bin/bash
# Sweep coordinator: every 5 minutes, one `tools/sweep.py tick` (gates, then waves of units under the QOS cap,
# resubmission of interrupted tasks). One CPU, little memory. After 46 h it submits its successor and exits,
# so the sweep keeps going for as long as it needs. Stops when every unit is finished or a gate fails.
#SBATCH --job-name=sweep-coordinator
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/common.sh
prepare_environment
prepare_codes
start=$(date +%s)
while true; do
    python3 tools/sweep.py tick
    rc=$?
    case "$rc" in
        0) ;;
        10) printf 'Sweep finished at %s\n' "$(date -u)"; exit 0 ;;
        3) printf 'A gate failed; the coordinator stops. See sweep_state/status.txt\n' >&2; exit 3 ;;
        *) printf 'tick failed with %s at %s; retrying in 5 minutes\n' "$rc" "$(date -u)" >&2 ;;
    esac
    if (( $(date +%s) - start > 46*3600 )); then
        # Without this job's own SLURM_* variables, which --export=ALL would pass on to the successor.
        next="$(env $(compgen -e | grep '^SLURM_' | grep -vx SLURM_CONF | sed 's/^/-u /') \
            sbatch --parsable --chdir="$PWD" --export=ALL --dependency="afterany:$SLURM_JOB_ID" jobs/20_sweep_coordinator.sbatch)"
        printf '%s %s\n' "$next" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> sweep_state/coordinator.txt
        printf 'Handing over to coordinator job %s\n' "$next"; exit 0
    fi
    sleep 300
done
