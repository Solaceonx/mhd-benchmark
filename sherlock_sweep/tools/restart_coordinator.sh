#!/bin/bash
# Submit a new coordinator for a sweep already started (e.g. after the old one died). Ticks are idempotent.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f sweep_state/gates.json ]] || { printf 'No sweep here (sweep_state/gates.json missing); use tools/start_sweep.sh\n' >&2; exit 2; }
if squeue -u "$USER" -h -n sweep-coordinator -o %i | grep -q .; then
    printf 'A coordinator is already queued or running:\n' >&2; squeue -u "$USER" -n sweep-coordinator >&2; exit 2
fi
source tools/sweep_env.sh
job="$(sbatch --parsable --chdir="$PWD" --export=ALL jobs/20_sweep_coordinator.sbatch)"
printf '%s %s\n' "$job" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> sweep_state/coordinator.txt
printf 'Coordinator job %s\n' "$job"
