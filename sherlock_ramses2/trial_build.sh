#!/bin/bash
# The trial build of check_build.sh (see there): RAMSES II built without a GPU, in a directory of its own. Ends by
# writing "OK <compiler module>" or "FAILED" to logs/trial_status.txt.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"; mkdir -p logs; status="$here/logs/trial_status.txt"
echo STARTED > "$status"
export MHD_CODES="${MHD_CODES:-${SCRATCH:?SCRATCH is not set}/mhd-codes-sweep}"
export RAMSES2_BUILD_ROOT="$MHD_CODES/ramses2-build-check" RAMSES2_CUDA_ARCH="${RAMSES2_CUDA_ARCH:-sm_90}"      # sm_90: the H100
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
type module >/dev/null 2>&1 || { echo "no module command in this shell (run with bash -l)"; echo FAILED > "$status"; exit 2; }
module load devel 2>/dev/null || true; module load python/3.12.1 2>/dev/null || true
rm -f "$RAMSES2_BUILD_ROOT/ot_ns1_p8/bin/ramses_ot3d" "$RAMSES2_BUILD_ROOT/nvhpc_module.txt"       # always a fresh trial
echo "== trial build on $(hostname), $(date -u), $SLURM_CPUS_PER_TASK compilations at a time"
source "$here/compiler.sh"
if choose_compiler_and_build; then echo "OK $nvhpc_chosen" > "$status"; else echo FAILED > "$status"; exit 1; fi
