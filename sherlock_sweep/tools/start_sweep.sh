#!/bin/bash
# Start the sweep on Sherlock (login node): fetch the pinned codes, then submit the gates and the coordinator.
#   bash tools/start_sweep.sh [--dry-run] [tools/sweep.py start options]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source tools/sweep_env.sh
printf 'Codes: %s\nResults: %s\n' "$MHD_CODES" "$MHD_RESULTS"
python3 -c 'import sys, numpy, tomllib; assert sys.version_info >= (3,11); print("Python",sys.version.split()[0],"NumPy",numpy.__version__)'
if [[ " $* " != *" --dry-run "* ]]; then bash tools/fetch_codes.sh "$MHD_CODES"; fi
exec python3 tools/sweep.py start "$@"
