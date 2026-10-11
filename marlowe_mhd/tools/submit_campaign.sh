#!/bin/bash
# One-command Marlowe GPU campaign; jobs are gated by Slurm afterok dependencies.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f site/marlowe.env ]]; then
    # shellcheck disable=SC1091
    source site/marlowe.env
fi
export MHD_SITE=marlowe
source jobs/common.sh
load_site_profile
prepare_environment
prepare_codes
"$MHD_JULIA" --version
python3 -c 'import sys, numpy, tomllib; assert sys.version_info >= (3,11); print("Python",sys.version.split()[0],"NumPy",numpy.__version__)'
exec python3 tools/submit_campaign.py --site=marlowe --gpu-only "$@"
