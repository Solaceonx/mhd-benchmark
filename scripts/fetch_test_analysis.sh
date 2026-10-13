#!/bin/bash
# Mac only: fetch the missing analysis inputs after the user authenticates to Sherlock.
# No simulation or job submission. Missing remote files are skipped and reported.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-movie/bin/python scripts/plan_test_analysis_fetch.py
ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu \
 'cd /scratch/users/lyxia/mhd-results-sweep && while IFS= read -r f; do if [ -f "$f" ]; then printf "%s\n" "$f"; else printf "not on scratch: %s\n" "$f" >&2; fi; done | tar -czf - -T -' \
 < scripts/test_analysis_fetch_needed.txt | tar -xzf - -C sweep_results
echo "Fetched available requested inputs. Rebuild with .venv-movie/bin/python scripts/test_diagnostics.py"
