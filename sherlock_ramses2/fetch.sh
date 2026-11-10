#!/bin/bash
# Get the RAMSES II source at the pinned commit (run once, on a login node) and show which compiler modules exist.
set -euo pipefail
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/ramses2"
repo=https://github.com/ramses-organisation/ramses2.git; commit=ec61647730de1bcc9fec9a6c9a66814d6fded434
# 2026-11-09: develop of 2026-11-08, with E. Moseley's fused-HLLD GPU work (PRs #45, #46); the first test (2026-10-06,
# results in $SCRATCH/mhd-results-ramses2, fix_results/ on the Mac) was aafd2abbffb37a0a4141e06a7ba791a566afa528
mkdir -p "$codes"
[[ -d "$src/.git" ]] || git clone --quiet "$repo" "$src"
(cd "$src" && { git cat-file -e "$commit^{commit}" 2>/dev/null || git fetch --quiet origin; } && git checkout --quiet "$commit")      # cd, not `git -C`: the login nodes' git predates that option
echo "RAMSES II at $(cd "$src" && git rev-parse HEAD) in $src"
echo "== compiler modules (the job picks the newest nvhpc; set RAMSES2_NVHPC_MODULE to choose)"
if ! module -t avail nvhpc 2>&1 | grep -i '^nvhpc'; then
    echo "no nvhpc module in the default view; module spider nvhpc says:"; module spider nvhpc 2>&1 | head -n 20
fi
