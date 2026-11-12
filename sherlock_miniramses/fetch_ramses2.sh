#!/bin/bash
# The official RAMSES II at the pinned commit (run once, on a login node), for ramses2.sbatch.
set -euo pipefail
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"
repo=https://github.com/ramses-organisation/ramses2.git; commit=04abdd9e0f44bbcd85a865d5b00c3fd9eb22bf9f
# develop of 2026-11-12: E. Moseley's UCT-HLLD on CUDA (PR #62), the LLF-switch fix (#64), the forcing-projection fix (#57)
src="$codes/ramses2-${commit:0:7}"; mkdir -p "$codes"
[[ -d "$src/.git" ]] || git clone --quiet "$repo" "$src"
cd "$src"                                   # not `git -C`: the login nodes' git predates it
git cat-file -e "$commit^{commit}" 2>/dev/null || git fetch --quiet origin
git checkout --quiet "$commit"
echo "RAMSES II at $(git rev-parse HEAD) in $src"
