#!/bin/bash
# Get the mini-ramses-ism source at the pinned commit (run once, on a login node) and show which compiler modules exist.
set -euo pipefail
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/miniramses-ism"
repo=https://bitbucket.org/ermoseley/mini-ramses-ism.git; commit=579d8423bbc1420dc4068d572b007ec35694e292   # develop, 2026-10-22
mkdir -p "$codes"
[[ -d "$src/.git" ]] || git clone --quiet --branch develop "$repo" "$src"
(cd "$src" && git fetch --quiet origin develop && git checkout --quiet "$commit")   # cd, not `git -C`: old git on the login nodes
echo "mini-ramses-ism at $(cd "$src" && git rev-parse HEAD) in $src"
echo "== compiler modules (the job picks the newest nvhpc that builds the code; MINIRAMSES_NVHPC_MODULE chooses one)"
if ! module -t avail nvhpc 2>&1 | grep -i '^nvhpc'; then
    echo "no nvhpc module in the default view; module spider nvhpc says:"; module spider nvhpc 2>&1 | head -n 20
fi
