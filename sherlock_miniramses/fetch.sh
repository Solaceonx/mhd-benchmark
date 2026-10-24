#!/bin/bash
# Get the mini-ramses-ism source at the pinned commit (run once, on a login node) and show which compiler modules exist.
set -euo pipefail
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/miniramses-ism"
repo=https://bitbucket.org/ermoseley/mini-ramses-ism.git; commit=579d8423bbc1420dc4068d572b007ec35694e292   # develop, 2026-10-22
mkdir -p "$codes"
# The repository went private on 2026-10-23 (404 without a login): Bitbucket is contacted only when the pinned commit is
# not here yet. A copy cloned on 2026-10-22 has it.
if [[ ! -d "$src/.git" ]]; then
    GIT_TERMINAL_PROMPT=0 git clone --quiet --branch develop "$repo" "$src" || {
        echo "Cannot clone $repo (private now?): ask Eric Moseley for access, or copy the source to $src" >&2; exit 2; }
fi
cd "$src"                                     # cd, not `git -C`: old git on the login nodes
if ! git cat-file -e "$commit^{commit}" 2>/dev/null; then
    GIT_TERMINAL_PROMPT=0 git fetch --quiet origin develop || {
        echo "Commit $commit is not in $src and $repo cannot be read (private now?)" >&2; exit 2; }
fi
git checkout --quiet "$commit"
cd - >/dev/null
echo "mini-ramses-ism at $(cd "$src" && git rev-parse HEAD) in $src"
echo "== compiler modules (the job picks the newest nvhpc that builds the code; MINIRAMSES_NVHPC_MODULE chooses one)"
if ! module -t avail nvhpc 2>&1 | grep -i '^nvhpc'; then
    echo "no nvhpc module in the default view; module spider nvhpc says:"; module spider nvhpc 2>&1 | head -n 20
fi
