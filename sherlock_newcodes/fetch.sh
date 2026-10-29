#!/bin/bash
# Get the new codes' sources at pinned commits (run on a login node): bash fetch.sh
# -> $MHD_CODES/newcodes/src/{athenapk (with its Parthenon and Kokkos submodules), cholla, gpluto, enzo, phantom}.
# Nothing is built here (build.sh, in a job). The sources are never changed: build.sh patches copies.
set -euo pipefail
# git 2.45 (module): AthenaPK pins a Parthenon commit only reachable by its id, which the system git (1.8) cannot fetch
module load system git/2.45.1 >/dev/null 2>&1 || true
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/newcodes/src"; mkdir -p "$src"; cd "$src"
get() {     # directory, repository, full commit id
    local dir="$1" url="$2" ref="$3"
    [[ -d "$dir/.git" ]] || git clone --quiet "$url" "$dir"
    # cd, not `git -C`: the login nodes' git is old; fetch a commit made after the clone
    (cd "$dir" && { git cat-file -e "$ref^{commit}" 2>/dev/null || git fetch --quiet origin; } && git checkout --quiet "$ref")
    echo "$dir: $(cd "$dir" && git rev-parse HEAD)"
}
get athenapk https://github.com/parthenon-hpc-lab/athenapk.git 6d31f708650acdc0e9f54059a4fc5e54f81a1350
(cd athenapk && { git submodule update --init --recursive --quiet 2>/dev/null || {
    # the pinned Parthenon commit lies on a merged branch: fetch it by its id, then update again
    git submodule update --init --quiet external/Kokkos external/catch2 || true
    p="$(git ls-tree HEAD external/parthenon | awk '{print $3}')"
    [[ -d external/parthenon/.git || -f external/parthenon/.git ]] || git submodule update --init --quiet --no-fetch external/parthenon 2>/dev/null || true
    (cd external/parthenon && git fetch --quiet origin "$p" && git checkout --quiet "$p")
    git submodule update --init --recursive --quiet; }; } && git submodule status | sed 's/^/   /')
get cholla https://github.com/cholla-hydro/cholla.git 191e59b063b4a4075738b0f016fd2203ab8b1523
get gpluto https://gitlab.com/PLUTO-code/gPLUTO.git 870aa8079326d48271123dbe3ed642c09c3c9fb4
get enzo https://github.com/enzo-project/enzo-dev.git 45f4c64b93105969b5e862386661ca75d8dd506c
get phantom https://github.com/danieljprice/phantom.git f535c0742cea3344109bb79595b565a37834e03a
