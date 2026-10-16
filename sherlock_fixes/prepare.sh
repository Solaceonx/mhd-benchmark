#!/bin/bash
# On the Mac, before sending: put the sweep-fixes branch of Tom Abel's kernels into vendor/ (git-ignored: his source is
# not committed to this repository) and record which commit it is. Takes the files from the branch itself (git show),
# so the clone's working tree and any local edits do not matter.
#   bash sherlock_fixes/prepare.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; clone="$here/../GLMMHDTurb"; branch=sweep-fixes
git -C "$clone" rev-parse --verify -q "$branch" >/dev/null || { echo "no branch $branch in $clone"; exit 1; }
mkdir -p "$here/vendor"
for f in glmmhd_turb.jl ct_mhd.jl; do git -C "$clone" show "$branch:$f" > "$here/vendor/$f"; done
commit="$(git -C "$clone" rev-parse "$branch")"
printf '{"repository": "github.com/yipihey/GLMMHDTurb", "branch": "%s", "commit": "%s", "base": "%s",\n "sha256": {"glmmhd_turb.jl": "%s", "ct_mhd.jl": "%s"}}\n' \
    "$branch" "$commit" "$(git -C "$clone" merge-base "$branch" origin/main)" \
    "$(shasum -a 256 "$here/vendor/glmmhd_turb.jl" | cut -d' ' -f1)" "$(shasum -a 256 "$here/vendor/ct_mhd.jl" | cut -d' ' -f1)" > "$here/vendor/BRANCH.json"
echo "vendor/ holds $branch at ${commit:0:7}"
