#!/bin/bash
# Clone the pinned external codes (AthenaK with its Kokkos submodule, RAMSES, Athena++).
# Run once on a Sherlock login node: bash tools/fetch_codes.sh [DEST]; default $MHD_CODES.
set -euo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$bundle/codes/versions.env"
dest="${1:-${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-gem}}"
mkdir -p "$dest"
git_in() { (cd "$1" && shift && git "$@"); }
fetch() {  # name repo commit [--recursive]
    local dir="$dest/$1"
    if [[ ! -d "$dir/.git" ]]; then git clone "$2" "$dir"; fi
    git_in "$dir" fetch --quiet origin "$3" || git_in "$dir" fetch --quiet origin
    git_in "$dir" checkout --quiet --detach "$3"
    if [[ "${4:-}" == --recursive ]]; then git_in "$dir" submodule update --init --recursive --quiet; fi
    printf '%-9s %s\n' "$1" "$(git_in "$dir" rev-parse HEAD)"
}
fetch athenak "$ATHENAK_REPO" "$ATHENAK_COMMIT" --recursive
fetch ramses "$RAMSES_REPO" "$RAMSES_COMMIT"
fetch athena "$ATHENAPP_REPO" "$ATHENAPP_COMMIT"
printf 'Sources ready in %s. Next: bash tools/submit.sh build-gpu-codes; bash tools/submit.sh build-cpu-codes\n' "$dest"
