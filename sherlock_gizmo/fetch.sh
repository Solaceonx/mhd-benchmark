#!/bin/bash
# Get the GIZMO source at the pinned commit (run once, on a login node).
set -euo pipefail
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/gizmo"
repo=https://github.com/pfhopkins/gizmo.git; commit=519be7dccb18d6b5783a03d9ef8f7a5d0c4a0b1c   # 2026-10-04
mkdir -p "$codes"
[[ -d "$src/.git" ]] || git clone --quiet "$repo" "$src"
(cd "$src" && git checkout --quiet "$commit") || { (cd "$src" && git fetch --quiet origin && git checkout --quiet "$commit"); }   # cd: old git on the login nodes
echo "GIZMO at $(cd "$src" && git rev-parse HEAD) in $src"
