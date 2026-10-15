#!/bin/bash
# Run on the Mac: upload an isolated bundle, then submit Mach 30/50 comparison movies.
# Original solver sources and original result trees are not changed.
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --exclude='__pycache__' --exclude='logs' \
    -czf - sherlock_turb | ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu 'bash -l -c '\''set -euo pipefail
bundle=$(mktemp -d /scratch/users/lyxia/mhd-study/turb-movie-compare.XXXXXX)
tar -xzf - -C "$bundle"
printf "Isolated movie bundle: %s\n" "$bundle"
bash -l "$bundle/sherlock_turb/compare_movies.sh" --submit'\'''
