#!/bin/bash
# Rebuild the GPU-suite page (progress/) from the verified Sherlock archives without keeping an
# unpacked copy: sherlock_mhd/returns, sherlock_rerun/returns and sherlock_gem/returns are checked and unpacked into a
# temporary directory outside the project (about 60 GB while this runs), the page is built, and the
# temporary copy is deleted. For a movie-only refresh use: build_progress_site.py --movies-only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";cd "$ROOT"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/mhd-page.XXXXXX")";trap 'rm -rf "$tmp"' EXIT
unpack() {   # movie_* archives are video runs, rendered by render_movies_from_archives.sh instead
    mkdir -p "$2"
    (cd "$1" && shopt -s nullglob && for f in *.tar.gz; do [[ "$f" == movie_* ]] && continue; shasum -a 256 -c --quiet "$f.sha256" || { echo "checksum failed: $1/$f" >&2; exit 1; }; tar -xzf "$f" -C "$2"; done)
}
echo "Unpacking archives into $tmp"
unpack sherlock_mhd/returns "$tmp/campaign"
unpack sherlock_rerun/returns "$tmp/rerun"
unpack sherlock_gem/returns "$tmp/gem"
.venv-movie/bin/python scripts/build_progress_site.py --gpu-only --gpu-returns "$tmp/campaign" --gpu-override "sherlock_rerun=$tmp/rerun" --gpu-override "sherlock_gem=$tmp/gem" | grep -v '^GPU suite'
rm -rf scripts/__pycache__
echo "Temporary copy removed."
