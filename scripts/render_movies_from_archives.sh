#!/bin/bash
# Render the dense-output movies from sherlock_movies/returns without keeping raw frames. One test at a
# time: verify its archives, unpack them into a temporary directory outside the project, render
# progress/media/movies/<test>-*.mp4, delete the temporary copy. Then refresh the page's movie list.
#   bash scripts/render_movies_from_archives.sh [--delete-archives]
# --delete-archives removes the downloaded movie archives once every movie rendered (the originals stay
# on Sherlock in sherlock_movies/returns until scratch is purged).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";cd "$ROOT"
src="${MOVIE_ARCHIVES:-sherlock_movies/returns}"   # overridable for testing
shopt -s nullglob
archives=("$src"/movie_*.tar.gz)
(( ${#archives[@]} )) || { echo "No movie archives in $src; download them first." >&2; exit 1; }
bases=$(for f in "${archives[@]}"; do b=$(basename "$f");b=${b#movie_};echo "${b%%__*}";done | sort -u)
[[ -n "${MOVIE_ONLY:-}" ]] && bases="$MOVIE_ONLY"      # only these tests, e.g. MOVIE_ONLY="rotor orszag_tang"
tmp=""
trap '[[ -n "$tmp" ]] && rm -rf "$tmp"' EXIT   # never leave raw frames behind, even on failure
for base in $bases; do
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/mhd-movies.XXXXXX")"
    for f in "$src"/movie_"$base"__*.tar.gz; do
        (cd "$src" && shasum -a 256 -c --quiet "$(basename "$f").sha256") || { echo "checksum failed: $f" >&2; rm -rf "$tmp"; exit 1; }
        tar -xzf "$f" -C "$tmp"
    done
    # shellcheck disable=SC2086
    .venv-movie/bin/python scripts/render_suite_movies.py "$tmp" --tests "$base" ${MOVIE_RENDER_ARGS:-} | { grep -E '^movies|skipped|no GLM' || true; }
    rm -rf "$tmp";tmp=""
done
.venv-movie/bin/python scripts/build_progress_site.py --movies-only
rm -rf scripts/__pycache__
if [[ "${1:-}" == --delete-archives ]]; then
    rm -f "$src"/movie_*.tar.gz "$src"/movie_*.tar.gz.sha256 "$src"/smoke-*.tar.gz "$src"/smoke-*.tar.gz.sha256
    echo "Deleted the downloaded movie archives."
fi
