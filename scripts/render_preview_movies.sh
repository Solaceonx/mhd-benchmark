#!/bin/bash
# Render the short preview movies of the GPU-suite page: the tests that have no dense movie run and are shown from
# the ordinary 128^2 returns (11 to 33 saved outputs). Like render_movies_from_archives.sh, nothing unpacked is kept:
# the archives of these tests are verified, unpacked into a temporary directory outside the project, rendered to
# progress/media/movies/<test>-*.mp4 and the copy is deleted; then the page's movie list is refreshed.
#   bash scripts/render_preview_movies.sh
# MOVIE_RENDER_ARGS is passed on to render_suite_movies.py (e.g. --fields density,magnetic,vorticity).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";cd "$ROOT"
CAMPAIGN="${PREVIEW_TESTS:-alfven alfven_oblique blast_2d_b0 blast_2d_b100 brio_wu brio_wu_rotated entropy field_loop field_loop_novz}"   # sherlock_mhd/returns
RERUN="kh_localized_dtjump"                                                                                           # sherlock_rerun/returns
tmp="$(mktemp -d "${TMPDIR:-/tmp}/mhd-previews.XXXXXX")";trap 'rm -rf "$tmp"' EXIT
unpack() {      # archive directory, destination, test names: the archives named <test>-*.tar.gz
    local src="$1" dest="$2" t f;shift 2;mkdir -p "$dest"
    for t in "$@"; do
        for f in "$src/$t"-*.tar.gz; do
            [[ -e "$f" ]] || continue
            (cd "$src" && shasum -a 256 -c --quiet "$(basename "$f").sha256") || { echo "checksum failed: $f" >&2; exit 1; }
            tar -xzf "$f" -C "$dest"
        done
    done
}
# shellcheck disable=SC2086
unpack sherlock_mhd/returns "$tmp/campaign" $CAMPAIGN
# shellcheck disable=SC2086
MHD_ANALYSIS_BUNDLE=sherlock_mhd .venv-movie/bin/python scripts/render_suite_movies.py "$tmp/campaign" --grid 128 --prefix "" \
    --tests "${CAMPAIGN// /,}" ${MOVIE_RENDER_ARGS:-} | { grep -E '^movies|skipped|no GLM' || true; }
# shellcheck disable=SC2086
unpack sherlock_rerun/returns "$tmp/rerun" $RERUN
# shellcheck disable=SC2086
MHD_ANALYSIS_BUNDLE=sherlock_rerun .venv-movie/bin/python scripts/render_suite_movies.py "$tmp/rerun" --grid 128 --prefix "" \
    --tests "${RERUN// /,}" ${MOVIE_RENDER_ARGS:-} | { grep -E '^movies|skipped|no GLM' || true; }
.venv-movie/bin/python scripts/build_progress_site.py --movies-only
rm -rf scripts/__pycache__
echo "Temporary copy removed."
