#!/bin/bash
# Submit the GEM reconnection test and its movie runs on Sherlock (gpu,owners, H100 SXM5, --requeue):
#   smoke of every config at 32^3 -> GPU driver: gem (GLM + CT at 64/128/256, 5 repeats) and
#   movie_gem__glm / movie_gem__ct (512^2, 241 outputs);
#   AthenaK build with the GEM problem header -> gem matched/best runs, gem reference (2048^2),
#   movie reference (2048^2 -> 512^2).
# Sources are cloned into $SCRATCH/mhd-codes-gem on first use; other campaigns' builds are not touched.
#   bash tools/submit_gem.sh [--dry-run]
#   bash tools/submit_gem.sh --retry NAME:gpu NAME:athenak:VARIANT NAME:reference ...
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JULIA_MODULE="${JULIA_MODULE:-julia/1.12.6}"
export MHD_PYTHON_MODULES="${MHD_PYTHON_MODULES:-python/3.12.1 py-numpy/2.2.6_py312}"
export MHD_GPU_BUILD_MODULES="${MHD_GPU_BUILD_MODULES:-devel cmake/3.31.4 cuda/12.4.0 gcc/12.4.0}"
export MHD_GPU_RUN_MODULES="${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
export MHD_GPU_PARTITION="${MHD_GPU_PARTITION:-gpu,owners}"
feature="${MHD_GPU_FEATURE:-GPU_SKU:H100_SXM5}"
source jobs/common.sh
prepare_environment
prepare_codes
case "${MHD_CODES%/}" in
    "${SCRATCH%/}/mhd-codes"|"${SCRATCH%/}/mhd-codes-rerun")
        printf 'Refusing to build into %s (used by another campaign); unset MHD_CODES.\n' "$MHD_CODES" >&2; exit 2 ;;
esac
printf 'Codes: %s\nResults: %s\nPartitions: %s  Feature: %s\n' "$MHD_CODES" "$MHD_RESULTS" "$MHD_GPU_PARTITION" "$feature"
common=(--parsable "--constraint=$feature" "--partition=$MHD_GPU_PARTITION" --requeue)
mkdir -p campaigns
ledger="campaigns/gem-$(date -u +%Y%m%dT%H%M%SZ).tsv"
submit() { local label="$1"; shift; local id; id="$(bash tools/submit.sh "$@" | cut -d';' -f1)"; printf '%s\t%s\n' "$label" "$id" | tee -a "$ledger" >&2; printf '%s' "$id"; }
part_args() {   # gpu | athenak:matched | athenak:best | reference
    case "$1" in gpu) ;; reference) echo --reference ;; athenak:*) echo "--code $1" ;; *) printf 'bad part %s\n' "$1" >&2; exit 2 ;; esac; }
if [[ "${1:-}" == --retry ]]; then
    shift;printf 'stage\tjob_id\n' > "$ledger"
    for item in "$@"; do t="${item%%:*}";p="${item#*:}";[[ -f "configs/$t.toml" ]] || { printf 'No config %s\n' "$t" >&2; exit 2; }
        # shellcheck disable=SC2046
        submit "$item" "$t" $(part_args "$p") "${common[@]}" >/dev/null; done
    printf 'Resubmitted. Ledger: %s\n' "$ledger";exit 0
fi
parts=("gem:gpu" "gem:athenak:matched" "gem:athenak:best" "gem:reference" "movie_gem__glm:gpu" "movie_gem__ct:gpu" "movie_gem__glm:reference")
if [[ "${1:-}" == --dry-run ]]; then printf 'smoke, build-gpu-codes, then: %s\n' "${parts[*]}";exit 0;fi
[[ -d "$MHD_CODES/athenak/.git" ]] || bash tools/fetch_codes.sh
printf 'stage\tjob_id\n' > "$ledger"
smoke="$(submit smoke smoke "${common[@]}" --time=01:00:00 --mem=16G)"
build="$(submit build-gpu build-gpu-codes "${common[@]}" --time=02:00:00)"
for item in "${parts[@]}"; do
    t="${item%%:*}";p="${item#*:}"
    if [[ "$p" == gpu ]]; then dep="$smoke"; else dep="$build"; fi
    # shellcheck disable=SC2046
    submit "$item" "$t" $(part_args "$p") "${common[@]}" "--dependency=afterok:$dep" >/dev/null
done
printf 'Submitted. Ledger: %s\n' "$ledger"
