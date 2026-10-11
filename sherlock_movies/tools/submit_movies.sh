#!/bin/bash
# Submit the dense-output movie runs: a 32^3 smoke check of every movie config, then one GPU-driver
# job per test and method (241 outputs; 2D at 384^2, 3D at 256^3 saving the z midplane), plus the
# AthenaK truth panel given by movie_truth (2D: PPM/HLLD/RK3 at 1536^2 block-averaged to 384^2;
# 3D: the same grid). Reuses the AthenaK build of the four-test rerun
# ($SCRATCH/mhd-codes-rerun, corrected problem header); nothing is fetched or rebuilt.
#   bash tools/submit_movies.sh [--dry-run]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JULIA_MODULE="${JULIA_MODULE:-julia/1.12.6}"
export MHD_PYTHON_MODULES="${MHD_PYTHON_MODULES:-python/3.12.1 py-numpy/2.2.6_py312}"
export MHD_GPU_RUN_MODULES="${MHD_GPU_RUN_MODULES:-devel cuda/12.4.0 gcc/12.4.0}"
export MHD_GPU_PARTITION="${MHD_GPU_PARTITION:-gpu,owners}"
feature="${MHD_GPU_FEATURE:-GPU_SKU:H100_SXM5}"
source jobs/common.sh
prepare_environment
prepare_codes
exe="$MHD_CODES/athenak/build-HOPPER90-fp64/src/athena"
[[ -x "$exe" ]] || { printf 'Missing %s: run the four-test rerun (sherlock_rerun) build first.\n' "$exe" >&2; exit 2; }
printf 'Codes: %s\nResults: %s\nPartitions: %s  Feature: %s\n' "$MHD_CODES" "$MHD_RESULTS" "$MHD_GPU_PARTITION" "$feature"
common=(--parsable "--constraint=$feature" "--partition=$MHD_GPU_PARTITION" --requeue)
tests=()
for c in configs/movie_*.toml; do tests+=("$(basename "$c" .toml)"); done
# movie_truth: athenak:reference (2D, high-resolution truth) or athenak:best|matched (3D, same grid).
truth_args() { local v; v="$(sed -n 's/^movie_truth *= *"\(.*\)"/\1/p' "configs/$1.toml")"
    case "$v" in athenak:reference) echo --reference ;; athenak:*) echo "--code $v" ;; "") ;; *) printf 'bad movie_truth %s\n' "$v" >&2; exit 2 ;; esac; }
if [[ "${1:-}" == --retry ]]; then
    # Resubmit individual parts without the smoke gate, e.g.
    #   bash tools/submit_movies.sh --retry movie_kh_uniform__glm:gpu movie_kh_localized__glm:truth
    shift;mkdir -p campaigns;ledger="campaigns/movies-retry-$(date -u +%Y%m%dT%H%M%SZ).tsv";printf 'stage\tjob_id\n' > "$ledger"
    for item in "$@"; do
        t="${item%%:*}";part="${item#*:}";[[ -f "configs/$t.toml" ]] || { printf 'No config %s\n' "$t" >&2; exit 2; }
        case "$part" in
            gpu) id="$(bash tools/submit.sh "$t" "${common[@]}" | cut -d';' -f1)" ;;
            truth) a="$(truth_args "$t")";[[ -n "$a" ]] || { printf '%s has no truth run\n' "$t" >&2; exit 2; }
                   # shellcheck disable=SC2086
                   id="$(bash tools/submit.sh "$t" $a "${common[@]}" | cut -d';' -f1)" ;;
            *) printf 'Use NAME:gpu or NAME:truth, not %s\n' "$item" >&2; exit 2 ;;
        esac
        printf '%s\t%s\n' "$item" "$id" | tee -a "$ledger"
    done
    printf 'Resubmitted. Ledger: %s\n' "$ledger";exit 0
fi
if [[ "${1:-}" == --dry-run ]]; then
    printf 'smoke, then:\n'
    for t in "${tests[@]}"; do printf '  bash tools/submit.sh %s %s --dependency=afterok:SMOKE\n' "$t" "${common[*]}"
        a="$(truth_args "$t")"; [[ -n "$a" ]] && printf '  bash tools/submit.sh %s %s %s\n' "$t" "$a" "${common[*]}"; done
    exit 0
fi
mkdir -p campaigns
ledger="campaigns/movies-$(date -u +%Y%m%dT%H%M%SZ).tsv"
printf 'stage\tjob_id\n' > "$ledger"
submit() { local label="$1"; shift; local id; id="$(bash tools/submit.sh "$@" | cut -d';' -f1)"; printf '%s\t%s\n' "$label" "$id" | tee -a "$ledger" >&2; printf '%s' "$id"; }
smoke="$(submit smoke smoke "${common[@]}" --time=01:30:00 --mem=16G)"
for t in "${tests[@]}"; do
    submit "$t:gpu" "$t" "${common[@]}" "--dependency=afterok:$smoke" >/dev/null
    # AthenaK does not depend on the GPU-driver smoke check; its build was validated by the rerun.
    a="$(truth_args "$t")"
    # shellcheck disable=SC2086
    [[ -n "$a" ]] && submit "$t:${a#--}" "$t" $a "${common[@]}" >/dev/null
done
printf 'Submitted. Ledger: %s\n' "$ledger"
