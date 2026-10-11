#!/bin/bash
# Submit one Slurm job (or array). Nothing here launches the whole study; see
# `python3 codes/plan.py matrix` for every command of the full comparison.
#   bash tools/submit.sh setup | smoke | build-gpu-codes | build-cpu-codes | list | matrix
#   bash tools/submit.sh TEST [sbatch options]                  GLM variants + CT (GPU driver)
#   bash tools/submit.sh TEST --code athenak:best [sbatch ...]  external code (athenak|ramses : variant)
#   bash tools/submit.sh TEST --reference                       AthenaK high-resolution reference
#   bash tools/submit.sh TEST --reference-check                 Athena++ reference cross-check (CPU)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs returns
export MHD_BUNDLE="$PWD"
source jobs/common.sh
load_site_profile

site_options=()
if [[ "${MHD_SITE:-}" == marlowe ]]; then
    : "${MARLOWE_ACCOUNT:?Set MARLOWE_PROJECT and source site/marlowe.env before submitting}"
    site_options+=(--account="$MARLOWE_ACCOUNT" --partition="${MARLOWE_PARTITION:-preempt}")
fi
usage() {
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//' >&2
    printf 'Tests: %s\n' "$(cd configs && ls *.toml | sed 's/\.toml$//' | tr '\n' ' ')" >&2
    exit 2
}
test_name="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$test_name" in
    setup) exec sbatch --chdir="$PWD" "${site_options[@]}" --gpus=1 "$@" jobs/00_setup.sbatch ;;
    smoke) exec sbatch --chdir="$PWD" "${site_options[@]}" "$@" jobs/01_smoke.sbatch ;;
    build-gpu-codes) exec sbatch --chdir="$PWD" "${site_options[@]}" "$@" jobs/10_build_gpu_codes.sbatch ;;
    build-cpu-codes) exec sbatch --chdir="$PWD" "${site_options[@]}" "$@" jobs/11_build_cpu_codes.sbatch ;;
    list) cd configs && ls *.toml | sed 's/\.toml$//'; exit 0 ;;
    matrix) exec python3 codes/plan.py matrix ;;
    ""|*/*|.*|-*) usage ;;
esac
config="configs/$test_name.toml"
[[ -f "$config" ]] || usage
code=gpu; variant=""; mode=run; extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --code) code="${2%%:*}"; variant="${2#*:}"; shift 2 ;;
        --code=*) value="${1#--code=}"; code="${value%%:*}"; variant="${value#*:}"; shift ;;
        --reference) code=athenak; variant=reference; mode=reference; shift ;;
        --reference-check) code=athenapp; variant=reference; mode=reference; shift ;;
        *) extra+=("$1"); shift ;;
    esac
done
has_option() { printf '%s\n' "${extra[@]+"${extra[@]}"}" | grep -q -- "^$1"; }
config_value() { sed -n "s/^$1[[:space:]]*=[[:space:]]*\"\(.*\)\"/\1/p" "$config"; }
if [[ "$code" == gpu ]]; then
    tasks="$(python3 codes/plan.py count "$config" gpu run)"
    options=(--job-name="mhd-$test_name" --export="ALL,MHD_TEST=$test_name")
    job=jobs/test.sbatch
else
    [[ "$code" == athenak || "$code" == ramses || "$code" == athenapp ]] || usage
    [[ -n "$variant" && "$variant" != "$code" ]] || { printf 'Give a variant, e.g. --code %s:best\n' "$code" >&2; exit 2; }
    tasks="$(python3 codes/plan.py count "$config" "$code" "$mode")"
    read -r -a resources <<< "$(python3 codes/plan.py sbatch "$code" "$mode")"
    options=(--job-name="mhd-$test_name-$code-$variant" "${resources[@]}"
             --export="ALL,MHD_TEST=$test_name,MHD_CODE=$code,MHD_VARIANT=$variant,MHD_MODE=$mode")
    job=jobs/external.sbatch
fi
if [[ "$tasks" -eq 0 ]]; then printf '%s does not run %s in %s mode (see codes/plan.py)\n' "$code" "$test_name" "$mode" >&2; exit 2; fi
# An explicit --array/--time/--mem replaces the config-derived value; one task at a time by default.
has_option --array || options+=(--array="0-$((tasks-1))%1")
time_limit="$(config_value slurm_time)"; [[ -n "$time_limit" ]] && ! has_option --time && options+=(--time="$time_limit")
memory="$(config_value slurm_mem)"; [[ -n "$memory" ]] && ! has_option --mem && options+=(--mem="$memory")
# Arguments are passed as arrays; no eval or shell-expanded remote commands.
sbatch --chdir="$PWD" "${options[@]}" "${site_options[@]}" "${extra[@]+"${extra[@]}"}" "$job"
